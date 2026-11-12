//! Persistent pinned worker pool with spin-wait barriers.
//!
//! Each participating thread has a stable id in `0..nthreads` (the calling
//! thread is always 0). Row ranges are a deterministic split of `[0, n)` by
//! id, so a thread keeps seeing the same slices of every weight matrix and
//! those slices can stay in its L2. One barrier is a few microseconds;
//! rayon fork-join per matmul is not.
//!
//! Session width is process-wide: `with_threads` also installs a persistent
//! rayon pool of the same width so leftover fp32 kernels that still use
//! `par_iter` honor the thread budget, and [`current_threads`] stays correct
//! even when that install runs the closure on a different thread.

use rayon::ThreadPool;
use std::collections::HashMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, Thread};
use std::time::Duration;

const MAX_THREADS: usize = 64;

/// Work published to the workers: `func(ctx, tid, nthreads)`.
struct Job {
    func: unsafe fn(*const (), usize, usize),
    ctx: *const (),
}

// The job pointer is only read between a Release store of `job_id` and the
// matching arrival barrier, and only the owning thread writes it.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Pool {
    job: std::cell::UnsafeCell<Job>,
    job_id: AtomicU64,
    arrived: AtomicU32,
    n_active: AtomicU32,
    session: AtomicBool,
    stop: AtomicBool,
    panicked: AtomicBool,
    workers: Mutex<Vec<(thread::JoinHandle<()>, Thread)>>,
    n_spawned: AtomicU32,
}

unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

fn global_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool {
        job: std::cell::UnsafeCell::new(Job {
            func: noop,
            ctx: std::ptr::null(),
        }),
        job_id: AtomicU64::new(0),
        arrived: AtomicU32::new(0),
        n_active: AtomicU32::new(1),
        session: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        panicked: AtomicBool::new(false),
        workers: Mutex::new(Vec::new()),
        n_spawned: AtomicU32::new(0),
    })
}

unsafe fn noop(_ctx: *const (), _tid: usize, _n: usize) {}

impl Pool {
    fn ensure_workers(&'static self, n: usize) {
        let want = n.clamp(1, MAX_THREADS);
        let mut guard = self.workers.lock().expect("worker list");
        while guard.len() + 1 < want {
            let tid = guard.len() + 1;
            let pool = self;
            let handle = thread::Builder::new()
                .name(format!("llm-cpu-{tid}"))
                .spawn(move || worker_loop(tid, pool))
                .expect("spawn worker");
            let th = handle.thread().clone();
            guard.push((handle, th));
        }
        self.n_spawned.store(guard.len() as u32, Ordering::Release);
    }

    fn enter(&'static self, n: usize) {
        let n = n.clamp(1, MAX_THREADS);
        self.ensure_workers(n);
        self.n_active.store(n as u32, Ordering::Relaxed);
        self.session.store(true, Ordering::Release);
        pin_thread(0);
        if let Ok(ws) = self.workers.lock() {
            for (_, th) in ws.iter() {
                th.unpark();
            }
        }
    }

    fn leave(&self) {
        self.session.store(false, Ordering::Release);
    }

    fn dispatch(&self, n: usize, func: unsafe fn(*const (), usize, usize), ctx: *const ()) {
        if n <= 1 {
            unsafe { func(ctx, 0, 1) };
            return;
        }
        // SAFETY: workers only read `job` after observing a new `job_id` and
        // the caller waits for every participant before returning (and thus
        // before `ctx` can go out of scope).
        unsafe {
            *self.job.get() = Job { func, ctx };
        }
        self.arrived.store(0, Ordering::Relaxed);
        self.panicked.store(false, Ordering::Relaxed);
        self.n_active.store(n as u32, Ordering::Relaxed);
        self.job_id.fetch_add(1, Ordering::Release);
        let result = catch_unwind(AssertUnwindSafe(|| unsafe { func(ctx, 0, n) }));
        if result.is_err() {
            self.panicked.store(true, Ordering::Release);
        }
        self.arrived.fetch_add(1, Ordering::Release);
        while self.arrived.load(Ordering::Acquire) < n as u32 {
            std::hint::spin_loop();
        }
        if let Err(payload) = result {
            resume_unwind(payload);
        }
        if self.panicked.load(Ordering::Acquire) {
            panic!("worker panicked in parallel region");
        }
    }
}

fn worker_loop(tid: usize, pool: &'static Pool) {
    pin_thread(tid);
    let mut seen = 0u64;
    loop {
        if pool.stop.load(Ordering::Acquire) {
            return;
        }
        let job = pool.job_id.load(Ordering::Acquire);
        if job != seen && job != 0 {
            seen = job;
            let n = pool.n_active.load(Ordering::Acquire) as usize;
            // Only participating threads enter the arrival count. Extra
            // workers left over from a wider session must not signal, or
            // the caller can return before the real participants finish.
            if tid < n {
                // SAFETY: `job` was published with Release on `job_id`; the
                // pointed-to `ctx` is live until `arrived` hits `n`.
                let result = catch_unwind(AssertUnwindSafe(|| unsafe {
                    let j = &*pool.job.get();
                    (j.func)(j.ctx, tid, n);
                }));
                if result.is_err() {
                    pool.panicked.store(true, Ordering::Release);
                }
                pool.arrived.fetch_add(1, Ordering::Release);
            }
            continue;
        }
        if !pool.session.load(Ordering::Acquire) {
            thread::park_timeout(Duration::from_millis(50));
            continue;
        }
        std::hint::spin_loop();
    }
}

/// Pin the current thread to `cpu` (modulo the affinity width we can see).
fn pin_thread(cpu: usize) {
    #[cfg(target_os = "linux")]
    {
        const CPUSET_BITS: usize = 1024;
        const LONGS: usize = CPUSET_BITS / 64;
        #[repr(C)]
        struct CpuSet {
            bits: [u64; LONGS],
        }
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
        }
        let ncpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let cpu = cpu % ncpu;
        let mut set = CpuSet { bits: [0; LONGS] };
        set.bits[cpu / 64] |= 1u64 << (cpu % 64);
        // SAFETY: `CpuSet` matches the libc `cpu_set_t` layout for a 1024-CPU
        // mask; pid 0 means the calling thread.
        unsafe {
            sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set);
        }
    }
    let _ = cpu;
}

/// Process-wide session width. Must not be thread-local: `with_threads`
/// installs a rayon pool, and the closure (and therefore every
/// [`current_threads`] check that sizes a dispatch) may run on a worker.
static SESSION_N: AtomicUsize = AtomicUsize::new(1);

/// Number of threads in the current [`with_threads`] session (1 if none).
#[must_use]
pub fn current_threads() -> usize {
    SESSION_N.load(Ordering::Acquire).max(1)
}

/// Contiguous `[start, end)` row (or tile) range owned by `tid`.
#[must_use]
pub fn thread_range(tid: usize, nthreads: usize, n: usize) -> (usize, usize) {
    if nthreads == 0 {
        return (0, 0);
    }
    let start = tid * n / nthreads;
    let end = (tid + 1) * n / nthreads;
    (start, end)
}

/// Run `f(tid, start, end)` on every session thread for a split of `n_items`.
pub fn for_ranges<F: Fn(usize, usize, usize) + Sync>(n_items: usize, f: F) {
    let n = current_threads();
    if n <= 1 || n_items <= 1 {
        f(0, 0, n_items);
        return;
    }
    struct Ctx<F> {
        f: F,
        n_items: usize,
    }
    unsafe fn shim<F: Fn(usize, usize, usize)>(ctx: *const (), tid: usize, nthreads: usize) {
        let ctx = unsafe { &*(ctx as *const Ctx<F>) };
        let (start, end) = thread_range(tid, nthreads, ctx.n_items);
        (ctx.f)(tid, start, end);
    }
    let ctx = Ctx { f, n_items };
    global_pool().dispatch(n, shim::<F>, std::ptr::from_ref(&ctx) as *const ());
}

/// Run `f(tid, nthreads)` on every session thread, including the caller as 0.
pub fn for_threads<F: Fn(usize, usize) + Sync>(f: F) {
    let n = current_threads();
    if n <= 1 {
        f(0, 1);
        return;
    }
    struct Ctx<F> {
        f: F,
    }
    unsafe fn shim<F: Fn(usize, usize)>(ctx: *const (), tid: usize, nthreads: usize) {
        let ctx = unsafe { &*(ctx as *const Ctx<F>) };
        (ctx.f)(tid, nthreads);
    }
    let ctx = Ctx { f };
    global_pool().dispatch(n, shim::<F>, std::ptr::from_ref(&ctx) as *const ());
}

fn rayon_pools() -> &'static Mutex<HashMap<usize, &'static ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, &'static ThreadPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn persistent_rayon(threads: usize) -> &'static ThreadPool {
    let n = threads.max(1);
    let mut map = rayon_pools().lock().expect("rayon pool map");
    map.entry(n).or_insert_with(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon thread pool");
        Box::leak(Box::new(pool))
    })
}

fn session_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Run `f` with a persistent `threads`-wide session (pinned spin pool + rayon).
///
/// Only one session can be active in the process: the pool and the session
/// width are process-wide. Concurrent callers are serialized.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    let _session = session_lock().lock().expect("session lock");
    let n = threads.max(1);
    let prev = SESSION_N.swap(n, Ordering::AcqRel);
    global_pool().enter(n);
    let r = persistent_rayon(n).install(|| {
        pin_thread(0);
        f()
    });
    global_pool().leave();
    SESSION_N.store(prev.max(1), Ordering::Release);
    r
}

/// Touch every page of `bytes` so later reads do not pay first-touch faults.
pub fn prefault_bytes(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        const MADV_WILLNEED: i32 = 3;
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        // Best-effort; the touch loop below is the reliable pre-fault.
        unsafe {
            madvise(
                bytes.as_ptr() as *mut std::ffi::c_void,
                bytes.len(),
                MADV_WILLNEED,
            );
        }
    }
    let mut acc = 0u8;
    let mut i = 0usize;
    while i < bytes.len() {
        acc ^= bytes[i];
        i += 4096;
    }
    acc ^= bytes[bytes.len() - 1];
    std::hint::black_box(acc);
}

/// Allocate `len` bytes and pre-fault every page (touch, plus `MADV_WILLNEED`
/// on Linux so the kernel can start paging in).
#[must_use]
pub fn alloc_prefaulted(len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let v = vec![0u8; len];
    prefault_bytes(&v);
    v
}

/// Touch every page of an `f32` buffer.
pub fn prefault_f32(data: &[f32]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
    };
    prefault_bytes(bytes);
}

#[cfg(test)]
mod tests {
    use super::{current_threads, for_ranges, thread_range, with_threads};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn ranges_cover_without_overlap() {
        for nthreads in 1..=8 {
            for n in [0usize, 1, 7, 16, 17, 65536] {
                let mut seen = vec![0u8; n];
                for tid in 0..nthreads {
                    let (s, e) = thread_range(tid, nthreads, n);
                    for slot in seen.iter_mut().take(e).skip(s) {
                        *slot += 1;
                    }
                }
                assert!(seen.iter().all(|&c| c == 1), "n={n} t={nthreads}");
            }
        }
    }

    #[test]
    fn for_ranges_fills() {
        with_threads(4, || {
            let buf: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
            for_ranges(64, |_tid, start, end| {
                for (i, slot) in buf.iter().enumerate().take(end).skip(start) {
                    slot.store(i + 1, Ordering::Relaxed);
                }
            });
            for (i, slot) in buf.iter().enumerate() {
                assert_eq!(slot.load(Ordering::Relaxed), i + 1);
            }
        });
    }

    #[test]
    fn current_threads_is_process_wide() {
        // `with_threads` goes through rayon::install, which may run this
        // closure on a pool thread. Session width must still be 4.
        with_threads(4, || {
            assert_eq!(current_threads(), 4);
            let seen = AtomicUsize::new(0);
            for_ranges(32, |_tid, start, end| {
                assert_eq!(current_threads(), 4);
                seen.fetch_add(end - start, Ordering::Relaxed);
            });
            assert_eq!(seen.load(Ordering::Relaxed), 32);
        });
        assert_eq!(current_threads(), 1);
    }

    #[test]
    fn narrower_session_after_wider_still_covers() {
        // Spawn the 8-wide worker set, then dispatch a 4-wide job. Only the
        // four participants may count toward the barrier.
        with_threads(8, || {
            let hit = AtomicUsize::new(0);
            for_ranges(16, |_tid, start, end| {
                hit.fetch_add(end.saturating_sub(start), Ordering::Relaxed);
            });
            assert_eq!(hit.load(Ordering::Relaxed), 16);
        });
        with_threads(4, || {
            assert_eq!(current_threads(), 4);
            let buf: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
            for_ranges(64, |_tid, start, end| {
                for (i, slot) in buf.iter().enumerate().take(end).skip(start) {
                    slot.store(i + 1, Ordering::Relaxed);
                }
            });
            for (i, slot) in buf.iter().enumerate() {
                assert_eq!(slot.load(Ordering::Relaxed), i + 1);
            }
        });
    }
}

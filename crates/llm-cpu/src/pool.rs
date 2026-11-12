//! Persistent rayon pools so decode GEMV does not spawn a pool per call.

use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn pools() -> &'static Mutex<HashMap<usize, &'static ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, &'static ThreadPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A process-lifetime pool with exactly `threads` workers, created on first use.
pub fn persistent_pool(threads: usize) -> &'static ThreadPool {
    let n = threads.max(1);
    let mut map = pools().lock().expect("thread-pool map");
    map.entry(n).or_insert_with(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon thread pool");
        Box::leak(Box::new(pool))
    })
}

/// Run `f` on a persistent pool with exactly `threads` workers.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    persistent_pool(threads).install(f)
}

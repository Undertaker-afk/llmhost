//! AMX-INT8 prefill tiles. Configured at runtime with inline asm because the
//! `x86_amx_intrinsics` feature is still unstable.
//!
//! Tile palette (matches the usual 16×32 × 32×16 int8 pattern):
//! - TMM0/TMM1 (B): 8 rows × 64 bytes, VNNI `{K/4, N*4}`
//! - TMM2/TMM3 (A): 16 rows × 32 bytes
//! - TMM4–TMM7 (C): 16 rows × 64 bytes (16 i32 columns)
//!
//! The kernel is blocked 2×2 (32 tokens × 32 output columns) so one K-block
//! of pre-packed B is reused across two A tiles. Each K-block still has its
//! own scale, so the int32 tiles are stored and folded into an f32
//! accumulator before the next K. Permission for XTILEDATA is requested once
//! via `arch_prctl`; `ldtilecfg` is issued on each participating thread.

#![allow(clippy::too_many_arguments)]

use crate::gemv_q4::{ActQ8Batch, Q4Matrix, TILE_N, TILE_STRIDE};
use llm_quant::BLOCK_SIZE;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Bytes in one pre-packed AMX B panel (8 rows × 64 bytes).
pub const AMX_B_STRIDE: usize = 8 * 64;
const TILE_QS: usize = TILE_N * BLOCK_SIZE / 2;

static PERM_OK: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 ok, 2 fail
static FEATURE: AtomicBool = AtomicBool::new(false);

/// True when the CPU, OS, and this process can run AMX-INT8 tiles.
#[must_use]
pub fn amx_ready() -> bool {
    if PERM_OK.load(Ordering::Acquire) == 0 {
        probe();
    }
    PERM_OK.load(Ordering::Acquire) == 1
}

fn probe() {
    let cpu = crate::features::CpuFeatures::detect().amx_int8;
    FEATURE.store(cpu, Ordering::Release);
    if !cpu {
        PERM_OK.store(2, Ordering::Release);
        return;
    }
    if request_xcomp_perm() {
        PERM_OK.store(1, Ordering::Release);
    } else {
        PERM_OK.store(2, Ordering::Release);
    }
}

fn request_xcomp_perm() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        const SYS_ARCH_PRCTL: i64 = 158;
        const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
        const XFEATURE_XTILEDATA: usize = 18;
        extern "C" {
            fn syscall(n: i64, ...) -> i64;
        }
        // SAFETY: arch_prctl(REQ_XCOMP_PERM, XTILEDATA) is the documented
        // Linux interface to enable AMX state in this process.
        let rc = unsafe { syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) };
        rc == 0
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        false
    }
}

#[repr(C, align(64))]
struct TileCfg {
    palette_id: u8,
    start_row: u8,
    reserved0: [u8; 14],
    colsb: [u16; 16],
    rows: [u8; 16],
}

fn default_cfg() -> TileCfg {
    let mut cfg = TileCfg {
        palette_id: 1,
        start_row: 0,
        reserved0: [0; 14],
        colsb: [0; 16],
        rows: [0; 16],
    };
    // TMM0, TMM1: B 8×64
    cfg.rows[0] = 8;
    cfg.colsb[0] = 64;
    cfg.rows[1] = 8;
    cfg.colsb[1] = 64;
    // TMM2, TMM3: A 16×32
    cfg.rows[2] = 16;
    cfg.colsb[2] = 32;
    cfg.rows[3] = 16;
    cfg.colsb[3] = 32;
    // TMM4–TMM7: C 16×64 (16 i32)
    for t in 4..8 {
        cfg.rows[t] = 16;
        cfg.colsb[t] = 64;
    }
    cfg
}

thread_local! {
    static CFG_LOADED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Load the AMX tile config on this thread if AMX is available.
pub fn ensure_amx_thread() -> bool {
    if !amx_ready() {
        return false;
    }
    if CFG_LOADED.with(|c| c.get()) {
        return true;
    }
    let cfg = default_cfg();
    // SAFETY: `cfg` is a 64-byte aligned tileconfig; AMX is enabled.
    unsafe { ldtilecfg(&cfg) };
    CFG_LOADED.with(|c| c.set(true));
    true
}

#[inline]
unsafe fn ldtilecfg(cfg: &TileCfg) {
    let ptr = std::ptr::from_ref(cfg);
    core::arch::asm!(
        "ldtilecfg [{cfg}]",
        cfg = in(reg) ptr,
        options(nostack)
    );
}

#[inline]
unsafe fn tilezero(tmm: u8) {
    match tmm {
        4 => core::arch::asm!("tilezero tmm4", options(nostack)),
        5 => core::arch::asm!("tilezero tmm5", options(nostack)),
        6 => core::arch::asm!("tilezero tmm6", options(nostack)),
        7 => core::arch::asm!("tilezero tmm7", options(nostack)),
        _ => debug_assert!(false, "unexpected tilezero tmm{tmm}"),
    }
}

#[inline]
unsafe fn tileloadd_a(tmm: u8, base: *const u8, stride: usize) {
    match tmm {
        2 => core::arch::asm!(
            "tileloadd tmm2, [{base} + {stride}]",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        3 => core::arch::asm!(
            "tileloadd tmm3, [{base} + {stride}]",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        _ => debug_assert!(false, "unexpected A tile tmm{tmm}"),
    }
}

#[inline]
unsafe fn tileloadd_b(tmm: u8, base: *const u8, stride: usize) {
    match tmm {
        0 => core::arch::asm!(
            "tileloadd tmm0, [{base} + {stride}]",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        1 => core::arch::asm!(
            "tileloadd tmm1, [{base} + {stride}]",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        _ => debug_assert!(false, "unexpected B tile tmm{tmm}"),
    }
}

#[inline]
unsafe fn tdpbssd(c: u8, a: u8, b: u8) {
    match (c, a, b) {
        (4, 2, 0) => core::arch::asm!("tdpbssd tmm4, tmm2, tmm0", options(nostack)),
        (5, 2, 1) => core::arch::asm!("tdpbssd tmm5, tmm2, tmm1", options(nostack)),
        (6, 3, 0) => core::arch::asm!("tdpbssd tmm6, tmm3, tmm0", options(nostack)),
        (7, 3, 1) => core::arch::asm!("tdpbssd tmm7, tmm3, tmm1", options(nostack)),
        (4, 3, 0) => core::arch::asm!("tdpbssd tmm4, tmm3, tmm0", options(nostack)),
        (5, 3, 1) => core::arch::asm!("tdpbssd tmm5, tmm3, tmm1", options(nostack)),
        _ => debug_assert!(false, "unexpected tdpbssd tmm{c}, tmm{a}, tmm{b}"),
    }
}

#[inline]
unsafe fn tilestored(tmm: u8, base: *mut u8, stride: usize) {
    match tmm {
        4 => core::arch::asm!(
            "tilestored [{base} + {stride}], tmm4",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        5 => core::arch::asm!(
            "tilestored [{base} + {stride}], tmm5",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        6 => core::arch::asm!(
            "tilestored [{base} + {stride}], tmm6",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        7 => core::arch::asm!(
            "tilestored [{base} + {stride}], tmm7",
            base = in(reg) base,
            stride = in(reg) stride,
            options(nostack)
        ),
        _ => debug_assert!(false, "unexpected tilestored tmm{tmm}"),
    }
}

/// Unpack one Q4 VNNI nibble panel into the AMX B layout (signed i8, 8×64).
pub fn pack_amx_b_panel(panel: &[u8], dst: &mut [i8]) {
    debug_assert!(dst.len() >= AMX_B_STRIDE);
    debug_assert!(panel.len() >= TILE_QS);
    for pair in 0..4 {
        let src = &panel[pair * 64..pair * 64 + 64];
        let start = pair * 2 * 64;
        let pair_rows = &mut dst[start..start + 128];
        let (r0, r1) = pair_rows.split_at_mut(64);
        for i in 0..64 {
            let b = src[i];
            r0[i] = (b & 0x0f) as i8 - 8;
            r1[i] = (b >> 4) as i8 - 8;
        }
    }
}

fn load_scales(panel: &[u8], sw: &mut [f32; TILE_N]) {
    let src = &panel[TILE_QS..TILE_QS + TILE_N * 4];
    for col in 0..TILE_N {
        sw[col] = f32::from_le_bytes(src[col * 4..col * 4 + 4].try_into().unwrap());
    }
}

fn pack_a_block(acts: &ActQ8Batch, kb: usize, m: usize, a_pack: &mut [i8]) {
    let k = acts.k;
    a_pack.fill(0);
    for t in 0..m {
        let src = unsafe { acts.qs.as_ptr().add(t * k + kb * BLOCK_SIZE) };
        unsafe {
            std::ptr::copy_nonoverlapping(src, a_pack.as_mut_ptr().add(t * 32), 32);
        }
    }
}

fn fold_c(
    c: &[i32; 16 * 16],
    acc: &mut [f32],
    acc_stride: usize,
    tok: usize,
    rows: usize,
    cols: usize,
    col_off: usize,
    sw: &[f32; TILE_N],
    scales: &[f32],
    scale_base: usize,
    bpr: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cols == TILE_N && is_x86_feature_detected!("avx512f") {
            unsafe {
                fold_c_avx512(
                    c, acc, acc_stride, tok, rows, col_off, sw, scales, scale_base, bpr,
                );
            }
            return;
        }
    }
    for r in 0..rows {
        let sa = scales[scale_base + (tok + r) * bpr];
        let dst = &mut acc[(tok + r) * acc_stride..];
        for col in 0..cols {
            dst[col_off + col] += c[r * TILE_N + col] as f32 * sw[col] * sa;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn fold_c_avx512(
    c: &[i32; 16 * 16],
    acc: &mut [f32],
    acc_stride: usize,
    tok: usize,
    rows: usize,
    col_off: usize,
    sw: &[f32; TILE_N],
    scales: &[f32],
    scale_base: usize,
    bpr: usize,
) {
    use std::arch::x86_64::*;
    let vsw = _mm512_loadu_ps(sw.as_ptr());
    for r in 0..rows {
        let sa = *scales.get_unchecked(scale_base + (tok + r) * bpr);
        let dst = acc.as_mut_ptr().add((tok + r) * acc_stride + col_off);
        let vacc = _mm512_loadu_ps(dst);
        let vc = _mm512_cvtepi32_ps(_mm512_loadu_si512(c.as_ptr().add(r * TILE_N).cast()));
        let vacc = _mm512_fmadd_ps(vc, _mm512_mul_ps(vsw, _mm512_set1_ps(sa)), vacc);
        _mm512_storeu_ps(dst, vacc);
    }
}

/// AMX GEMM over packed Q4 tiles `[t0, t1)` and a batch of `m` tokens.
///
/// Loop order is K then N-tiles so each packed A K-block is reused across
/// every output-tile pair. Accumulators and the A pack are allocated once.
///
/// # Safety
/// `out` must be `m * n` f32s; tiles in range must be valid.
pub unsafe fn gemm_tiles_amx(
    w: &Q4Matrix,
    acts: &ActQ8Batch,
    out: *mut f32,
    n: usize,
    m: usize,
    t0: usize,
    t1: usize,
) {
    if !ensure_amx_thread() {
        return;
    }
    if t0 >= t1 || m == 0 {
        return;
    }
    let bpr = w.blocks_per_row();
    let n_col_tiles = t1 - t0;
    let acc_stride = n_col_tiles * TILE_N;
    let mut acc = vec![0.0f32; m * acc_stride];
    let m_pad = m.next_multiple_of(16);
    let mut a_pack = vec![0i8; m_pad * 32];

    let mut c00 = [0i32; 16 * 16];
    let mut c01 = [0i32; 16 * 16];
    let mut c10 = [0i32; 16 * 16];
    let mut c11 = [0i32; 16 * 16];
    let mut b_tmp0 = [0i8; AMX_B_STRIDE];
    let mut b_tmp1 = [0i8; AMX_B_STRIDE];
    let mut sw0 = [0.0f32; TILE_N];
    let mut sw1 = [0.0f32; TILE_N];

    for kb in 0..bpr {
        pack_a_block(acts, kb, m, &mut a_pack);
        let mut tile = t0;
        let mut col_base = 0usize;
        while tile < t1 {
            let two_n = tile + 1 < t1;
            let row0 = tile * TILE_N;
            let cols0 = TILE_N.min(n.saturating_sub(row0));
            let cols1 = if two_n {
                TILE_N.min(n.saturating_sub(row0 + TILE_N))
            } else {
                0
            };
            let panel0 = w.tile(tile, kb);
            load_scales(panel0, &mut sw0);
            if two_n {
                load_scales(w.tile(tile + 1, kb), &mut sw1);
            }
            let b0_ptr = if let Some(p) = w.amx_b_panel(tile, kb) {
                p.as_ptr()
            } else {
                pack_amx_b_panel(panel0, &mut b_tmp0);
                b_tmp0.as_ptr()
            };
            let b1_ptr = if two_n {
                if let Some(p) = w.amx_b_panel(tile + 1, kb) {
                    p.as_ptr()
                } else {
                    pack_amx_b_panel(w.tile(tile + 1, kb), &mut b_tmp1);
                    b_tmp1.as_ptr()
                }
            } else {
                std::ptr::null()
            };

            // SAFETY: B panels are 8×64 signed i8; AMX is configured.
            unsafe {
                tileloadd_b(0, b0_ptr.cast(), 64);
                if two_n {
                    tileloadd_b(1, b1_ptr.cast(), 64);
                }
            }

            let mut tok = 0usize;
            while tok < m {
                let remain = m - tok;
                let rows0 = remain.min(16);
                let rows1 = remain.saturating_sub(16).min(16);
                let a0_ptr = a_pack.as_ptr().add(tok * 32).cast::<u8>();
                unsafe {
                    tilezero(4);
                    if two_n {
                        tilezero(5);
                    }
                    tileloadd_a(2, a0_ptr, 32);
                    tdpbssd(4, 2, 0);
                    if two_n {
                        tdpbssd(5, 2, 1);
                    }
                    tilestored(4, c00.as_mut_ptr().cast(), 16 * 4);
                    if two_n {
                        tilestored(5, c01.as_mut_ptr().cast(), 16 * 4);
                    }
                }
                fold_c(
                    &c00,
                    &mut acc,
                    acc_stride,
                    tok,
                    rows0,
                    cols0,
                    col_base,
                    &sw0,
                    &acts.scales,
                    kb,
                    bpr,
                );
                if two_n {
                    fold_c(
                        &c01,
                        &mut acc,
                        acc_stride,
                        tok,
                        rows0,
                        cols1,
                        col_base + TILE_N,
                        &sw1,
                        &acts.scales,
                        kb,
                        bpr,
                    );
                }
                if rows1 > 0 {
                    let a1_ptr = a_pack.as_ptr().add((tok + 16) * 32).cast::<u8>();
                    unsafe {
                        tilezero(6);
                        if two_n {
                            tilezero(7);
                        }
                        tileloadd_a(3, a1_ptr, 32);
                        tdpbssd(6, 3, 0);
                        if two_n {
                            tdpbssd(7, 3, 1);
                        }
                        tilestored(6, c10.as_mut_ptr().cast(), 16 * 4);
                        if two_n {
                            tilestored(7, c11.as_mut_ptr().cast(), 16 * 4);
                        }
                    }
                    fold_c(
                        &c10,
                        &mut acc,
                        acc_stride,
                        tok + 16,
                        rows1,
                        cols0,
                        col_base,
                        &sw0,
                        &acts.scales,
                        kb,
                        bpr,
                    );
                    if two_n {
                        fold_c(
                            &c11,
                            &mut acc,
                            acc_stride,
                            tok + 16,
                            rows1,
                            cols1,
                            col_base + TILE_N,
                            &sw1,
                            &acts.scales,
                            kb,
                            bpr,
                        );
                    }
                }
                tok += 32;
            }
            col_base += if two_n { 32 } else { TILE_N };
            tile += if two_n { 2 } else { 1 };
        }
    }

    for tile_i in 0..n_col_tiles {
        let row0 = (t0 + tile_i) * TILE_N;
        let cols = TILE_N.min(n.saturating_sub(row0));
        let col_off = tile_i * TILE_N;
        for tok in 0..m {
            for col in 0..cols {
                unsafe {
                    *out.add(tok * n + row0 + col) = acc[tok * acc_stride + col_off + col];
                }
            }
        }
    }
    let _ = TILE_STRIDE;
}

#[cfg(test)]
mod tests {
    use super::amx_ready;
    use crate::gemv_q4::{gemm_q4, gemv_q4, Q4Matrix};
    use crate::pool::with_threads;

    #[test]
    fn amx_gemm_matches_gemv_when_available() {
        if !amx_ready() {
            return;
        }
        let n = 48;
        let k = 64;
        let m = 40;
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as i32 * 3) % 17 - 8) as f32 * 0.1)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as i32 * 5) % 11 - 5) as f32 * 0.2)
            .collect();
        let mat = Q4Matrix::from_f32(&w, n, k);
        let mut gemm = vec![0.0f32; m * n];
        let mut gemv = vec![0.0f32; m * n];
        with_threads(2, || {
            gemm_q4(&mat, &x, &mut gemm, m);
            for t in 0..m {
                gemv_q4(&mat, &x[t * k..(t + 1) * k], &mut gemv[t * n..(t + 1) * n]);
            }
        });
        for i in 0..m * n {
            assert!(
                (gemm[i] - gemv[i]).abs() <= 1e-4 * (1.0 + gemv[i].abs()),
                "elem {i}: amx/gemm={} gemv={}",
                gemm[i],
                gemv[i]
            );
        }
    }
}

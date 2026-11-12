//! Q4_32 × Q8 GEMV / GEMM for decode and prefill.
//!
//! Weights are packed at load into 16-row VNNI tiles: for each K-block of 32,
//! 16 output rows share a 256-byte nibble panel (groups of 4 K, two groups
//! packed as low/high nibbles) plus 16 pre-converted f32 scales. Decode GEMV
//! and prefill GEMM both stream those tiles. A persistent pinned pool splits
//! tiles by thread id so each worker keeps the same row ranges in L2.
//!
//! AVX-512 VNNI uses `vpdpbusd` on unsigned nibbles `0..=15` with compensation
//! `sum((nibble-8)*a) = dpbusd(nibble, a) - 8*sum(a)`. AVX2 and scalar paths
//! use signed nibbles and must agree with VNNI bit-for-bit on the integer dot.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::pool::{alloc_prefaulted, for_ranges, prefault_bytes, thread_range};
use llm_quant::{quantize_q4, BlockQ4, BLOCK_SIZE};

/// Columns (output rows) in a packed weight tile.
pub const TILE_N: usize = 16;
/// Packed nibble bytes per tile per K-block (16 rows × 32 nibbles / 2).
const TILE_QS: usize = TILE_N * BLOCK_SIZE / 2;
/// Bytes per tile: nibble panel + 16 f32 scales. 320 is a multiple of 64.
pub const TILE_STRIDE: usize = TILE_QS + TILE_N * 4;

/// Quantized activation vector: one Q8-style block per 32 elements.
#[derive(Clone, Debug, Default)]
pub struct ActQ8 {
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
}

impl ActQ8 {
    /// Quantize `x` (length multiple of 32) into per-block signed int8.
    #[must_use]
    pub fn quantize(x: &[f32]) -> Self {
        let mut a = Self::default();
        a.quantize_into(x);
        a
    }

    /// Per-row activation quant: one scale for the whole vector, still stored
    /// per 32-element block (so existing kernels can consume it).
    #[must_use]
    pub fn quantize_row(x: &[f32]) -> Self {
        let mut a = Self::default();
        a.quantize_row_into(x);
        a
    }

    /// Reuse existing buffers when possible.
    pub fn quantize_into(&mut self, x: &[f32]) {
        quantize_blocks(x, self, false);
    }

    /// Like [`quantize_into`] but a single scale shared by every block.
    pub fn quantize_row_into(&mut self, x: &[f32]) {
        quantize_blocks(x, self, true);
    }
}

fn quantize_blocks(x: &[f32], dst: &mut ActQ8, per_row: bool) {
    assert!(
        x.len().is_multiple_of(BLOCK_SIZE),
        "activation length {} is not a multiple of {BLOCK_SIZE}",
        x.len()
    );
    let n_blocks = x.len() / BLOCK_SIZE;
    dst.qs.resize(x.len(), 0);
    dst.scales.resize(n_blocks, 0.0);
    dst.sums.resize(n_blocks, 0);
    let row_scale = if per_row {
        x.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0
    } else {
        0.0
    };
    for b in 0..n_blocks {
        let src = &x[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let qs = &mut dst.qs[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let scale = if per_row {
            row_scale
        } else {
            src.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0
        };
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut sum = 0i32;
        for (out, &v) in qs.iter_mut().zip(src.iter()) {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            *out = q;
            sum += i32::from(q);
        }
        dst.scales[b] = scale;
        dst.sums[b] = sum;
    }
}

/// A packed Q4 matrix with shape `[n, k]`, `k` a multiple of 32.
///
/// Tile layout: `tiles[(tile * bpr + kb) * TILE_STRIDE ..] +` 256 nibble
/// bytes followed by 16 f32 scales for output rows `tile*16 .. tile*16+16`.
#[derive(Clone, Debug)]
pub struct Q4Matrix {
    pub n: usize,
    pub k: usize,
    /// Packed VNNI tiles, length `n_tiles * bpr * TILE_STRIDE`.
    pub tiles: Vec<u8>,
    /// Pre-packed AMX B panels (signed i8, 8×64 per tile per K-block).
    /// Present when AMX-INT8 was available at pack time.
    amx_b: Option<Vec<u8>>,
    n_tiles: usize,
}

impl Q4Matrix {
    /// Quantize a row-major fp32 matrix `[n, k]` and pack VNNI tiles.
    #[must_use]
    pub fn from_f32(weight: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(weight.len(), n * k);
        assert!(
            k.is_multiple_of(BLOCK_SIZE),
            "K={k} is not a multiple of {BLOCK_SIZE}"
        );
        let blocks = quantize_q4(weight);
        Self::from_blocks(n, k, &blocks)
    }

    /// Pack already-quantized row-major [`BlockQ4`] weights.
    #[must_use]
    pub fn from_blocks(n: usize, k: usize, blocks: &[BlockQ4]) -> Self {
        let bpr = k / BLOCK_SIZE;
        assert_eq!(blocks.len(), n * bpr);
        let n_tiles = n.div_ceil(TILE_N);
        let mut tiles = alloc_prefaulted(n_tiles * bpr * TILE_STRIDE);
        for tile in 0..n_tiles {
            let row0 = tile * TILE_N;
            for kb in 0..bpr {
                let slot = &mut tiles[(tile * bpr + kb) * TILE_STRIDE..][..TILE_STRIDE];
                pack_tile(slot, blocks, n, bpr, row0, kb);
            }
        }
        let mut mat = Self {
            n,
            k,
            tiles,
            amx_b: None,
            n_tiles,
        };
        mat.pack_amx_b();
        mat
    }

    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.k / BLOCK_SIZE
    }

    #[inline]
    pub fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    #[inline]
    pub fn tile(&self, tile: usize, kb: usize) -> &[u8] {
        let bpr = self.blocks_per_row();
        let off = (tile * bpr + kb) * TILE_STRIDE;
        &self.tiles[off..off + TILE_STRIDE]
    }

    /// Fault in every page of the packed weights.
    pub fn prefault(&self) {
        prefault_bytes(&self.tiles);
        if let Some(b) = &self.amx_b {
            prefault_bytes(b);
        }
    }

    /// Pack signed AMX B panels when the process can run AMX-INT8.
    pub fn pack_amx_b(&mut self) {
        if !crate::amx::amx_ready() {
            self.amx_b = None;
            return;
        }
        let bpr = self.blocks_per_row();
        let stride = crate::amx::AMX_B_STRIDE;
        let mut buf = alloc_prefaulted(self.n_tiles * bpr * stride);
        for tile in 0..self.n_tiles {
            for kb in 0..bpr {
                let off = (tile * bpr + kb) * stride;
                let dst = &mut buf[off..off + stride];
                crate::amx::pack_amx_b_panel(self.tile(tile, kb), unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<i8>(), stride)
                });
            }
        }
        prefault_bytes(&buf);
        self.amx_b = Some(buf);
    }

    /// Pre-packed AMX B panel for `tile` / K-block, if packed.
    #[must_use]
    pub fn amx_b_panel(&self, tile: usize, kb: usize) -> Option<&[i8]> {
        let buf = self.amx_b.as_ref()?;
        let stride = crate::amx::AMX_B_STRIDE;
        let bpr = self.blocks_per_row();
        let off = (tile * bpr + kb) * stride;
        Some(unsafe {
            std::slice::from_raw_parts(buf[off..off + stride].as_ptr().cast::<i8>(), stride)
        })
    }
}

/// Pack 16 (or fewer) rows of one K-block into VNNI nibble layout + f32 scales.
pub(crate) fn pack_tile(
    out: &mut [u8],
    blocks: &[BlockQ4],
    n: usize,
    bpr: usize,
    row0: usize,
    kb: usize,
) {
    // For k-group pairs (0,1), (2,3), (4,5), (6,7): 64 bytes each.
    // byte[col*4 + j] = nibble(row, kg0*4+j) | (nibble(row, kg1*4+j) << 4)
    // where nibble is the unsigned 0..=15 payload.
    for pair in 0..4 {
        let kg0 = pair * 2;
        let kg1 = kg0 + 1;
        let dst = &mut out[pair * 64..pair * 64 + 64];
        for col in 0..TILE_N {
            let row = row0 + col;
            if row >= n {
                for j in 0..4 {
                    dst[col * 4 + j] = 0;
                }
                continue;
            }
            let block = &blocks[row * bpr + kb];
            for j in 0..4 {
                let lo = unsigned_nibble(block, kg0 * 4 + j);
                let hi = unsigned_nibble(block, kg1 * 4 + j);
                dst[col * 4 + j] = lo | (hi << 4);
            }
        }
    }
    let scales = &mut out[TILE_QS..TILE_STRIDE];
    for col in 0..TILE_N {
        let row = row0 + col;
        let s = if row < n {
            blocks[row * bpr + kb].scale.to_f32()
        } else {
            0.0
        };
        scales[col * 4..col * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
}

#[inline]
fn unsigned_nibble(block: &BlockQ4, i: usize) -> u8 {
    // BlockQ4 stores two's-complement nibbles. The VNNI tile stores the
    // unsigned offset form `s+8` in `0..=15` so `dpbusd(nibble, a) - 8*sum(a)`
    // equals the signed dot.
    (block.nibble(i) + 8) as u8
}

/// How activations are quantized before a Q4 GEMV/GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActQuant {
    /// One scale and int8 block per 32 elements (default).
    #[default]
    PerBlock,
    /// One scale for the whole vector, still stored per 32-element block.
    PerRow,
}

static ACT_QUANT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Set the process-wide activation-quant mode used by [`gemv_q4`] / [`gemm_q4`].
pub fn set_act_quant(mode: ActQuant) {
    let v = match mode {
        ActQuant::PerBlock => 0,
        ActQuant::PerRow => 1,
    };
    ACT_QUANT.store(v, std::sync::atomic::Ordering::Release);
}

/// Current activation-quant mode.
#[must_use]
pub fn current_act_quant() -> ActQuant {
    if ACT_QUANT.load(std::sync::atomic::Ordering::Acquire) == 1 {
        ActQuant::PerRow
    } else {
        ActQuant::PerBlock
    }
}

fn quantize_acts(acts: &mut ActQ8, x: &[f32]) {
    match current_act_quant() {
        ActQuant::PerBlock => acts.quantize_into(x),
        ActQuant::PerRow => acts.quantize_row_into(x),
    }
}

/// Integer-dot kernel used by the Q4 paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Kernel {
    Scalar,
    Avx2,
    Vnni,
}

/// Pick the fastest Q4 integer kernel this CPU can run.
#[must_use]
pub fn detect_q4_kernel() -> Q4Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_vnni_512() || has_vnni_256() {
            return Q4Kernel::Vnni;
        }
        if is_x86_feature_detected!("avx2") {
            return Q4Kernel::Avx2;
        }
    }
    Q4Kernel::Scalar
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_vnni_512() -> bool {
    is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_vnni_256() -> bool {
    (is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl"))
        || is_x86_feature_detected!("avxvnni")
}

thread_local! {
    static ACT_SCRATCH: std::cell::RefCell<ActQ8> = const { std::cell::RefCell::new(ActQ8 {
        qs: Vec::new(),
        scales: Vec::new(),
        sums: Vec::new(),
    }) };
}

/// `out[n] = W[n, k] @ x[k]` with Q4 weights and on-the-fly Q8 activations.
pub fn gemv_q4(w: &Q4Matrix, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.k);
    assert_eq!(out.len(), w.n);
    ACT_SCRATCH.with(|slot| {
        let mut acts = slot.borrow_mut();
        quantize_acts(&mut acts, x);
        gemv_q4_prep(w, &acts, out, detect_q4_kernel());
    });
}

/// Like [`gemv_q4`] but adds the product onto `dst` in f64 (FFN residual).
pub fn gemv_q4_add(w: &Q4Matrix, x: &[f32], dst: &mut [f32]) {
    assert_eq!(x.len(), w.k);
    assert_eq!(dst.len(), w.n);
    ACT_SCRATCH.with(|slot| {
        let mut acts = slot.borrow_mut();
        quantize_acts(&mut acts, x);
        gemv_q4_prep_add(w, &acts, dst, detect_q4_kernel());
    });
}

/// Like [`gemv_q4`] but reuses an already-quantized activation (for fused pairs).
pub fn gemv_q4_prep(w: &Q4Matrix, acts: &ActQ8, out: &mut [f32], kernel: Q4Kernel) {
    assert_eq!(acts.qs.len(), w.k);
    assert_eq!(out.len(), w.n);
    if w.n == 0 {
        return;
    }
    let out_addr = out.as_mut_ptr() as usize;
    let n = w.n;
    for_ranges(w.n_tiles, |_tid, t0, t1| {
        let out_ptr = out_addr as *mut f32;
        unsafe { gemv_tiles(w, acts, out_ptr, n, t0, t1, kernel, AccMode::Assign) };
    });
}

/// Fused add variant of [`gemv_q4_prep`].
pub fn gemv_q4_prep_add(w: &Q4Matrix, acts: &ActQ8, dst: &mut [f32], kernel: Q4Kernel) {
    assert_eq!(acts.qs.len(), w.k);
    assert_eq!(dst.len(), w.n);
    if w.n == 0 {
        return;
    }
    let out_addr = dst.as_mut_ptr() as usize;
    let n = w.n;
    for_ranges(w.n_tiles, |_tid, t0, t1| {
        let out_ptr = out_addr as *mut f32;
        unsafe { gemv_tiles(w, acts, out_ptr, n, t0, t1, kernel, AccMode::AddF64) };
    });
}

/// Two GEMVs sharing the same activation (SwiGLU gate and up projections).
pub fn gemv_q4_pair(
    w_gate: &Q4Matrix,
    w_up: &Q4Matrix,
    x: &[f32],
    out_gate: &mut [f32],
    out_up: &mut [f32],
) {
    assert_eq!(w_gate.k, w_up.k);
    assert_eq!(x.len(), w_gate.k);
    assert_eq!(w_gate.n, out_gate.len());
    assert_eq!(w_up.n, out_up.len());
    let acts = ACT_SCRATCH.with(|slot| {
        let mut acts = slot.borrow_mut();
        quantize_acts(&mut acts, x);
        // Clone the small scale/sum vectors plus qs so we can drop the TLS
        // borrow before the parallel region. For a 1536-d vector this is ~2 KB.
        acts.clone()
    });
    let kernel = detect_q4_kernel();
    // One barrier: each thread owns the same tile range of both matrices.
    // (Shapes match for SwiGLU gate/up.)
    assert_eq!(w_gate.n_tiles, w_up.n_tiles);
    let g_addr = out_gate.as_mut_ptr() as usize;
    let u_addr = out_up.as_mut_ptr() as usize;
    let n = w_gate.n;
    for_ranges(w_gate.n_tiles, |_tid, t0, t1| unsafe {
        gemv_tiles(
            w_gate,
            &acts,
            g_addr as *mut f32,
            n,
            t0,
            t1,
            kernel,
            AccMode::Assign,
        );
        gemv_tiles(
            w_up,
            &acts,
            u_addr as *mut f32,
            n,
            t0,
            t1,
            kernel,
            AccMode::Assign,
        );
    });
}

#[derive(Clone, Copy)]
enum AccMode {
    Assign,
    AddF64,
}

unsafe fn gemv_tiles(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    kernel: Q4Kernel,
    mode: AccMode,
) {
    match kernel {
        Q4Kernel::Vnni => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if has_vnni_512() {
                    gemv_tiles_vnni512(w, acts, out, n, t0, t1, mode);
                    return;
                }
            }
            gemv_tiles_scalar(w, acts, out, n, t0, t1, mode);
        }
        Q4Kernel::Avx2 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    gemv_tiles_avx2(w, acts, out, n, t0, t1, mode);
                    return;
                }
            }
            gemv_tiles_scalar(w, acts, out, n, t0, t1, mode);
        }
        Q4Kernel::Scalar => gemv_tiles_scalar(w, acts, out, n, t0, t1, mode),
    }
}

fn store_acc(out: *mut f32, n: usize, row0: usize, acc: &[f32; TILE_N], mode: AccMode) {
    let count = TILE_N.min(n.saturating_sub(row0));
    for col in 0..count {
        unsafe {
            let slot = out.add(row0 + col);
            match mode {
                AccMode::Assign => *slot = acc[col],
                AccMode::AddF64 => *slot = (f64::from(*slot) + f64::from(acc[col])) as f32,
            }
        }
    }
}

fn gemv_tiles_scalar(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    mode: AccMode,
) {
    let bpr = w.blocks_per_row();
    for tile in t0..t1 {
        let row0 = tile * TILE_N;
        let mut acc = [0.0f32; TILE_N];
        for kb in 0..bpr {
            let panel = w.tile(tile, kb);
            let a = &acts.qs[kb * BLOCK_SIZE..(kb + 1) * BLOCK_SIZE];
            let sa = acts.scales[kb];
            let mut idot = [0i32; TILE_N];
            for pair in 0..4 {
                let kg0 = pair * 2;
                let kg1 = kg0 + 1;
                let bytes = &panel[pair * 64..pair * 64 + 64];
                for col in 0..TILE_N {
                    let mut s = 0i32;
                    for j in 0..4 {
                        let packed = bytes[col * 4 + j];
                        let lo = (packed & 0x0f) as i8 - 8;
                        let hi = (packed >> 4) as i8 - 8;
                        s += i32::from(lo) * i32::from(a[kg0 * 4 + j]);
                        s += i32::from(hi) * i32::from(a[kg1 * 4 + j]);
                    }
                    idot[col] += s;
                }
            }
            for col in 0..TILE_N {
                let sw = f32::from_le_bytes(
                    panel[TILE_QS + col * 4..TILE_QS + col * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                acc[col] += idot[col] as f32 * sw * sa;
            }
        }
        store_acc(out, n, row0, &acc, mode);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gemv_tiles_avx2_inner(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    mode: AccMode,
) {
    // AVX2 path: two columns of 8 at a time, signed int16 madd.
    use std::arch::x86_64::*;
    let bpr = w.blocks_per_row();
    for tile in t0..t1 {
        let row0 = tile * TILE_N;
        let mut acc = [0.0f32; TILE_N];
        for kb in 0..bpr {
            let panel = w.tile(tile, kb);
            let a = &acts.qs[kb * BLOCK_SIZE..(kb + 1) * BLOCK_SIZE];
            let sa = acts.scales[kb];
            // Load 32 activations once, broadcast later via scalar extract of groups.
            let mut idot = [0i32; TILE_N];
            for pair in 0..4 {
                let kg0 = pair * 2;
                let kg1 = kg0 + 1;
                let bytes = panel.as_ptr().add(pair * 64);
                for col in 0..TILE_N {
                    let packed = _mm_cvtsi32_si128(std::ptr::read_unaligned(
                        bytes.add(col * 4) as *const i32
                    ));
                    let lo = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
                    let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f));
                    let eight = _mm_set1_epi8(8);
                    let lo_s = _mm_sub_epi8(lo, eight);
                    let hi_s = _mm_sub_epi8(hi, eight);
                    let a0 = std::ptr::read_unaligned(a.as_ptr().add(kg0 * 4) as *const i32);
                    let a1 = std::ptr::read_unaligned(a.as_ptr().add(kg1 * 4) as *const i32);
                    let wa = _mm_cvtepi8_epi16(lo_s);
                    let wb = _mm_cvtepi8_epi16(hi_s);
                    let aa = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(a0));
                    let ab = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(a1));
                    let p0 = _mm_madd_epi16(wa, aa);
                    let p1 = _mm_madd_epi16(wb, ab);
                    let sum = _mm_add_epi32(p0, p1);
                    // horizontal: 2 i32 in the low half (4 products → 2 madds)
                    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x31));
                    idot[col] += _mm_cvtsi128_si32(sum);
                }
            }
            for col in 0..TILE_N {
                let sw = f32::from_le_bytes(
                    panel[TILE_QS + col * 4..TILE_QS + col * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                acc[col] += idot[col] as f32 * sw * sa;
            }
        }
        store_acc(out, n, row0, &acc, mode);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemv_tiles_avx2(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    mode: AccMode,
) {
    gemv_tiles_avx2_inner(w, acts, out, n, t0, t1, mode);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
unsafe fn gemv_tiles_vnni512_inner(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    mode: AccMode,
) {
    use std::arch::x86_64::*;
    let bpr = w.blocks_per_row();
    let off = _mm512_set1_epi8(8);
    let low_mask = _mm512_set1_epi8(0x0f);
    for tile in t0..t1 {
        let row0 = tile * TILE_N;
        let mut vacc = _mm512_setzero_ps();
        let tile_base = w.tiles.as_ptr().add(tile * bpr * TILE_STRIDE);
        for kb in 0..bpr {
            let panel = tile_base.add(kb * TILE_STRIDE);
            let a_ptr = acts.qs.as_ptr().add(kb * BLOCK_SIZE) as *const i32;
            let mut va = [_mm512_setzero_si512(); 8];
            let mut vcomp = _mm512_setzero_si512();
            for k in 0..8 {
                va[k] = _mm512_set1_epi32(*a_ptr.add(k));
                vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
            }
            let mut vsum = _mm512_setzero_si512();
            for pair in 0..4 {
                let bytes = _mm512_loadu_si512(panel.add(pair * 64).cast());
                let vb0 = _mm512_and_si512(bytes, low_mask);
                let vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[pair * 2]);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[pair * 2 + 1]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);
            let sw = _mm512_loadu_ps(panel.add(TILE_QS).cast());
            let sa = _mm512_set1_ps(*acts.scales.get_unchecked(kb));
            vacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(sw, sa), vacc);
        }
        let mut acc = [0.0f32; TILE_N];
        _mm512_storeu_ps(acc.as_mut_ptr(), vacc);
        store_acc(out, n, row0, &acc, mode);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemv_tiles_vnni512(
    w: &Q4Matrix,
    acts: &ActQ8,
    out: *mut f32,
    n: usize,
    t0: usize,
    t1: usize,
    mode: AccMode,
) {
    gemv_tiles_vnni512_inner(w, acts, out, n, t0, t1, mode);
}

/// Prefill: `out[m, n] = X[m, k] @ W[n, k]^T` with Q4 `W`.
///
/// Each weight tile is reused across all `m` tokens. When AMX-INT8 is
/// available and `m >= 16`, the AMX kernel is used.
pub fn gemm_q4(w: &Q4Matrix, x: &[f32], out: &mut [f32], m: usize) {
    gemm_q4_mode(w, x, out, m, false);
}

/// Like [`gemm_q4`] but adds the product onto `out` in f64.
pub fn gemm_q4_add(w: &Q4Matrix, x: &[f32], out: &mut [f32], m: usize) {
    gemm_q4_mode(w, x, out, m, true);
}

fn gemm_q4_mode(w: &Q4Matrix, x: &[f32], out: &mut [f32], m: usize, add: bool) {
    assert_eq!(x.len(), m * w.k);
    assert_eq!(out.len(), m * w.n);
    if m == 0 || w.n == 0 {
        return;
    }
    if m == 1 {
        if add {
            gemv_q4_add(w, x, out);
        } else {
            gemv_q4(w, x, out);
        }
        return;
    }
    let acts = quantize_batch(x, m, w.k);
    let kernel = detect_q4_kernel();
    let use_amx = m >= 16 && crate::amx::amx_ready() && !add;
    let out_addr = out.as_mut_ptr() as usize;
    let n = w.n;
    for_ranges(w.n_tiles, |_tid, t0, t1| {
        let out_ptr = out_addr as *mut f32;
        if use_amx {
            unsafe { crate::amx::gemm_tiles_amx(w, &acts, out_ptr, n, m, t0, t1) };
        } else {
            unsafe { gemm_tiles(w, &acts, out_ptr, n, m, t0, t1, kernel, add) };
        }
    });
}

/// Packed activations for a batch of `m` tokens.
pub struct ActQ8Batch {
    #[allow(dead_code)]
    pub m: usize,
    pub k: usize,
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
}

impl ActQ8Batch {
    #[inline]
    fn block(&self, token: usize, kb: usize) -> (&[i8], f32, i32) {
        let bpr = self.k / BLOCK_SIZE;
        let off = token * self.k + kb * BLOCK_SIZE;
        (
            &self.qs[off..off + BLOCK_SIZE],
            self.scales[token * bpr + kb],
            self.sums[token * bpr + kb],
        )
    }
}

fn quantize_batch(x: &[f32], m: usize, k: usize) -> ActQ8Batch {
    let bpr = k / BLOCK_SIZE;
    let mut qs = vec![0i8; m * k];
    let mut scales = vec![0.0f32; m * bpr];
    let mut sums = vec![0i32; m * bpr];
    let nthreads = crate::pool::current_threads();
    // Parallelize over tokens; each thread writes a disjoint slice.
    let qs_addr = qs.as_mut_ptr() as usize;
    let sc_addr = scales.as_mut_ptr() as usize;
    let su_addr = sums.as_mut_ptr() as usize;
    let x_addr = x.as_ptr() as usize;
    for_ranges(m, |_tid, t0, t1| {
        for t in t0..t1 {
            let src = unsafe { std::slice::from_raw_parts((x_addr as *const f32).add(t * k), k) };
            let dst_q =
                unsafe { std::slice::from_raw_parts_mut((qs_addr as *mut i8).add(t * k), k) };
            let row_scale = if matches!(current_act_quant(), ActQuant::PerRow) {
                src.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0
            } else {
                0.0
            };
            for b in 0..bpr {
                let s = &src[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let d = &mut dst_q[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let scale = if matches!(current_act_quant(), ActQuant::PerRow) {
                    row_scale
                } else {
                    s.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) / 127.0
                };
                let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                let mut sum = 0i32;
                for (out, &v) in d.iter_mut().zip(s.iter()) {
                    let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    *out = q;
                    sum += i32::from(q);
                }
                unsafe {
                    *(sc_addr as *mut f32).add(t * bpr + b) = scale;
                    *(su_addr as *mut i32).add(t * bpr + b) = sum;
                }
            }
        }
    });
    let _ = nthreads;
    ActQ8Batch {
        m,
        k,
        qs,
        scales,
        sums,
    }
}

unsafe fn gemm_tiles(
    w: &Q4Matrix,
    acts: &ActQ8Batch,
    out: *mut f32,
    n: usize,
    m: usize,
    t0: usize,
    t1: usize,
    kernel: Q4Kernel,
    add: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if matches!(kernel, Q4Kernel::Vnni) && has_vnni_512() {
            gemm_tiles_vnni512(w, acts, out, n, m, t0, t1, add);
            return;
        }
    }
    let _ = kernel;
    gemm_tiles_scalar(w, acts, out, n, m, t0, t1, add);
}

fn gemm_tiles_scalar(
    w: &Q4Matrix,
    acts: &ActQ8Batch,
    out: *mut f32,
    n: usize,
    m: usize,
    t0: usize,
    t1: usize,
    add: bool,
) {
    let bpr = w.blocks_per_row();
    let mut acc = vec![0.0f32; m * TILE_N];
    for tile in t0..t1 {
        acc.fill(0.0);
        let row0 = tile * TILE_N;
        for kb in 0..bpr {
            let panel = w.tile(tile, kb);
            for t in 0..m {
                let (a, sa, _) = acts.block(t, kb);
                let mut idot = [0i32; TILE_N];
                for pair in 0..4 {
                    let kg0 = pair * 2;
                    let kg1 = kg0 + 1;
                    let bytes = &panel[pair * 64..pair * 64 + 64];
                    for col in 0..TILE_N {
                        let mut s = 0i32;
                        for j in 0..4 {
                            let packed = bytes[col * 4 + j];
                            let lo = (packed & 0x0f) as i8 - 8;
                            let hi = (packed >> 4) as i8 - 8;
                            s += i32::from(lo) * i32::from(a[kg0 * 4 + j]);
                            s += i32::from(hi) * i32::from(a[kg1 * 4 + j]);
                        }
                        idot[col] += s;
                    }
                }
                let dst = &mut acc[t * TILE_N..(t + 1) * TILE_N];
                for col in 0..TILE_N {
                    let sw = f32::from_le_bytes(
                        panel[TILE_QS + col * 4..TILE_QS + col * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    dst[col] += idot[col] as f32 * sw * sa;
                }
            }
        }
        for t in 0..m {
            let row_acc: [f32; TILE_N] = acc[t * TILE_N..(t + 1) * TILE_N].try_into().unwrap();
            let mode = if add {
                AccMode::AddF64
            } else {
                AccMode::Assign
            };
            store_acc(unsafe { out.add(t * n) }, n, row0, &row_acc, mode);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f", enable = "avx512vnni", enable = "avx512bw")]
unsafe fn gemm_tiles_vnni512_inner(
    w: &Q4Matrix,
    acts: &ActQ8Batch,
    out: *mut f32,
    n: usize,
    m: usize,
    t0: usize,
    t1: usize,
    add: bool,
) {
    use std::arch::x86_64::*;
    let bpr = w.blocks_per_row();
    let off = _mm512_set1_epi8(8);
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut acc = vec![0.0f32; m * TILE_N];
    for tile in t0..t1 {
        acc.fill(0.0);
        let row0 = tile * TILE_N;
        let tile_base = w.tiles.as_ptr().add(tile * bpr * TILE_STRIDE);
        for kb in 0..bpr {
            let panel = tile_base.add(kb * TILE_STRIDE);
            let sw = _mm512_loadu_ps(panel.add(TILE_QS).cast());
            for t in 0..m {
                let a_ptr = acts.qs.as_ptr().add(t * acts.k + kb * BLOCK_SIZE) as *const i32;
                let sa = _mm512_set1_ps(*acts.scales.get_unchecked(t * bpr + kb));
                let mut va = [_mm512_setzero_si512(); 8];
                let mut vcomp = _mm512_setzero_si512();
                for k in 0..8 {
                    va[k] = _mm512_set1_epi32(*a_ptr.add(k));
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                let mut vsum = _mm512_setzero_si512();
                for pair in 0..4 {
                    let bytes = _mm512_loadu_si512(panel.add(pair * 64).cast());
                    let vb0 = _mm512_and_si512(bytes, low_mask);
                    let vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low_mask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va[pair * 2]);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va[pair * 2 + 1]);
                }
                vsum = _mm512_sub_epi32(vsum, vcomp);
                let vacc = _mm512_loadu_ps(acc.as_ptr().add(t * TILE_N));
                let vacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(sw, sa), vacc);
                _mm512_storeu_ps(acc.as_mut_ptr().add(t * TILE_N), vacc);
            }
        }
        for t in 0..m {
            let row_acc: [f32; TILE_N] = acc[t * TILE_N..(t + 1) * TILE_N].try_into().unwrap();
            let mode = if add {
                AccMode::AddF64
            } else {
                AccMode::Assign
            };
            store_acc(out.add(t * n), n, row0, &row_acc, mode);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_tiles_vnni512(
    w: &Q4Matrix,
    acts: &ActQ8Batch,
    out: *mut f32,
    n: usize,
    m: usize,
    t0: usize,
    t1: usize,
    add: bool,
) {
    gemm_tiles_vnni512_inner(w, acts, out, n, m, t0, t1, add);
}

// ---- Per-block reference kernels (tests + fallback) --------------------

/// Signed-nibble · int8 dot product of one 32-element block.
#[must_use]
#[allow(dead_code)]
pub fn dot_q4_q8_scalar(block: &BlockQ4, a: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), BLOCK_SIZE);
    let mut sum = 0i32;
    for (i, &act) in a.iter().enumerate() {
        sum += i32::from(block.nibble(i)) * i32::from(act);
    }
    sum
}

#[cfg(test)]
fn avx2_dot(block: &BlockQ4, a: &[i8]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_q4_q8_avx2(block, a) };
        }
    }
    dot_q4_q8_scalar(block, a)
}

#[cfg(test)]
fn vnni_dot(block: &BlockQ4, a: &[i8], a_sum: i32) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_vnni_256() {
            return unsafe { dot_q4_q8_vnni(block, a, a_sum) };
        }
    }
    dot_q4_q8_scalar(block, a)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn unpack_signed_weights(block: &BlockQ4) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let packed = _mm_loadu_si128(block.qs.as_ptr().cast());
    let mask = _mm_set1_epi8(0x0f);
    let lo = sign_extend_nibbles(_mm_and_si128(packed, mask));
    let hi = sign_extend_nibbles(_mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    let ev = _mm_unpacklo_epi8(lo, hi);
    let od = _mm_unpackhi_epi8(lo, hi);
    _mm256_set_m128i(od, ev)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn sign_extend_nibbles(nibbles: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let eight = _mm_set1_epi8(8);
    let signs = _mm_cmpeq_epi8(_mm_and_si128(nibbles, eight), eight);
    _mm_or_si128(nibbles, _mm_and_si128(signs, _mm_set1_epi8(-16)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn dot_q4_q8_avx2(block: &BlockQ4, a: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let w = unpack_signed_weights(block);
    let aq = _mm256_loadu_si256(a.as_ptr().cast());
    let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
    let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
    let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(aq));
    let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(aq, 1));
    let p0 = _mm256_madd_epi16(w_lo, a_lo);
    let p1 = _mm256_madd_epi16(w_hi, a_hi);
    hsum_i32_256(_mm256_add_epi32(p0, p1))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512vnni", enable = "avx512vl", enable = "avx2")]
#[cfg(test)]
unsafe fn dot_q4_q8_vnni(block: &BlockQ4, a: &[i8], a_sum: i32) -> i32 {
    use std::arch::x86_64::*;
    let packed = _mm_loadu_si128(block.qs.as_ptr().cast());
    let mask = _mm_set1_epi8(0x0f);
    let eight = _mm_set1_epi8(8);
    let lo = _mm_and_si128(_mm_add_epi8(_mm_and_si128(packed, mask), eight), mask);
    let hi = _mm_and_si128(
        _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), mask), eight),
        mask,
    );
    let ev = _mm_unpacklo_epi8(lo, hi);
    let od = _mm_unpackhi_epi8(lo, hi);
    let w = _mm256_set_m128i(od, ev);
    let aq = _mm256_loadu_si256(a.as_ptr().cast());
    let prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, aq);
    hsum_i32_256(prod) - 8 * a_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
#[cfg(test)]
unsafe fn hsum_i32_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    _mm_cvtsi128_si32(s)
}

#[allow(dead_code)]
fn _use_thread_range() {
    let _ = thread_range(0, 1, 1);
}

#[cfg(test)]
mod tests {
    use super::{
        detect_q4_kernel, dot_q4_q8_scalar, gemm_q4, gemv_q4, gemv_q4_prep, ActQ8, Q4Kernel,
        Q4Matrix,
    };
    use crate::pool::with_threads;
    use llm_quant::{BlockQ4, BLOCK_SIZE};

    fn pattern_block(seed: i32) -> ([f32; BLOCK_SIZE], [i8; BLOCK_SIZE]) {
        let mut w = [0.0f32; BLOCK_SIZE];
        let mut a = [0i8; BLOCK_SIZE];
        for i in 0..BLOCK_SIZE {
            w[i] = ((i as i32 * 17 + seed) % 21 - 10) as f32 * 0.25;
            a[i] = ((i as i32 * 9 + seed) % 25 - 12) as i8;
        }
        (w, a)
    }

    #[test]
    fn vnni_and_avx2_match_scalar_dot() {
        for seed in 0..16 {
            let (w, a) = pattern_block(seed);
            let block = BlockQ4::quantize(&w);
            let scalar = dot_q4_q8_scalar(&block, &a);
            let avx = super::avx2_dot(&block, &a);
            assert_eq!(avx, scalar, "AVX2 mismatch at seed {seed}");
            let sum: i32 = a.iter().map(|&v| i32::from(v)).sum();
            let vnni = super::vnni_dot(&block, &a, sum);
            assert_eq!(vnni, scalar, "VNNI mismatch at seed {seed}");
        }
    }

    #[test]
    fn gemv_kernels_agree() {
        let n = 17;
        let k = 64;
        let mut w = vec![0.0f32; n * k];
        let mut x = vec![0.0f32; k];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = ((i as i32 * 13) % 19 - 9) as f32 * 0.15;
        }
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = ((i as i32 * 7) % 15 - 7) as f32 * 0.2;
        }
        let mat = Q4Matrix::from_f32(&w, n, k);
        let acts = ActQ8::quantize(&x);
        let mut scalar = vec![0.0f32; n];
        let mut avx = vec![0.0f32; n];
        let mut vnni = vec![0.0f32; n];
        with_threads(1, || {
            gemv_q4_prep(&mat, &acts, &mut scalar, Q4Kernel::Scalar);
            gemv_q4_prep(&mat, &acts, &mut avx, Q4Kernel::Avx2);
            gemv_q4_prep(&mat, &acts, &mut vnni, Q4Kernel::Vnni);
        });
        for i in 0..n {
            let s = scalar[i];
            assert!(
                (avx[i] - s).abs() <= 1e-6 * (1.0 + s.abs()),
                "AVX2 row {i}: {} vs {s}",
                avx[i]
            );
            assert!(
                (vnni[i] - s).abs() <= 1e-6 * (1.0 + s.abs()),
                "VNNI row {i}: {} vs {s}",
                vnni[i]
            );
        }
        let _ = detect_q4_kernel();
    }

    #[test]
    fn gemv_pool_matches_serial() {
        let n = 128;
        let k = 64;
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as i32 * 3) % 17 - 8) as f32 * 0.1)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as i32 * 5) % 11 - 5) as f32 * 0.2)
            .collect();
        let mat = Q4Matrix::from_f32(&w, n, k);
        let mut serial = vec![0.0f32; n];
        let mut pooled = vec![0.0f32; n];
        with_threads(1, || gemv_q4(&mat, &x, &mut serial));
        with_threads(4, || gemv_q4(&mat, &x, &mut pooled));
        for i in 0..n {
            assert!(
                (serial[i] - pooled[i]).abs() <= 1e-6 * (1.0 + serial[i].abs()),
                "row {i}: {} vs {}",
                pooled[i],
                serial[i]
            );
        }
    }

    #[test]
    fn gemm_matches_per_token_gemv() {
        let n = 48;
        let k = 64;
        let m = 20;
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as i32 * 3) % 17 - 8) as f32 * 0.1)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as i32 * 5) % 11 - 5) as f32 * 0.2)
            .collect();
        let mat = Q4Matrix::from_f32(&w, n, k);
        let mut gemm = vec![0.0f32; m * n];
        let mut gemv = vec![0.0f32; m * n];
        with_threads(4, || {
            gemm_q4(&mat, &x, &mut gemm, m);
            for t in 0..m {
                gemv_q4(&mat, &x[t * k..(t + 1) * k], &mut gemv[t * n..(t + 1) * n]);
            }
        });
        for i in 0..m * n {
            assert!(
                (gemm[i] - gemv[i]).abs() <= 1e-5 * (1.0 + gemv[i].abs()),
                "elem {i}: gemm={} gemv={}",
                gemm[i],
                gemv[i]
            );
        }
    }

    #[test]
    fn packed_gemv_matches_block_dots() {
        use super::TILE_N;
        use llm_quant::quantize_q4;
        let n = 17;
        let k = 64;
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as i32 * 13) % 19 - 9) as f32 * 0.15)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as i32 * 7) % 15 - 7) as f32 * 0.2)
            .collect();
        let blocks = quantize_q4(&w);
        let acts = ActQ8::quantize(&x);
        let bpr = k / BLOCK_SIZE;
        let mut expected = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let block = &blocks[row * bpr + b];
                let a = &acts.qs[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                let idot = dot_q4_q8_scalar(block, a);
                acc += idot as f32 * block.scale.to_f32() * acts.scales[b];
            }
            expected[row] = acc;
        }
        let mat = Q4Matrix::from_blocks(n, k, &blocks);
        assert_eq!(mat.n_tiles(), n.div_ceil(TILE_N));
        let mut got = vec![0.0f32; n];
        with_threads(1, || {
            gemv_q4_prep(&mat, &acts, &mut got, Q4Kernel::Scalar);
        });
        for i in 0..n {
            let e = expected[i];
            assert!(
                (got[i] - e).abs() <= 1e-6 * (1.0 + e.abs()),
                "row {i}: packed={} block_dot={e}",
                got[i]
            );
        }
        let mut vnni = vec![0.0f32; n];
        with_threads(1, || {
            gemv_q4_prep(&mat, &acts, &mut vnni, Q4Kernel::Vnni);
        });
        for i in 0..n {
            let e = expected[i];
            assert!(
                (vnni[i] - e).abs() <= 1e-6 * (1.0 + e.abs()),
                "vnni row {i}: packed={} block_dot={e}",
                vnni[i]
            );
        }
    }
}

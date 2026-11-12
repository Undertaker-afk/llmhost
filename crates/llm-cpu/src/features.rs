/// Instruction-set flags detected at runtime on this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512vnni: bool,
    pub amx_int8: bool,
}

impl CpuFeatures {
    /// Probe the current CPU. All flags are false on non-x86 targets.
    ///
    /// AVX2, AVX-512F, and AVX-512 VNNI are detected with
    /// [`is_x86_feature_detected`]. `amx-int8` is not yet stable in that
    /// macro (Rust issue 126622), so it is detected with CPUID and XCR0
    /// instead, using the same bits the standard probe will use once it
    /// stabilizes.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512vnni: is_x86_feature_detected!("avx512vnni"),
                amx_int8: detect_amx_int8(),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self {
                avx2: false,
                avx512f: false,
                avx512vnni: false,
                amx_int8: false,
            }
        }
    }

    /// Preferred decode kernel: AVX-512 VNNI, then AVX-512F, then AVX2, then scalar.
    #[must_use]
    pub fn decode_kernels(self) -> KernelSet {
        if self.avx512vnni {
            KernelSet::Avx512Vnni
        } else if self.avx512f {
            KernelSet::Avx512
        } else if self.avx2 {
            KernelSet::Avx2
        } else {
            KernelSet::Scalar
        }
    }

    /// Preferred prefill kernel: AMX int8 tiles when present, otherwise decode pick.
    #[must_use]
    pub fn prefill_kernels(self) -> KernelSet {
        if self.amx_int8 {
            KernelSet::AmxInt8
        } else {
            self.decode_kernels()
        }
    }
}

/// Selected kernel family for a matmul / GEMV path.
///
/// This is a dispatch tag, not a promise that the named kernels are
/// implemented yet. M2 fills in VNNI, AMX, and the quantized variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelSet {
    Scalar,
    Avx2,
    Avx512,
    Avx512Vnni,
    AmxInt8,
}

/// Detect AMX-INT8 plus OS support for the tile register state.
///
/// CPUID.(EAX=7, ECX=0):EDX bit 24 is AMX-TILE, bit 25 is AMX-INT8.
/// XCR0 bits 17 and 18 must also be set (TILECFG and TILEDATA), which
/// requires OSXSAVE (CPUID.1:ECX bit 27).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_amx_int8() -> bool {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__cpuid, __cpuid_count, _xgetbv};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // SAFETY: `__cpuid` / `__cpuid_count` are safe to call on x86/x86_64.
    // `_xgetbv` is only used after OSXSAVE is confirmed.
    unsafe {
        if __cpuid(0).eax < 7 {
            return false;
        }
        let leaf7 = __cpuid_count(7, 0);
        const AMX_TILE: u32 = 1 << 24;
        const AMX_INT8: u32 = 1 << 25;
        if leaf7.edx & (AMX_TILE | AMX_INT8) != (AMX_TILE | AMX_INT8) {
            return false;
        }
        const OSXSAVE: u32 = 1 << 27;
        if __cpuid(1).ecx & OSXSAVE == 0 {
            return false;
        }
        const XCR0_TILECFG: u64 = 1 << 17;
        const XCR0_TILEDATA: u64 = 1 << 18;
        let xcr0 = _xgetbv(0);
        xcr0 & (XCR0_TILECFG | XCR0_TILEDATA) == (XCR0_TILECFG | XCR0_TILEDATA)
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuFeatures, KernelSet};

    #[test]
    fn detect_runs() {
        let features = CpuFeatures::detect();
        // The call must be safe on every target; on this x86_64 box AVX2 is
        // expected, but the assertion only checks internal consistency.
        if features.avx512vnni {
            assert!(features.avx512f);
        }
        if features.avx512f || features.avx2 {
            let decode = features.decode_kernels();
            assert_ne!(decode, KernelSet::Scalar);
        }
        if features.amx_int8 {
            assert_eq!(features.prefill_kernels(), KernelSet::AmxInt8);
        }
    }

    #[test]
    fn decode_prefers_vnni() {
        let features = CpuFeatures {
            avx2: true,
            avx512f: true,
            avx512vnni: true,
            amx_int8: true,
        };
        assert_eq!(features.decode_kernels(), KernelSet::Avx512Vnni);
        assert_eq!(features.prefill_kernels(), KernelSet::AmxInt8);
    }
}

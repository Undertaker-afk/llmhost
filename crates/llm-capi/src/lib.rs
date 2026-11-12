//! C ABI for llmhost.
//!
//! Only [`llmhost_version`] is exported in this skeleton. The rest of the
//! surface (model load, session, generate) lands in milestone 6, with a
//! cbindgen header.

use std::ffi::c_char;

/// Return a pointer to a static, NUL-terminated version string (for example
/// `"0.1.0"`). The pointer is valid for the lifetime of the program and must
/// not be freed.
#[no_mangle]
pub extern "C" fn llmhost_version() -> *const c_char {
    c"0.1.0".as_ptr()
}

#[cfg(test)]
mod tests {
    use super::llmhost_version;
    use std::ffi::CStr;

    #[test]
    fn version_matches_package() {
        let ptr = llmhost_version();
        assert!(!ptr.is_null());
        let value = unsafe { CStr::from_ptr(ptr) };
        assert_eq!(value.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}

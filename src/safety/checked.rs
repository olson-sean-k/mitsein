use core::num::NonZeroUsize;

use crate::safety;

impl safety::NonZeroExt<usize> for NonZeroUsize {
    #[cfg(all(not(miri), test))]
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap()
    }
}

impl<T> safety::OptionExt<T> for Option<T> {
    #[cfg(all(not(miri), test))]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap()
    }
}

impl<T, E> safety::ResultExt<T, E> for Result<T, E> {
    #[cfg(all(not(miri), test))]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        match self {
            Ok(value) => value,
            Err(_) => panic!("called `Result::unwrap_maybe_unchecked` on an `Err` value"),
        }
    }
}

/// # Safety
pub const unsafe fn non_zero_from_usize_maybe_unchecked(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        _ => panic!(),
    }
}

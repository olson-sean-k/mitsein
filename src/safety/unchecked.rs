use core::num::NonZeroUsize;

use crate::safety;

impl safety::NonZeroExt<usize> for NonZeroUsize {
    #[inline(always)]
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        NonZeroUsize::new_unchecked(n)
    }
}

impl<T> safety::OptionExt<T> for Option<T> {
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap_unchecked()
    }
}

impl<T, E> safety::ResultExt<T, E> for Result<T, E> {
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap_unchecked()
    }
}

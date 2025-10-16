use core::hint;
use core::num::NonZeroUsize;
use core::slice::SliceIndex;

use crate::safety;

#[cfg(feature = "arrayvec")]
impl<T, const N: usize> safety::ArrayVecExt<T> for arrayvec::ArrayVec<T, N> {
    #[inline(always)]
    unsafe fn push_maybe_unchecked(&mut self, item: T) {
        unsafe { self.push_unchecked(item) }
    }
}

impl safety::NonZeroExt<usize> for NonZeroUsize {
    #[inline(always)]
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        unsafe { NonZeroUsize::new_unchecked(n) }
    }
}

impl<T> safety::OptionExt<T> for Option<T> {
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        unsafe { self.unwrap_unchecked() }
    }
}

impl<T, E> safety::ResultExt<T, E> for Result<T, E> {
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        unsafe { self.unwrap_unchecked() }
    }
}

impl<T> safety::SliceExt<T> for [T] {
    #[inline(always)]
    unsafe fn get_maybe_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        unsafe { self.get_unchecked(index) }
    }

    #[inline(always)]
    unsafe fn get_maybe_unchecked_mut<I>(&mut self, index: I) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        unsafe { self.get_unchecked_mut(index) }
    }
}

/// # Safety
///
/// `n` must be non-zero.
#[inline(always)]
pub const unsafe fn non_zero_from_usize_maybe_unchecked(n: usize) -> NonZeroUsize {
    unsafe { NonZeroUsize::new_unchecked(n) }
}

/// # Safety
///
/// Reaching this function is undefined behavior.
#[inline(always)]
pub const unsafe fn unreachable_maybe_unchecked() -> ! {
    unsafe { hint::unreachable_unchecked() }
}

/// # Safety
///
/// `option` must be [`Some`].
#[inline(always)]
pub const unsafe fn unwrap_option_maybe_unchecked<T>(option: Option<T>) -> T {
    unsafe { option.unwrap_unchecked() }
}

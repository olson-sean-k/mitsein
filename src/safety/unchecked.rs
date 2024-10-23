#[cfg(feature = "arrayvec")]
use arrayvec::ArrayVec;
use core::hint;
use core::num::NonZeroUsize;
use core::slice::SliceIndex;

use crate::safety;

#[cfg(feature = "arrayvec")]
impl<T, const N: usize> safety::ArrayVecExt<T> for ArrayVec<T, N> {
    #[inline(always)]
    unsafe fn push_maybe_unchecked(&mut self, item: T) {
        self.push_unchecked(item)
    }
}

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

impl<T> safety::SliceExt<T> for [T] {
    #[inline(always)]
    unsafe fn get_maybe_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        self.get_unchecked(index)
    }

    #[inline(always)]
    unsafe fn get_maybe_unchecked_mut<I>(&mut self, index: I) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        self.get_unchecked_mut(index)
    }
}

/// # Safety
///
/// `n` must be non-zero.
#[inline(always)]
pub const unsafe fn non_zero_from_usize_maybe_unchecked(n: usize) -> NonZeroUsize {
    NonZeroUsize::new_unchecked(n)
}

/// # Safety
///
/// Reaching this function is undefined behavior.
#[inline(always)]
pub const unsafe fn unreachable_maybe_unchecked() -> ! {
    hint::unreachable_unchecked()
}

use core::num::NonZeroUsize;
use core::slice::SliceIndex;

use crate::safety;

#[cfg(feature = "arrayvec")]
impl<T, const N: usize> safety::ArrayVecExt<T> for arrayvec::ArrayVec<T, N> {
    unsafe fn push_maybe_unchecked(&mut self, item: T) {
        self.push(item)
    }
}

impl safety::NonZeroExt<usize> for NonZeroUsize {
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap()
    }
}

impl<T> safety::OptionExt<T> for Option<T> {
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap()
    }
}

impl<T, E> safety::ResultExt<T, E> for Result<T, E> {
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        match self {
            Ok(value) => value,
            Err(_) => panic!("called `Result::unwrap_maybe_unchecked` on an `Err` value"),
        }
    }
}

impl<T> safety::SliceExt<T> for [T] {
    unsafe fn get_maybe_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        self.get(index).unwrap()
    }

    unsafe fn get_maybe_unchecked_mut<I>(&mut self, index: I) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>,
    {
        self.get_mut(index).unwrap()
    }
}

/// # Safety
///
/// `n` must be non-zero.
pub const unsafe fn non_zero_from_usize_maybe_unchecked(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        _ => panic!(),
    }
}

/// # Safety
///
/// Reaching this function is undefined behavior.
pub const unsafe fn unreachable_maybe_unchecked() -> ! {
    unreachable!()
}

/// # Safety
///
/// `option` must be `Some`.
pub const unsafe fn unwrap_option_maybe_unchecked<T>(option: Option<T>) -> T {
    option.unwrap()
}

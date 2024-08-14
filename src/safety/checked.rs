use core::num::NonZeroUsize;
use core::slice::SliceIndex;

use crate::safety;

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
pub const unsafe fn non_zero_from_usize_maybe_unchecked(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        _ => panic!(),
    }
}

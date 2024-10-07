// Avoid `cfg` attributes in these APIs. This requires more careful auditing, since the linter
// cannot warn when an API is dead.
#![allow(dead_code)]

// Checked implementation of extension traits. Failures panic.
#[cfg(all(not(miri), test))]
#[path = "checked.rs"]
mod maybe;
// Unchecked implementation of extension traits. Failures are ignored or unobserved. This is UB if
// the crate implementation is incorrect or its APIs are unsound!
#[cfg(not(all(not(miri), test)))]
#[path = "unchecked.rs"]
mod maybe;

use core::slice::SliceIndex;

// TODO: At time of writing, traits cannot expose `const` functions. Remove this in favor of
//       extension traits when this is possible.
#[allow(unused_imports)]
pub use maybe::non_zero_from_usize_maybe_unchecked;

pub trait NonZeroExt<T> {
    unsafe fn new_maybe_unchecked(n: T) -> Self;
}

pub trait OptionExt<T> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

pub trait ResultExt<T, E> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

pub trait SliceExt<T> {
    unsafe fn get_maybe_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

    unsafe fn get_maybe_unchecked_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;
}

//! APIs for construction and unwrapping that are checked or unchecked based on the build
//! configuration.
//!
//! This module provides "maybe unchecked" functions for fallible operations. These functions have
//! a `_maybe_unchecked` suffix and mirror `_unchecked` counterparts. When checks are enabled (and
//! so these functions do in fact check), the non-empty guarantee is checked at runtime and so the
//! surface area for potential unsoundness is dramatically reduced. Only unsafe transmutation
//! through transparent types remains (it is unaffected by this module and the build
//! configuration).
//!
//! Functions in this module must always be used just as their `_unchecked` counterparts:
//! invariants must be maintained by callers! The "maybe" merely means that a check may or may not
//! actually occur. This is why `_maybe_unchecked` functions are `unsafe`.
//!
//! Checks are enabled in test builds (i.e., `[#cfg(test)]`) so that bugs manifest as deterministic
//! panics rather than (often silent) undefined behavior. However, checks are **not** enabled for
//! Miri builds (despite also being test builds), because this may prevent Miri from discovering
//! memory safety problems.

// LINT: APIs in this module are pervasive and the set of features that require an item can be
//       volatile and difficult to determine. Dead code is allowed instead, though it requires more
//       careful auditing. Items with a more obvious or simple set of dependent features are
//       annotated with `cfg` or `cfg_attr` when possible. Contrast `OptionExt` with `ArrayVecExt`,
//       for example.
#![allow(dead_code)]

// Checked implementation of extension traits that fails or panics in error conditions.
//
// This implementation is used in test builds, but not Miri builds.
#[cfg(all(not(miri), test))]
#[path = "checked.rs"]
mod maybe;
// Unchecked implementation of extension traits that ignores error conditions and so has undefined
// behavior if such a condition occurs.
//
// This implementation is used in non-test builds and Miri builds.
#[cfg(not(all(not(miri), test)))]
#[path = "unchecked.rs"]
mod maybe;

use core::slice::SliceIndex;

// TODO: At time of writing, traits cannot expose `const` functions. Remove this in favor of
//       extension traits when this is possible.
// LINT: Some of these functions are unused depending on which features are enabled. The set of
//       features may be complicated, so this module prefers `allow` over `cfg_attr` and `expect`.
#[allow(unused_imports)]
pub use maybe::{
    non_zero_from_usize_maybe_unchecked, unreachable_maybe_unchecked, unwrap_option_maybe_unchecked,
};

/// Maybe unchecked extension methods for [`ArrayVec`].
///
/// [`ArrayVec`]: arrayvec::ArrayVec
#[cfg(feature = "arrayvec")]
pub trait ArrayVecExt<T> {
    /// # Safety
    ///
    /// `self` must have non-zero vacancy (length must be less than capacity).
    unsafe fn push_maybe_unchecked(&mut self, item: T);
}

/// Maybe unchecked extension methods for [`NonZero`].
///
/// [`NonZero`]: core::num::NonZero
pub trait NonZeroExt<T> {
    /// # Safety
    ///
    /// `n` must be non-zero.
    unsafe fn new_maybe_unchecked(n: T) -> Self;
}

/// Maybe unchecked extension methods for [`Option`].
pub trait OptionExt<T> {
    /// # Safety
    ///
    /// The `Option` must be [`Some`].
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

/// Maybe unchecked extension methods for [`Result`].
pub trait ResultExt<T, E> {
    /// # Safety
    ///
    /// The `Result` must be [`Ok`].
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

/// Maybe unchecked extension methods for [slices][prim@slice].
pub trait SliceExt<T> {
    /// # Safety
    ///
    /// `index` must be within the bounds of the slice.
    unsafe fn get_maybe_unchecked<I>(&self, index: I) -> &<I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;

    /// # Safety
    ///
    /// `index` must be within the bounds of the slice.
    unsafe fn get_maybe_unchecked_mut<I>(
        &mut self,
        index: I,
    ) -> &mut <I as SliceIndex<[T]>>::Output
    where
        I: SliceIndex<[T]>;
}

/// Tells the compiler to assume that `items` is non-empty. This function is extremely unsafe and
/// should be used only when it has unambiguous performance benefits. See the
/// [documentation for `core::hint::unreachable_unchecked`](core::hint::unreachable_unchecked) for
/// more details.
///
/// # Safety
/// [`items.is_empty()`](crate::MaybeEmptyExt::is_empty) must be false. 
#[inline(always)]
pub unsafe fn assume_is_non_empty_unchecked<T>(items: &T)
    where
        T: crate::sealed::MaybeEmpty,
{
    debug_assert!(!crate::MaybeEmptyExt::is_empty(items));

    if crate::MaybeEmptyExt::is_empty(items) {
        // SAFETY: we inherit the safety guarantees of the function, and so this branch can never be
        // reached
        unsafe { core::hint::unreachable_unchecked() }
    }
}

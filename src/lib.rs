//! Mitsein provides strongly typed APIs for non-empty collections and views, including (but not
//! limited to) iterators, slices, and containers. Where possible, unnecessary branches are omitted
//! and APIs enforce and reflect the non-empty invariant.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, -3];
//! xs.push(2);
//!
//! assert_eq!(&0, xs.first());
//! assert_eq!(6, xs.into_iter1().map(|x| x * 2).map(i64::abs).max());
#![doc = "```"]
//!
//! Note in the above example that operations on `xs` yield non-[optional][`Option`] outputs,
//! because `xs` is non-empty.
//!
//! Modules in this crate reflect corresponding modules in [`core`], [`alloc`], and [`std`], though
//! collection modules are exported at the crate root rather than a `collections` module. For
//! example, [`Vec`] is exported in `alloc::vec::Vec` and its non-empty counterpart [`Vec1`] is
//! exported in `mitsein::vec1::Vec1`. APIs in this crate are typically named by appending `1` to
//! the names of their counterparts.
//!
//! At time of writing, `rustdoc` ignores input type parameters in the "Methods from
//! `Deref<Target = _>`" section. For types that implement `Deref<Target = NonEmpty<_>>`, **the API
//! documentation may be misleading** and list all methods of [`NonEmpty`] regardless of its input
//! type parameter. This is mostly a problem for types that dereference to [`Slice1`], such as
//! [`Vec1`]. See [this `rustdoc` bug](https://github.com/rust-lang/rust/issues/24686).
//!
//! # Non-Empty Types
//!
//! Types that represent non-empty collections, containers, or views present APIs that reflect the
//! non-empty guarantee. The names of these types are formed by appending `1` to the name of their
//! counterparts. For example, the non-empty [`Vec`] type is [`Vec1`].
//!
//! ## Collections
//!
//! This crate provides the following non-empty collections and supporting APIs (depending on which
//! [feature flags](#integrations-and-feature-flags) are enabled):
//!
//! - [`ArrayVec1`][`array_vec1`]
//! - [`BTreeMap1`][`btree_map1`]
//! - [`BTreeSet1`][`btree_set1`]
//! - [`IndexMap1`][`index_map1`]
//! - [`IndexSet1`][`index_set1`]
//! - [`SmallVec1`][`small_vec1`]
//! - [`String1`][`string1`]
//! - [`Vec1`][`mod@vec1`]
//! - [`VecDeque1`][`vec_deque1`]
//!
//! Non-empty collections are represented with the [`NonEmpty`] type constructor. These types are
//! exported as type definitions in their respective modules. Similarly to `std::prelude`, the
//! [`prelude`] module notably re-exports [`Vec1`] and the [`vec1!`] macro.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = Vec1::from_head_and_tail(0i64, [1, 2, 3]);
//! while let Ok(_) = xs.pop_if_many().or_get_only() {}
//!
//! assert_eq!(xs.as_slice(), &[0]);
#![doc = "```"]
//!
//! ## Slices
//!
//! Like collections, non-empty slices are represented with the [`NonEmpty`] type constructor and
//! the [`Slice1`] and [`Str1`] type definitions. These types are unsized and so are accessed via
//! references just like standard slices. The [`prelude`] module re-exports [`Slice1`] and the
//! [`slice1!`] macro.
//!
//! ```rust
//! use mitsein::prelude::*;
//!
//! fn fold_terminals<T, U, F>(xs: &Slice1<T>, f: F) -> U
//! where
//!     F: FnOnce(&T, &T) -> U,
//! {
//!     f(xs.first(), xs.last())
//! }
//!
//! let xs = slice1![0i64, 1, 2, 3];
//! let y = fold_terminals(xs, |first, last| first + last);
//!
//! assert_eq!(y, 3);
//! ```
//!
//! See the [`slice1`][`mod@slice1`] and [`str1`][`mod@str1`] modules.
//!
//! ## Containers
//!
//! This crate provides the following non-empty containers for slices and strings (depending on
//! which [feature flags](#integrations-and-feature-flags) are enabled):
//!
//! - [`ArcSlice1`][`ArcSlice1Ext`]
//! - [`ArcStr1`][`ArcStr1Ext`]
//! - [`BoxedSlice1`][`BoxedSlice1Ext`]
//! - [`BoxedStr1`][`BoxedStr1Ext`]
//! - [`CowSlice1`][`CowSlice1Ext`]
//! - [`CowStr1`][`CowStr1Ext`]
//! - [`RcSlice1`][`RcSlice1Ext`]
//! - [`RcStr1`][`RcStr1Ext`]
//!
//! Each of these type definitions has an accompanying extension trait for operations and
//! conversions that take advantage of the non-empty guarantee. For example, [`ArcSlice1Ext`]
//! provides APIs for non-empty slices in an [`Arc`]. Some collection types like [`Vec1`] support
//! conversions from and into these containers.
//!
//! ## Iterators
//!
//! Non-empty iterators are provided by the [`Iterator1`] type constructor and directly support
//! combinators that never reduce cardinality to zero (e.g., [`map`][`Iterator1::map`]). Non-empty
//! collections and views naturally support iteration, collection, etc. via [`Iterator1`].
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let xs = vec1![0i32, 1, 2];
//! let ys: Vec1<_> = xs.iter1().copied().map(|x| x + 1).collect1();
//!
//! assert_eq!(ys.as_slice(), &[1, 2, 3]);
#![doc = "```"]
//!
//! Supporting traits are re-exported in the [`prelude`] module and provide methods for bridging
//! between non-empty [`Iterator1`] types and maybe-empty [`Iterator`] types.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::iter1;
//! use mitsein::prelude::*;
//!
//! let xs = iter1::head_and_tail(0i32, [1, 2]);
//! // `Iterator1` does not support `skip`, so `xs` is converted into an `Iterator` and then into a
//! // different `Iterator1` via `or_non_empty`.
//! let xs: Vec1<_> = xs.into_iter().skip(3).or_non_empty([3]).collect1();
//!
//! let ys = Vec::new();
//! let ys: Vec1<_> = ys.extend_non_empty([0i32]);
//!
//! assert_eq!(xs.as_slice(), &[3]);
//! assert_eq!(ys.as_slice(), &[0]);
#![doc = "```"]
//!
//! When the `rayon` feature is enabled, the [`ParallelIterator1`] type implements parallel and
//! non-empty iterators.
//!
//! See the [`iter1`] module.
//!
//! ## Arrays
//!
//! Because primitive arrays must contain initialized items at capacity in safe code, the only
//! empty array types are `[_; 0]`. The [`Array1`] trait is implemented for arrays with a non-zero
//! cardinality and provides conversions and operations that take advantage of the non-empty
//! guarantee of such arrays.
//!
//! ```rust
//! use mitsein::prelude::*;
//!
//! let mut xs = [0i64, 1, 2, 3];
//! let x = xs.as_mut_slice1().first_mut();
//! *x = 4;
//!
//! assert_eq!(xs.as_slice(), &[4, 1, 2, 3]);
//! ```
//!
//! At time of writing, it is not possible to implement [`Array1`] for any and all non-empty array
//! types, so this trait is only implemented for arrays with one to 64 items.
//!
//! See the [`array1`] module.
//!
//! # Segmentation
//!
//! A [`Segment`] is a view over a subset of a collection that can mutate both the items and
//! topology of its target. This is somewhat similar to a mutable slice, but items can also be
//! inserted and removed. This crate implements segmentation for both standard and non-empty
//! collections and is one of the most efficient ways to remove and drain items from non-empty
//! collections.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, 2, 3, 4];
//! xs.tail().clear(); // Efficiently clears the tail segment of `xs`.
//!
//! assert_eq!(xs.as_slice(), &[0]);
#![doc = "```"]
//!
//! See the [`segment`] module.
//!
//! # Integrations and Feature Flags
//!
//! Mitsein supports `no_std` environments and provides feature flags for integrating as needed
//! with [`alloc`] and [`std`]. By default, the `std` feature is enabled for complete support of
//! the standard library.
//!
//! The following table summarizes supported feature flags and integrations.
//!
//! | Feature     | Also Enables | Default | Crate         | Description                                               |
//! |-------------|--------------|---------|---------------|-----------------------------------------------------------|
//! | `alloc`     |              | No      | `alloc`       | Non-empty collections that allocate, like `Vec1`.         |
//! | `arbitrary` | `std`        | No      | [`arbitrary`] | Construction of arbitrary non-empty collections.          |
//! | `arrayvec`  |              | No      | [`arrayvec`]  | Non-empty implementations of [`arrayvec`] types.          |
//! | `either`    |              | No      | [`either`]    | Non-empty iterator implementation for `Either`.           |
//! | `indexmap`  | `alloc`      | No      | [`indexmap`]  | Non-empty implementations of [`indexmap`] types.          |
//! | `itertools` | `either`     | No      | [`itertools`] | Combinators from [`itertools`] for `Iterator1`.           |
//! | `rayon`     | `std`        | No      | [`rayon`]     | Parallel operations for non-empty types.                  |
//! | `schemars`  | `alloc`      | No      | [`schemars`]  | JSON schema generation for non-empty types.               |
//! | `serde`     |              | No      | [`serde`]     | De/serialization of non-empty collections with [`serde`]. |
//! | `smallvec`  | `alloc`      | No      | [`smallvec`]  | Non-empty implementations of [`smallvec`] types.          |
//! | `std`       | `alloc`      | Yes     | `std`         | Integrations with `std::io`.                              |
//!
//! [`Arc`]: alloc::sync::Arc
//! [`ArcSlice1Ext`]: crate::sync1::ArcSlice1Ext
//! [`ArcStr1Ext`]: crate::sync1::ArcStr1Ext
//! [`Array1`]: crate::array1::Array1
//! [`arbitrary`]: https://crates.io/crates/arbitrary
//! [`arrayvec`]: https://crates.io/crates/arrayvec
//! [`BoxedSlice1Ext`]: crate::boxed1::BoxedSlice1Ext
//! [`BoxedStr1Ext`]: crate::boxed1::BoxedStr1Ext
//! [`CowSlice1Ext`]: crate::borrow1::CowSlice1Ext
//! [`CowStr1Ext`]: crate::borrow1::CowStr1Ext
//! [`either`]: https://crates.io/crates/either
//! [`indexmap`]: https://crates.io/crates/indexmap
//! [`Iterator1`]: crate::iter1::Iterator1
//! [`Iterator1::map`]: crate::iter1::Iterator1::map
//! [`itertools`]: https://crates.io/crates/itertools
//! [`ParallelIterator1`]: crate::iter1::ParallelIterator1
//! [`rayon`]: https://crates.io/crates/rayon
//! [`RcSlice1Ext`]: crate::rc1::RcSlice1Ext
//! [`RcStr1Ext`]: crate::rc1::RcStr1Ext
//! [`schemars`]: https://crates.io/crates/schemars
//! [`segment`]: crate::segment
//! [`Segment`]: crate::segment::Segment
//! [`serde`]: https://crates.io/crates/serde
//! [`Slice1`]: crate::slice1::Slice1
//! [`smallvec`]: https://crates.io/crates/smallvec
//! [`Str1`]: crate::str1::Str1
//! [`Vec`]: alloc::vec::Vec
//! [`Vec1`]: crate::vec1::Vec1

// TODO: At time of writing, it is not possible to specify or enable features required for
//       documentation examples without explicitly applying `doc` attributes. These attributes harm
//       the legibility of non-rendered documentation. Migrate this to a cleaner mechanism when
//       possible.

// SAFETY: This crate implements non-empty collections, slices, and iterators. This non-empty
//         invariant is critical to memory safety and soundness, because these implementations use
//         unsafe code to omit branches and checks that are unnecessary if and only if the
//         invariant holds.
//
//         The non-empty invariant is pervasive: code within `unsafe` blocks tends to be
//         uninteresting and is very unlikely to be the source of memory safety bugs. It is trivial
//         and more likely for safe code to break this invariant! Sometimes maintaining this
//         invariant can be subtle, such as interactions with `drain` and `Ord`.
//
//         Most unsafe code in this crate falls into the following basic categories:
//
//           1. The code implements a `NonEmpty` type and therefore assumes that `self` is indeed
//              non-empty.
//           2. The code converts into or from another `NonEmpty` type and so assumes that the
//              input is non-empty.
//           3. The code explicitly checks the invariant or constructs non-empty data.
//
//         Unsafe code in categories (1) and (2) relies on the implementation and public APIs of
//         `NonEmpty` types. Unsafe code in category (3) relies on much more local checks or
//         enforcement of the non-empty invariant.
//
//         This crate must trust undocumented or unpromised behavior of foreign code. For example,
//         `Iterator1` combinators assume that implementations in `core` and `itertools` function
//         in a particular way. If `Map` were ever to discard an item, then `Iterator1::map` would
//         be unsound, for example. As another example, `SwapDrainSegment` relies on details of the
//         `Vec::drain` implementation. For the overwhelming majority of this code, a change in
//         behavior that this crate relies upon for memory safety would be a major bug in the
//         upstream package and is very unlikely to occur.

#![cfg_attr(docsrs, feature(doc_cfg))]
// LINT: The serialization implementations for `NonEmpty<T>` rely on conversions between
//       `Serde<NonEmpty<T>>` and `NonEmpty<T>`. These implementations require `NonEmpty<T>:
//       Clone`, which implies `T: Sized` (because `Clone` requires `Sized`). This is expected,
//       because serialization of `NonEmpty<T>` where `T` is unsized is not possible. These
//       implementations apply only to `Sized` and owning `NonEmpty` types, such as
//       `NonEmpty<Vec<U>>`.
//
//       This attribute is applied to the crate, because it is not possible to apply it to the
//       implementations generated by procedural macros.
#![cfg_attr(feature = "serde", expect(clippy::needless_maybe_sized))]
#![deny(
    clippy::cast_lossless,
    clippy::checked_conversions,
    clippy::cloned_instead_of_copied,
    clippy::explicit_into_iter_loop,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::from_iter_instead_of_collect,
    clippy::if_not_else,
    clippy::manual_ok_or,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_else,
    clippy::unreadable_literal,
    clippy::unused_self
)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod safety;
mod schemars;
mod serde;
mod take;

pub mod array1;
pub mod array_vec1;
pub mod borrow1;
pub mod boxed1;
pub mod btree_map1;
pub mod btree_set1;
pub mod cmp;
pub mod index_map1;
pub mod index_set1;
pub mod iter1;
pub mod rc1;
pub mod segment;
pub mod slice1;
pub mod small_vec1;
pub mod str1;
pub mod string1;
pub mod sync1;
pub mod vec1;
pub mod vec_deque1;

mod sealed {
    use crate::Cardinality;

    /// # Safety
    ///
    /// The implementation of this trait determines whether or not a collection or view is empty.
    /// This query is used to construct non-empty types, and an inconsistent implementation is
    /// unsound. In particular, it is unsound for [`MaybeEmpty::cardinality`] implementations to
    /// return `Some` when `Self` is empty.
    pub unsafe trait MaybeEmpty: Sized {
        fn cardinality(&self) -> Option<Cardinality<(), ()>>;
    }
}
use crate::sealed::*;

pub mod prelude {
    //! Re-exports of recommended APIs and extension traits for glob imports.

    pub use crate::array1::Array1;
    #[cfg(feature = "indexmap")]
    pub use crate::index_map1::OrOnlyEntryExt as _;
    #[cfg(feature = "either")]
    pub use crate::iter1::EitherExt as _;
    pub use crate::iter1::{
        Extend1, FromIterator1, IntoIterator1, IteratorExt as _, ThenIterator1,
    };
    #[cfg(feature = "rayon")]
    pub use crate::iter1::{FromParallelIterator1, IntoParallelIterator1};
    #[cfg(any(feature = "arrayvec", feature = "alloc"))]
    pub use crate::segment::{Segmentation, Tail};
    pub use crate::slice1::{Slice1, slice1};
    pub use crate::str1::Str1;
    #[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
    pub use crate::sync1::{
        ArcSlice1Ext as _, ArcStr1Ext as _, WeakSlice1Ext as _, WeakStr1Ext as _,
    };
    #[cfg(feature = "alloc")]
    pub use {
        crate::borrow1::{CowSlice1Ext as _, CowStr1Ext as _},
        crate::boxed1::{BoxedSlice1Ext as _, BoxedStr1Ext as _},
        crate::btree_map1::OrOnlyEntryExt as _,
        crate::rc1::{RcSlice1Ext as _, RcStr1Ext as _, WeakSlice1Ext as _, WeakStr1Ext as _},
        crate::string1::String1,
        crate::vec1::{Vec1, vec1},
    };
}

#[cfg(feature = "serde")]
use ::serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::mem;
use core::num::NonZeroUsize;

#[cfg(feature = "serde")]
use crate::serde::Serde;

#[cfg(any(feature = "arrayvec", feature = "alloc"))]
pub use take::TakeIfMany;

const EMPTY_ERROR_MESSAGE: &str = "failed to construct non-empty collection: no items";

trait MaybeEmptyExt: MaybeEmpty {
    fn map_non_empty<T, F>(self, f: F) -> Result<T, EmptyError<Self>>
    where
        F: FnOnce(Self) -> T;

    fn is_empty(&self) -> bool;
}

// This blanket implementation of an extension trait prevents different behaviors of
// `map_non_empty` for different `MaybeEmpty` types (as opposed to a default implementation in
// `MaybeEmpty`, for example). This is important for memory safety, because `FromMaybeEmpty` relies
// on this behavior with unsafe code.
impl<T> MaybeEmptyExt for T
where
    T: MaybeEmpty,
{
    fn map_non_empty<U, F>(self, f: F) -> Result<U, EmptyError<Self>>
    where
        F: FnOnce(Self) -> U,
    {
        if self.is_empty() {
            Err(EmptyError::from_empty(self))
        }
        else {
            Ok(f(self))
        }
    }

    fn is_empty(&self) -> bool {
        self.cardinality().is_none()
    }
}

trait NonZeroExt<T> {
    fn clamped(n: T) -> Self;
}

impl NonZeroExt<usize> for NonZeroUsize {
    fn clamped(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }
}

trait FromMaybeEmpty<T>: Sized
where
    T: MaybeEmpty,
{
    fn try_from_maybe_empty(items: T) -> Result<Self, EmptyError<T>> {
        items.map_non_empty(|items| {
            // SAFETY: The `map_non_empty` function only executes this code if `items` is
            //         non-empty.
            unsafe { Self::from_maybe_empty_unchecked(items) }
        })
    }

    /// # Safety
    ///
    /// `items` must be non-empty. See [`MaybeEmpty::cardinality`].
    unsafe fn from_maybe_empty_unchecked(items: T) -> Self;
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct EmptyError<T> {
    items: T,
}

impl<T> EmptyError<T> {
    fn from_empty(items: T) -> Self {
        EmptyError { items }
    }

    pub fn into_empty(self) -> T {
        self.items
    }

    pub fn take(self) -> (T, EmptyError<()>) {
        (self.items, EmptyError::from_empty(()))
    }

    pub fn take_and_drop(self) -> EmptyError<()> {
        EmptyError::from_empty(())
    }

    pub fn as_empty(&self) -> &T {
        &self.items
    }
}

impl<T> EmptyError<&'_ T> {
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_owning(self) -> EmptyError<T::Owned>
    where
        T: ToOwned,
    {
        EmptyError::from_empty(self.items.to_owned())
    }
}

impl<T> Debug for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("EmptyError").finish_non_exhaustive()
    }
}

impl<T> Display for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{EMPTY_ERROR_MESSAGE}")
    }
}

impl<T> Error for EmptyError<T> {}

#[cfg_attr(
    feature = "serde",
    derive(::serde_derive::Deserialize, ::serde_derive::Serialize)
)]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            deserialize = "Self: TryFrom<Serde<T>, Error = EmptyError<T>>, \
                           T: Deserialize<'de>,",
            serialize = "T: Clone + Serialize,",
        ),
        try_from = "Serde<T>",
        into = "Serde<T>",
    )
)]
#[derive(Clone, Copy, Hash)]
#[repr(transparent)]
pub struct NonEmpty<T>
where
    T: ?Sized,
{
    items: T,
}

impl<T> NonEmpty<T>
where
    T: Sized,
{
    pub fn and_then_try<F>(self, f: F) -> Result<Self, EmptyError<T>>
    where
        // A bound on `FromMaybeEmpty` would be more direct, but that trait is not part of the
        // public API. `TryFrom` is used instead, since it is implemented in terms of
        // `FromMaybeEmpty`.
        Self: TryFrom<T, Error = EmptyError<T>>,
        F: FnOnce(&mut T),
    {
        let NonEmpty { mut items } = self;
        f(&mut items);
        <NonEmpty<T> as TryFrom<T>>::try_from(items)
    }
}

#[cfg(any(feature = "arrayvec", feature = "alloc"))]
impl<T> NonEmpty<T>
where
    T: MaybeEmpty + ?Sized,
{
    fn cardinality(&self) -> Cardinality<(), ()> {
        match self.items.cardinality() {
            // SAFETY: `self.items` must be non-empty.
            None => unsafe { safety::unreachable_maybe_unchecked() },
            Some(cardinality) => cardinality,
        }
    }

    #[cfg(feature = "alloc")]
    fn as_cardinality_items_mut(&mut self) -> Cardinality<&mut T, &mut T> {
        match self.cardinality() {
            Cardinality::One(_) => Cardinality::One(&mut self.items),
            Cardinality::Many(_) => Cardinality::Many(&mut self.items),
        }
    }
}

impl<T> AsRef<T> for NonEmpty<T>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &self.items
    }
}

impl<T> Eq for NonEmpty<T>
where
    Self: PartialEq,
    T: Eq + ?Sized,
{
}

impl<T> FromMaybeEmpty<T> for NonEmpty<T>
where
    T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: T) -> Self {
        NonEmpty { items }
    }
}

impl<'a, T> FromMaybeEmpty<&'a T> for &'a NonEmpty<T>
where
    T: ?Sized,
    &'a T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: &'a T) -> Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `T` and
        //         `NonEmpty<T>` are the same.
        unsafe { mem::transmute::<&'_ T, &'_ NonEmpty<T>>(items) }
    }
}

impl<'a, T> FromMaybeEmpty<&'a mut T> for &'a mut NonEmpty<T>
where
    T: ?Sized,
    &'a mut T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: &'a mut T) -> Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `T` and
        //         `NonEmpty<T>` are the same.
        unsafe { mem::transmute::<&'_ mut T, &'_ mut NonEmpty<T>>(items) }
    }
}

impl<T> Ord for NonEmpty<T>
where
    Self: PartialOrd,
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.items, &other.items)
    }
}

impl<T, U> PartialEq<NonEmpty<U>> for NonEmpty<T>
where
    T: PartialEq<U> + ?Sized,
    U: ?Sized,
{
    fn eq(&self, other: &NonEmpty<U>) -> bool {
        PartialEq::eq(&self.items, &other.items)
    }
}

impl<T, U> PartialOrd<NonEmpty<U>> for NonEmpty<T>
where
    T: PartialOrd<U> + ?Sized,
    U: ?Sized,
{
    fn partial_cmp(&self, other: &NonEmpty<U>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.items, &other.items)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Cardinality<O, M> {
    One(O),
    Many(M),
}

impl<O, M> Cardinality<O, M> {
    pub fn one(self) -> Option<O> {
        match self {
            Cardinality::One(one) => Some(one),
            _ => None,
        }
    }

    pub fn many(self) -> Option<M> {
        match self {
            Cardinality::Many(many) => Some(many),
            _ => None,
        }
    }

    pub fn map_one<U, F>(self, f: F) -> Cardinality<U, M>
    where
        F: FnOnce(O) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(f(one)),
            Cardinality::Many(many) => Cardinality::Many(many),
        }
    }

    pub fn map_many<U, F>(self, f: F) -> Cardinality<O, U>
    where
        F: FnOnce(M) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(one),
            Cardinality::Many(many) => Cardinality::Many(f(many)),
        }
    }
}

impl<T> Cardinality<T, T> {
    pub fn map<U, F>(self, f: F) -> Cardinality<U, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(f(one)),
            Cardinality::Many(many) => Cardinality::Many(f(many)),
        }
    }
}

macro_rules! with_literals {
    ($f:ident$(,)?) => {};
    ($f:ident, [$($N:literal $(,)?)+]$(,)?) => {
        $(
            $f!($N);
        )+
    };
}
pub(crate) use with_literals;

macro_rules! with_tuples {
    ($f:ident, ($head:ident $(,)?) $(,)?) => {
        $f!(($head,));
    };
    ($f:ident, ($head:ident, $($tail:ident $(,)?)+) $(,)?) => {
        $f!(($head, $($tail,)+));
        $crate::with_tuples!($f, ($($tail,)+));
    };
}
pub(crate) use with_tuples;

macro_rules! impl_partial_eq_for_non_empty {
    (
        [$(for $right:ident$(,)?)? in $rhs:ty]$(,)?
        ==
        [$(for $left:ident$(,)?)? in $lhs:ty]$(,)?
    ) => {
        impl<$($right,)? $($left)?> ::core::cmp::PartialEq<$rhs> for $lhs
        $(where
            $left: ::core::cmp::PartialEq<$right>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, &rhs.items)
            }
        }
    };
    (
        [$(for $right:ident $(,const $n:ident: usize)?$(,)?)? in $rhs:ty]
        <=
        [$(for $left:ident$(,)?)? in $lhs:ty]$(,)?
    ) => {
        impl<$($right,)? $($left,)? $($(const $n: usize)?)?> ::core::cmp::PartialEq<$rhs> for $lhs
        $(where
            $left: ::core::cmp::PartialEq<$right>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, rhs)
            }
        }
    };
    (
        [$(for $right:ident$(,)?)? in $rhs:ty]
        =>
        [$(for $left:ident $(,const $n:ident: usize)?$(,)?)? in $lhs:ty]$(,)?
    ) => {
        impl<$($right,)? $($left,)? $($(const $n: usize)?)?> ::core::cmp::PartialEq<$rhs> for $lhs
        $(where
            $left: ::core::cmp::PartialEq<$right>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(self, &rhs.items)
            }
        }
    };
}
pub(crate) use impl_partial_eq_for_non_empty;

#[cfg(all(test, feature = "alloc"))]
pub mod harness {
    pub trait KeyValueRef {
        type Cloned;

        fn cloned(&self) -> Self::Cloned;
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a mut V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }
}

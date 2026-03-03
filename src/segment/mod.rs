//! Segmentation of items in ordered collections.
//!
//! A [`Segment`] isolates a contiguous and ordered subset of an ordered collection via a range.
//! **The primary feature of segmentation is efficient mass removal of items from [`NonEmpty`]
//! collections,** though standard maybe-empty collections and other features are also supported.
//! Roughly speaking, [`Segment`]s are somewhat like [slices][`prim@slice`]: they provide a view
//! into a collection through a reference. However, unlike [slices][`prim@slice`], [`Segment`]s can
//! change the _topology_ of a collection by inserting and removing items.
//!
//! Segmentation is strictly mutable and always requires an exclusive borrow of a collection.
//! [`Segment`]s are ephemeral types that are meant to be used very locally. It is possible to
//! stash them in data structures, but this is not recommended; prefer moving or borrowing
//! collections instead.
//!
//! # Construction
//!
//! [`Segment`]s can be constructed from range types supported by a collection or nominally from a
//! predefined range (namely a tail or reverse tail).
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, -3];
//!
//! // Construct a `Segment` over the range `1..` in `xs`. This is fallible.
//! let xss = xs.segment(1..).unwrap();
//! assert_eq!(xss.as_slice(), &[1, -3]);
//!
//! // Construct a nominal `Segment` over the tail of `xs`. This is infallible.
//! let xss = xs.tail();
//! assert_eq!(xss.as_slice(), &[1, -3]);
#![doc = "```"]
//!
//! Explicit segmentation over supported range types is provided by the [`ByRange`] trait and
//! nominal segmentation over the tail and reverse tail ranges is provided by the [`ByTail`] trait.
//! Both of these traits are anonymously exported in the [`prelude`] module.
//!
//! # Non-Empty vs. Maybe-Empty Collections
//!
//! Segmentation differs in one crucial way regarding non-empty and maybe-empty collections: the
//! strictness of subsets. **[`NonEmpty`] types can only be segmented into strict subsets,** and so
//! a [`Segment`] can never span all items of a non-empty collection.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec![0i64, 1];
//! // This segment contains all items in `xs`. This is okay, because `xs` is a maybe-empty `Vec`.
//! let mut xss = xs.segment(..).unwrap();
//! xss.clear();
//!
//!
//! let mut xs = vec1![0i64, 1];
//! // This panics, because the range `..` contains all items in `xs`, which is a non-empty `Vec1`.
//! let mut xss = xs.segment(..).unwrap();
//! xss.clear();
#![doc = "```"]
//!
//! # Positional vs. Relational Segments
//!
//! [`Segment`]s are parameterized by internal range types that are chosen for efficient operation
//! against a given collection. These range types differ, but are roughly divided into one of two
//! exclusive modes: _positional_ and _relational_.
//!
//! Positional [`Segment`]s represent a subset based on items covered by a range over positions or
//! indices in a collection. These [`Segment`]s support insertions and removals at the terminals of
//! the range, meaning that **positional [`Segment`]s can grow and shrink.** Collections that index
//! items by position produce positional [`Segment`]s. For example, [`ArrayVec1`] and [`Vec1`] are
//! both positional.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, 2, 3];
//!
//! let mut xss = xs.segment(1..4).unwrap();
//! xss.insert_back(42);
//! assert_eq!(xss.as_slice(), &[1, 2, 42]);
//! assert_eq!(xs.as_slice(), &[0, 1, 2, 42, 3]);
//!
//! let mut xss = xs.segment(1..4).unwrap();
//! let x = xss.remove_back();
//! assert_eq!(xss.as_slice(), &[1, 2]);
//! assert_eq!(xs.as_slice(), &[1, 2, 3]);
//! assert_eq!(x, Some(42));
#![doc = "```"]
//!
//! Relational [`Segment`]s represent a subset based on the intrinsic [ordering][`Ord`] of the
//! items in a collection. The terminals of these [`Segment`]s are fixed and items cannot be
//! inserted beyond the range. However, because the range represents a minimum and maximum for the
//! item type, the number of items within this range can vary wildly and is completely dependent on
//! the total ordering of the item type.
//!
//! # Examples
//!
//! TBD.
//!
//! [`ArrayVec1`]: crate::array_vec1::ArrayVec1
//! [`NonEmpty`]: crate::NonEmpty
//! [`prelude`]: crate::prelude
//! [`Vec1`]: crate::vec1::Vec1

// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound. The
//         segmentation APIs interact with non-empty collections and bugs here may break the
//         non-empty invariant. In particular, range types, intersection, and projection must be
//         correct.

#![cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#![cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]

pub(crate) mod range;

use core::cmp;
use core::fmt::{self, Debug, Formatter};
use core::ops::RangeBounds;

#[cfg(feature = "alloc")]
use crate::segment::range::TrimRange;
use crate::segment::range::{IndexRange, Intersect, Project};

pub use crate::segment::range::{IntoRangeBounds, OutOfBoundsError, RangeError, UnorderedError};

pub trait Segmentation {
    type Kind: Segmentation<Target = Self::Target> + ?Sized;
    type Target: ?Sized;
}

// TODO: Support segmentation over a query type `Q` borrowed from a key or owned index type `K`.
//       Note that any type `Q` must be isomorphic with `K` (i.e., implement `UnsafeOrdIsomorph`),
//       because the `Ord` implementations can disagree and expose items outside of the segment's
//       range. This can be unsound, such as removing an item from a non-empty collection that is
//       out of the segment's range.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be segmented by the type `{R}` over `{N}`",
    label = "segmented by the type `{R}` here",
    note = "positional collections are typically segmented by ranges over `usize`",
    note = "relational collections are typically segmented by ranges over the item type"
)]
pub trait ByRange<N, R>: Segmentation
where
    N: ?Sized,
{
    type Range;
    type Error;

    // LINT: Though this type is quite complex, the indirection introduced by a type defintion is
    //       arguably less clear and a bit trickier to understand.
    #[allow(clippy::type_complexity)]
    fn segment(
        &mut self,
        range: R,
    ) -> Result<Segment<'_, Self::Kind, Self::Target, Self::Range>, Self::Error>;
}

pub trait ByTail: Segmentation {
    type Range;

    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Range>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Range>;
}

#[must_use]
pub struct Segment<'a, K, T, R>
where
    K: Segmentation<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: R,
}

impl<'a, K, T, R> Segment<'a, K, T, R>
where
    K: Segmentation<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn unchecked(items: &'a mut K::Target, range: R) -> Self {
        Segment { items, range }
    }

    pub(crate) fn rekind<L>(self) -> Segment<'a, L, T, R>
    where
        L: Segmentation<Target = T> + ?Sized,
    {
        let Segment { items, range } = self;
        Segment::unchecked(items, range)
    }
}

impl<K, T> Segment<'_, K, T, IndexRange>
where
    K: Segmentation<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn from_tail_range(items: &mut T, n: usize) -> Segment<'_, K, T, IndexRange> {
        // A segment over the range `[1,n)` is always valid for a positional (indexed) collection.
        // For an empty collection, this range is adjacent to `[0,1)` at the index `1` and so the
        // segment can perform insertions and removals (even though it is empty).
        Segment::unchecked(items, IndexRange::unchecked(1, cmp::max(1, n)))
    }

    pub(crate) fn from_rtail_range(items: &mut T, n: usize) -> Segment<'_, K, T, IndexRange> {
        // A segment over the range `[0,n-1)` is always valid for a positional (indexed)
        // collection. For an empty collection, this range is adjacent to `[0,1)` at the index `0`
        // and so the segment can perform insertions and removals (even though it is empty).
        Segment::unchecked(items, IndexRange::unchecked(0, n.saturating_sub(1)))
    }

    pub(crate) fn intersected<R>(
        items: &mut T,
        n: usize,
        range: R,
    ) -> Result<Segment<'_, K, T, IndexRange>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        IndexRange::unchecked(0, n)
            .intersect(range)
            .map(|range| Segment::unchecked(items, range))
    }

    pub(crate) fn intersected_strict_subset<R>(
        items: &mut T,
        n: usize,
        range: R,
    ) -> Result<Segment<'_, K, T, IndexRange>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        let all = IndexRange::unchecked(0, n);
        let range = all.intersect(range)?;
        if range.is_strict_subset_of(&all) {
            Ok(Segment::unchecked(items, range))
        }
        else {
            let (start, end) = range.into();
            Err(OutOfBoundsError::Range(start, end).into())
        }
    }

    pub(crate) fn project_and_intersect<R>(
        &mut self,
        range: R,
    ) -> Result<Segment<'_, K, T, IndexRange>, RangeError<usize>>
    where
        IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
        R: RangeBounds<usize>,
    {
        let range = self.range.intersect(self.range.project(range)?)?;
        Ok(Segment::unchecked(self.items, range))
    }

    pub(crate) fn project_tail_range(&mut self) -> Segment<'_, K, T, IndexRange> {
        let range = self.range.project(1..).unwrap();
        Segment::unchecked(self.items, range)
    }

    pub(crate) fn project_rtail_range(&mut self, n: usize) -> Segment<'_, K, T, IndexRange> {
        let range = self.range.project(..n.saturating_sub(1)).unwrap();
        Segment::unchecked(self.items, range)
    }
}

#[cfg(feature = "alloc")]
impl<K, T> Segment<'_, K, T, TrimRange>
where
    K: Segmentation<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn advance_tail_range(&mut self) -> Segment<'_, K, T, TrimRange> {
        let range = self.range.tail();
        Segment::unchecked(self.items, range)
    }

    pub(crate) fn advance_rtail_range(&mut self) -> Segment<'_, K, T, TrimRange> {
        let range = self.range.rtail();
        Segment::unchecked(self.items, range)
    }

    pub(crate) fn untrimmed_item_count(&self, n: usize) -> usize {
        n.saturating_sub(self.range.trimmed_item_count())
    }
}

impl<K, T, R> Debug for Segment<'_, K, T, R>
where
    K: Segmentation<Target = T> + ?Sized,
    K::Target: Debug,
    T: Debug,
    R: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Segment")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

impl<K, T, R> Segmentation for Segment<'_, K, T, R>
where
    K: Segmentation<Target = T> + ?Sized,
    T: ?Sized,
{
    type Kind = K;
    type Target = K::Target;
}

//! Segmentation of items in ordered collections.

// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound. The
//         segmentation APIs interact with non-empty collections and bugs here may break the
//         non-empty invariant. In particular, range types, intersection, and projection must be
//         correct.

#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

pub(crate) mod range;

use core::cmp;
use core::fmt::{self, Debug, Formatter};
use core::ops::RangeBounds;

#[cfg(feature = "alloc")]
use crate::segment::range::TrimRange;
use crate::segment::range::{IndexRange, Intersect, Project};

pub use crate::segment::range::{IntoRangeBounds, OutOfBoundsError, RangeError, UnorderedError};

pub trait Segmentation: Sized {
    type Kind: Segmentation<Target = Self::Target>;
    type Target;
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
pub trait Query<N, R>: Segmentation
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

pub trait Tail: Segmentation {
    type Range;

    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Range>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Range>;
}

pub struct Segment<'a, K, T, R>
where
    K: Segmentation<Target = T>,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: R,
}

impl<'a, K, T, R> Segment<'a, K, T, R>
where
    K: Segmentation<Target = T> + ?Sized,
{
    pub(crate) fn unchecked(items: &'a mut K::Target, range: R) -> Self {
        Segment { items, range }
    }

    pub(crate) fn rekind<L>(self) -> Segment<'a, L, T, R>
    where
        L: Segmentation<Target = T>,
    {
        let Segment { items, range } = self;
        Segment::unchecked(items, range)
    }
}

impl<K, T> Segment<'_, K, T, IndexRange>
where
    K: Segmentation<Target = T> + ?Sized,
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
    K: Segmentation<Target = T>,
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
    K: Segmentation<Target = T>,
{
    type Kind = K;
    type Target = K::Target;
}

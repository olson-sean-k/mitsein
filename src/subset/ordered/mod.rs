//! Subsets of collections by range.

#![cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#![cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]

pub(crate) mod range;

use core::cmp;
use core::fmt::{self, Debug, Formatter};
use core::ops::RangeBounds;

use crate::subset::SubsetFor;
#[cfg(feature = "alloc")]
use crate::subset::ordered::range::TrimRange;
use crate::subset::ordered::range::{IndexRange, Intersect, Project};

pub use crate::subset::ordered::range::{
    IntoRangeBounds, OutOfBoundsError, RangeError, UnorderedError,
};

// TODO: Support ranges over a query type `Q` borrowed from a key or owned index type `K`. Note that
//       any type `Q` must be isomorphic with `K` (i.e., implement `UnsafeOrdIsomorph`), because the
//       `Ord` implementations can disagree and expose items outside of the subset.
#[diagnostic::on_unimplemented(
    message = "A subset of `{Self}` cannot be constructed from the range type `{R}` over `{N}`.",
    label = "constructed from the range type `{R}` here",
    note = "subsets of positional collections are typically constructed from ranges over `usize`",
    note = "subsets of relational collections are typically constructed from ranges over the item type"
)]
pub trait ByRange<N, R>: SubsetFor
where
    N: ?Sized,
{
    type Range;
    type Error;

    // LINT: Though this type is quite complex, the indirection introduced by a type defintion is
    //       arguably less clear and a bit trickier to understand.
    #[allow(clippy::type_complexity)]
    fn only(
        &mut self,
        range: R,
    ) -> Result<OnlyRangeSubset<'_, Self::Kind, Self::Target, Self::Range>, Self::Error>;
}

pub trait ByTail: SubsetFor {
    type Range;

    fn tail(&mut self) -> OnlyRangeSubset<'_, Self::Kind, Self::Target, Self::Range>;

    fn rtail(&mut self) -> OnlyRangeSubset<'_, Self::Kind, Self::Target, Self::Range>;
}

#[must_use]
pub struct OnlyRangeSubset<'a, K, T, R>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: R,
}

impl<'a, K, T, R> OnlyRangeSubset<'a, K, T, R>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn unchecked(items: &'a mut K::Target, range: R) -> Self {
        OnlyRangeSubset { items, range }
    }

    pub(crate) fn rekind<L>(self) -> OnlyRangeSubset<'a, L, T, R>
    where
        L: SubsetFor<Target = T> + ?Sized,
    {
        let OnlyRangeSubset { items, range } = self;
        OnlyRangeSubset::unchecked(items, range)
    }
}

impl<K, T> OnlyRangeSubset<'_, K, T, IndexRange>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn from_tail_range(
        items: &mut T,
        n: usize,
    ) -> OnlyRangeSubset<'_, K, T, IndexRange> {
        // A subset over the range `[1,n)` is always valid for a positional (indexed) collection.
        // For an empty collection, this range is adjacent to `[0,1)` at the index `1` and so the
        // subset can perform insertions and removals (even though it is empty).
        OnlyRangeSubset::unchecked(items, IndexRange::unchecked(1, cmp::max(1, n)))
    }

    pub(crate) fn from_rtail_range(
        items: &mut T,
        n: usize,
    ) -> OnlyRangeSubset<'_, K, T, IndexRange> {
        // A subset over the range `[0,n-1)` is always valid for a positional (indexed) collection.
        // For an empty collection, this range is adjacent to `[0,1)` at the index `0` and so the
        // subset can perform insertions and removals (even though it is empty).
        OnlyRangeSubset::unchecked(items, IndexRange::unchecked(0, n.saturating_sub(1)))
    }

    pub(crate) fn intersected<R>(
        items: &mut T,
        n: usize,
        range: R,
    ) -> Result<OnlyRangeSubset<'_, K, T, IndexRange>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        IndexRange::unchecked(0, n)
            .intersect(range)
            .map(|range| OnlyRangeSubset::unchecked(items, range))
    }

    pub(crate) fn intersected_strict_subset<R>(
        items: &mut T,
        n: usize,
        range: R,
    ) -> Result<OnlyRangeSubset<'_, K, T, IndexRange>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        let all = IndexRange::unchecked(0, n);
        let range = all.intersect(range)?;
        if range.is_strict_subset_of(&all) {
            Ok(OnlyRangeSubset::unchecked(items, range))
        }
        else {
            let (start, end) = range.into();
            Err(OutOfBoundsError::Range(start, end).into())
        }
    }

    pub(crate) fn project_and_intersect<R>(
        &mut self,
        range: R,
    ) -> Result<OnlyRangeSubset<'_, K, T, IndexRange>, RangeError<usize>>
    where
        IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
        R: RangeBounds<usize>,
    {
        let range = self.range.intersect(self.range.project(range)?)?;
        Ok(OnlyRangeSubset::unchecked(self.items, range))
    }

    pub(crate) fn project_tail_range(&mut self) -> OnlyRangeSubset<'_, K, T, IndexRange> {
        let range = self.range.project(1..).unwrap();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn project_rtail_range(
        &mut self,
        n: usize,
    ) -> OnlyRangeSubset<'_, K, T, IndexRange> {
        let range = self.range.project(..n.saturating_sub(1)).unwrap();
        OnlyRangeSubset::unchecked(self.items, range)
    }
}

#[cfg(feature = "alloc")]
impl<K, T> OnlyRangeSubset<'_, K, T, TrimRange>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: ?Sized,
{
    pub(crate) fn advance_tail_range(&mut self) -> OnlyRangeSubset<'_, K, T, TrimRange> {
        let range = self.range.tail();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn advance_rtail_range(&mut self) -> OnlyRangeSubset<'_, K, T, TrimRange> {
        let range = self.range.rtail();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn untrimmed_item_count(&self, n: usize) -> usize {
        n.saturating_sub(self.range.trimmed_item_count())
    }
}

impl<K, T, R> Debug for OnlyRangeSubset<'_, K, T, R>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: Debug + ?Sized,
    R: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OnlyRangeSubset")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

impl<K, T, R> SubsetFor for OnlyRangeSubset<'_, K, T, R>
where
    K: SubsetFor<Target = T> + ?Sized,
    T: ?Sized,
{
    type Kind = K;
    type Target = K::Target;
}

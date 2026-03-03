//! Ordered subsets of non-empty collections by range.

#![cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#![cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]

pub(crate) mod range;

use core::cmp;
use core::ops::RangeBounds;

#[cfg(feature = "alloc")]
use crate::subset::ordered::range::TrimRange;
use crate::subset::ordered::range::{IndexRange, Intersect, Project};

pub use crate::subset::ordered::range::{OutOfBoundsError, RangeError, UnorderedError};

// TODO: Support ranges over a query type `Q` borrowed from a key or owned index type `K`. Note that
//       any type `Q` must be isomorphic with `K` (i.e., implement `UnsafeOrdIsomorph`), because the
//       `Ord` implementations can disagree and expose items outside of the subset. See the B-tree
//       collection types.

/// A contiguous and ordered subset of a non-empty collection over a range.
///
/// This is a very general type constructor: refer to more specific type definitions to see the
/// relevant APIs for a particular collection type. For example, see [`vec1::OnlyRangeSubset`] to
/// see supported APIs for [`Vec1`]. Every supported non-empty collection type has such a subset
/// type definition.
///
/// [`vec1::OnlyRangeSubset`]: crate::vec1::OnlyRangeSubset
/// [`Vec1`]: crate::vec1::Vec1
#[derive(Debug)]
#[must_use]
pub struct OnlyRangeSubset<'a, T, R>
where
    T: ?Sized,
{
    pub(crate) items: &'a mut T,
    pub(crate) range: R,
}

impl<'a, T, R> OnlyRangeSubset<'a, T, R>
where
    T: ?Sized,
{
    pub(crate) fn unchecked(items: &'a mut T, range: R) -> Self {
        OnlyRangeSubset { items, range }
    }
}

impl<T> OnlyRangeSubset<'_, T, IndexRange>
where
    T: ?Sized,
{
    pub(crate) fn from_tail_range(items: &mut T, n: usize) -> OnlyRangeSubset<'_, T, IndexRange> {
        // A subset over the range `[1,n)` is always valid for a positional (indexed) collection.
        // For an empty collection, this range is adjacent to `[0,1)` at the index `1` and so the
        // subset can perform insertions and removals (even though it is empty).
        OnlyRangeSubset::unchecked(items, IndexRange::unchecked(1, cmp::max(1, n)))
    }

    pub(crate) fn from_rtail_range(items: &mut T, n: usize) -> OnlyRangeSubset<'_, T, IndexRange> {
        // A subset over the range `[0,n-1)` is always valid for a positional (indexed) collection.
        // For an empty collection, this range is adjacent to `[0,1)` at the index `0` and so the
        // subset can perform insertions and removals (even though it is empty).
        OnlyRangeSubset::unchecked(items, IndexRange::unchecked(0, n.saturating_sub(1)))
    }

    pub(crate) fn intersected_strict_subset<R>(
        items: &mut T,
        n: usize,
        range: R,
    ) -> Result<OnlyRangeSubset<'_, T, IndexRange>, RangeError<usize>>
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
    ) -> Result<OnlyRangeSubset<'_, T, IndexRange>, RangeError<usize>>
    where
        IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
        R: RangeBounds<usize>,
    {
        let range = self.range.intersect(self.range.project(range)?)?;
        Ok(OnlyRangeSubset::unchecked(self.items, range))
    }

    pub(crate) fn project_tail_range(&mut self) -> OnlyRangeSubset<'_, T, IndexRange> {
        let range = self.range.project(1..).unwrap();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn project_rtail_range(&mut self, n: usize) -> OnlyRangeSubset<'_, T, IndexRange> {
        let range = self.range.project(..n.saturating_sub(1)).unwrap();
        OnlyRangeSubset::unchecked(self.items, range)
    }
}

#[cfg(feature = "alloc")]
impl<T> OnlyRangeSubset<'_, T, TrimRange>
where
    T: ?Sized,
{
    pub(crate) fn advance_tail_range(&mut self) -> OnlyRangeSubset<'_, T, TrimRange> {
        let range = self.range.tail();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn advance_rtail_range(&mut self) -> OnlyRangeSubset<'_, T, TrimRange> {
        let range = self.range.rtail();
        OnlyRangeSubset::unchecked(self.items, range)
    }

    pub(crate) fn untrimmed_item_count(&self, n: usize) -> usize {
        n.saturating_sub(self.range.trimmed_item_count())
    }
}

// TODO: Replace references to this module with `core::range::legacy` when it is stabilized. See
//       https://github.com/rust-lang/rust/issues/125687
mod legacy {
    pub use core::ops::Range;
}

use core::cmp;
use core::mem;
use core::ops::{Bound, RangeBounds};
use core::range::Range;

use crate::range1::IntoRangeBounds;
use crate::subset::ranged::range::{self, OutOfBoundsError, RangeError, UnorderedError};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IndexRange {
    start: usize,
    end: usize,
}

impl IndexRange {
    pub(crate) fn unchecked(start: usize, end: usize) -> Self {
        IndexRange { start, end }
    }

    pub fn new(start: usize, end: usize) -> Result<Self, UnorderedError<usize>> {
        if start <= end {
            Ok(IndexRange::unchecked(start, end))
        }
        else {
            Err(UnorderedError(start, end))
        }
    }

    pub fn empty_at(at: usize) -> Self {
        IndexRange::unchecked(at, at)
    }

    // This function succeeds for both intersections and adjacencies. For an adjacency, the output
    // range is empty at the boundary (the point of contact).
    pub fn contact<R>(self, other: R) -> Result<Self, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        use Bound::{Excluded, Included, Unbounded};

        let start = match other.start_bound() {
            Excluded(start) => start
                .checked_add(1)
                .unwrap_or_else(|| range::panic_start_overflow()),
            Included(start) => *start,
            Unbounded => self.start,
        };
        let end = match other.end_bound() {
            Excluded(end) => *end,
            Included(end) => end
                .checked_add(1)
                .unwrap_or_else(|| range::panic_end_overflow()),
            Unbounded => self.end,
        };
        let other = IndexRange::new(start, end)?;

        if self.start <= other.end && self.end >= other.start {
            Ok(IndexRange::unchecked(
                cmp::max(self.start, other.start),
                cmp::min(self.end, other.end),
            ))
        }
        else {
            Err(RangeError::OutOfBounds((other.start, other.end).into()))
        }
    }

    pub fn project_point(self, index: usize) -> Result<usize, OutOfBoundsError<usize>> {
        let index = self
            .start
            .checked_add(index)
            .unwrap_or_else(|| range::panic_start_overflow());
        if index <= self.end {
            Ok(index)
        }
        else {
            Err(index.into())
        }
    }

    pub fn project_range_bounds<R>(self, other: R) -> Result<Self, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        use Bound::{Excluded, Included, Unbounded};

        let start = match other.start_bound() {
            Excluded(start) => self.project_point(
                start
                    .checked_add(1)
                    .unwrap_or_else(|| range::panic_start_overflow()),
            ),
            Included(start) => self.project_point(*start),
            Unbounded => Ok(self.start),
        }?;
        let end = match other.end_bound() {
            Excluded(end) => self.project_point(*end),
            Included(end) => self.project_point(
                end.checked_add(1)
                    .unwrap_or_else(|| range::panic_end_overflow()),
            ),
            Unbounded => Ok(self.end),
        }?;
        IndexRange::new(start, end).map_err(From::from)
    }

    pub fn get_and_clear_from_end(&mut self) -> Self {
        let start = self.start;
        mem::replace(self, IndexRange::empty_at(start))
    }

    pub fn resize_from_start(&mut self, len: usize) {
        self.start = self
            .end
            .checked_sub(len)
            .unwrap_or_else(|| range::panic_end_underflow());
    }

    pub fn resize_from_end(&mut self, len: usize) {
        self.end = self
            .start
            .checked_add(
                len.checked_add(1)
                    .unwrap_or_else(|| range::panic_len_overflow()),
            )
            .unwrap_or_else(|| range::panic_start_overflow());
    }

    pub fn take_from_start(&mut self, n: usize) {
        let start = self
            .start
            .checked_add(n)
            .unwrap_or_else(|| range::panic_start_overflow());
        assert!(start <= self.end, "exhausted range");
        self.start = start;
    }

    pub fn take_from_end(&mut self, n: usize) {
        let end = self
            .end
            .checked_sub(n)
            .unwrap_or_else(|| range::panic_end_underflow());
        assert!(self.start <= end, "exhausted range");
        self.end = end;
    }

    pub fn put_from_start(&mut self, n: usize) {
        self.start = self
            .start
            .checked_sub(n)
            .unwrap_or_else(|| range::panic_start_underflow());
    }

    pub fn put_from_end(&mut self, n: usize) {
        self.end = self
            .end
            .checked_add(n)
            .unwrap_or_else(|| range::panic_end_overflow());
    }

    pub fn advance_by(&mut self, n: usize) {
        self.put_from_end(n);
        self.take_from_start(n);
    }

    #[cfg(any(feature = "alloc", feature = "arrayvec", feature = "smallvec"))]
    pub(crate) fn truncate_from_end(&mut self, len: usize) -> Option<legacy::Range<usize>> {
        let from = self.len();
        let to = len;
        (to < from).then(|| {
            let n = from - to;
            self.take_from_end(n);
            (self.end - n)..self.end
        })
    }

    #[cfg(feature = "indexmap")]
    pub(crate) fn retain_from_end<'a, T, F>(&'a mut self, mut f: F) -> impl 'a + FnMut(&T) -> bool
    where
        F: 'a + FnMut(&T) -> bool,
    {
        // See comments in `retain_mut_in_bounds` below; these functions are nearly identical.
        let mut index = 0;
        let before = *self;
        let after = self;
        move |item| {
            let is_retained = if before.contains(index) {
                f(item)
            }
            else {
                true
            };
            if !is_retained {
                after.take_from_end(1);
            }
            index = index.saturating_add(1);
            is_retained
        }
    }

    pub(crate) fn retain_mut_from_end<'a, T, F>(
        &'a mut self,
        mut f: F,
    ) -> impl 'a + FnMut(&mut T) -> bool
    where
        F: 'a + FnMut(&mut T) -> bool,
    {
        let mut index = 0;
        let before = *self;
        let after = self;
        move |item| {
            // Always retain items that are **not** contained by the **original** range (`before`),
            // otherwise apply the given predicate.
            let is_retained = if before.contains(index) {
                f(item)
            }
            else {
                true
            };
            if !is_retained {
                after.take_from_end(1);
            }
            // Saturation is sufficient here, because collections cannot index nor contain more
            // than `usize::MAX` items (only `isize::MAX` for sized item types in most collections)
            // and `index` is initialized to zero. This function will never be called again if this
            // overflows.
            index = index.saturating_add(1);
            is_retained
        }
    }

    #[cfg(feature = "indexmap")]
    pub(crate) fn retain_key_value_from_end<'a, K, V, F>(
        &'a mut self,
        mut f: F,
    ) -> impl 'a + FnMut(&K, &mut V) -> bool
    where
        F: 'a + FnMut(&K, &mut V) -> bool,
    {
        // See comments in `retain_mut_in_bounds` above; these functions are nearly identical.
        let mut index = 0;
        let before = *self;
        let after = self;
        move |key, value| {
            let is_retained = if before.contains(index) {
                f(key, value)
            }
            else {
                true
            };
            if !is_retained {
                after.take_from_end(1);
            }
            index = index.saturating_add(1);
            is_retained
        }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn len(&self) -> usize {
        self.end
            .checked_sub(self.start)
            .unwrap_or_else(|| range::panic_len_underflow())
    }

    pub fn contains(&self, index: usize) -> bool {
        self.start <= index && index < self.end
    }

    pub fn is_strict_subset_of(&self, other: &Self) -> bool {
        self != other && self.start >= other.start && self.end <= other.end
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn is_prefix(&self) -> bool {
        self.start == 0
    }
}

impl From<IndexRange> for (usize, usize) {
    fn from(range: IndexRange) -> Self {
        (range.start, range.end)
    }
}

impl IntoRangeBounds<usize> for IndexRange {
    fn into_bounds(self) -> (Bound<usize>, Bound<usize>) {
        let IndexRange { start, end } = self;
        (Bound::Included(start), Bound::Excluded(end))
    }
}

impl RangeBounds<usize> for IndexRange {
    fn start_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&usize> {
        Bound::Excluded(&self.end)
    }
}

impl TryFrom<Range<usize>> for IndexRange {
    type Error = UnorderedError<usize>;

    fn try_from(range: Range<usize>) -> Result<Self, Self::Error> {
        IndexRange::new(range.start, range.end)
    }
}

impl TryFrom<legacy::Range<usize>> for IndexRange {
    type Error = <IndexRange as TryFrom<Range<usize>>>::Error;

    fn try_from(range: legacy::Range<usize>) -> Result<Self, Self::Error> {
        IndexRange::try_from(Range::from(range))
    }
}

#[cfg(test)]
mod tests {
    use core::ops::RangeBounds;
    use rstest::rstest;

    use crate::subset::ranged::range::{IndexRange, OutOfBoundsError, RangeError};

    #[rstest]
    #[case::superset(IndexRange::unchecked(0, 9), 1..3, Ok(IndexRange::unchecked(1, 3)))]
    #[case::subset(IndexRange::unchecked(1, 3), 0..9, Ok(IndexRange::unchecked(1, 3)))]
    #[case::adjacent(IndexRange::unchecked(1, 3), 3..9, Ok(IndexRange::unchecked(3, 3)))]
    #[case::disjoint(
        IndexRange::unchecked(1, 3),
        4..9,
        Err(RangeError::OutOfBounds(OutOfBoundsError::Range(4, 9))),
    )]
    fn contact_index_range_and_range_bounds_then_contact_eq<R>(
        #[case] lhs: IndexRange,
        #[case] rhs: R,
        #[case] expected: Result<IndexRange, RangeError<usize>>,
    ) where
        R: RangeBounds<usize>,
    {
        assert_eq!(lhs.contact(rhs), expected);
    }

    #[rstest]
    #[case::bounded(IndexRange::unchecked(1, 9), 2..5, Ok(IndexRange::unchecked(3, 6)))]
    #[case::unbounded_end(IndexRange::unchecked(1, 9), 2.., Ok(IndexRange::unchecked(3, 9)))]
    #[case::unbounded_start(IndexRange::unchecked(1, 9), ..2, Ok(IndexRange::unchecked(1, 3)))]
    #[case::out_of_bounds(
        IndexRange::unchecked(1, 3),
        1..5,
        Err(RangeError::OutOfBounds(OutOfBoundsError::Point(6))),
    )]
    fn project_range_bounds_onto_index_range_then_projection_eq<R>(
        #[case] lhs: IndexRange,
        #[case] rhs: R,
        #[case] expected: Result<IndexRange, RangeError<usize>>,
    ) where
        R: RangeBounds<usize>,
    {
        assert_eq!(lhs.project_range_bounds(rhs), expected);
    }
}

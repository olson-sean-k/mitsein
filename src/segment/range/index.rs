use core::cmp;
use core::mem;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::segment::range::{
    self, Intersect, IntoRangeBounds, OutOfBoundsError, Project, RangeError, UnorderedError,
};

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
            .checked_add(len.checked_add(1).expect("overflow in segment length"))
            .unwrap_or_else(|| range::panic_start_overflow());
    }

    pub fn take_from_start(&mut self, n: usize) {
        let start = self
            .start
            .checked_add(n)
            .unwrap_or_else(|| range::panic_start_overflow());
        assert!(start <= self.end, "segment out of bounds");
        self.start = start;
    }

    pub fn take_from_end(&mut self, n: usize) {
        let end = self
            .end
            .checked_sub(n)
            .unwrap_or_else(|| range::panic_end_underflow());
        assert!(self.start <= end, "segment out of bounds");
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

    pub(crate) fn truncate_from_end(&mut self, len: usize) -> Option<Range<usize>> {
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

    fn intersect(self, other: Self) -> Result<Self, OutOfBoundsError<usize>> {
        // Accept empty input ranges and adjacencies.
        if self.start <= other.end && self.end >= other.start {
            Ok(IndexRange::unchecked(
                cmp::max(self.start, other.start),
                cmp::min(self.end, other.end),
            ))
        }
        else {
            Err((other.start, other.end).into())
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
            .expect("underflow in segment length")
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

impl<R> Intersect<R> for IndexRange
where
    R: RangeBounds<usize>,
{
    type Output = Self;
    type Error = RangeError<usize>;

    fn intersect(self, range: R) -> Result<Self::Output, Self::Error> {
        use Bound::{Excluded, Included, Unbounded};

        let start = match range.start_bound() {
            Excluded(start) => start
                .checked_add(1)
                .unwrap_or_else(|| range::panic_start_overflow()),
            Included(start) => *start,
            Unbounded => self.start,
        };
        let end = match range.end_bound() {
            Excluded(end) => *end,
            Included(end) => end
                .checked_add(1)
                .unwrap_or_else(|| range::panic_end_overflow()),
            Unbounded => self.end,
        };
        let other = IndexRange::new(start, end)?;
        IndexRange::intersect(self, other).map_err(From::from)
    }
}

impl IntoRangeBounds<usize> for IndexRange {
    fn into_bounds(self) -> (Bound<usize>, Bound<usize>) {
        let IndexRange { start, end } = self;
        (Bound::Included(start), Bound::Excluded(end))
    }
}

impl Project<Range<usize>> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: Range<usize>) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<RangeFrom<usize>> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: RangeFrom<usize>) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<RangeFull> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: RangeFull) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<RangeInclusive<usize>> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: RangeInclusive<usize>) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<RangeTo<usize>> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: RangeTo<usize>) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<RangeToInclusive<usize>> for IndexRange {
    type Output = Self;
    type Error = RangeError<usize>;

    fn project(self, range: RangeToInclusive<usize>) -> Result<Self::Output, Self::Error> {
        self::project_range_bounds_onto_index_range(self, range)
    }
}

impl Project<usize> for IndexRange {
    type Output = usize;
    type Error = OutOfBoundsError<usize>;

    fn project(self, index: usize) -> Result<Self::Output, Self::Error> {
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

fn project_range_bounds_onto_index_range<R>(
    range: IndexRange,
    bounds: R,
) -> Result<IndexRange, RangeError<usize>>
where
    R: RangeBounds<usize>,
{
    use Bound::{Excluded, Included, Unbounded};

    let start = match bounds.start_bound() {
        Excluded(start) => range.project(
            start
                .checked_add(1)
                .unwrap_or_else(|| range::panic_start_overflow()),
        ),
        Included(start) => range.project(*start),
        Unbounded => Ok(range.start),
    }?;
    let end = match bounds.end_bound() {
        Excluded(end) => range.project(*end),
        Included(end) => range.project(
            end.checked_add(1)
                .unwrap_or_else(|| range::panic_end_overflow()),
        ),
        Unbounded => Ok(range.end),
    }?;
    IndexRange::try_from(start..end).map_err(From::from)
}

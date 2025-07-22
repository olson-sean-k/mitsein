#![cfg(any(feature = "alloc", feature = "arrayvec"))]

use core::cmp;
use core::fmt::Debug;
use core::mem;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::segment::{Indexer, Ranged, Segment, SegmentedOver};

pub type RangeFor<K> = <<K as SegmentedOver>::Target as Ranged>::Range;

impl<'a, K, T> Segment<'a, K, T>
where
    K: SegmentedOver<Target = T> + ?Sized,
    T: Ranged,
{
    pub(crate) fn unchecked(items: &'a mut K::Target, range: RangeFor<K>) -> Self {
        Segment { items, range }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn empty<I>(items: &'a mut K::Target) -> Self
    where
        K::Target: Ranged<Range = RelationalRange<I>>,
    {
        Segment::unchecked(items, RelationalRange::Empty)
    }

    pub(crate) fn intersect<R>(items: &'a mut K::Target, range: &R) -> Self
    where
        RangeFor<K>: Intersect<R, Output = RangeFor<K>>,
    {
        let range = items.range().intersect(range).expect_in_bounds();
        Segment::unchecked(items, range)
    }

    pub(crate) fn intersect_strict_subset<R>(items: &'a mut K::Target, range: &R) -> Self
    where
        RangeFor<K>: Intersect<R, Output = RangeFor<K>> + IsStrictSubset<RangeFor<K>>,
    {
        let basis = items.range();
        let range = basis.intersect(range).expect_in_bounds();
        if range.is_strict_subset(&basis) {
            Segment::unchecked(items, range)
        }
        else {
            panic!("segment is not a strict subset")
        }
    }

    pub(crate) fn project<R>(&self, range: &R) -> <RangeFor<K> as Project<R>>::Output
    where
        RangeFor<K>: Project<R>,
        R: RangeBounds<<RangeFor<K> as Indexer>::Index>,
    {
        self.range.project(range).expect_in_bounds()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PositionalRange {
    pub start: usize,
    pub end: usize,
}

impl PositionalRange {
    pub fn unchecked(start: usize, end: usize) -> Self {
        PositionalRange { start, end }
    }

    pub fn ordered(start: usize, end: usize) -> Self {
        assert!(start <= end, "segment starts at {start} but ends at {end}");
        PositionalRange::unchecked(start, end)
    }

    pub fn empty_at(at: usize) -> Self {
        PositionalRange::unchecked(at, at)
    }

    pub fn get_and_clear_from_end(&mut self) -> Self {
        let start = self.start;
        mem::replace(self, PositionalRange::empty_at(start))
    }

    pub fn resize_from_start(&mut self, len: usize) {
        self.start = self
            .end
            .checked_sub(len)
            .unwrap_or_else(|| self::panic_end_underflow());
    }

    pub fn resize_from_end(&mut self, len: usize) {
        self.end = self
            .start
            .checked_add(len.checked_add(1).expect("overflow in segment length"))
            .unwrap_or_else(|| self::panic_start_overflow());
    }

    pub fn take_from_start(&mut self, n: usize) {
        let start = self
            .start
            .checked_add(n)
            .unwrap_or_else(|| self::panic_start_overflow());
        assert!(start <= self.end, "segment out of bounds");
        self.start = start;
    }

    pub fn take_from_end(&mut self, n: usize) {
        let end = self
            .end
            .checked_sub(n)
            .unwrap_or_else(|| self::panic_end_underflow());
        assert!(self.start <= end, "segment out of bounds");
        self.end = end;
    }

    pub fn put_from_start(&mut self, n: usize) {
        self.start = self
            .start
            .checked_sub(n)
            .unwrap_or_else(|| self::panic_start_underflow());
    }

    pub fn put_from_end(&mut self, n: usize) {
        self.end = self
            .end
            .checked_add(n)
            .unwrap_or_else(|| self::panic_end_overflow());
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

    pub fn len(&self) -> usize {
        self.end
            .checked_sub(self.start)
            .expect("underflow in segment length")
    }

    pub fn contains(&self, index: usize) -> bool {
        self.start <= index && index < self.end
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn is_prefix(&self) -> bool {
        self.start == 0
    }
}

impl From<Range<usize>> for PositionalRange {
    fn from(range: Range<usize>) -> Self {
        PositionalRange {
            start: range.start,
            end: range.end,
        }
    }
}

impl Indexer for PositionalRange {
    type Index = usize;
}

impl RangeBounds<usize> for PositionalRange {
    fn start_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&usize> {
        Bound::Excluded(&self.end)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationalRange<T> {
    Empty,
    NonEmpty { start: T, end: T },
}

impl<T> RelationalRange<T> {
    pub fn unchecked(start: T, end: T) -> Self {
        RelationalRange::NonEmpty { start, end }
    }

    pub fn try_into_range_inclusive(self) -> Option<RangeInclusive<T>> {
        match self {
            RelationalRange::Empty => None,
            RelationalRange::NonEmpty { start, end } => Some(RangeInclusive::new(start, end)),
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn retain_in_range<'a, F>(&'a self, mut f: F) -> impl 'a + FnMut(&T) -> bool
    where
        T: Ord,
        F: 'a + FnMut(&T) -> bool,
    {
        let mut by_key_value = self.retain_key_value_in_range(move |key, _| f(key));
        move |item| by_key_value(item, &mut ())
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn retain_key_value_in_range<'a, U, F>(
        &'a self,
        mut f: F,
    ) -> impl 'a + FnMut(&T, &mut U) -> bool
    where
        T: Ord,
        F: 'a + FnMut(&T, &mut U) -> bool,
    {
        move |key, value| {
            // Always retain items that are **not** contained by the range, otherwise apply the
            // given predicate.
            if self.contains(key) {
                f(key, value)
            }
            else {
                true
            }
        }
    }

    #[cfg(feature = "alloc")]
    pub fn contains(&self, key: &T) -> bool
    where
        T: Ord,
    {
        match self {
            RelationalRange::Empty => false,
            RelationalRange::NonEmpty { ref start, ref end } => start <= key && end >= key,
        }
    }

    #[cfg(feature = "alloc")]
    pub fn is_empty(&self) -> bool {
        matches!(self, RelationalRange::Empty)
    }
}

impl<T> RelationalRange<T>
where
    T: Ord,
{
    #[cfg(feature = "alloc")]
    pub fn ordered(start: T, end: T) -> Self {
        assert!(start <= end, "segment starts after it ends");
        RelationalRange::unchecked(start, end)
    }
}

impl<T> From<Option<(T, T)>> for RelationalRange<T> {
    fn from(range: Option<(T, T)>) -> Self {
        match range {
            Some((start, end)) => RelationalRange::NonEmpty { start, end },
            _ => RelationalRange::Empty,
        }
    }
}

impl<T> Indexer for RelationalRange<T> {
    type Index = T;
}

pub type Projection<T> = Result<T, T>;

pub trait ProjectionExt<T> {
    fn into_output(self) -> T;

    fn expect_in_bounds(self) -> T;
}

impl<T> ProjectionExt<T> for Projection<T> {
    fn into_output(self) -> T {
        match self {
            Ok(output) | Err(output) => output,
        }
    }

    fn expect_in_bounds(self) -> T {
        match self {
            Ok(output) => output,
            _ => panic!("index out of bounds"),
        }
    }
}

pub trait Project<T> {
    type Output;

    fn project(&self, other: &T) -> Projection<Self::Output>;
}

impl Project<Range<usize>> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &Range<usize>) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<RangeFrom<usize>> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &RangeFrom<usize>) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<RangeFull> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &RangeFull) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<RangeInclusive<usize>> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &RangeInclusive<usize>) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<RangeTo<usize>> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &RangeTo<usize>) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<RangeToInclusive<usize>> for PositionalRange {
    type Output = Self;

    fn project(&self, range: &RangeToInclusive<usize>) -> Projection<Self::Output> {
        self::project_range_bounds_onto_positional_range(self, range)
    }
}

impl Project<usize> for PositionalRange {
    type Output = usize;

    fn project(&self, offset: &usize) -> Projection<Self::Output> {
        let projection = self
            .start
            .checked_add(*offset)
            .unwrap_or_else(|| self::panic_start_overflow());
        // TODO: This doesn't seem right (`<=` vs. `<`). It may be important to distinguish between
        //       `Exlcuded` and `Included` offsets here.
        if projection <= self.end {
            Ok(projection)
        }
        else {
            // Out of bounds.
            Err(projection)
        }
    }
}

pub type Intersection<T> = Option<T>;

pub trait IntersectionExt<T> {
    fn expect_in_bounds(self) -> T;
}

impl<T> IntersectionExt<T> for Intersection<T> {
    fn expect_in_bounds(self) -> T {
        match self {
            Some(intersection) => intersection,
            _ => panic!("index out of bounds"),
        }
    }
}

pub trait Intersect<R>: Sized {
    type Output;

    fn intersect(&self, range: &R) -> Intersection<Self::Output>;
}

impl<R> Intersect<R> for PositionalRange
where
    R: RangeBounds<usize>,
{
    type Output = Self;

    fn intersect(&self, range: &R) -> Intersection<Self::Output> {
        use Bound::{Excluded, Included, Unbounded};

        let start = match range.start_bound() {
            Excluded(start) => start
                .checked_add(1)
                .unwrap_or_else(|| self::panic_start_overflow()),
            Included(start) => *start,
            Unbounded => self.start,
        };
        let end = match range.end_bound() {
            Excluded(end) => *end,
            Included(end) => end
                .checked_add(1)
                .unwrap_or_else(|| self::panic_end_overflow()),
            Unbounded => self.end,
        };
        // Accept empty input ranges and adjacencies.
        if self.start <= end && self.end >= start {
            Some(PositionalRange::unchecked(
                cmp::max(self.start, start),
                cmp::min(self.end, end),
            ))
        }
        else {
            None
        }
    }
}

impl<T> Intersect<Self> for RelationalRange<T>
where
    T: Clone + Ord,
{
    type Output = Self;

    fn intersect(&self, range: &Self) -> Intersection<Self::Output> {
        match range.clone().try_into_range_inclusive() {
            Some(range) => self.intersect(&range),
            // Accept empty input ranges.
            _ => Some(RelationalRange::Empty),
        }
    }
}

impl<T> Intersect<RangeFrom<T>> for RelationalRange<T>
where
    T: Clone + Ord,
{
    type Output = Self;

    fn intersect(&self, range: &RangeFrom<T>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            RelationalRange::Empty => Some(RelationalRange::Empty),
            RelationalRange::NonEmpty { ref start, ref end } => {
                if end >= &range.start {
                    Some(RelationalRange::unchecked(
                        cmp::max(start, &range.start).clone(),
                        end.clone(),
                    ))
                }
                else {
                    None
                }
            },
        }
    }
}

impl<T> Intersect<RangeInclusive<T>> for RelationalRange<T>
where
    T: Clone + Ord,
{
    type Output = Self;

    fn intersect(&self, range: &RangeInclusive<T>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            RelationalRange::Empty => Some(RelationalRange::Empty),
            RelationalRange::NonEmpty { ref start, ref end } => {
                if start <= range.end() && end >= range.start() {
                    Some(RelationalRange::unchecked(
                        cmp::max(start, range.start()).clone(),
                        cmp::min(end, range.end()).clone(),
                    ))
                }
                else {
                    None
                }
            },
        }
    }
}

impl<T> Intersect<RangeToInclusive<T>> for RelationalRange<T>
where
    T: Clone + Ord,
{
    type Output = Self;

    fn intersect(&self, range: &RangeToInclusive<T>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            RelationalRange::Empty => Some(RelationalRange::Empty),
            RelationalRange::NonEmpty { ref start, ref end } => {
                if start <= &range.end {
                    Some(RelationalRange::unchecked(
                        start.clone(),
                        cmp::min(end, &range.end).clone(),
                    ))
                }
                else {
                    None
                }
            },
        }
    }
}

pub trait IsStrictSubset<R> {
    fn is_strict_subset(&self, other: &R) -> bool;
}

impl IsStrictSubset<Self> for PositionalRange {
    fn is_strict_subset(&self, other: &Self) -> bool {
        self != other && self.start >= other.start && self.end <= other.end
    }
}

impl<T> IsStrictSubset<Self> for RelationalRange<T>
where
    T: Ord,
{
    fn is_strict_subset(&self, other: &Self) -> bool {
        use RelationalRange::{Empty, NonEmpty};

        match (self, other) {
            (Empty, Empty) | (NonEmpty { .. }, Empty) => false,
            (Empty, NonEmpty { .. }) => true,
            (
                NonEmpty { ref start, ref end },
                NonEmpty {
                    start: ref from,
                    end: ref to,
                },
            ) => from < start || to > end,
        }
    }
}

pub fn ordered_range_offsets<R>(range: R) -> R
where
    R: RangeBounds<usize>,
{
    use Bound::{Excluded, Included, Unbounded};

    let start = match range.start_bound() {
        Excluded(start) => start
            .checked_add(1)
            .unwrap_or_else(|| panic_start_overflow()),
        Included(start) => *start,
        Unbounded => {
            return range;
        },
    };
    let end = match range.end_bound() {
        Excluded(end) => *end,
        Included(end) => end.checked_add(1).unwrap_or_else(|| panic_end_overflow()),
        Unbounded => {
            return range;
        },
    };

    assert!(start <= end, "segment starts at {start} but ends at {end}");
    range
}

#[cfg(feature = "alloc")]
pub fn ordered_range_bounds<T, R>(range: R) -> R
where
    T: Ord,
    R: RangeBounds<T>,
{
    use Bound::{Excluded, Included, Unbounded};

    assert!(
        match (range.start_bound(), range.end_bound()) {
            // Unlike the opposite bounds, `start == end` is considered misordered (rather than
            // empty).
            (Excluded(start), Included(end)) => start < end,
            // At time of writing, no such range is supported in `core` and such an exlusive range
            // is a bit unusual. However, any crate can implement such a range.
            (Excluded(_), Excluded(_)) => panic!("unexpected exclusive range"),
            (Included(start), Excluded(end) | Included(end)) => start <= end,
            (Unbounded, _) | (_, Unbounded) => true,
        },
        "segment starts after it ends",
    );
    range
}

fn project_range_bounds_onto_positional_range<R>(
    basis: &PositionalRange,
    range: &R,
) -> Projection<PositionalRange>
where
    R: RangeBounds<usize>,
{
    use Bound::{Excluded, Included, Unbounded};

    let start = match range.start_bound() {
        Excluded(start) => basis.project(
            &start
                .checked_add(1)
                .unwrap_or_else(|| self::panic_start_overflow()),
        ),
        Included(start) => basis.project(start),
        Unbounded => Ok(basis.start),
    };
    let end = match range.end_bound() {
        Excluded(end) => basis.project(end),
        Included(end) => basis.project(
            &end.checked_add(1)
                .unwrap_or_else(|| self::panic_end_overflow()),
        ),
        Unbounded => Ok(basis.end),
    };
    if let (Ok(start), Ok(end)) = (start, end) {
        Ok(From::from(start..end))
    }
    else {
        // Out of bounds.
        Err(From::from(start.into_output()..end.into_output()))
    }
}

const fn panic_start_underflow() -> ! {
    panic!("underflow in segment start")
}

const fn panic_start_overflow() -> ! {
    panic!("overflow in segment start")
}

const fn panic_end_underflow() -> ! {
    panic!("underflow in segment end")
}

const fn panic_end_overflow() -> ! {
    panic!("overflow in segment end")
}

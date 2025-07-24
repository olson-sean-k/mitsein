#![cfg(any(feature = "alloc", feature = "arrayvec"))]

#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::borrow::Borrow;
use core::cmp;
use core::fmt::Debug;
use core::mem;
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::segment::{Ranged, Segment, SegmentedOver};

impl<'a, K, T, R> Segment<'a, K, T, R>
where
    K: SegmentedOver<Target = T> + ?Sized,
    T: Ranged + ?Sized,
{
    pub(crate) fn unchecked(items: &'a mut K::Target, range: R) -> Self {
        Segment { items, range }
    }

    pub(crate) fn intersect<Q>(items: &'a mut K::Target, range: &Q) -> Self
    where
        T: Ranged<NominalRange = R>,
        R: Intersect<Q, Output = R>,
    {
        Segment::intersect_with(items, range, |_, all| all)
    }

    pub(crate) fn intersect_with<Q, F>(items: &'a mut K::Target, range: &Q, f: F) -> Self
    where
        R: Intersect<Q, Output = R>,
        F: FnOnce(&T, T::NominalRange) -> R,
    {
        let range = f(&*items, items.all()).intersect(range).expect_in_bounds();
        Segment::unchecked(items, range)
    }

    pub(crate) fn intersect_strict_subset<Q>(items: &'a mut K::Target, range: &Q) -> Self
    where
        T: Ranged<NominalRange = R>,
        R: Intersect<Q, Output = R> + IsStrictSubset<R>,
    {
        Segment::intersect_strict_subset_with(items, range, |_, all| all)
    }

    pub(crate) fn intersect_strict_subset_with<Q, F>(
        items: &'a mut K::Target,
        range: &Q,
        f: F,
    ) -> Self
    where
        R: Intersect<Q, Output = R> + IsStrictSubset<R>,
        F: FnOnce(&T, T::NominalRange) -> R,
    {
        let all = f(&*items, items.all());
        let range = all.intersect(range).expect_in_bounds();
        if range.is_strict_subset(&all) {
            Segment::unchecked(items, range)
        }
        else {
            panic!("segment is not a strict subset")
        }
    }

    pub(crate) fn map_range<Q, F>(self, f: F) -> Segment<'a, K, T, Q>
    where
        F: FnOnce(R) -> Q,
    {
        let Segment { items, range } = self;
        Segment {
            items,
            range: f(range),
        }
    }

    pub(crate) fn project<Q>(&self, range: &Q) -> <R as Project<Q>>::Output
    where
        R: Project<Q>,
    {
        self.range.project(range).expect_in_bounds()
    }
}

#[cfg(feature = "alloc")]
impl<'a, K, T, N> Segment<'a, K, T, ItemRange<N>>
where
    K: SegmentedOver<Target = T> + ?Sized,
    T: Ranged + ?Sized,
{
    pub(crate) fn empty(items: &'a mut K::Target) -> Self {
        Segment::unchecked(items, ItemRange::Empty)
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'a, K, T, N> Segment<'a, K, T, ItemRange<&'_ N>>
where
    K: SegmentedOver<Target = T>,
    T: Ranged + ?Sized,
    N: ?Sized + ToOwned,
{
    pub fn into_owning_range(self) -> Segment<'a, K, T, ItemRange<N::Owned>> {
        let Segment { items, range } = self;
        Segment {
            items,
            range: range.into_owning(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a, K, T, N> Segment<'a, K, T, RelationalRange<N>>
where
    K: SegmentedOver<Target = T> + ?Sized,
    T: Ranged + ?Sized,
{
    pub(crate) fn empty(items: &'a mut K::Target) -> Self {
        Segment::unchecked(items, RelationalRange::from(ItemRange::Empty))
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'a, K, T, N> From<Segment<'a, K, T, ItemRange<N>>> for Segment<'a, K, T, RelationalRange<N>>
where
    K: SegmentedOver<Target = T>,
    T: Ranged + ?Sized,
{
    fn from(segment: Segment<'a, K, T, ItemRange<N>>) -> Self {
        let Segment { items, range } = segment;
        Segment {
            items,
            range: range.into(),
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'a, K, T, N> From<Segment<'a, K, T, TrimRange>> for Segment<'a, K, T, RelationalRange<N>>
where
    TrimRange: Resolve<'a, T, RelationalRange<N>>,
    K: SegmentedOver<Target = T>,
    T: Ranged + ?Sized,
{
    fn from(segment: Segment<'a, K, T, TrimRange>) -> Self {
        let Segment { items, range } = segment;
        let range = range.resolve(items);
        Segment {
            items,
            range,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ItemRange<N> {
    Empty,
    NonEmpty { start: N, end: N },
}

impl<N> ItemRange<N> {
    pub fn unchecked(start: N, end: N) -> Self {
        ItemRange::NonEmpty { start, end }
    }

    pub fn try_into_range_inclusive(self) -> Option<RangeInclusive<N>> {
        match self {
            ItemRange::Empty => None,
            ItemRange::NonEmpty { start, end } => Some(RangeInclusive::new(start, end)),
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn retain_in_range<'a, F>(&'a self, mut f: F) -> impl 'a + FnMut(&N) -> bool
    where
        N: Ord,
        F: 'a + FnMut(&N) -> bool,
    {
        let mut by_key_value = self.retain_key_value_in_range(move |key, _| f(key));
        move |item| by_key_value(item, &mut ())
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn retain_key_value_in_range<'a, M, F>(
        &'a self,
        mut f: F,
    ) -> impl 'a + FnMut(&N, &mut M) -> bool
    where
        N: Ord,
        F: 'a + FnMut(&N, &mut M) -> bool,
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

    pub fn as_ref(&self) -> ItemRange<&N> {
        match self {
            ItemRange::Empty => ItemRange::Empty,
            ItemRange::NonEmpty { start, end } => ItemRange::NonEmpty { start, end },
        }
    }

    pub fn borrow<T>(&self) -> ItemRange<&T>
    where
        N: Borrow<T>,
        T: ?Sized,
    {
        match self {
            ItemRange::Empty => ItemRange::Empty,
            ItemRange::NonEmpty { start, end } => ItemRange::NonEmpty {
                start: start.borrow(),
                end: end.borrow(),
            },
        }
    }

    #[cfg(feature = "alloc")]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        //N: PartialOrd<Q>,
        //Q: PartialOrd<N> + ?Sized,
        N: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self {
            ItemRange::Empty => false,
            ItemRange::NonEmpty { start, end } => start.borrow() <= key && end.borrow() >= key,
        }
    }

    #[cfg(feature = "alloc")]
    pub fn is_empty(&self) -> bool {
        matches!(self, ItemRange::Empty)
    }
}

impl<N> ItemRange<N>
where
    N: Ord,
{
    #[cfg(feature = "alloc")]
    pub fn ordered(start: N, end: N) -> Self {
        assert!(start <= end, "segment starts after it ends");
        ItemRange::unchecked(start, end)
    }
}

impl<'n, N> ItemRange<&'n N>
where
    N: Clone,
{
    pub fn cloned(self) -> ItemRange<N> {
        match self {
            ItemRange::Empty => ItemRange::Empty,
            ItemRange::NonEmpty { start, end } => ItemRange::NonEmpty {
                start: start.clone(),
                end: end.clone(),
            },
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'n, N> ItemRange<&'n N>
where
    N: ?Sized + ToOwned,
{
    pub fn into_owning(self) -> ItemRange<N::Owned> {
        match self {
            ItemRange::Empty => ItemRange::Empty,
            ItemRange::NonEmpty { start, end } => ItemRange::NonEmpty {
                start: start.to_owned(),
                end: end.to_owned(),
            },
        }
    }
}

impl<N> Default for ItemRange<N> {
    fn default() -> Self {
        ItemRange::Empty
    }
}

impl<N> From<Option<(N, N)>> for ItemRange<N> {
    fn from(range: Option<(N, N)>) -> Self {
        match range {
            Some((start, end)) => ItemRange::NonEmpty { start, end },
            _ => ItemRange::Empty,
        }
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
        assert!(
            start <= end,
            "segment starts at {} but ends at {}",
            start,
            end
        );
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
        // See comments in `retain_mut_from_end` below; these functions are nearly identical.
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
        // See comments in `retain_mut_from_end` above; these functions are nearly identical.
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

impl RangeBounds<usize> for PositionalRange {
    fn start_bound(&self) -> Bound<&usize> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&usize> {
        Bound::Excluded(&self.end)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RelationalRange<N> {
    Item(ItemRange<N>),
    Trim(TrimRange),
}

impl<N> RelationalRange<N> {
    pub fn resolve_and_get<'a, T>(&mut self, items: &'a T) -> &ItemRange<N>
    where
        TrimRange: Resolve<'a, T, ItemRange<N>>,
    {
        match self {
            RelationalRange::Item(range) => range,
            RelationalRange::Trim(range) => {
                let range = (*range).resolve(items);
                *self = RelationalRange::Item(range);
                match self {
                    RelationalRange::Item(range) => range,
                    _ => unreachable!(),
                }
            },
        }
    }
}

impl<N> Default for RelationalRange<N> {
    fn default() -> Self {
        RelationalRange::Item(ItemRange::default())
    }
}

impl<N> From<ItemRange<N>> for RelationalRange<N> {
    fn from(range: ItemRange<N>) -> Self {
        RelationalRange::Item(range)
    }
}

impl<N> From<TrimRange> for RelationalRange<N> {
    fn from(range: TrimRange) -> Self {
        RelationalRange::Trim(range)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrimRange {
    pub tail: usize,
    pub rtail: usize,
}

impl TrimRange {
    pub const ALL: Self = TrimRange { tail: 0, rtail: 0 };
    pub const TAIL: Self = TrimRange { tail: 1, rtail: 0 };
    pub const RTAIL: Self = TrimRange { tail: 0, rtail: 1 };

    pub const fn tail(self) -> Self {
        let TrimRange { tail, rtail } = self;
        TrimRange {
            tail: match tail.checked_add(1) {
                Some(tail) => tail,
                _ => panic!("overflow computing tail of terminal range"),
            },
            rtail,
        }
    }

    pub const fn rtail(self) -> Self {
        let TrimRange { tail, rtail } = self;
        TrimRange {
            tail,
            rtail: match rtail.checked_add(1) {
                Some(rtail) => rtail,
                _ => panic!("overflow computing reverse tail of terminal range"),
            },
        }
    }

    pub const fn is_all(&self) -> bool {
        self.tail == 0 && self.rtail == 0
    }
}

pub trait Resolve<'a, T, R>
where
    T: ?Sized,
{
    #[must_use]
    fn resolve(self, items: &'a T) -> R;
}

impl<'a, T, R> Resolve<'a, T, R> for R {
    fn resolve(self, _items: &'a T) -> R {
        self
    }
}

pub type Projection<R> = Result<R, R>;

pub trait ProjectionExt<R> {
    fn into_output(self) -> R;

    fn expect_in_bounds(self) -> R;
}

impl<R> ProjectionExt<R> for Projection<R> {
    fn into_output(self) -> R {
        match self {
            Ok(output) | Err(output) => output,
        }
    }

    fn expect_in_bounds(self) -> R {
        match self {
            Ok(output) => output,
            _ => panic!("index out of bounds"),
        }
    }
}

pub trait Project<R> {
    type Output;

    fn project(&self, other: &R) -> Projection<Self::Output>;
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

pub type Intersection<R> = Option<R>;

pub trait IntersectionExt<R> {
    fn expect_in_bounds(self) -> R;
}

impl<R> IntersectionExt<R> for Intersection<R> {
    fn expect_in_bounds(self) -> R {
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

impl<N, M> Intersect<(Bound<M>, Bound<M>)> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Clone + Ord,
{
    type Output = ItemRange<M>;

    fn intersect(&self, range: &(Bound<M>, Bound<M>)) -> Intersection<Self::Output> {
        todo!()
    }
}

impl<N, M> Intersect<ItemRange<M>> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Clone + Ord,
{
    type Output = ItemRange<M>;

    fn intersect(&self, range: &ItemRange<M>) -> Intersection<Self::Output> {
        match range.clone().try_into_range_inclusive() {
            Some(range) => self.intersect(&range),
            // Accept empty input ranges.
            _ => Some(ItemRange::Empty),
        }
    }
}

impl<N, M> Intersect<RangeFrom<M>> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Clone + Ord,
{
    type Output = ItemRange<M>;

    fn intersect(&self, range: &RangeFrom<M>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            ItemRange::Empty => Some(ItemRange::Empty),
            ItemRange::NonEmpty { start, end } => {
                if end.borrow() >= &range.start {
                    Some(ItemRange::unchecked(
                        cmp::max(start.borrow(), &range.start).clone(),
                        end.borrow().clone(),
                    ))
                }
                else {
                    None
                }
            },
        }
    }
}

impl<N, M> Intersect<RangeInclusive<M>> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Clone + Ord,
{
    type Output = ItemRange<M>;

    fn intersect(&self, range: &RangeInclusive<M>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            ItemRange::Empty => Some(ItemRange::Empty),
            ItemRange::NonEmpty { start, end } => {
                if start.borrow() <= range.end() && end.borrow() >= range.start() {
                    Some(ItemRange::unchecked(
                        cmp::max(start.borrow(), range.start()).clone(),
                        cmp::min(end.borrow(), range.end()).clone(),
                    ))
                }
                else {
                    None
                }
            },
        }
    }
}

impl<N, M> Intersect<RangeToInclusive<M>> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Clone + Ord,
{
    type Output = ItemRange<M>;

    fn intersect(&self, range: &RangeToInclusive<M>) -> Intersection<Self::Output> {
        match self {
            // Accept empty input ranges.
            ItemRange::Empty => Some(ItemRange::Empty),
            ItemRange::NonEmpty { start, end } => {
                if start.borrow() <= &range.end {
                    Some(ItemRange::unchecked(
                        start.borrow().clone(),
                        cmp::min(end.borrow(), &range.end).clone(),
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

impl<N, M> IsStrictSubset<ItemRange<M>> for ItemRange<N>
where
    N: Borrow<M> + Ord,
    M: Ord,
{
    fn is_strict_subset(&self, other: &ItemRange<M>) -> bool {
        use ItemRange::{Empty, NonEmpty};

        match (self, other) {
            (Empty, Empty) | (NonEmpty { .. }, Empty) => false,
            (Empty, NonEmpty { .. }) => true,
            (
                NonEmpty { start, end },
                NonEmpty {
                    start: from,
                    end: to,
                },
            ) => from.borrow() < start.borrow() || to.borrow() > end.borrow(),
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

    assert!(
        start <= end,
        "segment starts at {} but ends at {}",
        start,
        end
    );
    range
}

#[cfg(feature = "alloc")]
pub fn ordered_range_bounds<N, R>(range: R) -> R
where
    N: Ord + ?Sized,
    R: RangeBounds<N>,
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

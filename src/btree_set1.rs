#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};

use crate::array1::Array1;
use crate::cmp::UnsafeOrd;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, Intersect, RelationalRange};
use crate::segment::{self, Ranged, Segment, Segmentation, SegmentedOver};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::NonEmpty;

segment::impl_target_forward_type_and_definition!(
    for <T> where T: Clone + Ord => BTreeSet,
    BTreeSetTarget,
    BTreeSetSegment,
);

impl<T> Ranged for BTreeSet<T>
where
    T: Clone + Ord,
{
    type Range = RelationalRange<T>;

    fn range(&self) -> Self::Range {
        self.first().cloned().zip(self.last().cloned()).into()
    }

    fn tail(&self) -> Self::Range {
        self.iter().nth(1).cloned().zip(self.last().cloned()).into()
    }

    fn rtail(&self) -> Self::Range {
        self.first()
            .cloned()
            .zip(self.iter().rev().nth(1).cloned())
            .into()
    }
}

impl<T> Segmentation for BTreeSet<T>
where
    T: Clone + Ord,
{
    fn tail(&mut self) -> BTreeSetSegment<'_, Self> {
        match Ranged::tail(self).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(self),
        }
    }

    fn rtail(&mut self) -> BTreeSetSegment<'_, Self> {
        match Ranged::rtail(self).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(self),
        }
    }
}

impl<T, R> segment::SegmentedBy<R> for BTreeSet<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Clone + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> BTreeSetSegment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_bounds(range))
    }
}

impl<T> SegmentedOver for BTreeSet<T>
where
    T: Clone + Ord,
{
    type Kind = BTreeSetTarget<Self>;
    type Target = Self;
}

type Cardinality<'a, T> = crate::Cardinality<&'a mut BTreeSet<T>, &'a mut BTreeSet<T>>;

pub type BTreeSet1<T> = NonEmpty<BTreeSet<T>>;

impl<T> BTreeSet1<T> {
    /// # Safety
    pub const unsafe fn from_btree_set_unchecked(items: BTreeSet<T>) -> Self {
        BTreeSet1 { items }
    }

    pub fn from_one(item: T) -> Self
    where
        T: Ord,
    {
        iter1::one(item).collect1()
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        T: Ord,
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        T: Ord,
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn into_btree_set(self) -> BTreeSet<T> {
        self.items
    }

    fn arity(&mut self) -> Cardinality<'_, T> {
        // `BTreeSet::len` is reliable even in the face of a non-conformant `Ord` implementation.
        match self.items.len() {
            0 => unreachable!(),
            1 => Cardinality::One(&mut self.items),
            _ => Cardinality::Many(&mut self.items),
        }
    }

    fn many_or_get_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        T: Ord,
        F: FnOnce(&mut BTreeSet<T>) -> T,
    {
        match self.arity() {
            // SAFETY:
            Cardinality::One(one) => Err(unsafe { one.first().unwrap_maybe_unchecked() }),
            Cardinality::Many(many) => Ok(f(many)),
        }
    }

    fn many_or_get<Q, F>(&mut self, query: &Q, f: F) -> Option<Result<T, &T>>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        F: FnOnce(&mut BTreeSet<T>) -> Option<T>,
    {
        let result = match self.arity() {
            Cardinality::One(one) => Err(one.get(query)),
            Cardinality::Many(many) => Ok(f(many)),
        };
        match result {
            Err(one) => one.map(Err),
            Ok(many) => many.map(Ok),
        }
    }

    pub fn split_off_tail(&mut self) -> BTreeSet<T>
    where
        T: Clone + UnsafeOrd,
    {
        match self.items.iter().nth(1).cloned() {
            // `BTreeSet::split_off` relies on the `Ord` implementation to determine where the
            // split begins. This requires `UnsafeOrd` here, because a non-conformant `Ord`
            // implementation may split at the first item (despite the matched expression) and
            // empty the `BTreeSet1`.
            Some(item) => self.items.split_off(&item),
            _ => BTreeSet::new(),
        }
    }

    pub fn append<R>(&mut self, items: R)
    where
        T: Ord,
        R: Into<BTreeSet<T>>,
    {
        self.items.append(&mut items.into())
    }

    pub fn insert(&mut self, item: T) -> bool
    where
        T: Ord,
    {
        self.items.insert(item)
    }

    pub fn replace(&mut self, item: T) -> Option<T>
    where
        T: Ord,
    {
        self.items.replace(item)
    }

    pub fn pop_first_or_get_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_get_only(|items| unsafe { items.pop_first().unwrap_maybe_unchecked() })
    }

    pub fn pop_first_until_only(&mut self) -> &T
    where
        T: Ord,
    {
        self.pop_first_until_only_with(|_| {})
    }

    pub fn pop_first_until_only_with<F>(&mut self, mut f: F) -> &T
    where
        T: Ord,
        F: FnMut(T),
    {
        while let Ok(item) = self.pop_first_or_get_only() {
            f(item);
        }
        // SAFETY:
        unsafe { self.pop_first_or_get_only().err().unwrap_maybe_unchecked() }
    }

    pub fn pop_last_or_get_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_get_only(|items| unsafe { items.pop_last().unwrap_maybe_unchecked() })
    }

    pub fn pop_last_until_only(&mut self) -> &T
    where
        T: Ord,
    {
        self.pop_last_until_only_with(|_| {})
    }

    pub fn pop_last_until_only_with<F>(&mut self, mut f: F) -> &T
    where
        T: Ord,
        F: FnMut(T),
    {
        while let Ok(item) = self.pop_last_or_get_only() {
            f(item);
        }
        // SAFETY:
        unsafe { self.pop_last_or_get_only().err().unwrap_maybe_unchecked() }
    }

    pub fn remove_or_get_only<Q>(&mut self, query: &Q) -> Result<bool, &T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self
            .many_or_get(query, move |items| items.take(query))
            .transpose()
        {
            Ok(item) => Ok(item.is_some()),
            Err(only) => Err(only),
        }
    }

    pub fn take_or_get_only<Q>(&mut self, query: &Q) -> Option<Result<T, &T>>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.many_or_get(query, move |items| items.take(query))
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get(query)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn first(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn range<Q, R>(&self, range: R) -> btree_set::Range<'_, T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        self.items.range(range)
    }

    pub fn difference<'a, R>(&'a self, other: &'a R) -> btree_set::Difference<'a, T>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.difference(other.as_ref())
    }

    pub fn symmetric_difference<'a, R>(
        &'a self,
        other: &'a R,
    ) -> btree_set::SymmetricDifference<'a, T>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.symmetric_difference(other.as_ref())
    }

    pub fn intersection<'a, R>(&'a self, other: &'a R) -> btree_set::Intersection<'a, T>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.intersection(other.as_ref())
    }

    pub fn union<'a, R>(&'a self, other: &'a R) -> Iterator1<btree_set::Union<'a, T>>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.union(other.as_ref())) }
    }

    pub fn iter1(&self) -> Iterator1<btree_set::Iter<'_, T>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn is_disjoint<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_disjoint(other.as_ref())
    }

    pub fn is_subset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_subset(other.as_ref())
    }

    pub fn is_superset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_superset(other.as_ref())
    }

    pub fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.contains(item)
    }

    pub const fn as_btree_set(&self) -> &BTreeSet<T> {
        &self.items
    }
}

impl<R, T> BitAnd<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() & rhs.as_ref()
    }
}

impl<R, T> BitOr<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet1<T>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        // SAFETY:
        unsafe { BTreeSet1::from_btree_set_unchecked(self.as_btree_set() | rhs.as_ref()) }
    }
}

impl<R, T> BitXor<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() ^ rhs.as_ref()
    }
}

impl<T> Debug for BTreeSet1<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T> Extend<T> for BTreeSet1<T>
where
    T: Ord,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<T, const N: usize> From<[T; N]> for BTreeSet1<T>
where
    [T; N]: Array1,
    T: Ord,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY:
        unsafe { BTreeSet1::from_btree_set_unchecked(BTreeSet::from(items)) }
    }
}

impl<T> From<BTreeSet1<T>> for BTreeSet<T> {
    fn from(items: BTreeSet1<T>) -> Self {
        items.items
    }
}

impl<T> FromIterator1<T> for BTreeSet1<T>
where
    T: Ord,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY:
        unsafe { BTreeSet1::from_btree_set_unchecked(items.into_iter1().collect()) }
    }
}

impl<T> IntoIterator for BTreeSet1<T> {
    type Item = T;
    type IntoIter = btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for BTreeSet1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> Segmentation for BTreeSet1<T>
where
    T: Clone + UnsafeOrd,
{
    fn tail(&mut self) -> BTreeSetSegment<'_, Self> {
        match Ranged::tail(&self.items).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(&mut self.items),
        }
    }

    fn rtail(&mut self) -> BTreeSetSegment<'_, Self> {
        match Ranged::rtail(&self.items).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(&mut self.items),
        }
    }
}

impl<T, R> segment::SegmentedBy<R> for BTreeSet1<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Clone + UnsafeOrd,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> BTreeSetSegment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_bounds(range))
    }
}

impl<T> SegmentedOver for BTreeSet1<T>
where
    T: Clone + UnsafeOrd,
{
    type Kind = BTreeSetTarget<Self>;
    type Target = BTreeSet<T>;
}

impl<R, T> Sub<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() - rhs.as_ref()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<BTreeSet<T>>> for BTreeSet1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<BTreeSet<T>>) -> Result<Self, Self::Error> {
        BTreeSet1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T> TryFrom<BTreeSet<T>> for BTreeSet1<T> {
    type Error = BTreeSet<T>;

    fn try_from(items: BTreeSet<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY:
            _ => Ok(unsafe { BTreeSet1::from_btree_set_unchecked(items) }),
        }
    }
}

impl<'a, K, T> BTreeSetSegment<'a, K>
where
    K: SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    pub fn insert_in_range(&mut self, item: T) -> Result<bool, T> {
        if self.range.contains(&item) {
            Ok(self.items.insert(item))
        }
        else {
            Err(item)
        }
    }

    pub fn append_in_range(&mut self, other: &mut BTreeSet<T>) {
        if let RelationalRange::NonEmpty { ref start, ref end } = self.range {
            let low = other;
            let mut middle = low.split_off(start);
            let mut high = middle.split_off(end);
            self.items.append(&mut middle);
            if let Some(first) = high.take(end) {
                self.items.insert(first);
            }
            low.append(&mut high);
        }
    }

    pub fn remove(&mut self, item: &T) -> bool {
        if self.range.contains(item) {
            self.items.remove(item)
        }
        else {
            false
        }
    }

    pub fn take(&mut self, item: &T) -> Option<T> {
        if self.range.contains(item) {
            self.items.take(item)
        }
        else {
            None
        }
    }

    pub fn clear(&mut self) {
        if let Some(range) = self.range.clone().try_into_range_inclusive() {
            self.items.retain(|item| !range.contains(item));
        }
    }

    pub fn get(&self, item: &T) -> Option<&T> {
        if self.range.contains(item) {
            self.items.get(item)
        }
        else {
            None
        }
    }

    pub fn first(&self) -> Option<&T> {
        match self.range {
            RelationalRange::Empty => None,
            RelationalRange::NonEmpty { ref start, .. } => self.items.get(start),
        }
    }

    pub fn last(&self) -> Option<&T> {
        match self.range {
            RelationalRange::Empty => None,
            RelationalRange::NonEmpty { ref end, .. } => self.items.get(end),
        }
    }

    pub fn contains(&self, item: &T) -> bool {
        self.range.contains(item) && self.items.contains(item)
    }
}

impl<'a, K, T> Segmentation for BTreeSetSegment<'a, K>
where
    K: SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    fn tail(&mut self) -> BTreeSetSegment<'_, K> {
        match self.range.clone().try_into_range_inclusive() {
            Some(range) => match BTreeSet::range(self.items, range.clone()).nth(1) {
                Some(start) => Segment::unchecked(
                    self.items,
                    RelationalRange::unchecked(start.clone(), range.end().clone()),
                ),
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }

    fn rtail(&mut self) -> BTreeSetSegment<'_, K> {
        match self.range.clone().try_into_range_inclusive() {
            Some(range) => match BTreeSet::range(self.items, range.clone()).rev().nth(1) {
                Some(end) => Segment::unchecked(
                    self.items,
                    RelationalRange::unchecked(range.start().clone(), end.clone()),
                ),
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }
}

impl<'a, K, T, R> segment::SegmentedBy<R> for BTreeSetSegment<'a, K>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    K: segment::SegmentedBy<R> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> BTreeSetSegment<'_, K> {
        Segment::intersect(self.items, &range::ordered_range_bounds(range))
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::btree_set1::BTreeSet1;
    use crate::iter1::{self, FromIterator1};

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> BTreeSet1<u8> {
        BTreeSet1::from_iter1(iter1::harness::xs1(end))
    }

    #[fixture]
    pub fn terminals1(#[default(0)] first: u8, #[default(9)] last: u8) -> BTreeSet1<u8> {
        BTreeSet1::from_iter1([first, last])
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::btree_set1::harness::{self, terminals1};
    use crate::btree_set1::BTreeSet1;
    use crate::iter1::FromIterator1;
    use crate::Segmentation;

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_btree_set1_then_btree_set1_eq_head(#[case] mut xs1: BTreeSet1<u8>) {
        xs1.tail().clear();
        assert_eq!(xs1, BTreeSet1::from_one(0));
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_btree_set1_then_btree_set1_eq_tail(#[case] mut xs1: BTreeSet1<u8>) {
        let tail = *xs1.last();
        xs1.rtail().clear();
        assert_eq!(xs1, BTreeSet1::from_one(tail));
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_btree_set1_then_btree_set1_eq_head_and_tail(
        #[case] mut xs1: BTreeSet1<u8>,
    ) {
        let n = xs1.len().get();
        let head_and_tail = [0, *xs1.last()];
        xs1.tail().rtail().clear();
        assert_eq!(
            xs1,
            BTreeSet1::try_from_iter(if n > 1 {
                head_and_tail[..].iter().copied()
            }
            else {
                head_and_tail[..1].iter().copied()
            })
            .unwrap(),
        );
    }

    #[rstest]
    #[case::absent_in_range(4, 4, Ok(true))]
    #[case::present_in_range(4, 9, Ok(false))]
    #[case::out_of_range(4, 0, Err(0))]
    #[case::out_of_range(4, 1, Err(1))]
    fn insert_into_btree_set1_segment_range_from_then_output_eq(
        #[from(terminals1)] mut xs1: BTreeSet1<u8>,
        #[case] from: u8,
        #[case] item: u8,
        #[case] expected: Result<bool, u8>,
    ) {
        let mut segment = xs1.segment(from..);
        assert_eq!(segment.insert_in_range(item), expected);
    }
}

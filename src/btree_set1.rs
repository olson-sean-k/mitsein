#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};

use crate::array1::Array1;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
use crate::segment::range::{self, Intersect, RelationalRange};
use crate::segment::{self, Ranged, Segment, Segmentation, Segmented};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::NonEmpty;

impl<T> Ranged for BTreeSet<T>
where
    // This is a defensive bound. A `Segment` cannot reference items in a `BTreeSet` to form a
    // range since it must also have a mutable reference to the `BTreeSet`. This means that items
    // must be copied to form the range, which could cause segmentation to be very costly for
    // non-trivial `Clone` types. Consider `BTreeSet<Vec<i64>>`, for example.
    T: Copy + Ord,
{
    type Range = RelationalRange<T>;

    fn range(&self) -> Self::Range {
        self.first().copied().zip(self.last().copied()).into()
    }

    fn tail(&self) -> Self::Range {
        self.iter().nth(1).copied().zip(self.last().copied()).into()
    }

    fn rtail(&self) -> Self::Range {
        self.first()
            .copied()
            .zip(self.iter().rev().nth(1).copied())
            .into()
    }
}

impl<T> Segmentation for BTreeSet<T>
where
    T: Copy + Ord,
{
    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match Ranged::tail(self).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(self),
        }
    }

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match Ranged::rtail(self).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(self),
        }
    }
}

impl<T> Segmented for BTreeSet<T>
where
    T: Copy + Ord,
{
    type Kind = Self;
    type Target = Self;
}

// TODO: Support borrowing a key from items and querying with a range over such a key (rather than
//       only a range over the item `T`). Note that this is less useful for `Copy` types than
//       non-`Copy` types though.
impl<T, R> segment::SegmentedBy<R> for BTreeSet<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Copy + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target> {
        Segment::intersect(self, &range::ordered_range_bounds(range))
    }
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

    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        T: Ord,
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(BTreeSet1::from_iter1)
    }

    pub fn into_btree_set(self) -> BTreeSet<T> {
        self.items
    }

    fn arity(&mut self) -> Cardinality<'_, T> {
        match self.items.len() {
            0 => unreachable!(),
            1 => Cardinality::One(&mut self.items),
            _ => Cardinality::Many(&mut self.items),
        }
    }

    fn many_or_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        T: Ord,
        F: FnOnce(&mut BTreeSet<T>) -> T,
    {
        match self.arity() {
            // SAFETY:
            Cardinality::One(one) => Err(unsafe { one.first().unwrap_unchecked() }),
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
        T: Clone + Ord,
    {
        match self.items.iter().nth(1).cloned() {
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

    pub fn pop_first_or_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_first().unwrap_unchecked() })
    }

    pub fn pop_last_or_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_last().unwrap_unchecked() })
    }

    pub fn remove_or_only<Q>(&mut self, query: &Q) -> Result<bool, &T>
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

    pub fn take_or_only<Q>(&mut self, query: &Q) -> Option<Result<T, &T>>
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
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub fn first(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.first().unwrap_unchecked() }
    }

    pub fn last(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.last().unwrap_unchecked() }
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
    T: Copy + Ord,
{
    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match Ranged::tail(&self.items).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(&mut self.items),
        }
    }

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match Ranged::rtail(&self.items).try_into_range_inclusive() {
            Some(range) => self.segment(range),
            _ => Segment::empty(&mut self.items),
        }
    }
}

impl<T> Segmented for BTreeSet1<T>
where
    T: Copy + Ord,
{
    type Kind = Self;
    type Target = BTreeSet<T>;
}

impl<T, R> segment::SegmentedBy<R> for BTreeSet1<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Copy + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_bounds(range))
    }
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

pub type BTreeSetSegment<'a, T> = Segment<'a, BTreeSet<T>, BTreeSet<T>>;

pub type BTreeSet1Segment<'a, T> = Segment<'a, BTreeSet1<T>, BTreeSet<T>>;

impl<'a, K, T> Segment<'a, K, BTreeSet<T>>
where
    K: Segmented<Target = BTreeSet<T>>,
    T: Copy + Ord,
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

    // This function cannot query the set via some `Q` where `T: Borrow<Q>`, because it is possible
    // that the ordering of `Q` disagrees with the ordering of `T` and so items outside of the
    // segment could be removed. Removing an item outside of the segment is unsound!
    pub fn remove(&mut self, item: &T) -> bool {
        if self.range.contains(item) {
            self.items.remove(item)
        }
        else {
            false
        }
    }

    // This function cannot query the set via some `Q` where `T: Borrow<Q>`, because it is possible
    // that the ordering of `Q` disagrees with the ordering of `T` and so items outside of the
    // segment could be removed. Removing an item outside of the segment is unsound!
    pub fn take(&mut self, item: &T) -> Option<T> {
        if self.range.contains(item) {
            self.items.take(item)
        }
        else {
            None
        }
    }

    pub fn clear(&mut self) {
        if let Some(range) = self.range.try_into_range_inclusive() {
            self.items.retain(|item| !range.contains(item));
        }
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.range.contains(query) {
            self.items.get(query)
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

    pub fn contains<Q>(&self, query: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.range.contains(query) && self.items.contains(query)
    }
}

impl<'a, K, T> Segmentation for Segment<'a, K, BTreeSet<T>>
where
    K: Segmented<Target = BTreeSet<T>>,
    K::Target: Ranged<Range = RelationalRange<T>>,
    T: Copy + Ord,
{
    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match self.range.try_into_range_inclusive() {
            Some(range) => match BTreeSet::range(self.items, range.clone()).nth(1) {
                Some(start) => {
                    Segment::unchecked(self.items, RelationalRange::unchecked(*start, *range.end()))
                },
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target> {
        match self.range.try_into_range_inclusive() {
            Some(range) => match BTreeSet::range(self.items, range.clone()).rev().nth(1) {
                Some(end) => {
                    Segment::unchecked(self.items, RelationalRange::unchecked(*range.start(), *end))
                },
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }
}

impl<'a, K, T> Segmented for Segment<'a, K, BTreeSet<T>>
where
    K: Segmented<Target = BTreeSet<T>>,
    T: Copy + Ord,
{
    type Kind = K;
    type Target = K::Target;
}

impl<'a, K, T, R> segment::SegmentedBy<R> for Segment<'a, K, BTreeSet<T>>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    K: segment::SegmentedBy<R, Target = BTreeSet<T>>,
    T: Copy + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target> {
        Segment::intersect(self.items, &range::ordered_range_bounds(range))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::btree_set1::BTreeSet1;
    use crate::Segmentation;

    #[test]
    fn segmentation() {
        let mut xs = BTreeSet1::from([0i32, 1, 2, 3]);
        xs.tail().clear();
        assert_eq!(xs.into_iter().collect::<Vec<_>>().as_slice(), &[0]);

        let mut xs = BTreeSet1::from([0i32, 1, 2, 3]);
        xs.rtail().clear();
        assert_eq!(xs.into_iter().collect::<Vec<_>>().as_slice(), &[3]);

        let mut xs = BTreeSet1::from([0i32, 1, 2, 3]);
        xs.tail().rtail().clear();
        assert_eq!(xs.into_iter().collect::<Vec<_>>().as_slice(), &[0, 3]);

        let mut xs = BTreeSet1::from([0i32]);
        xs.tail().clear();
        assert_eq!(xs.into_iter().collect::<Vec<_>>().as_slice(), &[0]);

        let mut xs = BTreeSet1::from([0i32]);
        xs.rtail().clear();
        assert_eq!(xs.into_iter().collect::<Vec<_>>().as_slice(), &[0]);

        let mut xs = BTreeSet1::from([0i32, 9]);
        let mut segment = xs.segment(4..);
        assert_eq!(segment.insert_in_range(4), Ok(true));
        assert_eq!(segment.insert_in_range(9), Ok(false));
        assert_eq!(segment.insert_in_range(0), Err(0));
        assert_eq!(segment.insert_in_range(1), Err(1));
    }
}

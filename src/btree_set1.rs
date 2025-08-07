//! A non-empty [`BTreeSet`][`btree_set`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, Bound, RangeBounds, Sub};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::array1::Array1;
use crate::cmp::{UnsafeIsomorph, UnsafeOrd};
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, Intersect, IntersectionExt, ItemRange, Resolve, TrimRange};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::take;
use crate::{EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedBTreeSet>::Item;

pub trait ClosedBTreeSet {
    type Item;

    fn as_btree_set(&self) -> &BTreeSet<Self::Item>;
}

impl<T> ClosedBTreeSet for BTreeSet<T> {
    type Item = T;

    fn as_btree_set(&self) -> &BTreeSet<Self::Item> {
        self
    }
}

impl<T> Extend1<T> for BTreeSet<T>
where
    T: Ord,
{
    fn extend_non_empty<I>(mut self, items: I) -> BTreeSet1<T>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { BTreeSet1::from_btree_set_unchecked(self) }
    }
}

unsafe impl<T> MaybeEmpty for BTreeSet<T> {
    fn cardinality(&self) -> Option<crate::Cardinality<(), ()>> {
        // `BTreeSet::len` is reliable even in the face of a non-conformant `Ord` implementation.
        // The `BTreeSet1` implementation relies on this to maintain its non-empty invariant
        // without bounds on `UnsafeOrd`.
        match self.len() {
            0 => None,
            1 => Some(crate::Cardinality::One(())),
            _ => Some(crate::Cardinality::Many(())),
        }
    }
}

impl<T> Ranged for BTreeSet<T> {
    type NominalRange = TrimRange;

    fn all(&self) -> Self::NominalRange {
        TrimRange::ALL
    }

    fn tail(&self) -> Self::NominalRange {
        TrimRange::TAIL1
    }

    fn rtail(&self) -> Self::NominalRange {
        TrimRange::RTAIL1
    }
}

impl<T> Segmentation for BTreeSet<T> {
    type Tail = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Tail> {
        Segment::unchecked(self, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Tail> {
        Segment::unchecked(self, TrimRange::RTAIL1)
    }
}

impl<T, Q, R> SegmentedBy<Q, R> for BTreeSet<T>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + Ord,
    Q: Ord + ?Sized,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, Self, Self::Range> {
        segment::Segment::<_, _, ItemRange<T>>::intersect_with(
            self,
            &range::ordered_range_bounds(range),
            |items, all: TrimRange| all.resolve(items),
        )
    }
}

impl<T> SegmentedOver for BTreeSet<T> {
    type Kind = Self;
    type Target = Self;
}

type Take<'a, T, U, N = ()> = take::Take<'a, BTreeSet<T>, U, N>;

pub type Pop<'a, K> = Take<'a, ItemFor<K>, ItemFor<K>>;

pub type DropRemove<'a, 'q, K, Q> = Take<'a, ItemFor<K>, bool, &'q Q>;

pub type TakeRemove<'a, 'q, K, Q> = Take<'a, ItemFor<K>, Option<ItemFor<K>>, &'q Q>;

impl<'a, T, U, N> Take<'a, T, U, N>
where
    T: Ord,
{
    pub fn or_get_only(self) -> Result<U, &'a T> {
        self.take_or_else(|items, _| items.first())
    }
}

impl<'a, T, Q> Take<'a, T, bool, &'_ Q>
where
    T: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    pub fn or_get(self) -> Result<bool, Option<&'a T>> {
        self.take_or_else(|items, query| items.get(query))
    }
}

impl<'a, T, Q> Take<'a, T, Option<T>, &'_ Q>
where
    T: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    pub fn or_get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, query| items.get(query))
    }
}

pub type BTreeSet1<T> = NonEmpty<BTreeSet<T>>;

impl<T> BTreeSet1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`BTreeSet::new()`][`BTreeSet::new`].
    ///
    /// [`BTreeSet::new`]: alloc::collections::btree_set::BTreeSet::new
    pub unsafe fn from_btree_set_unchecked(items: BTreeSet<T>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
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

    pub fn append(&mut self, items: &mut BTreeSet<T>)
    where
        T: Ord,
    {
        self.items.append(items)
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

    pub fn pop_first(&mut self) -> Pop<'_, Self>
    where
        T: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        Take::with(self, (), |items, _| unsafe {
            items.items.pop_first().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_first_until_only(&mut self) -> PopFirstUntilOnly<'_, T>
    where
        T: Ord,
    {
        PopFirstUntilOnly { items: self }
    }

    pub fn pop_last(&mut self) -> Pop<'_, Self>
    where
        T: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        Take::with(self, (), |items, _| unsafe {
            items.items.pop_last().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_last_until_only(&mut self) -> PopLastUntilOnly<'_, T>
    where
        T: Ord,
    {
        PopLastUntilOnly { items: self }
    }

    pub fn remove<'a, 'q, Q>(&'a mut self, query: &'q Q) -> DropRemove<'a, 'q, Self, Q>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Take::with(self, query, |items, query| items.items.remove(query))
    }

    pub fn take<'a, 'q, Q>(&'a mut self, query: &'q Q) -> TakeRemove<'a, 'q, Self, Q>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        Take::with(self, query, |items, query| items.items.take(query))
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get(query)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn first(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY: `self` must be non-empty.
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
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.difference(other.as_btree_set())
    }

    pub fn symmetric_difference<'a, R>(
        &'a self,
        other: &'a R,
    ) -> btree_set::SymmetricDifference<'a, T>
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.symmetric_difference(other.as_btree_set())
    }

    pub fn intersection<'a, R>(&'a self, other: &'a R) -> btree_set::Intersection<'a, T>
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.intersection(other.as_btree_set())
    }

    pub fn union<'a, R>(&'a self, other: &'a R) -> Iterator1<btree_set::Union<'a, T>>
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        // SAFETY: `self` must be non-empty and `BTreeSet::union` cannot reduce the cardinality of
        //         its inputs.
        unsafe { Iterator1::from_iter_unchecked(self.items.union(other.as_btree_set())) }
    }

    pub fn iter1(&self) -> Iterator1<btree_set::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn is_disjoint<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.is_disjoint(other.as_btree_set())
    }

    pub fn is_subset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.is_subset(other.as_btree_set())
    }

    pub fn is_superset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: ClosedBTreeSet<Item = T>,
    {
        self.items.is_superset(other.as_btree_set())
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

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> BTreeSet1<T>
where
    T: Ord,
{
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        T: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T> Arbitrary<'a> for BTreeSet1<T>
where
    T: Arbitrary<'a> + Ord,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(T::arbitrary(unstructured), unstructured.arbitrary_iter()?).collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (T::size_hint(depth).0, None)
    }
}

impl<R, T> BitAnd<&'_ R> for &'_ BTreeSet1<T>
where
    R: ClosedBTreeSet<Item = T>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() & rhs.as_btree_set()
    }
}

impl<T> BitAnd<&'_ BTreeSet1<T>> for &'_ BTreeSet<T>
where
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitand(self, rhs: &'_ BTreeSet1<T>) -> Self::Output {
        self & rhs.as_btree_set()
    }
}

impl<R, T> BitOr<&'_ R> for &'_ BTreeSet1<T>
where
    R: ClosedBTreeSet<Item = T>,
    T: Clone + Ord,
{
    type Output = BTreeSet1<T>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        // SAFETY: `self` must be non-empty and `BTreeSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { BTreeSet1::from_btree_set_unchecked(self.as_btree_set() | rhs.as_btree_set()) }
    }
}

impl<T> BitOr<&'_ BTreeSet1<T>> for &'_ BTreeSet<T>
where
    T: Clone + Ord,
{
    type Output = BTreeSet1<T>;

    fn bitor(self, rhs: &'_ BTreeSet1<T>) -> Self::Output {
        // SAFETY: `rhs` must be non-empty and `BTreeSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { BTreeSet1::from_btree_set_unchecked(self | rhs.as_btree_set()) }
    }
}

impl<R, T> BitXor<&'_ R> for &'_ BTreeSet1<T>
where
    R: ClosedBTreeSet<Item = T>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() ^ rhs.as_btree_set()
    }
}

impl<T> BitXor<&'_ BTreeSet1<T>> for &'_ BTreeSet<T>
where
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitxor(self, rhs: &'_ BTreeSet1<T>) -> Self::Output {
        self ^ rhs.as_btree_set()
    }
}

impl<T> ClosedBTreeSet for BTreeSet1<T> {
    type Item = T;

    fn as_btree_set(&self) -> &BTreeSet<Self::Item> {
        self.as_ref()
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
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
        unsafe { BTreeSet1::from_btree_set_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> FromParallelIterator1<T> for BTreeSet1<T>
where
    T: Ord + Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe {
            BTreeSet1::from_btree_set_unchecked(items.into_par_iter1().into_par_iter().collect())
        }
    }
}

impl<T> IntoIterator for BTreeSet1<T> {
    type Item = T;
    type IntoIter = btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a BTreeSet1<T> {
    type Item = &'a T;
    type IntoIter = btree_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<T> IntoIterator1 for BTreeSet1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> IntoIterator1 for &'_ BTreeSet1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator for BTreeSet1<T>
where
    T: Ord + Send,
{
    type Item = T;
    type Iter = <BTreeSet<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a BTreeSet1<T>
where
    T: Ord + Sync,
{
    type Item = &'a T;
    type Iter = <&'a BTreeSet<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for BTreeSet1<T>
where
    T: Ord + Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for &'_ BTreeSet1<T>
where
    T: Ord + Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

impl<T> Segmentation for BTreeSet1<T>
where
    T: UnsafeOrd,
{
    type Tail = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Tail> {
        Segment::unchecked(&mut self.items, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Tail> {
        Segment::unchecked(&mut self.items, TrimRange::RTAIL1)
    }
}

impl<T, Q, R> SegmentedBy<Q, R> for BTreeSet1<T>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + UnsafeIsomorph<Q>,
    Q: ?Sized + UnsafeOrd,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, Self, Self::Range> {
        segment::Segment::<_, _, ItemRange<T>>::intersect_strict_subset_with(
            &mut self.items,
            &range::ordered_range_bounds(range),
            |items, all: TrimRange| all.resolve(items),
        )
    }
}

impl<T> SegmentedOver for BTreeSet1<T>
where
    T: UnsafeOrd,
{
    type Kind = Self;
    type Target = BTreeSet<T>;
}

impl<R, T> Sub<&'_ R> for &'_ BTreeSet1<T>
where
    R: ClosedBTreeSet<Item = T>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() - rhs.as_btree_set()
    }
}

impl<T> Sub<&'_ BTreeSet1<T>> for &'_ BTreeSet<T>
where
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn sub(self, rhs: &'_ BTreeSet1<T>) -> Self::Output {
        self - rhs.as_btree_set()
    }
}

impl<T> TryFrom<BTreeSet<T>> for BTreeSet1<T> {
    type Error = EmptyError<BTreeSet<T>>;

    fn try_from(items: BTreeSet<T>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[derive(Debug)]
pub struct PopFirstUntilOnly<'a, T>
where
    T: Ord,
{
    items: &'a mut BTreeSet1<T>,
}

impl<T> Drop for PopFirstUntilOnly<'_, T>
where
    T: Ord,
{
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

impl<T> Iterator for PopFirstUntilOnly<'_, T>
where
    T: Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_first().or_none()
    }
}

#[derive(Debug)]
pub struct PopLastUntilOnly<'a, T>
where
    T: Ord,
{
    items: &'a mut BTreeSet1<T>,
}

impl<T> Drop for PopLastUntilOnly<'_, T>
where
    T: Ord,
{
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

impl<T> Iterator for PopLastUntilOnly<'_, T>
where
    T: Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_last().or_none()
    }
}

//pub type Segment<'a, K> =
//    segment::Segment<'a, K, BTreeSet<ItemFor<K>>, RelationalRange<ItemFor<K>>>;
pub type Segment<'a, K, R> = segment::Segment<'a, K, BTreeSet<ItemFor<K>>, R>;

impl<'a, T> Resolve<'a, BTreeSet<T>, ItemRange<T>> for TrimRange
where
    T: Clone + Ord,
{
    fn resolve(self, items: &'a BTreeSet<T>) -> ItemRange<T> {
        let range: ItemRange<&T> = self.resolve(items);
        range.cloned()
    }
}

impl<'a, T> Resolve<'a, BTreeSet<T>, ItemRange<&'a T>> for TrimRange
where
    T: Ord,
{
    fn resolve(self, items: &'a BTreeSet<T>) -> ItemRange<&'a T> {
        let TrimRange { tail, rtail } = self;
        items
            .iter()
            .nth(tail)
            .zip(items.iter().rev().nth(rtail))
            .into()
    }
}

impl<K, T> Segment<'_, K, ItemRange<T>>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Ord,
{
    fn remove_isomorph_unchecked<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.take_isomorph_unchecked(key).is_some()
    }

    fn take_isomorph_unchecked<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.contains(key) {
            self.items.take(key)
        }
        else {
            None
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.items.retain(self.range.retain_in_range(f))
    }

    pub fn insert_in_range(&mut self, item: T) -> Result<bool, T> {
        if self.contains(&item) {
            Ok(self.items.insert(item))
        }
        else {
            Err(item)
        }
    }

    pub fn append_in_range(&mut self, other: &mut BTreeSet<T>) {
        if let ItemRange::NonEmpty { start, end } = &self.range {
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

    pub fn clear(&mut self) {
        self.retain(|_| false)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.range.contains(key) {
            self.items.get(key)
        }
        else {
            None
        }
    }

    pub fn first(&self) -> Option<&T> {
        match &self.range {
            ItemRange::Empty => None,
            ItemRange::NonEmpty { start, .. } => self.items.get(start),
        }
    }

    pub fn last(&self) -> Option<&T> {
        match &self.range {
            ItemRange::Empty => None,
            ItemRange::NonEmpty { end, .. } => self.items.get(end),
        }
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.range.contains(key) && self.items.contains(key)
    }
}

impl<T> Segment<'_, BTreeSet<T>, ItemRange<T>>
where
    T: Ord,
{
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove_isomorph_unchecked(key)
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.take_isomorph_unchecked(key)
    }
}

impl<T> Segment<'_, BTreeSet1<T>, ItemRange<T>>
where
    T: UnsafeOrd,
{
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q> + UnsafeIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.remove_isomorph_unchecked(key)
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q> + UnsafeIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.take_isomorph_unchecked(key)
    }
}

impl<K, T> Segment<'_, K, TrimRange>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Ord,
{
    fn remove_isomorph_unchecked<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.take_isomorph_unchecked(key).is_some()
    }

    fn take_isomorph_unchecked<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.contains(key) {
            self.items.take(key)
        }
        else {
            None
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let TrimRange { tail, rtail } = self.range;
        let rtail = self.items.len().saturating_sub(rtail);
        let mut index = 0usize;
        self.items.retain(|item| {
            let is_in_range = index > tail || index < rtail;
            index = index.checked_add(1).expect("overflow computing item index");
            (!is_in_range) || f(item)
        });
    }

    pub fn insert_in_range(&mut self, item: T) -> Result<bool, T> {
        if self.contains(&item) {
            Ok(self.items.insert(item))
        }
        else {
            Err(item)
        }
    }

    pub fn append_in_range(&mut self, other: &mut BTreeSet<T>)
    where
        T: Clone,
    {
        let range: ItemRange<_> = self.range.resolve(self.items);
        if let ItemRange::NonEmpty { start, end } = range.cloned() {
            let low = other;
            let mut middle = low.split_off(&start);
            let mut high = middle.split_off(&end);
            self.items.append(&mut middle);
            if let Some(first) = high.take(&end) {
                self.items.insert(first);
            }
            low.append(&mut high);
        }
    }

    pub fn clear(&mut self) {
        self.retain(|_| false)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self::get_in_trim_range(self.items, key, &self.range)
    }

    pub fn first(&self) -> Option<&T> {
        self.items.iter().nth(self.range.tail)
    }

    pub fn last(&self) -> Option<&T> {
        self.items.iter().rev().nth(self.range.rtail)
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self::get_in_trim_range(self.items, key, &self.range).is_some()
    }
}

impl<T> Segment<'_, BTreeSet<T>, TrimRange>
where
    T: Ord,
{
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove_isomorph_unchecked(key)
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.take_isomorph_unchecked(key)
    }
}

impl<T> Segment<'_, BTreeSet1<T>, TrimRange>
where
    T: UnsafeOrd,
{
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q> + UnsafeIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.remove_isomorph_unchecked(key)
    }

    pub fn take<Q>(&mut self, key: &Q) -> Option<T>
    where
        T: Borrow<Q> + UnsafeIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.take_isomorph_unchecked(key)
    }
}

impl<K, T> Segmentation for Segment<'_, K, ItemRange<T>>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    type Tail = ItemRange<T>;

    fn tail(&mut self) -> Segment<'_, K, Self::Tail> {
        match &self.range {
            ItemRange::Empty => Segment::empty(self.items),
            ItemRange::NonEmpty { start, end } => match self
                .items
                .range((Bound::Included(start), Bound::Included(end)))
                .nth(1)
            {
                Some(start) => Segment::unchecked(
                    self.items,
                    ItemRange::unchecked(start.clone(), end.clone()).into(),
                ),
                _ => Segment::empty(self.items),
            },
        }
    }

    fn rtail(&mut self) -> Segment<'_, K, Self::Tail> {
        match &self.range {
            ItemRange::Empty => Segment::empty(self.items),
            ItemRange::NonEmpty { start, end } => match self
                .items
                .range((Bound::Included(start), Bound::Included(end)))
                .rev()
                .nth(1)
            {
                Some(end) => Segment::unchecked(
                    self.items,
                    ItemRange::unchecked(start.clone(), end.clone()).into(),
                ),
                _ => Segment::empty(self.items),
            },
        }
    }
}

impl<K, T> Segmentation for Segment<'_, K, TrimRange>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    type Tail = TrimRange;

    fn tail(&mut self) -> Segment<'_, K, Self::Tail> {
        Segment::unchecked(self.items, self.range.tail())
    }

    fn rtail(&mut self) -> Segment<'_, K, Self::Tail> {
        Segment::unchecked(self.items, self.range.rtail())
    }
}

impl<'a, T, Q, R> SegmentedBy<Q, R> for Segment<'a, BTreeSet<T>, ItemRange<T>>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + Ord,
    Q: Ord + ?Sized,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, BTreeSet<T>, Self::Range> {
        let range = self
            .range
            .intersect(&range::ordered_range_bounds(range))
            .expect_in_bounds();
        Segment::unchecked(self.items, range)
    }
}

impl<'a, T, Q, R> SegmentedBy<Q, R> for Segment<'a, BTreeSet1<T>, ItemRange<T>>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + UnsafeIsomorph<Q>,
    Q: ?Sized + UnsafeOrd,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, BTreeSet1<T>, Self::Range> {
        let range = self
            .range
            .intersect(&range::ordered_range_bounds(range))
            .expect_in_bounds();
        Segment::unchecked(self.items, range)
    }
}

impl<'a, T, Q, R> SegmentedBy<Q, R> for Segment<'a, BTreeSet<T>, TrimRange>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + Ord,
    Q: Ord + ?Sized,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, BTreeSet<T>, Self::Range> {
        let range = Resolve::<_, ItemRange<_>>::resolve(self.range, self.items)
            .intersect(&range::ordered_range_bounds(range))
            .expect_in_bounds();
        Segment::unchecked(self.items, range)
    }
}

impl<'a, T, Q, R> SegmentedBy<Q, R> for Segment<'a, BTreeSet1<T>, TrimRange>
where
    ItemRange<T>: Intersect<R, Output = ItemRange<T>>,
    T: Borrow<Q> + Clone + UnsafeIsomorph<Q>,
    Q: ?Sized + UnsafeOrd,
    R: RangeBounds<Q>,
{
    type Range = ItemRange<T>;

    fn segment(&mut self, range: R) -> Segment<'_, BTreeSet1<T>, Self::Range> {
        let range = Resolve::<_, ItemRange<_>>::resolve(self.range, self.items)
            .intersect(&range::ordered_range_bounds(range))
            .expect_in_bounds();
        Segment::unchecked(self.items, range)
    }
}

fn get_in_trim_range<'a, T, Q>(items: &'a BTreeSet<T>, key: &Q, range: &TrimRange) -> Option<&'a T>
where
    T: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    let TrimRange { tail, rtail } = range;
    let predicate = |item: &&_| Borrow::<Q>::borrow(*item) == key;
    items.get(key).take_if(|_| {
        items.iter().take(*tail).find(predicate).is_none()
            && items.iter().rev().take(*rtail).find(predicate).is_none()
    })
}

#[cfg(test)]
pub mod harness {
    use alloc::string::String;
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

    #[fixture]
    pub fn alphabet1() -> BTreeSet1<String> {
        BTreeSet1::from_iter1([
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "e".into(),
            "f".into(),
            "g".into(),
            "h".into(),
            "i".into(),
            "j".into(),
            "k".into(),
            "l".into(),
            "m".into(),
            "n".into(),
            "o".into(),
            "p".into(),
            "q".into(),
            "r".into(),
            "s".into(),
            "t".into(),
            "u".into(),
            "v".into(),
            "w".into(),
            "x".into(),
            "y".into(),
            "z".into(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::ops::Bound;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::btree_set1::harness::{self, alphabet1, terminals1};
    use crate::btree_set1::BTreeSet1;
    use crate::iter1::FromIterator1;
    use crate::Segmentation;
    #[cfg(feature = "serde")]
    use crate::{
        btree_set1::harness::xs1,
        serde::{self, harness::sequence},
    };

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
    //#[case("w".., vec!["x", "y", "z"])]
    #[case((Bound::Included("w"), Bound::Unbounded), vec!["x", "y", "z"])]
    fn get_btree_set1_segment_by_isomorph_then_segment_eq(
        #[from(alphabet1)] mut xs1: BTreeSet1<String>,
        #[case] segment: (Bound<&'static str>, Bound<&'static str>),
        #[case] expected: Vec<&'static str>,
    ) {
        let xss = xs1.segment(segment);
        assert_eq!(xss.iter().collect(), expected,);
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

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_btree_set1_into_and_from_tokens_eq(
        xs1: BTreeSet1<u8>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_btree_set1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<BTreeSet1<u8>, Vec<_>>(sequence)
    }
}

//! A non-empty [`BTreeSet`][`btree_set`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::cmp::UnsafeOrd;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, Intersect, RelationalRange};
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
    fn tail(&mut self) -> Segment<'_, Self> {
        match Ranged::tail(self).try_into_range_inclusive() {
            Some(range) => Segmentation::segment(self, range),
            _ => Segment::empty(self),
        }
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        match Ranged::rtail(self).try_into_range_inclusive() {
            Some(range) => Segmentation::segment(self, range),
            _ => Segment::empty(self),
        }
    }
}

impl<T, R> SegmentedBy<R> for BTreeSet<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Clone + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_bounds(range))
    }
}

impl<T> SegmentedOver for BTreeSet<T>
where
    T: Clone + Ord,
{
    type Kind = Self;
    type Target = Self;
}

type TakeIfMany<'a, T, U, N = ()> = take::TakeIfMany<'a, BTreeSet<T>, U, N>;

pub type PopIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, ItemFor<K>>;

pub type DropRemoveIfMany<'a, 'q, K, Q> = TakeIfMany<'a, ItemFor<K>, bool, &'q Q>;

pub type TakeRemoveIfMany<'a, 'q, K, Q> = TakeIfMany<'a, ItemFor<K>, Option<ItemFor<K>>, &'q Q>;

impl<'a, T, U, N> TakeIfMany<'a, T, U, N>
where
    T: Ord,
{
    pub fn or_get_only(self) -> Result<U, &'a T> {
        self.take_or_else(|items, _| items.first())
    }
}

impl<'a, T, Q> TakeIfMany<'a, T, bool, &'_ Q>
where
    T: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    pub fn or_get(self) -> Result<bool, Option<&'a T>> {
        self.take_or_else(|items, query| items.get(query))
    }
}

impl<'a, T, Q> TakeIfMany<'a, T, Option<T>, &'_ Q>
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

    pub fn pop_first_if_many(&mut self) -> PopIfMany<'_, Self>
    where
        T: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop_first().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_first_until_only(&mut self) -> PopFirstUntilOnly<'_, T>
    where
        T: Ord,
    {
        PopFirstUntilOnly { items: self }
    }

    pub fn pop_last_if_many(&mut self) -> PopIfMany<'_, Self>
    where
        T: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop_last().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_last_until_only(&mut self) -> PopLastUntilOnly<'_, T>
    where
        T: Ord,
    {
        PopLastUntilOnly { items: self }
    }

    pub fn remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> DropRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.remove(query))
    }

    pub fn take_if_many<'a, 'q, Q>(&'a mut self, query: &'q Q) -> TakeRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.take(query))
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

    /// # Safety
    ///
    /// The [`BTreeSet`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::btree_set1::BTreeSet1;
    ///
    /// let mut xs = BTreeSet1::from([0i32, 1, 2, 3]);
    /// // This block is unsound. The `&mut BTreeSet` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_btree_set().clear();
    /// }
    /// let x = xs.first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_btree_set(&mut self) -> &mut BTreeSet<T> {
        &mut self.items
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

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<T> JsonSchema for BTreeSet1<T>
where
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        BTreeSet::<T>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<BTreeSet<T>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        BTreeSet::<T>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        BTreeSet::<T>::schema_id()
    }
}

impl<T> Segmentation for BTreeSet1<T>
where
    T: Clone + UnsafeOrd,
{
    fn tail(&mut self) -> Segment<'_, Self> {
        match Ranged::tail(&self.items).try_into_range_inclusive() {
            Some(range) => Segmentation::segment(self, range),
            _ => Segment::empty(&mut self.items),
        }
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        match Ranged::rtail(&self.items).try_into_range_inclusive() {
            Some(range) => Segmentation::segment(self, range),
            _ => Segment::empty(&mut self.items),
        }
    }
}

impl<T, R> SegmentedBy<R> for BTreeSet1<T>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    T: Clone + UnsafeOrd,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_bounds(range))
    }
}

impl<T> SegmentedOver for BTreeSet1<T>
where
    T: Clone + UnsafeOrd,
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
        self.items.pop_first_if_many().or_none()
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
        self.items.pop_last_if_many().or_none()
    }
}

pub type Segment<'a, K> = segment::Segment<'a, K, BTreeSet<ItemFor<K>>>;

impl<K, T> Segment<'_, K>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.items.retain(self.range.retain_in_range(f))
    }

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

impl<K, T> Segmentation for Segment<'_, K>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
{
    fn tail(&mut self) -> Segment<'_, K> {
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

    fn rtail(&mut self) -> Segment<'_, K> {
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

impl<K, T, R> SegmentedBy<R> for Segment<'_, K>
where
    RelationalRange<T>: Intersect<R, Output = RelationalRange<T>>,
    K: ClosedBTreeSet<Item = T> + SegmentedBy<R> + SegmentedOver<Target = BTreeSet<T>>,
    T: Clone + Ord,
    R: RangeBounds<T>,
{
    fn segment(&mut self, range: R) -> Segment<'_, K> {
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
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::btree_set1::harness::{self, terminals1};
    use crate::btree_set1::BTreeSet1;
    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
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

    #[cfg(feature = "schemars")]
    #[rstest]
    fn btree_set1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<BTreeSet1<u8>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
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

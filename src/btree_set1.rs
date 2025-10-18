//! A non-empty [`BTreeSet`][`btree_set`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::iter::{Skip, Take};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, Bound, RangeBounds, Sub};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::cmp::{UnsafeIsomorph, UnsafeOrd};
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{
    self, IntoRangeBounds, ItemRange, OptionExt as _, OutOfBoundsError, RangeError,
    ResolveTrimRange, TrimRange, UnorderedError,
};
use crate::segment::{self, Segmentation, SegmentedBy, SegmentedOver, Tail};
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

impl<T> ResolveTrimRange<Option<ItemRange<T>>> for BTreeSet<T>
where
    T: Clone + Ord,
{
    fn resolve_trim_range(&self, range: TrimRange) -> Option<ItemRange<T>> {
        let TrimRange { tail, rtail } = range;
        self.iter()
            .nth(tail)
            .zip(self.iter().rev().nth(rtail))
            .and_then(|(start, end)| range::ordered_range_bounds(start.clone()..=end.clone()).ok())
            .map(|range| {
                let (start, end) = range.into_bounds();
                ItemRange::unchecked(start, end)
            })
    }
}

impl<T> Segmentation for BTreeSet<T> where T: Ord {}

impl<T, R> SegmentedBy<T, R> for BTreeSet<T>
where
    T: Ord,
    R: IntoRangeBounds<T>,
{
    type Range = Option<ItemRange<T>>;
    type Error = UnorderedError<Bound<T>>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self, Self::Range>, Self::Error> {
        range::ordered_range_bounds(range)
            .map(|range| {
                let (start, end) = range.into_bounds();
                Segment::unchecked(self, Some(ItemRange::unchecked(start, end)))
            })
            .map_err(|range| {
                let (start, end) = range.into_bounds();
                UnorderedError(start, end)
            })
    }
}

impl<T> Tail for BTreeSet<T> {
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(self, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(self, TrimRange::RTAIL1)
    }
}

impl<T> SegmentedOver for BTreeSet<T> {
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
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
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

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<BTreeSet<T>>>
    where
        T: Ord,
        F: FnMut(&T) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }

    pub fn retain_until_only<F>(&mut self, mut f: F) -> Option<&'_ T>
    where
        T: Ord,
        F: FnMut(&T) -> bool,
    {
        // Segmentation for relational collections can be expensive and involves very different
        // tradeoffs for collections with many items and collections with large items. For this
        // reason, the first item is filtered directly rather than using `tail` or `rtail` here.
        let mut index = 0usize;
        self.items.retain(|item| {
            let is_retained = index == 0 || f(item);
            index += 1;
            is_retained
        });
        if self.len().get() == 1 {
            let first = self.first();
            if f(first) { None } else { Some(first) }
        }
        else {
            if !f(self.first()) {
                // The first item is **not** retained and there is more than one item.
                self.pop_first_if_many().or_none();
            }
            None
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

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.contains(key)
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

impl<T> Segmentation for BTreeSet1<T> where T: Clone + UnsafeOrd {}

impl<T, R> SegmentedBy<T, R> for BTreeSet1<T>
where
    T: UnsafeOrd,
    R: IntoRangeBounds<T>,
{
    type Range = Option<ItemRange<T>>;
    type Error = RangeError<Bound<T>>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self, Self::Range>, Self::Error> {
        range::ordered_range_bounds(range)
            .map_err(|range| {
                let (start, end) = range.into_bounds();
                UnorderedError(start, end).into()
            })
            .and_then(|range| {
                if range.contains(self.first()) && range.contains(self.last()) {
                    let (start, end) = range.into_bounds();
                    Err(OutOfBoundsError::Range(start, end).into())
                }
                else {
                    let (start, end) = range.into_bounds();
                    Ok(Segment::unchecked(
                        &mut self.items,
                        Some(ItemRange::unchecked(start, end)),
                    ))
                }
            })
    }
}

impl<T> SegmentedOver for BTreeSet1<T> {
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

impl<T> Tail for BTreeSet1<T> {
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(&mut self.items, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(&mut self.items, TrimRange::RTAIL1)
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

pub type Segment<'a, K, R> = segment::Segment<'a, K, BTreeSet<ItemFor<K>>, R>;

impl<K, T> Segment<'_, K, Option<ItemRange<ItemFor<K>>>>
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
        if let Some(range) = self.range.as_ref() {
            self.items.retain(range.retain_in_range(f))
        }
    }

    pub fn insert_in_range(&mut self, item: T) -> Result<bool, T> {
        if self.range.contains(&item) {
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
        if let Some(range) = self.range.as_ref() {
            // To append within the range of the segment, `other` is split into `low`, `middle`,
            // and `high`. The `middle` set contains any and all items in range, and so it extends
            // the segment. `low` and `high` are out of bounds of the range, and so these items are
            // not inserted into the segment and must remain in `other`.
            //
            // Note that `low` is just an alias for `other` here, and so it is an exclusive
            // reference to the input `BTreeSet` (unlike `middle` and `high`).
            let low = other;
            let mut middle = match range.start_bound() {
                Bound::Excluded(start) => {
                    let mut middle = low.split_off(start);
                    low.extend(middle.take(start));
                    middle
                },
                Bound::Included(start) => low.split_off(start),
                Bound::Unbounded => {
                    if let Some(first) = low.first().cloned() {
                        // The segment has no lower bound, so all of `low` is split off into
                        // `middle` (leaving `low` empty).
                        low.split_off(&first)
                    }
                    else {
                        // If `other` is empty (and so `low.first()` is `None`), then the middle
                        // items are also empty.
                        BTreeSet::new()
                    }
                },
            };
            let high = match range.end_bound() {
                Bound::Excluded(end) => middle.split_off(end),
                Bound::Included(end) => {
                    let mut high = middle.split_off(end);
                    middle.extend(high.take(end));
                    high
                },
                Bound::Unbounded => BTreeSet::new(),
            };
            self.items.extend(middle);
            low.extend(high);
        }
    }

    pub fn clear(&mut self) {
        self.retain(|_| false);
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.contains(key).then(|| self.items.get(key)).flatten()
    }

    pub fn first(&self) -> Option<&T> {
        self.range
            .as_ref()
            .and_then(|range| match range.start_bound() {
                Bound::Excluded(start) => self.items.range(start..).nth(1),
                Bound::Included(start) => self.items.range(start..).next(),
                Bound::Unbounded => self.items.first(),
            })
    }

    pub fn last(&self) -> Option<&T> {
        self.range
            .as_ref()
            .and_then(|range| match range.end_bound() {
                Bound::Excluded(end) => self.items.range(..end).next_back(),
                Bound::Included(end) => self.items.range(..=end).next_back(),
                Bound::Unbounded => self.items.last(),
            })
    }

    pub fn len(&self) -> usize {
        self.range.as_ref().map_or(0, |range| {
            self.items
                .range((range.start_bound(), range.end_bound()))
                .count()
        })
    }

    pub fn iter(&self) -> impl '_ + Clone + Iterator<Item = &'_ T> {
        self.range
            .as_ref()
            .map(|range| self.items.range((range.start_bound(), range.end_bound())))
            .into_iter()
            .flatten()
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.range.contains(key) && self.items.contains(key)
    }
}

impl<T> Segment<'_, BTreeSet<T>, Option<ItemRange<T>>>
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

impl<T> Segment<'_, BTreeSet1<T>, Option<ItemRange<T>>>
where
    T: Ord,
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

impl<K, T> Tail for Segment<'_, K, Option<ItemRange<T>>>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    // A `T: UnsafeOrd` bound is not needed here, because segments over an `ItemRange` can only be
    // constructed for a `BTreeSet1` via `SegmentedBy`, which has that bound. This means that there
    // is no need to separate `Tail` implementations for `BTreeSet` and `BTreeSet1`.
    T: Clone + Ord,
{
    type Range = Option<ItemRange<T>>;

    fn tail(&mut self) -> Segment<'_, K, Self::Range> {
        if let Some(range) = self.range.clone() {
            let (start, end) = range.into_bounds();
            let start = match start {
                Bound::Excluded(start) => self.items.range(start..).nth(2),
                Bound::Included(start) => self.items.range(start..).nth(1),
                Bound::Unbounded => self.items.iter().nth(1),
            };
            let range = start
                .cloned()
                .and_then(|start| range::ordered_range_bounds((Bound::Included(start), end)).ok())
                .map(|(start, end)| ItemRange::unchecked(start, end));
            Segment::unchecked(self.items, range)
        }
        else {
            Segment::unchecked(self.items, None)
        }
    }

    fn rtail(&mut self) -> Segment<'_, K, Self::Range> {
        if let Some(range) = self.range.clone() {
            let (start, end) = range.into_bounds();
            let end = match end {
                Bound::Excluded(end) => self.items.range(..end).rev().nth(1),
                Bound::Included(end) => self.items.range(..=end).rev().nth(1),
                Bound::Unbounded => self.items.iter().rev().nth(1),
            };
            let range = end
                .cloned()
                .and_then(|end| range::ordered_range_bounds((start, Bound::Included(end))).ok())
                .map(|(start, end)| ItemRange::unchecked(start, end));
            Segment::unchecked(self.items, range)
        }
        else {
            Segment::unchecked(self.items, None)
        }
    }
}

impl<'a, K, T> Segment<'a, K, TrimRange>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
    T: Ord,
{
    pub fn by_item(self) -> Segment<'a, K, Option<ItemRange<T>>>
    where
        T: Clone,
    {
        let Segment { items, range } = self;
        let range = items.resolve_trim_range(range);
        Segment::unchecked(items, range)
    }

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
            let is_in_range = index >= tail && index < rtail;
            index = index.checked_add(1).expect("overflow in item index");
            (!is_in_range) || f(item)
        })
    }

    pub fn insert_in_range(&mut self, item: T) -> Result<bool, T>
    where
        T: Clone,
    {
        let range: Option<ItemRange<_>> = self.items.resolve_trim_range(self.range);
        Segment::<K, _>::unchecked(self.items, range).insert_in_range(item)
    }

    pub fn append_in_range(&mut self, other: &mut BTreeSet<T>)
    where
        T: Clone,
    {
        let range: Option<ItemRange<_>> = self.items.resolve_trim_range(self.range);
        Segment::<K, _>::unchecked(self.items, range).append_in_range(other)
    }

    pub fn clear(&mut self) {
        self.retain(|_| false);
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let TrimRange { tail, rtail } = self.range;
        let is_key = |item: &_| Borrow::<Q>::borrow(item) == key;
        self.items.get(key).take_if(|_| {
            (!self.items.iter().take(tail).any(is_key))
                && (!self.items.iter().rev().take(rtail).any(is_key))
        })
    }

    pub fn first(&self) -> Option<&T> {
        self.items.iter().nth(self.range.tail)
    }

    pub fn last(&self) -> Option<&T> {
        self.items.iter().rev().nth(self.range.rtail)
    }

    pub fn len(&self) -> usize {
        self.untrimmed_item_count(self.items.len())
    }

    pub fn iter(&self) -> Take<Skip<btree_set::Iter<'_, T>>> {
        self.items.iter().skip(self.range.tail).take(self.len())
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
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
    T: Ord,
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

impl<K, T> Tail for Segment<'_, K, TrimRange>
where
    K: ClosedBTreeSet<Item = T> + SegmentedOver<Target = BTreeSet<T>>,
{
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, K, Self::Range> {
        self.advance_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, K, Self::Range> {
        self.advance_rtail_range()
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
    use alloc::vec::Vec;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::btree_set1::BTreeSet1;
    use crate::btree_set1::harness::{self, terminals1, xs1};
    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::segment::range::IntoRangeBounds;
    use crate::segment::{Segmentation, Tail};
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};

    // SAFETY: The `FnMut`s constructed in cases (the parameter `f`) must not stash or otherwise
    //         allow access to the parameter beyond the scope of their bodies. (This is difficult
    //         to achieve in this context.)
    #[rstest]
    #[case::ignore_and_retain(|_| true, (None, BTreeSet1::from([0, 1, 2, 3, 4])))]
    #[case::ignore_and_do_not_retain(|_| false, (Some(0), BTreeSet1::from([0])))]
    #[case::compare_and_retain_none(
        |x: *const _| unsafe {
            *x > 4
        },
        (Some(0), BTreeSet1::from([0])),
    )]
    #[case::compare_and_retain_some(
        |x: *const _| unsafe {
            *x < 3
        },
        (None, BTreeSet1::from([0, 1, 2])),
    )]
    fn retain_until_only_from_btree_set1_then_output_and_btree_set1_eq<F>(
        mut xs1: BTreeSet1<u8>,
        #[case] mut f: F,
        #[case] expected: (Option<u8>, BTreeSet1<u8>),
    ) where
        F: FnMut(*const u8) -> bool,
    {
        // TODO: The type parameter `F` must be a `FnMut` over `*const u8` instead of `&u8` here,
        //       because `rstest` constructs the case in a way that the `&u8` has a lifetime that
        //       is too specific and too long (it would borrow the item beyond
        //       `retain_until_only`). Is there a way to prevent this without introducing `*const
        //       u8` and unsafe code in cases for `f`? If so, do that instead!
        let x = xs1.retain_until_only(|x| f(x as *const u8)).copied();
        assert_eq!((x, xs1), expected);
    }

    #[rstest]
    #[case::empty_at_front(0..0, &[])]
    #[case::empty_at_back(4..4, &[])]
    #[case::one_at_front(0..1, &[0])]
    #[case::one_at_back(4.., &[4])]
    #[case::middle(1..4, &[1, 2, 3])]
    #[case::tail(1.., &[1, 2, 3, 4])]
    #[case::rtail(..4, &[0, 1, 2, 3])]
    fn collect_segment_iter_of_btree_set1_into_vec_then_eq<S>(
        mut xs1: BTreeSet1<u8>,
        #[case] segment: S,
        #[case] expected: &[u8],
    ) where
        S: IntoRangeBounds<u8>,
    {
        let segment = xs1.segment(segment).unwrap();
        let xs: Vec<_> = segment.iter().copied().collect();
        assert_eq!(xs.as_slice(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0), &[])]
    #[case::one_tail(harness::xs1(1), &[1])]
    #[case::many_tail(harness::xs1(4), &[1, 2, 3, 4])]
    fn collect_tail_iter_of_btree_set1_into_vec_then_eq(
        #[case] mut xs1: BTreeSet1<u8>,
        #[case] expected: &[u8],
    ) {
        let segment = xs1.tail();
        let xs: Vec<_> = segment.iter().copied().collect();
        assert_eq!(xs.as_slice(), expected);
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0), &[])]
    #[case::one_rtail(harness::xs1(1), &[0])]
    #[case::many_rtail(harness::xs1(4), &[0, 1, 2, 3])]
    fn collect_rtail_iter_of_btree_set1_into_vec_then_eq(
        #[case] mut xs1: BTreeSet1<u8>,
        #[case] expected: &[u8],
    ) {
        let segment = xs1.rtail();
        let xs: Vec<_> = segment.iter().copied().collect();
        assert_eq!(xs.as_slice(), expected);
    }

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
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_intersected_tail_rtail_of_btree_set1_then_btree_set1_eq_self(
        #[case] mut xs1: BTreeSet1<u8>,
    ) {
        let expected = xs1.clone();
        let mut segment = xs1.tail();
        let mut segment = segment.tail();
        let mut segment = segment.tail();
        let mut segment = segment.rtail();
        let mut segment = segment.rtail();
        let mut segment = segment.rtail();
        segment.clear();
        assert_eq!(xs1, expected);
    }

    #[rstest]
    #[case::absent_in_range(4.., 4, Ok(true))]
    #[case::absent_in_range(..=4, 4, Ok(true))]
    #[case::present_in_range(4.., 9, Ok(false))]
    #[case::out_of_range_lower_bound(4.., 0, Err(0))]
    #[case::out_of_range_lower_bound(4.., 1, Err(1))]
    #[case::out_of_range_upper_bound(..5, 5, Err(5))]
    #[case::out_of_range_upper_bound(3..=5, 6, Err(6))]
    #[case::out_of_range_upper_bound(..5, 6, Err(6))]
    fn insert_into_btree_set1_segment_then_output_eq<R>(
        #[from(terminals1)] mut xs1: BTreeSet1<u8>,
        #[case] range: R,
        #[case] item: u8,
        #[case] expected: Result<bool, u8>,
    ) where
        R: IntoRangeBounds<u8>,
    {
        let mut segment = xs1.segment(range).unwrap();
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

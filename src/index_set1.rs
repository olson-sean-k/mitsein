//! A non-empty [`IndexSet`].
//!
//! [`IndexSet`]: indexmap::set

#![cfg(feature = "indexmap")]
#![cfg_attr(docsrs, doc(cfg(feature = "indexmap")))]

use alloc::boxed::Box;
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::hash::{BuildHasher, Hash};
use core::iter::{Skip, Take};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, Deref, RangeBounds, Sub};
use indexmap::set::{self as index_set, IndexSet, Slice};
use indexmap::{Equivalent, TryReserveError};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "std")]
use std::hash::RandomState;
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

#[cfg(feature = "std")]
use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{self, NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, IndexRange, Intersect, Project, RangeError};
use crate::segment::{self, Segmentation, SegmentedBy, SegmentedOver, Tail};
use crate::take;
use crate::{EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedIndexSet>::Item;
type StateFor<K> = <K as ClosedIndexSet>::State;

pub trait ClosedIndexSet {
    type Item;
    type State;

    fn as_index_set(&self) -> &IndexSet<Self::Item, Self::State>;
}

impl<T, S> ClosedIndexSet for IndexSet<T, S> {
    type Item = T;
    type State = S;

    fn as_index_set(&self) -> &IndexSet<Self::Item, Self::State> {
        self
    }
}

impl<T, S> Extend1<T> for IndexSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend_non_empty<I>(mut self, items: I) -> IndexSet1<T, S>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { IndexSet1::from_index_set_unchecked(self) }
    }
}

unsafe impl<T, S> MaybeEmpty for IndexSet<T, S> {
    fn cardinality(&self) -> Option<crate::Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(crate::Cardinality::One(())),
            _ => Some(crate::Cardinality::Many(())),
        }
    }
}

impl<T, S> Segmentation for IndexSet<T, S> {}

impl<T, S, R> SegmentedBy<usize, R> for IndexSet<T, S>
where
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self>, Self::Error> {
        let n = self.len();
        Segment::intersected(self, n, range)
    }
}

impl<T, S> SegmentedOver for IndexSet<T, S> {
    type Kind = Self;
    type Target = Self;
}

impl<T, S> Tail for IndexSet<T, S> {
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self> {
        let n = self.len();
        Segment::from_tail_range(self, n)
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        let n = self.len();
        Segment::from_rtail_range(self, n)
    }
}

type TakeIfMany<'a, T, S, U, N = ()> = take::TakeIfMany<'a, IndexSet<T, S>, U, N>;

pub type PopIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, StateFor<K>, ItemFor<K>>;

pub type DropRemoveIfMany<'a, 'q, K, Q> = TakeIfMany<'a, ItemFor<K>, StateFor<K>, bool, &'q Q>;

pub type TakeRemoveIfMany<'a, K, N = usize> =
    TakeIfMany<'a, ItemFor<K>, StateFor<K>, Option<ItemFor<K>>, N>;

pub type TakeRemoveFullIfMany<'a, 'q, K, Q> =
    TakeIfMany<'a, ItemFor<K>, StateFor<K>, Option<(usize, ItemFor<K>)>, &'q Q>;

impl<'a, T, S, U, N> TakeIfMany<'a, T, S, U, N> {
    pub fn or_get_only(self) -> Result<U, &'a T> {
        self.take_or_else(|items, _| items.first())
    }
}

impl<'a, T, S> TakeIfMany<'a, T, S, Option<T>, usize>
where
    S: BuildHasher,
{
    pub fn or_get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, index| items.get_index(index))
    }
}

impl<'a, T, S, Q> TakeIfMany<'a, T, S, bool, &'_ Q>
where
    T: Borrow<Q>,
    S: BuildHasher,
    Q: Equivalent<T> + Hash + ?Sized,
{
    pub fn or_get(self) -> Result<bool, Option<&'a T>> {
        self.take_or_else(|items, query| items.get(query))
    }
}

impl<'a, T, S, Q> TakeIfMany<'a, T, S, Option<T>, &'_ Q>
where
    T: Borrow<Q>,
    S: BuildHasher,
    Q: Equivalent<T> + Hash + ?Sized,
{
    pub fn or_get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, query| items.get(query))
    }
}

pub type Slice1<T> = NonEmpty<Slice<T>>;

// TODO: At time of writing, `const` functions are not supported in traits, so
//       `FromMaybeEmpty::from_maybe_empty_unchecked` cannot be used to construct a `Slice1` yet.
//       Use that function instead of `mem::transmute` when possible.
impl<T> Slice1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice [`Slice::new()`][`Slice::new`].
    pub const unsafe fn from_slice_unchecked(items: &Slice<T>) -> &Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `Slice<T>` and
        //         `Slice1<T>` are the same.
        mem::transmute::<&'_ Slice<T>, &'_ Slice1<T>>(items)
    }

    pub fn split_first(&self) -> (&T, &Slice<T>) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.split_first().unwrap_maybe_unchecked() }
    }

    pub fn split_last(&self) -> (&T, &Slice<T>) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.split_last().unwrap_maybe_unchecked() }
    }

    pub fn first(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<index_set::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.len()) }
    }
}

impl<T> Deref for Slice1<T> {
    type Target = Slice<T>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

// TODO: Remove unnecessary bounds on `Borrow`. The `Equivalent` trait encapsulates this.

#[cfg(feature = "std")]
pub type IndexSet1<T, S = RandomState> = NonEmpty<IndexSet<T, S>>;

#[cfg(not(feature = "std"))]
pub type IndexSet1<T, S> = NonEmpty<IndexSet<T, S>>;

impl<T, S> IndexSet1<T, S> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`IndexSet::new()`][`IndexSet::new`].
    ///
    /// [`IndexSet::new`]: indexmap::set::IndexSet::new
    pub unsafe fn from_index_set_unchecked(items: IndexSet<T, S>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn into_index_set(self) -> IndexSet<T, S> {
        self.items
    }

    pub fn into_boxed_slice1(self) -> Box<Slice1<T>> {
        let items = Box::into_raw(self.items.into_boxed_slice());
        // SAFETY: This cast is safe, because `Slice<T>` and `Slice1<T>` have the same
        //         representation (`Slice1<T>` is `repr(transparent)`). Moreover, the allocator
        //         only requires that the memory location and layout are the same when
        //         deallocating, so dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut Slice1<T>) }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional)
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.items.try_reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.items.reserve_exact(additional)
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.items.try_reserve_exact(additional)
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.items.sort()
    }

    pub fn sort_by<F>(&mut self, f: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.items.sort_by(f)
    }

    pub fn sorted_by<F>(self, f: F) -> Iterator1<index_set::IntoIter<T>>
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.sorted_by(f)) }
    }

    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.items.sort_unstable()
    }

    pub fn sort_unstable_by<F>(&mut self, f: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.items.sort_unstable_by(f)
    }

    pub fn sorted_unstable_by<F>(self, f: F) -> Iterator1<index_set::IntoIter<T>>
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.sorted_unstable_by(f)) }
    }

    pub fn sort_by_cached_key<K, F>(&mut self, f: F)
    where
        K: Ord,
        F: FnMut(&T) -> K,
    {
        self.items.sort_by_cached_key(f)
    }

    pub fn reverse(&mut self) {
        self.items.reverse()
    }

    pub fn split_off_tail(&mut self) -> IndexSet<T, S>
    where
        S: Clone,
    {
        self.items.split_off(1)
    }

    pub fn move_index(&mut self, from: usize, to: usize) {
        self.items.move_index(from, to)
    }

    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.items.swap_indices(a, b)
    }

    pub fn pop_if_many(&mut self) -> PopIfMany<'_, Self>
    where
        T: Eq + Hash,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn shift_remove_index_if_many(&mut self, index: usize) -> TakeRemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| {
            items.items.shift_remove_index(index)
        })
    }

    pub fn swap_remove_index_if_many(&mut self, index: usize) -> TakeRemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| {
            items.items.swap_remove_index(index)
        })
    }

    pub fn get_index(&self, index: usize) -> Option<&T> {
        self.items.get_index(index)
    }

    pub fn get_range<R>(&self, range: R) -> Option<&Slice<T>>
    where
        R: RangeBounds<usize>,
    {
        self.items.get_range(range)
    }

    pub fn binary_search(&self, query: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.items.binary_search(query)
    }

    pub fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        self.items.binary_search_by(f)
    }

    pub fn binary_search_by_key<'a, K, F>(&'a self, key: &K, f: F) -> Result<usize, usize>
    where
        K: Ord,
        F: FnMut(&'a T) -> K,
    {
        self.items.binary_search_by_key(key, f)
    }

    pub fn partition_point<F>(&self, f: F) -> usize
    where
        F: FnMut(&T) -> bool,
    {
        self.items.partition_point(f)
    }

    pub fn first(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn iter1(&self) -> Iterator1<index_set::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn hasher(&self) -> &S {
        self.items.hasher()
    }

    pub const fn as_index_set(&self) -> &IndexSet<T, S> {
        &self.items
    }

    /// # Safety
    ///
    /// The [`IndexSet`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::index_set1::IndexSet1;
    ///
    /// let mut xs = IndexSet1::from([0i32, 1, 2, 3]);
    /// // This block is unsound. The `&mut IndexSet` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_index_set().clear();
    /// }
    /// let x = xs.first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_index_set(&mut self) -> &mut IndexSet<T, S> {
        &mut self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        // SAFETY:
        unsafe { Slice1::from_slice_unchecked(self.items.as_slice()) }
    }
}

impl<T, S> IndexSet1<T, S>
where
    S: BuildHasher,
{
    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        self.items.get(query)
    }

    pub fn shift_remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> DropRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.shift_remove(query))
    }

    pub fn swap_remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> DropRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.swap_remove(query))
    }

    pub fn shift_remove_full_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> TakeRemoveFullIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| {
            items.items.shift_remove_full(query)
        })
    }

    pub fn swap_remove_full_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> TakeRemoveFullIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| {
            items.items.swap_remove_full(query)
        })
    }

    pub fn shift_take_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> TakeRemoveIfMany<'a, Self, &'q Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.shift_take(query))
    }

    pub fn swap_take_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> TakeRemoveIfMany<'a, Self, &'q Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.swap_take(query))
    }

    pub fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        self.items.contains(item)
    }
}

impl<T, S> IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    pub fn from_one(item: T) -> Self
    where
        S: Default,
    {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_hasher(item: T, hasher: S) -> Self {
        IndexSet1::from_iter1_with_hasher([item], hasher)
    }

    pub fn from_iter1_with_hasher<U>(items: U, hasher: S) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = IndexSet::with_hasher(hasher);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { IndexSet1::from_index_set_unchecked(items) }
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        S: Default,
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        S: Default,
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn append<SR>(&mut self, items: &mut IndexSet<T, SR>) {
        self.items.append(items)
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.items.insert(item)
    }

    pub fn insert_full(&mut self, item: T) -> (usize, bool) {
        self.items.insert_full(item)
    }

    pub fn insert_sorted(&mut self, item: T) -> (usize, bool)
    where
        T: Ord,
    {
        self.items.insert_sorted(item)
    }

    pub fn replace(&mut self, item: T) -> Option<T> {
        self.items.replace(item)
    }

    pub fn replace_full(&mut self, item: T) -> (usize, Option<T>) {
        self.items.replace_full(item)
    }

    pub fn difference<'a, R, SR>(&'a self, other: &'a R) -> index_set::Difference<'a, T, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.difference(other.as_index_set())
    }

    pub fn symmetric_difference<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> index_set::SymmetricDifference<'a, T, S, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.symmetric_difference(other.as_index_set())
    }

    pub fn intersection<'a, R, SR>(&'a self, other: &'a R) -> index_set::Intersection<'a, T, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.intersection(other.as_index_set())
    }

    pub fn union<'a, R, SR>(&'a self, other: &'a R) -> Iterator1<index_set::Union<'a, T, S>>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: 'a + BuildHasher,
    {
        // SAFETY: `self` must be non-empty and `IndexSet::union` cannot reduce the cardinality of
        //         its inputs.
        unsafe { Iterator1::from_iter_unchecked(self.items.union(other.as_index_set())) }
    }

    pub fn is_disjoint<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.is_disjoint(other.as_index_set())
    }

    pub fn is_subset<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.is_subset(other.as_index_set())
    }

    pub fn is_superset<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher,
    {
        self.items.is_superset(other.as_index_set())
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IndexSet1<T, S> {
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        T: Sync,
        S: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IndexSet1<T, S>
where
    T: Eq + Hash + Sync,
    S: BuildHasher + Sync,
{
    pub fn par_difference<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> index_set::rayon::ParDifference<'a, T, S, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_difference(other.as_index_set())
    }

    pub fn par_symmetric_difference<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> index_set::rayon::ParSymmetricDifference<'a, T, S, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_symmetric_difference(other.as_index_set())
    }

    pub fn par_intersection<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> index_set::rayon::ParIntersection<'a, T, S, SR>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_intersection(other.as_index_set())
    }

    pub fn par_union<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> ParallelIterator1<index_set::rayon::ParUnion<'a, T, S, SR>>
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: 'a + BuildHasher + Sync,
    {
        // SAFETY: `self` must be non-empty and `IndexSet::par_union` cannot reduce the cardinality
        //         of its inputs.
        unsafe {
            ParallelIterator1::from_par_iter_unchecked(self.items.par_union(other.as_index_set()))
        }
    }

    pub fn par_eq<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_eq(other.as_index_set())
    }

    pub fn par_is_disjoint<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_is_disjoint(other.as_index_set())
    }

    pub fn par_is_subset<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_is_subset(other.as_index_set())
    }

    pub fn par_is_superset<R, SR>(&self, other: &R) -> bool
    where
        R: ClosedIndexSet<Item = T, State = SR>,
        SR: BuildHasher + Sync,
    {
        self.items.par_is_superset(other.as_index_set())
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IndexSet1<T, S>
where
    T: Eq + Hash + Send + Sync,
    S: BuildHasher + Sync,
{
    pub fn par_sort(&mut self)
    where
        T: Ord,
    {
        self.items.par_sort()
    }

    pub fn par_sort_by<F>(&mut self, f: F)
    where
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        self.items.par_sort_by(f)
    }

    pub fn par_sorted_by<F>(self, f: F) -> ParallelIterator1<index_set::rayon::IntoParIter<T>>
    where
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        // SAFETY: `self` must be non-empty and `IndexSet::par_sorted_by` cannot reduce the
        //         cardinality of its inputs.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_sorted_by(f)) }
    }

    pub fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.items.par_sort_unstable()
    }

    pub fn par_sort_unstable_by<F>(&mut self, f: F)
    where
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        self.items.par_sort_unstable_by(f)
    }

    pub fn par_sorted_unstable_by<F>(
        self,
        f: F,
    ) -> ParallelIterator1<index_set::rayon::IntoParIter<T>>
    where
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        // SAFETY: `self` must be non-empty and `IndexSet::par_sorted_unstable_by` cannot reduce
        //         the cardinality of its inputs.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_sorted_unstable_by(f)) }
    }

    pub fn par_sort_by_cached_key<K, F>(&mut self, f: F)
    where
        K: Ord + Send,
        F: Fn(&T) -> K + Sync,
    {
        self.items.par_sort_by_cached_key(f)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<T> IndexSet1<T, RandomState>
where
    T: Eq + Hash,
{
    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        IndexSet1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = IndexSet::with_capacity(capacity);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { IndexSet1::from_index_set_unchecked(items) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T> Arbitrary<'a> for IndexSet1<T>
where
    T: Arbitrary<'a> + Eq + Hash,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(T::arbitrary(unstructured), unstructured.arbitrary_iter()?).collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (T::size_hint(depth).0, None)
    }
}

impl<R, T, S> BitAnd<&'_ R> for &'_ IndexSet1<T, S>
where
    R: ClosedIndexSet<Item = T>,
    R::State: BuildHasher,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() & rhs.as_index_set()
    }
}

impl<T, S, S1> BitAnd<&'_ IndexSet1<T, S1>> for &'_ IndexSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
    S1: BuildHasher,
{
    type Output = IndexSet<T, S>;

    fn bitand(self, rhs: &'_ IndexSet1<T, S1>) -> Self::Output {
        self & rhs.as_index_set()
    }
}

impl<R, T, S> BitOr<&'_ R> for &'_ IndexSet1<T, S>
where
    R: ClosedIndexSet<Item = T>,
    R::State: BuildHasher,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet1<T, S>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        // SAFETY: `self` must be non-empty and `IndexSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { IndexSet1::from_index_set_unchecked(self.as_index_set() | rhs.as_index_set()) }
    }
}

impl<T, S, S1> BitOr<&'_ IndexSet1<T, S1>> for &'_ IndexSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
    S1: BuildHasher,
{
    type Output = IndexSet1<T, S>;

    fn bitor(self, rhs: &'_ IndexSet1<T, S1>) -> Self::Output {
        // SAFETY: `rhs` must be non-empty and `IndexSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { IndexSet1::from_index_set_unchecked(self | rhs.as_index_set()) }
    }
}

impl<R, T, S> BitXor<&'_ R> for &'_ IndexSet1<T, S>
where
    R: ClosedIndexSet<Item = T>,
    R::State: BuildHasher,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() ^ rhs.as_index_set()
    }
}

impl<T, S, S1> BitXor<&'_ IndexSet1<T, S1>> for &'_ IndexSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
    S1: BuildHasher,
{
    type Output = IndexSet<T, S>;

    fn bitxor(self, rhs: &'_ IndexSet1<T, S1>) -> Self::Output {
        self ^ rhs.as_index_set()
    }
}

impl<T, S> ClosedIndexSet for IndexSet1<T, S> {
    type Item = T;
    type State = S;

    fn as_index_set(&self) -> &IndexSet<Self::Item, Self::State> {
        self.as_ref()
    }
}

impl<T, S> Debug for IndexSet1<T, S>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T, S> Extend<T> for IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<T, const N: usize> From<[T; N]> for IndexSet1<T, RandomState>
where
    [T; N]: Array1,
    T: Eq + Hash,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { IndexSet1::from_index_set_unchecked(IndexSet::from(items)) }
    }
}

impl<T, S> From<IndexSet1<T, S>> for IndexSet<T, S> {
    fn from(items: IndexSet1<T, S>) -> Self {
        items.items
    }
}

impl<T, S> FromIterator1<T> for IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { IndexSet1::from_index_set_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> FromParallelIterator1<T> for IndexSet1<T, S>
where
    T: Eq + Hash + Send,
    S: BuildHasher + Default + Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe {
            IndexSet1::from_index_set_unchecked(items.into_par_iter1().into_par_iter().collect())
        }
    }
}

impl<T, S> IntoIterator for IndexSet1<T, S> {
    type Item = T;
    type IntoIter = index_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T, S> IntoIterator for &'a IndexSet1<T, S> {
    type Item = &'a T;
    type IntoIter = index_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<T, S> IntoIterator1 for IndexSet1<T, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, S> IntoIterator1 for &'_ IndexSet1<T, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator for IndexSet1<T, S>
where
    T: Send,
    S: Send,
{
    type Item = T;
    type Iter = index_set::rayon::IntoParIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T, S> IntoParallelIterator for &'a IndexSet1<T, S>
where
    T: Sync,
    S: Sync,
{
    type Item = &'a T;
    type Iter = <&'a IndexSet<T, S> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator1 for IndexSet1<T, S>
where
    T: Send,
    S: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator1 for &'_ IndexSet1<T, S>
where
    T: Sync,
    S: Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<T, S> JsonSchema for IndexSet1<T, S>
where
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        IndexSet::<T, S>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<IndexSet<T, S>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        IndexSet::<T, S>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        IndexSet::<T, S>::schema_id()
    }
}

impl<T, S> Segmentation for IndexSet1<T, S> {}

impl<T, S, R> SegmentedBy<usize, R> for IndexSet1<T, S>
where
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self>, Self::Error> {
        let n = self.items.len();
        Segment::intersected_strict_subset(&mut self.items, n, range)
    }
}

impl<T, S> SegmentedOver for IndexSet1<T, S> {
    type Kind = Self;
    type Target = IndexSet<T, S>;
}

impl<R, T, S> Sub<&'_ R> for &'_ IndexSet1<T, S>
where
    R: ClosedIndexSet<Item = T>,
    R::State: BuildHasher,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() - rhs.as_index_set()
    }
}

impl<T, S, S1> Sub<&'_ IndexSet1<T, S1>> for &'_ IndexSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
    S1: BuildHasher,
{
    type Output = IndexSet<T, S>;

    fn sub(self, rhs: &'_ IndexSet1<T, S1>) -> Self::Output {
        self - rhs.as_index_set()
    }
}

impl<T, S> Tail for IndexSet1<T, S> {
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self> {
        self.items.tail().rekind()
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        self.items.rtail().rekind()
    }
}

impl<T, S> TryFrom<IndexSet<T, S>> for IndexSet1<T, S> {
    type Error = EmptyError<IndexSet<T, S>>;

    fn try_from(items: IndexSet<T, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

pub type Segment<'a, K> = segment::Segment<'a, K, IndexSet<ItemFor<K>, StateFor<K>>, IndexRange>;

// TODO: It should be possible to safely implement `swap_drain` for segments over `IndexSet1`. The
//       `IndexSet::drain` iterator immediately culls its indices but then defers to `vec::Drain`
//       for removing buckets. `IndexSet::swap_indices` can be used much like `slice::swap` here.
impl<K, T, S> Segment<'_, K>
where
    K: ClosedIndexSet<Item = T, State = S> + SegmentedOver<Target = IndexSet<T, S>>,
{
    pub fn truncate(&mut self, len: usize) {
        if let Some(range) = self.range.truncate_from_end(len) {
            self.items.drain(range);
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.items.retain(self.range.retain_from_end(f))
    }

    pub fn move_index(&mut self, from: usize, to: usize) {
        let from = self
            .range
            .project(from)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        let to = self
            .range
            .project(to)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        self.items.move_index(from, to)
    }

    pub fn swap_indices(&mut self, a: usize, b: usize) {
        let a = self
            .range
            .project(a)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        let b = self
            .range
            .project(b)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        self.items.swap_indices(a, b)
    }

    pub fn shift_remove_index(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(index).ok()?;
        self.items
            .shift_remove_index(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_index(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(index).ok()?;
        self.items
            .swap_remove_index(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn clear(&mut self) {
        self.items.drain(self.range.get_and_clear_from_end());
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn iter(&self) -> Take<Skip<index_set::Iter<'_, T>>> {
        self.items.iter().skip(self.range.start()).take(self.len())
    }
}

impl<K, T, S> Segment<'_, K>
where
    K: ClosedIndexSet<Item = T, State = S> + SegmentedOver<Target = IndexSet<T, S>>,
    T: Eq + Hash,
    S: BuildHasher,
{
    pub fn shift_insert(&mut self, index: usize, item: T) -> bool {
        let index = self
            .range
            .project(index)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        if self.items.shift_insert(index, item) {
            self.range.put_from_end(1);
            true
        }
        else {
            false
        }
    }
}

impl<K, T, S> Segmentation for Segment<'_, K> where
    K: ClosedIndexSet<Item = T, State = S> + SegmentedOver<Target = IndexSet<T, S>>
{
}

impl<K, T, S, R> SegmentedBy<usize, R> for Segment<'_, K>
where
    IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
    K: ClosedIndexSet<Item = T, State = S> + SegmentedOver<Target = IndexSet<T, S>>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, K>, Self::Error> {
        let range = self.range.intersect(self.range.project(range)?)?;
        Ok(Segment::unchecked(self.items, range))
    }
}

impl<K, T, S> Tail for Segment<'_, K>
where
    K: ClosedIndexSet<Item = T, State = S> + SegmentedOver<Target = IndexSet<T, S>>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, K> {
        self.project_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, K> {
        let n = self.len();
        self.project_rtail_range(n)
    }
}

#[cfg(all(test, feature = "std"))]
pub mod harness {
    use rstest::fixture;

    use crate::index_set1::IndexSet1;
    use crate::iter1::{self, FromIterator1};

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> IndexSet1<u8> {
        IndexSet1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(all(test, feature = "schemars", feature = "std"))]
mod tests {
    use rstest::rstest;

    use crate::index_set1::IndexSet1;
    use crate::schemars;

    #[rstest]
    fn index_set1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<IndexSet1<u8>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
    }
}

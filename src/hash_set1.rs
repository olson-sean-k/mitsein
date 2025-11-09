//! A non-empty [`HashSet`][`hash_set`].

#![cfg(feature = "std")]
#![cfg_attr(docsrs, doc(cfg(feature = "std")))]

#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::hash::{BuildHasher, Hash};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, Sub};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::hash_set::{self, HashSet};
use std::hash::RandomState;
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::hash::UnsafeHash;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::take;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedHashSet>::Item;
type StateFor<K> = <K as ClosedHashSet>::State;

pub trait ClosedHashSet {
    type Item;
    type State;

    fn as_hash_set(&self) -> &HashSet<Self::Item, Self::State>;
}

impl<T, S> ClosedHashSet for HashSet<T, S> {
    type Item = T;
    type State = S;

    fn as_hash_set(&self) -> &HashSet<Self::Item, Self::State> {
        self
    }
}

impl<T, S> Extend1<T> for HashSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend_non_empty<I>(mut self, items: I) -> HashSet1<T, S>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { HashSet1::from_hash_set_unchecked(self) }
    }
}

unsafe impl<T, S> MaybeEmpty for HashSet<T, S> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        // SAFETY: This implementation is critical to memory safety. `HashSet::len` is reliable
        //         here, because it does not break the contract by returning a non-zero value for
        //         an empty set, even if the `Eq` or `Hash` implementations for `T` are
        //         non-compliant. This is why `HashSet1` APIs do not require unsafe trait
        //         implementations for `T`.
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

type TakeIfMany<'a, T, S, U, N = ()> = take::TakeIfMany<'a, HashSet<T, S>, U, N>;

pub type DropRemoveIfMany<'a, 'q, K, Q> = TakeIfMany<'a, ItemFor<K>, StateFor<K>, bool, &'q Q>;

pub type TakeRemoveIfMany<'a, 'q, K, Q> =
    TakeIfMany<'a, ItemFor<K>, StateFor<K>, Option<ItemFor<K>>, &'q Q>;

impl<'a, T, S, U, N> TakeIfMany<'a, T, S, U, N>
where
    T: Eq + Hash,
{
    pub fn or_get_only(self) -> Result<U, &'a T> {
        self.take_or_else(|items, _| items.iter1().first())
    }
}

impl<'a, T, S, Q> TakeIfMany<'a, T, S, bool, &'_ Q>
where
    T: Borrow<Q> + Eq + Hash,
    S: BuildHasher,
    Q: Eq + Hash + ?Sized,
{
    pub fn or_get(self) -> Result<bool, Option<&'a T>> {
        self.take_or_else(|items, query| items.get(query))
    }
}

impl<'a, T, S, Q> TakeIfMany<'a, T, S, Option<T>, &'_ Q>
where
    T: Borrow<Q> + Eq + Hash,
    S: BuildHasher,
    Q: Eq + Hash + ?Sized,
{
    pub fn or_get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, query| items.get(query))
    }
}

pub type HashSet1<T, S = RandomState> = NonEmpty<HashSet<T, S>>;

impl<T, S> HashSet1<T, S> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`HashSet::new()`][`HashSet::new`].
    ///
    /// [`HashSet::new`]: std::collections::hash_set::HashSet::new
    pub unsafe fn from_hash_set_unchecked(items: HashSet<T, S>) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    pub fn try_from_ref(items: &HashSet<T, S>) -> Result<&'_ Self, EmptyError<&'_ HashSet<T, S>>> {
        items.try_into()
    }

    pub fn try_from_mut_ref(
        items: &mut HashSet<T, S>,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut HashSet<T, S>>> {
        items.try_into()
    }

    pub fn into_hash_set(self) -> HashSet<T, S> {
        self.items
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<HashSet<T, S>>>
    where
        T: Eq + Hash,
        F: FnMut(&T) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }

    pub fn iter1(&self) -> Iterator1<hash_set::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn hasher(&self) -> &S {
        self.items.hasher()
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub const fn as_hash_set(&self) -> &HashSet<T, S> {
        &self.items
    }

    /// # Safety
    ///
    /// The [`HashSet`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::hash_set1::HashSet1;
    ///
    /// let mut xs = HashSet1::from([0i32, 1, 2, 3]);
    /// // This block is unsound. The `&mut HashSet` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_hash_set().clear();
    /// }
    /// let x = xs.iter1().first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_hash_set(&mut self) -> &mut HashSet<T, S> {
        &mut self.items
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> HashSet1<T, S>
where
    T: Eq + Hash,
{
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        T: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }
}

impl<T, S> HashSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    pub fn from_one_with_hasher(item: T, hasher: S) -> Self {
        HashSet1::from_iter1_with_hasher([item], hasher)
    }

    pub fn from_one_with_capacity_and_hasher(item: T, capacity: usize, hasher: S) -> Self {
        HashSet1::from_iter1_with_capacity_and_hasher([item], capacity, hasher)
    }

    pub fn from_iter1_with_hasher<U>(items: U, hasher: S) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = HashSet::with_hasher(hasher);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { HashSet1::from_hash_set_unchecked(items) }
    }

    pub fn from_iter1_with_capacity_and_hasher<U>(items: U, capacity: usize, hasher: S) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = HashSet::with_capacity_and_hasher(capacity, hasher);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { HashSet1::from_hash_set_unchecked(items) }
    }

    pub fn retain_until_only<F>(&mut self, mut f: F) -> Option<&'_ T>
    where
        T: Clone,
        F: FnMut(&T) -> bool,
    {
        // Unordered collections cannot support segmentation and hashed collections not only
        // exhibit arbitrary ordering, but that ordering is virtually never deterministic.
        // Moreover, this code cannot trust the `Eq` and `Hash` implementations for `T` to maintain
        // the non-empty invariant. Given that, the first observed item is retained and cloned. The
        // clone is necessary, because there is no other way to reliably reference it while also
        // mutating the underlying `HashSet`. This first item must be tested against the predicate
        // function when more than one item remains after this first `retain`.
        let mut first = None;
        let mut index = 0usize;
        self.items.retain(|item| {
            let is_retained = if index == 0 {
                first = Some(item.clone());
                true
            }
            else {
                f(item)
            };
            index += 1;
            is_retained
        });
        // SAFETY: `self` must be non-empty, so a first item is always observed in `retain`.
        let first = unsafe { first.unwrap_maybe_unchecked() };
        if self.len().get() == 1 {
            if f(&first) {
                None
            }
            else {
                Some(self.iter1().first())
            }
        }
        else {
            if !f(&first) {
                // The first observed item is **not** retained, but there is more than one item in
                // the collection.
                self.remove_if_many(&first);
            }
            None
        }
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.items.insert(item)
    }

    pub fn replace(&mut self, item: T) -> Option<T> {
        self.items.replace(item)
    }

    pub fn remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> DropRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.remove(query))
    }

    pub fn take_if_many<'a, 'q, Q>(&'a mut self, query: &'q Q) -> TakeRemoveIfMany<'a, 'q, Self, Q>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.take(query))
    }

    pub fn except<'a>(&'a mut self, key: &'a T) -> Option<Except<'a, T, S>>
    where
        T: UnsafeHash,
    {
        self.contains(key).then_some(Except {
            items: &mut self.items,
            key,
        })
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.items.get(query)
    }

    pub fn difference<'a, R>(&'a self, other: &'a R) -> hash_set::Difference<'a, T, S>
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.difference(other.as_hash_set())
    }

    pub fn symmetric_difference<'a, R>(
        &'a self,
        other: &'a R,
    ) -> hash_set::SymmetricDifference<'a, T, S>
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.symmetric_difference(other.as_hash_set())
    }

    pub fn intersection<'a, R>(&'a self, other: &'a R) -> hash_set::Intersection<'a, T, S>
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.intersection(other.as_hash_set())
    }

    pub fn union<'a, R>(&'a self, other: &'a R) -> Iterator1<hash_set::Union<'a, T, S>>
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        // SAFETY: `self` must be non-empty and `HashSet::union` cannot reduce the cardinality of
        //         its inputs.
        unsafe { Iterator1::from_iter_unchecked(self.items.union(other.as_hash_set())) }
    }

    pub fn is_disjoint<R>(&self, other: &R) -> bool
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.is_disjoint(other.as_hash_set())
    }

    pub fn is_subset<R>(&self, other: &R) -> bool
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.is_subset(other.as_hash_set())
    }

    pub fn is_superset<R>(&self, other: &R) -> bool
    where
        R: ClosedHashSet<Item = T, State = S>,
    {
        self.items.is_superset(other.as_hash_set())
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.items.contains(key)
    }
}

impl<T, S> HashSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }
}

impl<T> HashSet1<T, RandomState>
where
    T: Eq + Hash,
{
    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        HashSet1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = HashSet::with_capacity(capacity);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { HashSet1::from_hash_set_unchecked(items) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T, S> Arbitrary<'a> for HashSet1<T, S>
where
    T: Arbitrary<'a> + Eq + Hash,
    S: BuildHasher + Default,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(T::arbitrary(unstructured), unstructured.arbitrary_iter()?).collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (T::size_hint(depth).0, None)
    }
}

impl<R, T, S> BitAnd<&'_ R> for &'_ HashSet1<T, S>
where
    R: ClosedHashSet<Item = T, State = S>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_hash_set() & rhs.as_hash_set()
    }
}

impl<T, S> BitAnd<&'_ HashSet1<T, S>> for &'_ HashSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn bitand(self, rhs: &'_ HashSet1<T, S>) -> Self::Output {
        self & rhs.as_hash_set()
    }
}

impl<R, T, S> BitOr<&'_ R> for &'_ HashSet1<T, S>
where
    R: ClosedHashSet<Item = T, State = S>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet1<T, S>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        // SAFETY: `self` must be non-empty and `HashSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { HashSet1::from_hash_set_unchecked(self.as_hash_set() | rhs.as_hash_set()) }
    }
}

impl<T, S> BitOr<&'_ HashSet1<T, S>> for &'_ HashSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet1<T, S>;

    fn bitor(self, rhs: &'_ HashSet1<T, S>) -> Self::Output {
        // SAFETY: `rhs` must be non-empty and `HashSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { HashSet1::from_hash_set_unchecked(self | rhs.as_hash_set()) }
    }
}

impl<R, T, S> BitXor<&'_ R> for &'_ HashSet1<T, S>
where
    R: ClosedHashSet<Item = T, State = S>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_hash_set() ^ rhs.as_hash_set()
    }
}

impl<T, S> BitXor<&'_ HashSet1<T, S>> for &'_ HashSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn bitxor(self, rhs: &'_ HashSet1<T, S>) -> Self::Output {
        self ^ rhs.as_hash_set()
    }
}

impl<T, S> ClosedHashSet for HashSet1<T, S> {
    type Item = T;
    type State = S;

    fn as_hash_set(&self) -> &HashSet<Self::Item, Self::State> {
        self.as_ref()
    }
}

impl<T, S> Debug for HashSet1<T, S>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T, S> Extend<T> for HashSet1<T, S>
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

impl<T, const N: usize> From<[T; N]> for HashSet1<T, RandomState>
where
    [T; N]: Array1,
    T: Eq + Hash,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { HashSet1::from_hash_set_unchecked(HashSet::from(items)) }
    }
}

impl<T, S> From<HashSet1<T, S>> for HashSet<T, S> {
    fn from(items: HashSet1<T, S>) -> Self {
        items.items
    }
}

impl<T, S> FromIterator1<T> for HashSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { HashSet1::from_hash_set_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> FromParallelIterator1<T> for HashSet1<T, S>
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
            HashSet1::from_hash_set_unchecked(items.into_par_iter1().into_par_iter().collect())
        }
    }
}

impl<T, S> IntoIterator for HashSet1<T, S> {
    type Item = T;
    type IntoIter = hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T, S> IntoIterator for &'a HashSet1<T, S> {
    type Item = &'a T;
    type IntoIter = hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<T, S> IntoIterator1 for HashSet1<T, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, S> IntoIterator1 for &'_ HashSet1<T, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator for HashSet1<T, S>
where
    T: Send,
{
    type Item = T;
    type Iter = <HashSet<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T, S> IntoParallelIterator for &'a HashSet1<T, S>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = <&'a HashSet<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator1 for HashSet1<T, S>
where
    T: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T, S> IntoParallelIterator1 for &'_ HashSet1<T, S>
where
    T: Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<T, S> JsonSchema for HashSet1<T, S>
where
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        HashSet::<T, S>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<HashSet<T, S>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        HashSet::<T, S>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        HashSet::<T, S>::schema_id()
    }
}

impl<R, T, S> Sub<&'_ R> for &'_ HashSet1<T, S>
where
    R: ClosedHashSet<Item = T, State = S>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_hash_set() - rhs.as_hash_set()
    }
}

impl<T, S> Sub<&'_ HashSet1<T, S>> for &'_ HashSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = HashSet<T, S>;

    fn sub(self, rhs: &'_ HashSet1<T, S>) -> Self::Output {
        self - rhs.as_hash_set()
    }
}

impl<T, S> TryFrom<HashSet<T, S>> for HashSet1<T, S> {
    type Error = EmptyError<HashSet<T, S>>;

    fn try_from(items: HashSet<T, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T, S> TryFrom<&'a HashSet<T, S>> for &'a HashSet1<T, S> {
    type Error = EmptyError<&'a HashSet<T, S>>;

    fn try_from(items: &'a HashSet<T, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T, S> TryFrom<&'a mut HashSet<T, S>> for &'a mut HashSet1<T, S> {
    type Error = EmptyError<&'a mut HashSet<T, S>>;

    fn try_from(items: &'a mut HashSet<T, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

// TODO: Though it is likely even less useful, this concept can be generalized to maybe-empty
//       collections just like `Segment` is. It can also be applied to other unordered non-empty
//       collections.
// TODO: Support isomorphic keys.
#[must_use]
pub struct Except<'a, T, S> {
    items: &'a mut HashSet<T, S>,
    key: &'a T,
}

impl<T, S> Except<'_, T, S> {
    pub fn key(&self) -> &T {
        self.key
    }
}

impl<T, S> Except<'_, T, S>
where
    T: Eq + Hash,
{
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.items.retain(|item| {
            let is_retained = item == self.key;
            is_retained || f(item)
        });
    }

    pub fn clear(&mut self) {
        self.retain(|_| false)
    }

    pub fn iter(&self) -> impl '_ + Clone + Iterator<Item = &'_ T> {
        self.items.iter().filter(|&item| item == self.key)
    }
}

impl<T, S> Debug for Except<'_, T, S>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Except")
            .field("items", &self.items)
            .field("key", self.key)
            .finish()
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::hash_set1::HashSet1;
    use crate::iter1::{self, FromIterator1};

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> HashSet1<u8> {
        HashSet1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::hash_set1::HashSet1;
    use crate::hash_set1::harness::xs1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};

    // SAFETY: The `FnMut`s constructed in cases (the parameter `f`) must not stash or otherwise
    //         allow access to the parameter beyond the scope of their bodies. (This is difficult
    //         to achieve in this context.)
    // Unlike ordered collections, not only is the remaining only item unspecified, it is also
    // nondeterministic when using `RandomState`. This test instead asserts the length of the
    // `HashSet1` when its contents cannot be known.
    #[rstest]
    #[case::ignore_and_retain(|_| true, Ok(HashSet1::from([0, 1, 2, 3, 4])))]
    #[case::ignore_and_do_not_retain(|_| false, Err(NonZeroUsize::MIN))]
    #[case::compare_and_retain_none(
        |x: *const _| unsafe {
            *x > 4
        },
        Err(NonZeroUsize::MIN),
    )]
    #[case::compare_and_retain_some(
        |x: *const _| unsafe {
            *x < 3
        },
        Ok(HashSet1::from([0, 1, 2])),
    )]
    fn retain_until_only_from_hash_set1_then_len_or_hash_set1_eq<F>(
        mut xs1: HashSet1<u8>,
        #[case] mut f: F,
        #[case] expected: Result<HashSet1<u8>, NonZeroUsize>,
    ) where
        F: FnMut(*const u8) -> bool,
    {
        // TODO: The type parameter `F` must be a `FnMut` over `*const u8` instead of `&u8` here,
        //       because `rstest` constructs the case in a way that the `&u8` has a lifetime that
        //       is too specific and too long (it would borrow the item beyond
        //       `retain_until_only`). Is there a way to prevent this without introducing `*const
        //       u8` and unsafe code in cases for `f`? If so, do that instead!
        let _ = xs1.retain_until_only(|x| f(x as *const u8)).copied();
        match expected {
            Ok(expected) => assert_eq!(xs1, expected),
            Err(expected) => assert_eq!(xs1.len(), expected),
        }
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    fn clear_except_of_hash_set1_then_hash_set1_eq_key(mut xs1: HashSet1<u8>, #[case] key: u8) {
        xs1.except(&key).unwrap().clear();
        assert_eq!(xs1, HashSet1::from_one(key));
    }

    #[cfg(feature = "schemars")]
    #[rstest]
    fn hash_set1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<HashSet1<u8>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_hash_set1_into_and_from_tokens_eq(
        #[with(0)] xs1: HashSet1<u8>,
        #[with(1)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_hash_set1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<HashSet1<u8>, Vec<_>>(sequence)
    }
}

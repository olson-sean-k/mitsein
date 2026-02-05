//! A non-empty [`BTreeMap`][`btree_map`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_map::{self, BTreeMap, VacantEntry};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::iter::{FusedIterator, Skip, Take};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Bound, RangeBounds, RangeFull};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::cmp::{UnsafeOrd, UnsafeOrdIsomorph};
use crate::except::{self, ByKey, Exception, KeyNotFoundError};
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{
    self, IntoRangeBounds, ItemRange, OptionExt as _, OutOfBoundsError, RangeError,
    ResolveTrimRange, TrimRange, UnorderedError,
};
use crate::segment::{self, ByRange, ByTail, Segmentation};
use crate::take;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type KeyFor<T> = <T as ClosedBTreeMap>::Key;
type ValueFor<T> = <T as ClosedBTreeMap>::Value;
type EntryFor<T> = (KeyFor<T>, ValueFor<T>);

pub trait ClosedBTreeMap {
    type Key;
    type Value;

    fn as_btree_map(&self) -> &BTreeMap<Self::Key, Self::Value>;
}

impl<K, V> ClosedBTreeMap for BTreeMap<K, V> {
    type Key = K;
    type Value = V;

    fn as_btree_map(&self) -> &BTreeMap<Self::Key, Self::Value> {
        self
    }
}

impl<K, V, Q> ByKey<Q> for BTreeMap<K, V>
where
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    fn except<'a>(
        &'a mut self,
        key: &'a Q,
    ) -> Result<Except<'a, Self, Q>, KeyNotFoundError<&'a Q>> {
        self.contains_key(key)
            .then_some(Except::unchecked(self, key))
            .ok_or_else(|| KeyNotFoundError::from_key(key))
    }
}

impl<K, V, R> ByRange<K, R> for BTreeMap<K, V>
where
    K: Ord,
    R: IntoRangeBounds<K>,
{
    type Range = Option<ItemRange<K>>;
    type Error = UnorderedError<Bound<K>>;

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

impl<K, V> ByTail for BTreeMap<K, V>
where
    K: Clone,
{
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(self, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(self, TrimRange::RTAIL1)
    }
}

impl<K, V> Exception for BTreeMap<K, V> {
    type Kind = Self;
    type Target = Self;
}

impl<K, V> Extend1<(K, V)> for BTreeMap<K, V>
where
    K: Ord,
{
    fn extend_non_empty<I>(mut self, items: I) -> BTreeMap1<K, V>
    where
        I: IntoIterator1<Item = (K, V)>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { BTreeMap1::from_btree_map_unchecked(self) }
    }
}

unsafe impl<K, V> MaybeEmpty for BTreeMap<K, V> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        // SAFETY: This implementation is critical to memory safety. `BTreeMap::len` is reliable
        //         here, because it does not break the contract by returning a non-zero value for
        //         an empty map, even if the `Eq` or `Ord` implementations for `K` are
        //         non-compliant. This is why `BTreeMap1` APIs do not require `K: UnsafeOrd`
        //         bounds (unlike segments, which rely on consistently isolating a range).
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

impl<K, V> ResolveTrimRange<Option<ItemRange<K>>> for BTreeMap<K, V>
where
    K: Clone + Ord,
{
    fn resolve_trim_range(&self, range: TrimRange) -> Option<ItemRange<K>> {
        let TrimRange { tail, rtail } = range;
        self.keys()
            .nth(tail)
            .zip(self.keys().rev().nth(rtail))
            .and_then(|(start, end)| range::ordered_range_bounds(start.clone()..=end.clone()).ok())
            .map(|range| {
                let (start, end) = range.into_bounds();
                ItemRange::unchecked(start, end)
            })
    }
}

impl<K, V> Segmentation for BTreeMap<K, V> {
    type Kind = Self;
    type Target = Self;
}

pub type ManyEntry<'a, K, V> = btree_map::OccupiedEntry<'a, K, V>;

#[derive(Debug)]
#[repr(transparent)]
pub struct OnlyEntry<'a, K, V>
where
    K: Ord,
{
    entry: btree_map::OccupiedEntry<'a, K, V>,
}

impl<'a, K, V> OnlyEntry<'a, K, V>
where
    K: Ord,
{
    fn from_occupied_entry(entry: btree_map::OccupiedEntry<'a, K, V>) -> Self {
        OnlyEntry { entry }
    }

    pub fn into_mut(self) -> &'a mut V {
        self.entry.into_mut()
    }

    pub fn insert(&mut self, value: V) -> V {
        self.entry.insert(value)
    }

    pub fn get(&self) -> &V {
        self.entry.get()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.entry.get_mut()
    }

    pub fn key(&self) -> &K {
        self.entry.key()
    }
}

pub type OccupiedEntry<'a, K, V> = Cardinality<OnlyEntry<'a, K, V>, ManyEntry<'a, K, V>>;

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: Ord,
{
    pub fn into_mut(self) -> &'a mut V {
        match self {
            OccupiedEntry::Many(many) => many.into_mut(),
            OccupiedEntry::One(only) => only.into_mut(),
        }
    }

    pub fn remove_entry_or_get_only(self) -> OrOnlyEntry<'a, (K, V), K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.remove_entry()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn remove_or_get_only(self) -> OrOnlyEntry<'a, V, K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.remove()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn insert(&mut self, value: V) -> V {
        match self {
            OccupiedEntry::Many(many) => many.insert(value),
            OccupiedEntry::One(only) => only.insert(value),
        }
    }

    pub fn get(&self) -> &V {
        match self {
            OccupiedEntry::Many(many) => many.get(),
            OccupiedEntry::One(only) => only.get(),
        }
    }

    pub fn get_mut(&mut self) -> &mut V {
        match self {
            OccupiedEntry::Many(many) => many.get_mut(),
            OccupiedEntry::One(only) => only.get_mut(),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            OccupiedEntry::Many(many) => many.key(),
            OccupiedEntry::One(only) => only.key(),
        }
    }
}

impl<'a, K, V> From<ManyEntry<'a, K, V>> for OccupiedEntry<'a, K, V>
where
    K: Ord,
{
    fn from(many: ManyEntry<'a, K, V>) -> Self {
        OccupiedEntry::Many(many)
    }
}

impl<'a, K, V> From<OnlyEntry<'a, K, V>> for OccupiedEntry<'a, K, V>
where
    K: Ord,
{
    fn from(only: OnlyEntry<'a, K, V>) -> Self {
        OccupiedEntry::One(only)
    }
}

#[derive(Debug)]
pub enum Entry<'a, K, V>
where
    K: Ord,
{
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Ord,
{
    /// # Safety
    ///
    /// The `BTreeMap1` from which `entry` has been obtained must have a non-empty cardinality of
    /// many (must have **more than one** item).
    unsafe fn from_entry_many(entry: btree_map::Entry<'a, K, V>) -> Self {
        match entry {
            btree_map::Entry::Vacant(vacant) => Entry::Vacant(vacant),
            btree_map::Entry::Occupied(occupied) => Entry::Occupied(occupied.into()),
        }
    }

    fn from_entry_only(entry: btree_map::Entry<'a, K, V>) -> Self {
        match entry {
            btree_map::Entry::Vacant(vacant) => Entry::Vacant(vacant),
            btree_map::Entry::Occupied(occupied) => {
                Entry::Occupied(OnlyEntry::from_occupied_entry(occupied).into())
            },
        }
    }

    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match &mut self {
            Entry::Vacant(_) => {},
            Entry::Occupied(occupied) => f(occupied.get_mut()),
        };
        self
    }

    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(move || default)
    }

    pub fn or_insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        self.or_insert_with_key(move |_| f())
    }

    pub fn or_insert_with_key<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Vacant(vacant) => {
                let value = f(vacant.key());
                vacant.insert(value)
            },
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Vacant(vacant) => vacant.key(),
            Entry::Occupied(occupied) => occupied.key(),
        }
    }
}

impl<'a, K, V> From<OnlyEntry<'a, K, V>> for Entry<'a, K, V>
where
    K: Ord,
{
    fn from(only: OnlyEntry<'a, K, V>) -> Self {
        Entry::Occupied(only.into())
    }
}

pub type OrOnlyEntry<'a, T, K, V> = Result<T, OnlyEntry<'a, K, V>>;

pub trait OrOnlyEntryExt<'a, K, V>
where
    K: Ord,
{
    fn get(&self) -> &V;

    fn get_mut(&mut self) -> &mut V;
}

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, V, K, V>
where
    K: Ord,
{
    fn get(&self) -> &V {
        match self {
            Ok(value) => value,
            Err(only) => only.get(),
        }
    }

    fn get_mut(&mut self) -> &mut V {
        match self {
            Ok(value) => value,
            Err(only) => only.get_mut(),
        }
    }
}

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, (K, V), K, V>
where
    K: Ord,
{
    fn get(&self) -> &V {
        match self {
            Ok((_, value)) => value,
            Err(only) => only.get(),
        }
    }

    fn get_mut(&mut self) -> &mut V {
        match self {
            Ok((_, value)) => value,
            Err(only) => only.get_mut(),
        }
    }
}

type TakeIfMany<'a, K, V, U, N = ()> = take::TakeIfMany<'a, BTreeMap<K, V>, U, N>;

pub type PopIfMany<'a, T> = TakeIfMany<'a, KeyFor<T>, ValueFor<T>, EntryFor<T>>;

pub type RemoveIfMany<'a, 'q, T, Q> =
    TakeIfMany<'a, KeyFor<T>, ValueFor<T>, Option<ValueFor<T>>, &'q Q>;

pub type RemoveEntryIfMany<'a, 'q, T, Q> =
    TakeIfMany<'a, KeyFor<T>, ValueFor<T>, Option<EntryFor<T>>, &'q Q>;

impl<'a, K, V, U, N> TakeIfMany<'a, K, V, U, N>
where
    K: Ord,
{
    pub fn or_get_only(self) -> Result<U, OnlyEntry<'a, K, V>> {
        self.take_or_else(|items, _| items.first_entry_as_only())
    }

    pub fn or_replace_only(self, value: V) -> Result<U, V> {
        self.or_else_replace_only(move || value)
    }

    pub fn or_else_replace_only<F>(self, f: F) -> Result<U, V>
    where
        F: FnOnce() -> V,
    {
        self.take_or_else(move |items, _| mem::replace(items.first_entry().get_mut(), f()))
    }
}

impl<'a, K, V, U, Q> TakeIfMany<'a, K, V, Option<U>, &'_ Q>
where
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    pub fn or_get(self) -> Option<Result<U, OnlyEntry<'a, K, V>>> {
        self.try_take_or_else(|items, query| {
            items
                .items
                .contains_key(query)
                .then(|| items.first_entry_as_only())
        })
    }

    pub fn or_replace(self, value: V) -> Option<Result<U, V>> {
        self.or_else_replace(move || value)
    }

    pub fn or_else_replace<F>(self, f: F) -> Option<Result<U, V>>
    where
        F: FnOnce() -> V,
    {
        self.try_take_or_else(|items, query| {
            items
                .get_mut(query)
                .map(move |item| mem::replace(item, f()))
        })
    }
}

pub type BTreeMap1<K, V> = NonEmpty<BTreeMap<K, V>>;

impl<K, V> BTreeMap1<K, V> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`BTreeMap::new()`][`BTreeMap::new`].
    ///
    /// [`BTreeMap::new`]: alloc::collections::btree_map::BTreeMap::new
    pub unsafe fn from_btree_map_unchecked(items: BTreeMap<K, V>) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    pub fn from_one(item: (K, V)) -> Self
    where
        K: Ord,
    {
        iter1::one(item).collect1()
    }

    pub fn from_head_and_tail<I>(head: (K, V), tail: I) -> Self
    where
        K: Ord,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_rtail_and_head<I>(tail: I, head: (K, V)) -> Self
    where
        K: Ord,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::rtail_and_head(tail, head).collect1()
    }

    pub fn try_from_ref(
        items: &BTreeMap<K, V>,
    ) -> Result<&'_ Self, EmptyError<&'_ BTreeMap<K, V>>> {
        items.try_into()
    }

    pub fn try_from_mut(
        items: &mut BTreeMap<K, V>,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut BTreeMap<K, V>>> {
        items.try_into()
    }

    pub fn into_btree_map(self) -> BTreeMap<K, V> {
        self.items
    }

    pub fn into_keys1(self) -> Iterator1<btree_map::IntoKeys<K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_keys()) }
    }

    pub fn into_values1(self) -> Iterator1<btree_map::IntoValues<K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_values()) }
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<BTreeMap<K, V>>>
    where
        K: Ord,
        F: FnMut(&K, &mut V) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }

    pub fn retain_until_only<F>(&mut self, mut f: F) -> Option<(&'_ K, &'_ V)>
    where
        K: Ord,
        F: FnMut(&K, &V) -> bool,
    {
        // Segmentation for relational collections can be expensive and involves very different
        // tradeoffs for collections with many items and collections with large items. For this
        // reason, the first item is filtered directly rather than using `tail` or `rtail` here.
        let mut index = 0usize;
        self.items.retain(|key, value| {
            let is_retained = index == 0 || f(key, &*value);
            index += 1;
            is_retained
        });
        if self.len().get() == 1 {
            let (key, value) = self.first_key_value();
            if f(key, value) {
                None
            }
            else {
                Some((key, value))
            }
        }
        else {
            let (key, value) = self.first_key_value();
            if !f(key, value) {
                // The first item is **not** retained and there is more than one item.
                self.pop_first_if_many();
            }
            None
        }
    }

    pub fn split_off_tail(&mut self) -> BTreeMap<K, V>
    where
        K: Clone + UnsafeOrd,
    {
        match self.items.keys().nth(1).cloned() {
            // `BTreeMap::split_off` relies on the `Ord` implementation to determine where the
            // split begins. This requires `UnsafeOrd` here, because a non-conformant `Ord`
            // implementation may split at the first item (despite the matched expression) and
            // empty the `BTreeMap1`.
            Some(key) => self.items.split_off(&key),
            _ => BTreeMap::new(),
        }
    }

    pub fn append(&mut self, items: &mut BTreeMap<K, V>)
    where
        K: Ord,
    {
        self.items.append(items)
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V>
    where
        K: Ord,
    {
        match self.as_cardinality_items_mut() {
            Cardinality::One(items) => Entry::from_entry_only(items.entry(key)),
            // SAFETY: The `items` method returns the correct non-empty cardinality based on the
            //         `MaybeEmpty` implementation.
            Cardinality::Many(items) => unsafe { Entry::from_entry_many(items.entry(key)) },
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        self.items.insert(key, value)
    }

    pub fn pop_first_if_many(&mut self) -> PopIfMany<'_, Self>
    where
        K: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop_first().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_first_until_only(&mut self) -> PopFirstUntilOnly<'_, K, V>
    where
        K: Ord,
    {
        PopFirstUntilOnly { items: self }
    }

    pub fn pop_last_if_many(&mut self) -> PopIfMany<'_, Self>
    where
        K: Ord,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop_last().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_last_until_only(&mut self) -> PopLastUntilOnly<'_, K, V>
    where
        K: Ord,
    {
        PopLastUntilOnly { items: self }
    }

    pub fn remove_if_many<'a, 'q, Q>(&'a mut self, query: &'q Q) -> RemoveIfMany<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.remove(query))
    }

    pub fn remove_entry_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveEntryIfMany<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.remove_entry(query))
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get(query)
    }

    pub fn get_key_value<Q>(&self, query: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get_key_value(query)
    }

    pub fn get_mut<Q>(&mut self, query: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get_mut(query)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` is non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn first_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY: `self` is non-empty.
        unsafe { self.items.first_key_value().unwrap_maybe_unchecked() }
    }

    pub fn first_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        self.as_cardinality_items_mut()
            // SAFETY: `self` is non-empty.
            .map(|items| unsafe { items.first_entry().unwrap_maybe_unchecked() })
            .map_one(OnlyEntry::from_occupied_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    fn first_entry_as_only(&mut self) -> OnlyEntry<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY: `self` is non-empty.
        OnlyEntry::from_occupied_entry(unsafe { self.items.first_entry().unwrap_maybe_unchecked() })
    }

    pub fn last_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY: `self` is non-empty.
        unsafe { self.items.last_key_value().unwrap_maybe_unchecked() }
    }

    pub fn last_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        self.as_cardinality_items_mut()
            // SAFETY: `self` is non-empty.
            .map(|items| unsafe { items.last_entry().unwrap_maybe_unchecked() })
            .map_one(OnlyEntry::from_occupied_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    pub fn iter1(&self) -> Iterator1<btree_map::Iter<'_, K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<btree_map::IterMut<'_, K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub fn keys1(&self) -> Iterator1<btree_map::Keys<'_, K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.keys()) }
    }

    pub fn values1(&self) -> Iterator1<btree_map::Values<'_, K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.values()) }
    }

    pub fn values1_mut(&mut self) -> Iterator1<btree_map::ValuesMut<'_, K, V>> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.values_mut()) }
    }

    pub fn contains_key<Q>(&self, query: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.contains_key(query)
    }

    pub const fn as_btree_map(&self) -> &BTreeMap<K, V> {
        &self.items
    }

    /// # Safety
    ///
    /// The [`BTreeMap`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::btree_map1::BTreeMap1;
    ///
    /// let mut xs = BTreeMap1::from([("a", 0i32), ("b", 1)]);
    /// // This block is unsound. The `&mut BTreeMap` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_btree_map().clear();
    /// }
    /// let x = xs.first_key_value(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_btree_map(&mut self) -> &mut BTreeMap<K, V> {
        &mut self.items
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> BTreeMap1<K, V>
where
    K: Ord,
{
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        K: Sync,
        V: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }

    pub fn par_iter1_mut(
        &mut self,
    ) -> ParallelIterator1<<&'_ mut Self as IntoParallelIterator>::Iter>
    where
        K: Sync,
        V: Send,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter_mut()) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, K, V> Arbitrary<'a> for BTreeMap1<K, V>
where
    (K, V): Arbitrary<'a>,
    K: Ord,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(
            <(K, V)>::arbitrary(unstructured),
            unstructured.arbitrary_iter()?,
        )
        .collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (<(K, V)>::size_hint(depth).0, None)
    }
}

impl<K, V, Q> ByKey<Q> for BTreeMap1<K, V>
where
    K: Borrow<Q> + UnsafeOrdIsomorph<Q>,
    Q: ?Sized + UnsafeOrd,
{
    fn except<'a>(
        &'a mut self,
        key: &'a Q,
    ) -> Result<Except<'a, Self, Q>, KeyNotFoundError<&'a Q>> {
        self.contains_key(key)
            .then_some(Except::unchecked(&mut self.items, key))
            .ok_or_else(|| KeyNotFoundError::from_key(key))
    }
}

impl<K, V, R> ByRange<K, R> for BTreeMap1<K, V>
where
    K: UnsafeOrd,
    R: IntoRangeBounds<K>,
{
    type Range = Option<ItemRange<K>>;
    type Error = RangeError<Bound<K>>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self, Self::Range>, Self::Error> {
        range::ordered_range_bounds(range)
            .map_err(|range| {
                let (start, end) = range.into_bounds();
                UnorderedError(start, end).into()
            })
            .and_then(|range| {
                if range.contains(self.keys1().first()) && range.contains(self.keys1().last()) {
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

impl<K, V> ByTail for BTreeMap1<K, V>
where
    K: Clone,
{
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(&mut self.items, TrimRange::TAIL1)
    }

    fn rtail(&mut self) -> Segment<'_, Self, Self::Range> {
        Segment::unchecked(&mut self.items, TrimRange::RTAIL1)
    }
}

impl<K, V> ClosedBTreeMap for BTreeMap1<K, V> {
    type Key = K;
    type Value = V;

    fn as_btree_map(&self) -> &BTreeMap<Self::Key, Self::Value> {
        self.as_ref()
    }
}

impl<K, V> Debug for BTreeMap1<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<K, V> Exception for BTreeMap1<K, V> {
    type Kind = Self;
    type Target = BTreeMap<K, V>;
}

impl<K, V> Extend<(K, V)> for BTreeMap1<K, V>
where
    K: Ord,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        self.items.extend(extension)
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for BTreeMap1<K, V>
where
    [(K, V); N]: Array1,
    K: Ord,
{
    fn from(items: [(K, V); N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { BTreeMap1::from_btree_map_unchecked(BTreeMap::from(items)) }
    }
}

impl<K, V> From<BTreeMap1<K, V>> for BTreeMap<K, V> {
    fn from(items: BTreeMap1<K, V>) -> Self {
        items.items
    }
}

impl<K, V> FromIterator1<(K, V)> for BTreeMap1<K, V>
where
    K: Ord,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = (K, V)>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { BTreeMap1::from_btree_map_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> FromParallelIterator1<(K, V)> for BTreeMap1<K, V>
where
    K: Ord + Send,
    V: Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = (K, V)>,
    {
        // SAFETY: `items` is non-empty.
        unsafe {
            BTreeMap1::from_btree_map_unchecked(items.into_par_iter1().into_par_iter().collect())
        }
    }
}

impl<K, V> IntoIterator for BTreeMap1<K, V> {
    type Item = (K, V);
    type IntoIter = btree_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a BTreeMap1<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = btree_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut BTreeMap1<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = btree_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<K, V> IntoIterator1 for BTreeMap1<K, V> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<K, V> IntoIterator1 for &'_ BTreeMap1<K, V> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<K, V> IntoIterator1 for &'_ mut BTreeMap1<K, V> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> IntoParallelIterator for BTreeMap1<K, V>
where
    K: Ord + Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = <BTreeMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V> IntoParallelIterator for &'a BTreeMap1<K, V>
where
    K: Ord + Sync,
    V: Sync,
{
    type Item = (&'a K, &'a V);
    type Iter = <&'a BTreeMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V> IntoParallelIterator for &'a mut BTreeMap1<K, V>
where
    K: Ord + Sync,
    V: Send,
{
    type Item = (&'a K, &'a mut V);
    type Iter = <&'a mut BTreeMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> IntoParallelIterator1 for BTreeMap1<K, V>
where
    K: Ord + Send,
    V: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` is non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> IntoParallelIterator1 for &'_ BTreeMap1<K, V>
where
    K: Ord + Sync,
    V: Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` is non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V> IntoParallelIterator1 for &'_ mut BTreeMap1<K, V>
where
    K: Ord + Sync,
    V: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` is non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&mut self.items) }
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<K, V> JsonSchema for BTreeMap1<K, V>
where
    K: JsonSchema,
    V: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        BTreeMap::<K, V>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<BTreeMap<K, V>>(
            schemars::NON_EMPTY_KEY_OBJECT,
            generator,
        )
    }

    fn inline_schema() -> bool {
        BTreeMap::<K, V>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        BTreeMap::<K, V>::schema_id()
    }
}

impl<K, V> Segmentation for BTreeMap1<K, V> {
    type Kind = Self;
    type Target = BTreeMap<K, V>;
}

impl<K, V> TryFrom<BTreeMap<K, V>> for BTreeMap1<K, V> {
    type Error = EmptyError<BTreeMap<K, V>>;

    fn try_from(items: BTreeMap<K, V>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, K, V> TryFrom<&'a BTreeMap<K, V>> for &'a BTreeMap1<K, V> {
    type Error = EmptyError<&'a BTreeMap<K, V>>;

    fn try_from(items: &'a BTreeMap<K, V>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, K, V> TryFrom<&'a mut BTreeMap<K, V>> for &'a mut BTreeMap1<K, V> {
    type Error = EmptyError<&'a mut BTreeMap<K, V>>;

    fn try_from(items: &'a mut BTreeMap<K, V>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[derive(Debug)]
pub struct PopFirstUntilOnly<'a, K, V>
where
    K: Ord,
{
    items: &'a mut BTreeMap1<K, V>,
}

impl<K, V> Drop for PopFirstUntilOnly<'_, K, V>
where
    K: Ord,
{
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

impl<K, V> Iterator for PopFirstUntilOnly<'_, K, V>
where
    K: Ord,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_first_if_many().or_none()
    }
}

#[derive(Debug)]
pub struct PopLastUntilOnly<'a, K, V>
where
    K: Ord,
{
    items: &'a mut BTreeMap1<K, V>,
}

impl<K, V> Drop for PopLastUntilOnly<'_, K, V>
where
    K: Ord,
{
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

impl<K, V> Iterator for PopLastUntilOnly<'_, K, V>
where
    K: Ord,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.items.pop_first_if_many().or_none()
    }
}

// Unfortunately, the type of the `ExtractIf` predicate `F` cannot be named in `Except::drain` and
// so prevents returning a complete type.
struct DrainExcept<'a, K, V, F>
where
    K: Ord,
    F: FnMut(&K, &mut V) -> bool,
{
    input: btree_map::ExtractIf<'a, K, V, RangeFull, F>,
}

impl<K, V, F> Debug for DrainExcept<'_, K, V, F>
where
    K: Debug + Ord,
    V: Debug,
    F: FnMut(&K, &mut V) -> bool,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DrainExcept")
            .field("input", &self.input)
            .finish()
    }
}

impl<K, V, F> Drop for DrainExcept<'_, K, V, F>
where
    K: Ord,
    F: FnMut(&K, &mut V) -> bool,
{
    fn drop(&mut self) {
        self.input.by_ref().for_each(|_| {});
    }
}

impl<K, V, F> FusedIterator for DrainExcept<'_, K, V, F>
where
    K: Ord,
    F: FnMut(&K, &mut V) -> bool,
{
}

impl<K, V, F> Iterator for DrainExcept<'_, K, V, F>
where
    K: Ord,
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next()
    }
}

pub type Except<'a, T, Q> = except::Except<'a, T, BTreeMap<KeyFor<T>, ValueFor<T>>, Q>;

impl<T, K, V, Q> Except<'_, T, Q>
where
    T: ClosedBTreeMap<Key = K, Value = V> + Exception<Target = BTreeMap<K, V>>,
    K: Borrow<Q> + Ord,
    Q: Ord + ?Sized,
{
    pub fn drain(&mut self) -> impl '_ + Drop + Iterator<Item = (K, V)> {
        DrainExcept {
            input: self.items.extract_if(.., |key, _| key.borrow() != self.key),
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.items.retain(|key, value| {
            let is_retained = key.borrow() == self.key;
            is_retained || f(key, value)
        });
    }

    pub fn clear(&mut self) {
        self.retain(|_, _| false)
    }

    pub fn iter(&self) -> impl '_ + Clone + Iterator<Item = (&'_ K, &'_ V)> {
        self.items
            .iter()
            .filter(|&(key, _)| key.borrow() != self.key)
    }
}

pub type Segment<'a, T, R> = segment::Segment<'a, T, BTreeMap<KeyFor<T>, ValueFor<T>>, R>;

impl<T, K, V> Segment<'_, T, Option<ItemRange<K>>>
where
    T: ClosedBTreeMap<Key = K, Value = V> + Segmentation<Target = BTreeMap<K, V>>,
    K: Ord,
{
    fn remove_isomorph_unchecked<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.range.contains(key) {
            self.items.remove(key)
        }
        else {
            None
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        if let Some(range) = self.range.as_ref() {
            self.items.retain(range.retain_key_value_in_range(f))
        }
    }

    pub fn insert_in_range(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)> {
        if self.range.contains(&key) {
            Ok(self.items.insert(key, value))
        }
        else {
            Err((key, value))
        }
    }

    pub fn append_in_range(&mut self, other: &mut BTreeMap<K, V>)
    where
        K: Clone,
    {
        if let Some(range) = self.range.as_ref() {
            // To append within the range of the segment, `other` is split into `low`, `middle`,
            // and `high`. The `middle` set contains any and all items in range, and so it extends
            // the segment. `low` and `high` are out of bounds of the range, and so these items are
            // not inserted into the segment and must remain in `other`.
            //
            // Note that `low` is just an alias for `other` here, and so it is an exclusive
            // reference to the input `BTreeMap` (unlike `middle` and `high`).
            let low = other;
            let mut middle = match range.start_bound() {
                Bound::Excluded(start) => {
                    let mut middle = low.split_off(start);
                    low.extend(middle.remove_entry(start));
                    middle
                },
                Bound::Included(start) => low.split_off(start),
                Bound::Unbounded => {
                    if let Some(first) = low.keys().next().cloned() {
                        // The segment has no lower bound, so all of `low` is split off into
                        // `middle` (leaving `low` empty).
                        low.split_off(&first)
                    }
                    else {
                        // If `other` is empty (and so `low.first()` is `None`), then the middle
                        // items are also empty.
                        BTreeMap::new()
                    }
                },
            };
            let high = match range.end_bound() {
                Bound::Excluded(end) => middle.split_off(end),
                Bound::Included(end) => {
                    let mut high = middle.split_off(end);
                    middle.extend(high.remove_entry(end));
                    high
                },
                Bound::Unbounded => BTreeMap::new(),
            };
            self.items.extend(middle);
            low.extend(high);
        }
    }

    pub fn clear(&mut self) {
        if let Some(range) = self.range.as_ref() {
            self.items.retain(|key, _| !range.contains(key))
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.range.contains(key) {
            self.items.get(key)
        }
        else {
            None
        }
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.range
            .as_ref()
            .and_then(|range| match range.start_bound() {
                Bound::Excluded(start) => self.items.range(start..).nth(1),
                Bound::Included(start) => self.items.range(start..).next(),
                Bound::Unbounded => self.items.first_key_value(),
            })
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.range
            .as_ref()
            .and_then(|range| match range.end_bound() {
                Bound::Excluded(end) => self.items.range(..end).next_back(),
                Bound::Included(end) => self.items.range(..=end).next_back(),
                Bound::Unbounded => self.items.last_key_value(),
            })
    }

    pub fn len(&self) -> usize {
        self.range.as_ref().map_or(0, |range| {
            self.items
                .range((range.start_bound(), range.end_bound()))
                .count()
        })
    }

    pub fn iter(&self) -> impl '_ + Clone + DoubleEndedIterator<Item = (&'_ K, &'_ V)> {
        self.range
            .as_ref()
            .map(|range| self.items.range((range.start_bound(), range.end_bound())))
            .into_iter()
            .flatten()
    }

    pub fn iter_mut(&mut self) -> impl '_ + DoubleEndedIterator<Item = (&'_ K, &'_ mut V)> {
        self.range
            .as_ref()
            .map(|range| {
                self.items
                    .range_mut((range.start_bound(), range.end_bound()))
            })
            .into_iter()
            .flatten()
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.range.contains(key) && self.items.contains_key(key)
    }
}

impl<K, V> Segment<'_, BTreeMap<K, V>, Option<ItemRange<K>>> {
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.remove_isomorph_unchecked(key)
    }
}

impl<K, V> Segment<'_, BTreeMap1<K, V>, Option<ItemRange<K>>> {
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + UnsafeOrdIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.remove_isomorph_unchecked(key)
    }
}

impl<T, K, V> ByTail for Segment<'_, T, Option<ItemRange<K>>>
where
    T: ClosedBTreeMap<Key = K, Value = V> + Segmentation<Target = BTreeMap<K, V>>,
    // A `K: UnsafeOrd` bound is not needed here, because segments over an `ItemRange` can only be
    // constructed for a `BTreeMap1` via `Query`, which has that bound. This means that there is no
    // need to separate `Tail` implementations for `BTreeMap` and `BTreeMap1`.
    K: Clone + Ord,
{
    type Range = Option<ItemRange<K>>;

    fn tail(&mut self) -> Segment<'_, T, Self::Range> {
        if let Some(range) = self.range.clone() {
            let (start, end) = range.into_bounds();
            let start = match start {
                Bound::Excluded(start) => self.items.range(start..).nth(2),
                Bound::Included(start) => self.items.range(start..).nth(1),
                Bound::Unbounded => self.items.iter().nth(1),
            };
            let range = start
                .map(|(key, _)| key)
                .cloned()
                .and_then(|start| range::ordered_range_bounds((Bound::Included(start), end)).ok())
                .map(|(start, end)| ItemRange::unchecked(start, end));
            Segment::unchecked(self.items, range)
        }
        else {
            Segment::unchecked(self.items, None)
        }
    }

    fn rtail(&mut self) -> Segment<'_, T, Self::Range> {
        if let Some(range) = self.range.clone() {
            let (start, end) = range.into_bounds();
            let end = match end {
                Bound::Excluded(end) => self.items.range(..end).rev().nth(1),
                Bound::Included(end) => self.items.range(..=end).rev().nth(1),
                Bound::Unbounded => self.items.iter().rev().nth(1),
            };
            let range = end
                .map(|(key, _)| key)
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

impl<'a, T, K, V> Segment<'a, T, TrimRange>
where
    T: ClosedBTreeMap<Key = K, Value = V> + Segmentation<Target = BTreeMap<K, V>>,
    K: Ord,
{
    pub fn by_key(self) -> Segment<'a, T, Option<ItemRange<K>>>
    where
        K: Clone,
    {
        let Segment { items, range } = self;
        let range = items.resolve_trim_range(range);
        Segment::unchecked(items, range)
    }

    fn remove_isomorph_unchecked<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.contains_key(key) {
            self.items.remove(key)
        }
        else {
            None
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let TrimRange { tail, rtail } = self.range;
        let rtail = self.items.len().saturating_sub(rtail);
        let mut index = 0usize;
        self.items.retain(|key, value| {
            let is_in_range = index >= tail && index < rtail;
            index = index.checked_add(1).expect("overflow in item index");
            (!is_in_range) || f(key, value)
        })
    }

    pub fn insert_in_range(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)>
    where
        K: Clone,
    {
        let range: Option<ItemRange<_>> = self.items.resolve_trim_range(self.range);
        Segment::<T, _>::unchecked(self.items, range).insert_in_range(key, value)
    }

    pub fn append_in_range(&mut self, other: &mut BTreeMap<K, V>)
    where
        K: Clone,
    {
        let range: Option<ItemRange<_>> = self.items.resolve_trim_range(self.range);
        Segment::<T, _>::unchecked(self.items, range).append_in_range(other)
    }

    pub fn clear(&mut self) {
        self.retain(|_, _| false);
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let TrimRange { tail, rtail } = self.range;
        let is_key = |item: &_| Borrow::<Q>::borrow(item) == key;
        self.items.get(key).take_if(|_| {
            (!self.items.keys().take(tail).any(is_key))
                && (!self.items.keys().rev().take(rtail).any(is_key))
        })
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.items.iter().nth(self.range.tail)
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.items.iter().rev().nth(self.range.rtail)
    }

    pub fn len(&self) -> usize {
        self.untrimmed_item_count(self.items.len())
    }

    pub fn iter(&self) -> Take<Skip<btree_map::Iter<'_, K, V>>> {
        self.items.iter().skip(self.range.tail).take(self.len())
    }

    pub fn iter_mut(&mut self) -> Take<Skip<btree_map::IterMut<'_, K, V>>> {
        let body = self.len();
        self.items.iter_mut().skip(self.range.tail).take(body)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }
}

impl<K, V> Segment<'_, BTreeMap<K, V>, TrimRange>
where
    K: Ord,
{
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.remove_isomorph_unchecked(key)
    }
}

impl<K, V> Segment<'_, BTreeMap1<K, V>, TrimRange>
where
    K: UnsafeOrd,
{
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + UnsafeOrdIsomorph<Q>,
        Q: ?Sized + UnsafeOrd,
    {
        self.remove_isomorph_unchecked(key)
    }
}

impl<T, K, V> ByTail for Segment<'_, T, TrimRange>
where
    T: ClosedBTreeMap<Key = K, Value = V> + Segmentation<Target = BTreeMap<K, V>>,
    K: Clone + Ord,
{
    type Range = TrimRange;

    fn tail(&mut self) -> Segment<'_, T, Self::Range> {
        self.advance_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, T, Self::Range> {
        self.advance_rtail_range()
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::btree_map1::BTreeMap1;
    use crate::iter1::{self, FromIterator1};

    pub const VALUE: char = 'x';

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> BTreeMap1<u8, char> {
        BTreeMap1::from_iter1(iter1::harness::xs1(end).map(|x| (x, VALUE)))
    }

    #[fixture]
    pub fn terminals1(#[default(0)] first: u8, #[default(9)] last: u8) -> BTreeMap1<u8, char> {
        BTreeMap1::from_iter1([(first, VALUE), (last, VALUE)])
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::btree_map1::BTreeMap1;
    use crate::btree_map1::harness::{self, VALUE, terminals1, xs1};
    use crate::except::ByKey;
    use crate::harness::KeyValueRef;
    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::segment::range::IntoRangeBounds;
    use crate::segment::{ByRange, ByTail};
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::map};

    #[rstest]
    #[case(0, &[(1, VALUE), (2, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(1, &[(0, VALUE), (2, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(2, &[(0, VALUE), (1, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(3, &[(0, VALUE), (1, VALUE), (2, VALUE), (4, VALUE)])]
    #[case(4, &[(0, VALUE), (1, VALUE), (2, VALUE), (3, VALUE)])]
    fn drain_except_of_btree_map1_then_drained_eq(
        mut xs1: BTreeMap1<u8, char>,
        #[case] key: u8,
        #[case] expected: &[(u8, char)],
    ) {
        let xs: Vec<_> = xs1.except(&key).unwrap().drain().collect();
        assert_eq!(xs.as_slice(), expected);
    }

    #[rstest]
    #[case((0, VALUE))]
    #[case((1, VALUE))]
    #[case((2, VALUE))]
    #[case((3, VALUE))]
    #[case((4, VALUE))]
    fn clear_except_of_btree_map1_then_btree_map1_eq_key_value(
        mut xs1: BTreeMap1<u8, char>,
        #[case] entry: (u8, char),
    ) {
        let (key, value) = entry;
        xs1.except(&key).unwrap().clear();
        assert_eq!(xs1, BTreeMap1::from_one((key, value)));
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    fn iter_except_of_btree_map1_then_iter_does_not_contain_key(
        mut xs1: BTreeMap1<u8, char>,
        #[case] key: u8,
    ) {
        let xs = xs1.except(&key).unwrap();
        assert!(!xs.iter().any(|(&x, _)| x == key));
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_btree_map1_then_btree_map1_eq_head(#[case] mut xs1: BTreeMap1<u8, char>) {
        xs1.tail().clear();
        assert_eq!(xs1, BTreeMap1::from_one((0, VALUE)));
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_btree_map1_then_btree_map1_eq_tail(#[case] mut xs1: BTreeMap1<u8, char>) {
        let tail = xs1.last_key_value().cloned();
        xs1.rtail().clear();
        assert_eq!(xs1, BTreeMap1::from_one(tail));
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_btree_map1_then_btree_map1_eq_head_and_tail(
        #[case] mut xs1: BTreeMap1<u8, char>,
    ) {
        let n = xs1.len().get();
        let head_and_tail = [(0, VALUE), xs1.last_key_value().cloned()];
        xs1.tail().rtail().clear();
        assert_eq!(
            xs1,
            BTreeMap1::try_from_iter(if n > 1 {
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
    fn clear_intersected_tail_rtail_of_btree_map1_then_btree_map1_eq_self(
        #[case] mut xs1: BTreeMap1<u8, char>,
    ) {
        let expected = xs1.clone();
        let mut xss = xs1.tail();
        let mut xss = xss.tail();
        let mut xss = xss.tail();
        let mut xss = xss.rtail();
        let mut xss = xss.rtail();
        let mut xss = xss.rtail();
        xss.clear();
        assert_eq!(xs1, expected);
    }

    #[rstest]
    #[case::absent_in_range(4.., 4, Ok(None))]
    #[case::absent_in_range(..=4, 4, Ok(None))]
    #[case::present_in_range(4.., 9, Ok(Some(VALUE)))]
    #[case::out_of_range_lower_bound(4.., 0, Err((0, VALUE)))]
    #[case::out_of_range_lower_bound(4.., 1, Err((1, VALUE)))]
    #[case::out_of_range_upper_bound(..5, 5, Err((5, VALUE)))]
    #[case::out_of_range_upper_bound(3..=5, 6, Err((6, VALUE)))]
    #[case::out_of_range_upper_bound(..5, 6, Err((6, VALUE)))]
    fn insert_into_btree_map1_segment_then_output_eq<R>(
        #[from(terminals1)] mut xs1: BTreeMap1<u8, char>,
        #[case] range: R,
        #[case] key: u8,
        #[case] expected: Result<Option<char>, (u8, char)>,
    ) where
        R: IntoRangeBounds<u8>,
    {
        let mut xss = xs1.segment(range).unwrap();
        assert_eq!(xss.insert_in_range(key, VALUE), expected);
    }

    #[cfg(feature = "schemars")]
    #[rstest]
    fn btree_map1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<BTreeMap1<u8, char>>(
            schemars::NON_EMPTY_KEY_OBJECT,
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_btree_map1_into_and_from_tokens_eq(
        xs1: BTreeMap1<u8, char>,
        map: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, map)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_btree_map1_from_empty_tokens_then_empty_error(
        #[with(0)] map: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<BTreeMap1<u8, char>, Vec<_>>(map)
    }
}

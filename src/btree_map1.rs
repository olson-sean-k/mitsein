//! A non-empty [`BTreeMap`][`btree_map`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_map::{self, BTreeMap, VacantEntry};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::RangeBounds;
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
use crate::cmp::UnsafeOrd;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, Intersect, RelationalRange};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
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
        // `BTreeMap::len` is reliable even in the face of a non-conformant `Ord` implementation.
        // The `BTreeMap1` implementation relies on this to maintain its non-empty invariant
        // without bounds on `UnsafeOrd`.
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

impl<K, V> Ranged for BTreeMap<K, V>
where
    K: Clone + Ord,
{
    type Range = RelationalRange<K>;

    fn range(&self) -> Self::Range {
        self.keys()
            .next()
            .cloned()
            .zip(self.keys().next_back().cloned())
            .into()
    }

    fn tail(&self) -> Self::Range {
        self.keys()
            .nth(1)
            .cloned()
            .zip(self.keys().next_back().cloned())
            .into()
    }

    fn rtail(&self) -> Self::Range {
        self.keys()
            .next()
            .cloned()
            .zip(self.keys().rev().nth(1).cloned())
            .into()
    }
}

impl<K, V> Segmentation for BTreeMap<K, V>
where
    K: Clone + Ord,
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

impl<K, V, R> SegmentedBy<R> for BTreeMap<K, V>
where
    RelationalRange<K>: Intersect<R, Output = RelationalRange<K>>,
    K: Clone + Ord,
    R: RangeBounds<K>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_bounds(range))
    }
}

impl<K, V> SegmentedOver for BTreeMap<K, V>
where
    K: Clone + Ord,
{
    type Target = Self;
    type Kind = Self;
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
            OccupiedEntry::Many(ref mut many) => many.insert(value),
            OccupiedEntry::One(ref mut only) => only.insert(value),
        }
    }

    pub fn get(&self) -> &V {
        match self {
            OccupiedEntry::Many(ref many) => many.get(),
            OccupiedEntry::One(ref only) => only.get(),
        }
    }

    pub fn get_mut(&mut self) -> &mut V {
        match self {
            OccupiedEntry::Many(ref mut many) => many.get_mut(),
            OccupiedEntry::One(ref mut only) => only.get_mut(),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            OccupiedEntry::Many(ref many) => many.key(),
            OccupiedEntry::One(ref only) => only.key(),
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
            Entry::Occupied(ref mut occupied) => f(occupied.get_mut()),
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
            Ok(ref value) => value,
            Err(ref only) => only.get(),
        }
    }

    fn get_mut(&mut self) -> &mut V {
        match self {
            Ok(ref mut value) => value,
            Err(ref mut only) => only.get_mut(),
        }
    }
}

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, (K, V), K, V>
where
    K: Ord,
{
    fn get(&self) -> &V {
        match self {
            Ok((_, ref value)) => value,
            Err(ref only) => only.get(),
        }
    }

    fn get_mut(&mut self) -> &mut V {
        match self {
            Ok((_, ref mut value)) => value,
            Err(ref mut only) => only.get_mut(),
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
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
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

    pub fn from_tail_and_head<I>(tail: I, head: (K, V)) -> Self
    where
        K: Ord,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn into_btree_map(self) -> BTreeMap<K, V> {
        self.items
    }

    pub fn into_keys1(self) -> Iterator1<btree_map::IntoKeys<K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_keys()) }
    }

    pub fn into_values1(self) -> Iterator1<btree_map::IntoValues<K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_values()) }
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
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn first_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first_key_value().unwrap_maybe_unchecked() }
    }

    pub fn first_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        self.as_cardinality_items_mut()
            // SAFETY: `self` must be non-empty.
            .map(|items| unsafe { items.first_entry().unwrap_maybe_unchecked() })
            .map_one(OnlyEntry::from_occupied_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    fn first_entry_as_only(&mut self) -> OnlyEntry<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY: `self` must be non-empty.
        OnlyEntry::from_occupied_entry(unsafe { self.items.first_entry().unwrap_maybe_unchecked() })
    }

    pub fn last_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last_key_value().unwrap_maybe_unchecked() }
    }

    pub fn last_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        self.as_cardinality_items_mut()
            // SAFETY: `self` must be non-empty.
            .map(|items| unsafe { items.last_entry().unwrap_maybe_unchecked() })
            .map_one(OnlyEntry::from_occupied_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    pub fn iter1(&self) -> Iterator1<btree_map::Iter<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<btree_map::IterMut<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub fn keys1(&self) -> Iterator1<btree_map::Keys<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.keys()) }
    }

    pub fn values1(&self) -> Iterator1<btree_map::Values<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.values()) }
    }

    pub fn values1_mut(&mut self) -> Iterator1<btree_map::ValuesMut<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
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
        // SAFETY: `self` must be non-empty.
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
        // SAFETY: `self` must be non-empty.
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
        // SAFETY: `self` must be non-empty.
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
        // SAFETY: `self` must be non-empty.
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

impl<K, V> Segmentation for BTreeMap1<K, V>
where
    K: Clone + UnsafeOrd,
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

impl<K, V, R> SegmentedBy<R> for BTreeMap1<K, V>
where
    RelationalRange<K>: Intersect<R, Output = RelationalRange<K>>,
    K: Clone + UnsafeOrd,
    R: RangeBounds<K>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_bounds(range))
    }
}

impl<K, V> SegmentedOver for BTreeMap1<K, V>
where
    K: Clone + UnsafeOrd,
{
    type Target = BTreeMap<K, V>;
    type Kind = Self;
}

impl<K, V> TryFrom<BTreeMap<K, V>> for BTreeMap1<K, V> {
    type Error = EmptyError<BTreeMap<K, V>>;

    fn try_from(items: BTreeMap<K, V>) -> Result<Self, Self::Error> {
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

pub type Segment<'a, T> = segment::Segment<'a, T, BTreeMap<KeyFor<T>, ValueFor<T>>>;

impl<T, K, V> Segment<'_, T>
where
    T: ClosedBTreeMap<Key = K, Value = V> + SegmentedOver<Target = BTreeMap<K, V>>,
    K: Clone + Ord,
{
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.items.retain(self.range.retain_key_value_in_range(f))
    }

    pub fn insert_in_range(&mut self, key: K, value: V) -> Result<Option<V>, (K, V)> {
        if self.range.contains(&key) {
            Ok(self.items.insert(key, value))
        }
        else {
            Err((key, value))
        }
    }

    pub fn append_in_range(&mut self, other: &mut BTreeMap<K, V>) {
        if let RelationalRange::NonEmpty { ref start, ref end } = self.range {
            let low = other;
            let mut middle = low.split_off(start);
            let mut high = middle.split_off(end);
            self.items.append(&mut middle);
            if let Some(first) = high.remove(end) {
                self.items.insert(end.clone(), first);
            }
            low.append(&mut high);
        }
    }

    // It is especially important here to query `K` and not another related type `Q`, even if `K:
    // Borrow<Q>`. A type `Q` can implement `Ord` differently than `K`, which can remove items
    // beyond the range of the segment. This is not great, but for non-empty collections this is
    // unsound!
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.range.contains(key) {
            self.items.remove(key)
        }
        else {
            None
        }
    }

    pub fn clear(&mut self) {
        if let Some(range) = self.range.clone().try_into_range_inclusive() {
            self.items.retain(|key, _| !range.contains(key));
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if self.range.contains(key) {
            self.items.get(key)
        }
        else {
            None
        }
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        match self.range {
            RelationalRange::Empty => None,
            RelationalRange::NonEmpty { ref start, .. } => {
                self.items.get(start).map(|first| (start, first))
            },
        }
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        match self.range {
            RelationalRange::Empty => None,
            RelationalRange::NonEmpty { ref end, .. } => {
                self.items.get(end).map(|last| (end, last))
            },
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.range.contains(key) && self.items.contains_key(key)
    }
}

impl<T, K, V> Segmentation for Segment<'_, T>
where
    T: ClosedBTreeMap<Key = K, Value = V> + SegmentedOver<Target = BTreeMap<K, V>>,
    K: Clone + Ord,
{
    fn tail(&mut self) -> Segment<'_, T> {
        match self.range.clone().try_into_range_inclusive() {
            Some(range) => match BTreeMap::range(self.items, range.clone()).nth(1) {
                Some((start, _)) => Segment::unchecked(
                    self.items,
                    RelationalRange::unchecked(start.clone(), range.end().clone()),
                ),
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }

    fn rtail(&mut self) -> Segment<'_, T> {
        match self.range.clone().try_into_range_inclusive() {
            Some(range) => match BTreeMap::range(self.items, range.clone()).rev().nth(1) {
                Some((end, _)) => Segment::unchecked(
                    self.items,
                    RelationalRange::unchecked(range.start().clone(), end.clone()),
                ),
                _ => Segment::empty(self.items),
            },
            _ => Segment::empty(self.items),
        }
    }
}

impl<T, K, V, R> SegmentedBy<R> for Segment<'_, T>
where
    RelationalRange<K>: Intersect<R, Output = RelationalRange<K>>,
    T: ClosedBTreeMap<Key = K, Value = V> + SegmentedBy<R> + SegmentedOver<Target = BTreeMap<K, V>>,
    K: Clone + Ord,
    R: RangeBounds<K>,
{
    fn segment(&mut self, range: R) -> Segment<'_, T> {
        Segment::intersect(self.items, &range::ordered_range_bounds(range))
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
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::btree_map1::harness::{self, terminals1, VALUE};
    use crate::btree_map1::BTreeMap1;
    use crate::harness::KeyValueRef;
    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::Segmentation;
    #[cfg(feature = "serde")]
    use crate::{
        btree_map1::harness::xs1,
        serde::{self, harness::map},
    };

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
    #[case::absent_in_range(4, 4, Ok(None))]
    #[case::present_in_range(4, 9, Ok(Some(VALUE)))]
    #[case::out_of_range(4, 0, Err((0, VALUE)))]
    #[case::out_of_range(4, 1, Err((1, VALUE)))]
    fn insert_into_btree_map1_segment_range_from_then_output_eq(
        #[from(terminals1)] mut xs1: BTreeMap1<u8, char>,
        #[case] from: u8,
        #[case] key: u8,
        #[case] expected: Result<Option<char>, (u8, char)>,
    ) {
        let mut segment = xs1.segment(from..);
        assert_eq!(segment.insert_in_range(key, VALUE), expected);
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

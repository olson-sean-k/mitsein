//! A non-empty [`IndexMap`].
//!
//! [`IndexMap`]: indexmap::map

#![cfg(feature = "indexmap")]
#![cfg_attr(docsrs, doc(cfg(feature = "indexmap")))]

use alloc::boxed::Box;
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::hash::{BuildHasher, Hash};
use core::iter::{FusedIterator, Skip, Take};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, RangeBounds};
use indexmap::Equivalent;
use indexmap::map::{self as index_map, IndexMap, Slice, VacantEntry};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "std")]
use std::hash::RandomState;
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

#[cfg(feature = "std")]
use crate::array1::Array1;
use crate::except::{self, ByKey, Exception, KeyNotFoundError};
use crate::hash::UnsafeHash;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{self, NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, IndexRange, Project, RangeError};
use crate::segment::{self, ByRange, ByTail, Segmentation};
use crate::take;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type KeyFor<T> = <T as ClosedIndexMap>::Key;
type ValueFor<T> = <T as ClosedIndexMap>::Value;
type EntryFor<T> = (KeyFor<T>, ValueFor<T>);
type StateFor<T> = <T as ClosedIndexMap>::State;

pub trait ClosedIndexMap {
    type Key;
    type Value;
    type State;

    fn as_index_map(&self) -> &IndexMap<Self::Key, Self::Value, Self::State>;
}

impl<K, V, S> ClosedIndexMap for IndexMap<K, V, S> {
    type Key = K;
    type Value = V;
    type State = S;

    fn as_index_map(&self) -> &IndexMap<Self::Key, Self::Value, Self::State> {
        self
    }
}

impl<K, V, S, Q> ByKey<Q> for IndexMap<K, V, S>
where
    S: BuildHasher,
    Q: Equivalent<K> + Hash + ?Sized,
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

impl<K, V, S, R> ByRange<usize, R> for IndexMap<K, V, S>
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

impl<K, V, S> ByTail for IndexMap<K, V, S> {
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

impl<K, V, S> Exception for IndexMap<K, V, S> {
    type Kind = Self;
    type Target = Self;
}

impl<K, V, S> Extend1<(K, V)> for IndexMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend_non_empty<I>(mut self, items: I) -> IndexMap1<K, V, S>
    where
        I: IntoIterator1<Item = (K, V)>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { IndexMap1::from_index_map_unchecked(self) }
    }
}

unsafe impl<K, V, S> MaybeEmpty for IndexMap<K, V, S> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

impl<K, V, S> Segmentation for IndexMap<K, V, S> {
    type Kind = Self;
    type Target = Self;
}

pub type ManyEntry<'a, K, V> = index_map::OccupiedEntry<'a, K, V>;

#[derive(Debug)]
#[repr(transparent)]
pub struct OnlyEntry<'a, K, V> {
    entry: index_map::OccupiedEntry<'a, K, V>,
}

// TODO: Implement additional operations.
impl<'a, K, V> OnlyEntry<'a, K, V> {
    fn from_occupied_entry(entry: index_map::OccupiedEntry<'a, K, V>) -> Self {
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

    pub fn index(&self) -> usize {
        self.entry.index()
    }
}

pub type OccupiedEntry<'a, K, V> = Cardinality<OnlyEntry<'a, K, V>, ManyEntry<'a, K, V>>;

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn into_mut(self) -> &'a mut V {
        match self {
            OccupiedEntry::Many(many) => many.into_mut(),
            OccupiedEntry::One(only) => only.into_mut(),
        }
    }

    pub fn move_index(self, to: usize) {
        match self {
            OccupiedEntry::Many(many) => many.move_index(to),
            OccupiedEntry::One(only) => only.entry.move_index(to),
        }
    }

    pub fn swap_indices(self, other: usize) {
        match self {
            OccupiedEntry::Many(many) => many.swap_indices(other),
            OccupiedEntry::One(only) => only.entry.swap_indices(other),
        }
    }

    pub fn shift_remove_entry_or_get_only(self) -> OrOnlyEntry<'a, (K, V), K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.shift_remove_entry()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn swap_remove_entry_or_get_only(self) -> OrOnlyEntry<'a, (K, V), K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.swap_remove_entry()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn shift_remove_or_get_only(self) -> OrOnlyEntry<'a, V, K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.shift_remove()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn swap_remove_or_get_only(self) -> OrOnlyEntry<'a, V, K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.swap_remove()),
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

    pub fn index(&self) -> usize {
        match self {
            OccupiedEntry::Many(many) => many.index(),
            OccupiedEntry::One(only) => only.index(),
        }
    }
}

impl<'a, K, V> From<ManyEntry<'a, K, V>> for OccupiedEntry<'a, K, V> {
    fn from(many: ManyEntry<'a, K, V>) -> Self {
        OccupiedEntry::Many(many)
    }
}

impl<'a, K, V> From<OnlyEntry<'a, K, V>> for OccupiedEntry<'a, K, V> {
    fn from(only: OnlyEntry<'a, K, V>) -> Self {
        OccupiedEntry::One(only)
    }
}

#[derive(Debug)]
pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V> {
    /// # Safety
    ///
    /// The `IndexMap1` from which `entry` has been obtained must have a non-empty cardinality of
    /// many (must have **more than one** item).
    unsafe fn from_entry_many(entry: index_map::Entry<'a, K, V>) -> Self {
        match entry {
            index_map::Entry::Vacant(vacant) => Entry::Vacant(vacant),
            index_map::Entry::Occupied(occupied) => Entry::Occupied(occupied.into()),
        }
    }

    fn from_entry_only(entry: index_map::Entry<'a, K, V>) -> Self {
        match entry {
            index_map::Entry::Vacant(vacant) => Entry::Vacant(vacant),
            index_map::Entry::Occupied(occupied) => {
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

    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Vacant(vacant) => vacant.insert_entry(value).into(),
            Entry::Occupied(mut occupied) => {
                occupied.insert(value);
                occupied
            },
        }
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

    pub fn index(&self) -> usize {
        match self {
            Entry::Vacant(vacant) => vacant.index(),
            Entry::Occupied(occupied) => occupied.index(),
        }
    }
}

impl<'a, K, V> From<IndexedOnlyEntry<'a, K, V>> for OnlyEntry<'a, K, V> {
    fn from(entry: IndexedOnlyEntry<'a, K, V>) -> Self {
        OnlyEntry {
            entry: entry.entry.into(),
        }
    }
}

impl<'a, K, V> From<OnlyEntry<'a, K, V>> for Entry<'a, K, V> {
    fn from(only: OnlyEntry<'a, K, V>) -> Self {
        Entry::Occupied(only.into())
    }
}

pub type OrOnlyEntry<'a, T, K, V> = Result<T, OnlyEntry<'a, K, V>>;

pub trait OrOnlyEntryExt<'a, K, V> {
    fn get(&self) -> &V;

    fn get_mut(&mut self) -> &mut V;
}

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, V, K, V> {
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

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, (K, V), K, V> {
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

pub type IndexedManyEntry<'a, K, V> = index_map::IndexedEntry<'a, K, V>;

#[derive(Debug)]
#[repr(transparent)]
pub struct IndexedOnlyEntry<'a, K, V> {
    entry: index_map::IndexedEntry<'a, K, V>,
}

// TODO: Implement additional operations.
impl<'a, K, V> IndexedOnlyEntry<'a, K, V> {
    fn from_indexed_entry(entry: index_map::IndexedEntry<'a, K, V>) -> Self {
        IndexedOnlyEntry { entry }
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

    pub fn index(&self) -> usize {
        self.entry.index()
    }
}

impl<'a, K, V> From<OnlyEntry<'a, K, V>> for IndexedOnlyEntry<'a, K, V> {
    fn from(entry: OnlyEntry<'a, K, V>) -> Self {
        IndexedOnlyEntry {
            entry: entry.entry.into(),
        }
    }
}

pub type IndexedEntry<'a, K, V> =
    Cardinality<IndexedOnlyEntry<'a, K, V>, IndexedManyEntry<'a, K, V>>;

impl<'a, K, V> IndexedEntry<'a, K, V> {
    pub fn into_mut(self) -> &'a mut V {
        match self {
            IndexedEntry::Many(many) => many.into_mut(),
            IndexedEntry::One(only) => only.into_mut(),
        }
    }

    pub fn shift_remove_entry_or_get_only(self) -> OrIndexedOnlyEntry<'a, (K, V), K, V> {
        match self {
            IndexedEntry::Many(many) => Ok(many.shift_remove_entry()),
            IndexedEntry::One(only) => Err(only),
        }
    }

    pub fn swap_remove_entry_or_get_only(self) -> OrIndexedOnlyEntry<'a, (K, V), K, V> {
        match self {
            IndexedEntry::Many(many) => Ok(many.swap_remove_entry()),
            IndexedEntry::One(only) => Err(only),
        }
    }

    pub fn shift_remove_or_get_only(self) -> OrIndexedOnlyEntry<'a, V, K, V> {
        match self {
            IndexedEntry::Many(many) => Ok(many.shift_remove()),
            IndexedEntry::One(only) => Err(only),
        }
    }

    pub fn swap_remove_or_get_only(self) -> OrIndexedOnlyEntry<'a, V, K, V> {
        match self {
            IndexedEntry::Many(many) => Ok(many.swap_remove()),
            IndexedEntry::One(only) => Err(only),
        }
    }

    pub fn insert(&mut self, value: V) -> V {
        match self {
            IndexedEntry::Many(many) => many.insert(value),
            IndexedEntry::One(only) => only.insert(value),
        }
    }

    pub fn get(&self) -> &V {
        match self {
            IndexedEntry::Many(many) => many.get(),
            IndexedEntry::One(only) => only.get(),
        }
    }

    pub fn get_mut(&mut self) -> &mut V {
        match self {
            IndexedEntry::Many(many) => many.get_mut(),
            IndexedEntry::One(only) => only.get_mut(),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            IndexedEntry::Many(many) => many.key(),
            IndexedEntry::One(only) => only.key(),
        }
    }

    pub fn index(&self) -> usize {
        match self {
            IndexedEntry::Many(many) => many.index(),
            IndexedEntry::One(only) => only.index(),
        }
    }
}

impl<'a, K, V> From<IndexedManyEntry<'a, K, V>> for IndexedEntry<'a, K, V> {
    fn from(many: IndexedManyEntry<'a, K, V>) -> Self {
        IndexedEntry::Many(many)
    }
}

impl<'a, K, V> From<IndexedOnlyEntry<'a, K, V>> for IndexedEntry<'a, K, V> {
    fn from(one: IndexedOnlyEntry<'a, K, V>) -> Self {
        IndexedEntry::One(one)
    }
}

pub type OrIndexedOnlyEntry<'a, T, K, V> = Result<T, IndexedOnlyEntry<'a, K, V>>;

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrIndexedOnlyEntry<'a, V, K, V> {
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

type TakeIfMany<'a, K, V, S, U, N = ()> = take::TakeIfMany<'a, IndexMap<K, V, S>, U, N>;

pub type PopIfMany<'a, T> = TakeIfMany<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, EntryFor<T>>;

pub type RemoveIfMany<'a, 'q, T, Q> =
    TakeIfMany<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, Option<ValueFor<T>>, &'q Q>;

pub type RemoveEntryIfMany<'a, 'q, T, Q> =
    TakeIfMany<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, Option<EntryFor<T>>, &'q Q>;

impl<'a, K, V, S, U, N> TakeIfMany<'a, K, V, S, U, N>
where
    S: BuildHasher,
{
    pub fn or_get_only(self) -> Result<U, IndexedOnlyEntry<'a, K, V>> {
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

impl<'a, K, V, S, U, Q> TakeIfMany<'a, K, V, S, Option<U>, &'_ Q>
where
    S: BuildHasher,
    Q: Equivalent<K> + Hash + ?Sized,
{
    pub fn or_get(self) -> Option<Result<U, OnlyEntry<'a, K, V>>> {
        self.try_take_or_else(|items, query| {
            items
                .items
                .contains_key(query)
                .then(|| items.first_entry_as_only().into())
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

pub type Slice1<K, V> = NonEmpty<Slice<K, V>>;

// TODO: At time of writing, `const` functions are not supported in traits, so
//       `FromMaybeEmpty::from_maybe_empty_unchecked` cannot be used to construct a `Slice1` yet.
//       Use that function instead of `mem::transmute` when possible.
impl<K, V> Slice1<K, V> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice [`Slice::new()`][`Slice::new`].
    pub const unsafe fn from_slice_unchecked(items: &Slice<K, V>) -> &Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `Slice<K, V>` and
        //         `Slice1<K, V>` are the same.
        unsafe { mem::transmute::<&'_ Slice<K, V>, &'_ Slice1<K, V>>(items) }
    }

    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice [`Slice::new()`][`Slice::new`].
    pub const unsafe fn from_mut_slice_unchecked(items: &mut Slice<K, V>) -> &mut Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `Slice<K, V>` and
        //         `Slice1<K, V>` are the same.
        unsafe { mem::transmute::<&'_ mut Slice<K, V>, &'_ mut Slice1<K, V>>(items) }
    }

    pub fn split_first(&self) -> ((&K, &V), &Slice<K, V>) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.split_first().unwrap_maybe_unchecked() }
    }

    pub fn split_last(&self) -> ((&K, &V), &Slice<K, V>) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.split_last().unwrap_maybe_unchecked() }
    }

    pub fn first(&self) -> (&K, &V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> (&K, &V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<index_map::Iter<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.len()) }
    }
}

impl<K, V> Deref for Slice1<K, V> {
    type Target = Slice<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

#[cfg(feature = "std")]
pub type IndexMap1<K, V, S = RandomState> = NonEmpty<IndexMap<K, V, S>>;

#[cfg(not(feature = "std"))]
pub type IndexMap1<K, V, S> = NonEmpty<IndexMap<K, V, S>>;

impl<K, V, S> IndexMap1<K, V, S> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`IndexMap::new()`][`IndexMap::new`].
    ///
    /// [`IndexMap::new`]: index_map::IndexMap::new
    pub unsafe fn from_index_map_unchecked(items: IndexMap<K, V, S>) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    pub fn try_from_ref(
        items: &IndexMap<K, V, S>,
    ) -> Result<&'_ Self, EmptyError<&'_ IndexMap<K, V, S>>> {
        items.try_into()
    }

    pub fn try_from_mut(
        items: &mut IndexMap<K, V, S>,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut IndexMap<K, V, S>>> {
        items.try_into()
    }

    pub fn into_index_map(self) -> IndexMap<K, V, S> {
        self.items
    }

    pub fn into_boxed_slice1(self) -> Box<Slice1<K, V>> {
        let items = Box::into_raw(self.items.into_boxed_slice());
        // SAFETY: This cast is safe, because `Slice<K, V>` and `Slice1<K, V>` have the same
        //         representation (`Slice1<K, V>` is `repr(transparent)`). Moreover, the allocator
        //         only requires that the memory location and layout are the same when
        //         deallocating, so dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut Slice1<K, V>) }
    }

    pub fn into_keys1(self) -> Iterator1<index_map::IntoKeys<K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_keys()) }
    }

    pub fn into_values1(self) -> Iterator1<index_map::IntoValues<K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.into_values()) }
    }

    pub fn sorted_by<F>(self, f: F) -> Iterator1<index_map::IntoIter<K, V>>
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.sorted_by(f)) }
    }

    pub fn sorted_unstable_by<F>(self, f: F) -> Iterator1<index_map::IntoIter<K, V>>
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.sorted_unstable_by(f)) }
    }

    pub fn split_off_tail(&mut self) -> IndexMap<K, V, S>
    where
        S: Clone,
    {
        self.items.split_off(1)
    }

    pub fn reverse(&mut self) {
        self.items.reverse()
    }

    pub fn sort_keys(&mut self)
    where
        K: Ord,
    {
        self.items.sort_keys()
    }

    pub fn sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.items.sort_unstable_keys()
    }

    pub fn sort_by<F>(&mut self, f: F)
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        self.items.sort_by(f)
    }

    pub fn sort_unstable_by<F>(&mut self, f: F)
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        self.items.sort_unstable_by(f)
    }

    pub fn sort_by_cached_key<T, F>(&mut self, f: F)
    where
        T: Ord,
        F: FnMut(&K, &V) -> T,
    {
        self.items.sort_by_cached_key(f)
    }

    pub fn move_index(&mut self, from: usize, to: usize) {
        self.items.move_index(from, to)
    }

    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.items.swap_indices(a, b)
    }

    pub fn get_index(&self, index: usize) -> Option<(&'_ K, &'_ V)> {
        self.items.get_index(index)
    }

    pub fn get_index_mut(&mut self, index: usize) -> Option<(&'_ K, &'_ mut V)> {
        self.items.get_index_mut(index)
    }

    pub fn get_index_entry(&mut self, index: usize) -> Option<IndexedEntry<'_, K, V>> {
        self.items.get_index_entry(index).map(From::from)
    }

    pub fn get_range<R>(&self, range: R) -> Option<&'_ Slice<K, V>>
    where
        R: RangeBounds<usize>,
    {
        self.items.get_range(range)
    }

    pub fn get_range_mut<R>(&mut self, range: R) -> Option<&'_ mut Slice<K, V>>
    where
        R: RangeBounds<usize>,
    {
        self.items.get_range_mut(range)
    }

    pub fn binary_search_keys(&self, key: &K) -> Result<usize, usize>
    where
        K: Ord,
    {
        self.items.binary_search_keys(key)
    }

    pub fn binary_search_by<'a, F>(&'a self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a K, &'a V) -> Ordering,
    {
        self.items.binary_search_by(f)
    }

    pub fn binary_search_by_key<'a, Q, F>(&'a self, query: &Q, f: F) -> Result<usize, usize>
    where
        Q: Ord,
        F: FnMut(&'a K, &'a V) -> Q,
    {
        self.items.binary_search_by_key(query, f)
    }

    pub fn partition_point<F>(&self, f: F) -> usize
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.items.partition_point(f)
    }

    pub fn iter1(&self) -> Iterator1<index_map::Iter<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<index_map::IterMut<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub fn keys1(&self) -> Iterator1<index_map::Keys<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.keys()) }
    }

    pub fn values1(&self) -> Iterator1<index_map::Values<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.values()) }
    }

    pub fn values1_mut(&mut self) -> Iterator1<index_map::ValuesMut<'_, K, V>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.values_mut()) }
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn hasher(&self) -> &S {
        self.items.hasher()
    }

    pub const fn as_index_map(&self) -> &IndexMap<K, V, S> {
        &self.items
    }

    /// # Safety
    ///
    /// The [`IndexMap`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::index_map1::IndexMap1;
    ///
    /// let mut xs = IndexMap1::from([("a", 0i32), ("b", 1)]);
    /// // This block is unsound. The `&mut IndexMap` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_index_map().clear();
    /// }
    /// let x = xs.first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_index_map(&mut self) -> &mut IndexMap<K, V, S> {
        &mut self.items
    }

    pub fn as_slice1(&self) -> &'_ Slice1<K, V> {
        unsafe { Slice1::from_slice_unchecked(self.items.as_slice()) }
    }

    pub fn as_mut_slice1(&mut self) -> &'_ mut Slice1<K, V> {
        unsafe { Slice1::from_mut_slice_unchecked(self.items.as_mut_slice()) }
    }
}

impl<K, V, S> IndexMap1<K, V, S>
where
    S: BuildHasher,
{
    pub fn pop_if_many(&mut self) -> PopIfMany<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn shift_remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveIfMany<'a, 'q, Self, Q>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.shift_remove(query))
    }

    pub fn swap_remove_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveIfMany<'a, 'q, Self, Q>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| items.items.swap_remove(query))
    }

    pub fn shift_remove_entry_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveEntryIfMany<'a, 'q, Self, Q>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| {
            items.items.shift_remove_entry(query)
        })
    }

    pub fn swap_remove_entry_if_many<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveEntryIfMany<'a, 'q, Self, Q>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeIfMany::with(self, query, |items, query| {
            items.items.swap_remove_entry(query)
        })
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&V>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get(query)
    }

    pub fn get_mut<Q>(&mut self, query: &Q) -> Option<&mut V>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_mut(query)
    }

    pub fn get_key_value<Q>(&self, query: &Q) -> Option<(&K, &V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_key_value(query)
    }

    pub fn get_full<Q>(&self, query: &Q) -> Option<(usize, &K, &V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_full(query)
    }

    pub fn get_full_mut<Q>(&mut self, query: &Q) -> Option<(usize, &K, &mut V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_full_mut(query)
    }

    pub fn get_index_of<Q>(&self, query: &Q) -> Option<usize>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_index_of(query)
    }

    pub fn first(&self) -> (&K, &V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn first_mut(&mut self) -> (&K, &mut V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first_mut().unwrap_maybe_unchecked() }
    }

    pub fn first_entry(&mut self) -> IndexedEntry<'_, K, V> {
        self.as_cardinality_items_mut()
            // SAFETY: `self` must be non-empty.
            .map(|items| unsafe { items.first_entry().unwrap_maybe_unchecked() })
            .map_one(IndexedOnlyEntry::from_indexed_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    fn first_entry_as_only(&mut self) -> IndexedOnlyEntry<'_, K, V> {
        // SAFETY: `self` must be non-empty.
        IndexedOnlyEntry::from_indexed_entry(unsafe {
            self.items.first_entry().unwrap_maybe_unchecked()
        })
    }

    pub fn last(&self) -> (&K, &V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn last_mut(&mut self) -> (&K, &mut V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last_mut().unwrap_maybe_unchecked() }
    }

    pub fn last_entry(&mut self) -> IndexedEntry<'_, K, V> {
        self.as_cardinality_items_mut()
            // SAFETY: `self` must be non-empty.
            .map(|items| unsafe { items.last_entry().unwrap_maybe_unchecked() })
            .map_one(IndexedOnlyEntry::from_indexed_entry)
            .map_one(From::from)
            .map_many(From::from)
    }

    pub fn contains_key<Q>(&self, query: &Q) -> bool
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.contains_key(query)
    }
}

impl<K, V, S> IndexMap1<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn from_one(item: (K, V)) -> Self
    where
        S: Default,
    {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_hasher(item: (K, V), hasher: S) -> Self {
        IndexMap1::from_iter1_with_hasher([item], hasher)
    }

    pub fn from_iter1_with_hasher<I>(items: I, hasher: S) -> Self
    where
        I: IntoIterator1<Item = (K, V)>,
    {
        let items = {
            let mut xs = IndexMap::with_hasher(hasher);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { IndexMap1::from_index_map_unchecked(items) }
    }

    pub fn from_head_and_tail<I>(head: (K, V), tail: I) -> Self
    where
        S: Default,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_rtail_and_head<I>(tail: I, head: (K, V)) -> Self
    where
        S: Default,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::rtail_and_head(tail, head).collect1()
    }

    pub fn append<SR>(&mut self, items: &mut IndexMap<K, V, SR>) {
        self.items.append(items)
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.as_cardinality_items_mut() {
            Cardinality::One(items) => Entry::from_entry_only(items.entry(key)),
            // SAFETY: The `items` method returns the correct non-empty cardinality based on the
            //         `MaybeEmpty` implementation.
            Cardinality::Many(items) => unsafe { Entry::from_entry_many(items.entry(key)) },
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.items.insert(key, value)
    }

    pub fn insert_full(&mut self, key: K, value: V) -> (usize, Option<V>) {
        self.items.insert_full(key, value)
    }

    pub fn insert_sorted(&mut self, key: K, value: V) -> (usize, Option<V>)
    where
        K: Ord,
    {
        self.items.insert_sorted(key, value)
    }

    pub fn insert_before(&mut self, index: usize, key: K, value: V) -> (usize, Option<V>) {
        self.items.insert_before(index, key, value)
    }

    pub fn shift_insert(&mut self, index: usize, key: K, value: V) -> Option<V> {
        self.items.shift_insert(index, key, value)
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IndexMap1<K, V, S> {
    pub fn par_sorted_by<F>(self, f: F) -> ParallelIterator1<index_map::rayon::IntoParIter<K, V>>
    where
        K: Send,
        V: Send,
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_sorted_by(f)) }
    }

    pub fn par_sorted_unstable_by<F>(
        self,
        f: F,
    ) -> ParallelIterator1<index_map::rayon::IntoParIter<K, V>>
    where
        K: Send,
        V: Send,
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_sorted_unstable_by(f)) }
    }

    pub fn par_sort_keys(&mut self)
    where
        K: Ord + Send,
        V: Send,
    {
        self.items.par_sort_keys()
    }

    pub fn par_sort_unstable_keys(&mut self)
    where
        K: Ord + Send,
        V: Send,
    {
        self.items.par_sort_unstable_keys()
    }

    pub fn par_sort_by<F>(&mut self, f: F)
    where
        K: Send,
        V: Send,
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.items.par_sort_by(f)
    }

    pub fn par_sort_unstable_by<F>(&mut self, f: F)
    where
        K: Send,
        V: Send,
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.items.par_sort_unstable_by(f)
    }

    pub fn par_sort_by_cached_key<T, F>(&mut self, f: F)
    where
        K: Send,
        V: Send,
        T: Ord + Send,
        F: Fn(&K, &V) -> T + Sync,
    {
        self.items.par_sort_by_cached_key(f)
    }

    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        K: Sync,
        V: Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }

    pub fn par_iter1_mut(
        &mut self,
    ) -> ParallelIterator1<<&'_ mut Self as IntoParallelIterator>::Iter>
    where
        K: Send + Sync,
        V: Send,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter_mut()) }
    }

    pub fn par_keys1(&self) -> ParallelIterator1<index_map::rayon::ParKeys<'_, K, V>>
    where
        K: Sync,
        V: Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_keys()) }
    }

    pub fn par_values1(&self) -> ParallelIterator1<index_map::rayon::ParValues<'_, K, V>>
    where
        K: Sync,
        V: Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_values()) }
    }

    pub fn par_values1_mut(&mut self) -> ParallelIterator1<index_map::rayon::ParValuesMut<'_, K, V>>
    where
        K: Send,
        V: Send,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_values_mut()) }
    }

    pub fn par_eq<R>(&self, other: &R) -> bool
    where
        K: Eq + Hash + Sync,
        V: PartialEq<R::Value> + Sync,
        S: BuildHasher,
        R: ClosedIndexMap<Key = K>,
        R::Value: Sync,
        R::State: BuildHasher + Sync,
    {
        self.items.par_eq(other.as_index_map())
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, K, V, S> Arbitrary<'a> for IndexMap1<K, V, S>
where
    K: Arbitrary<'a> + Eq + Hash,
    V: Arbitrary<'a>,
    S: BuildHasher + Default,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(
            <(K, V)>::arbitrary(unstructured),
            unstructured.arbitrary_iter()?,
        )
        .collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        let k = <K>::size_hint(depth).0;
        let v = <V>::size_hint(depth).0;
        (k.saturating_add(v), None)
    }
}

// TODO: Support isomorphic key types (via an `UnsafeHashIsomorph` trait).
impl<K, V, S> ByKey<K> for IndexMap1<K, V, S>
where
    K: UnsafeHash,
    S: BuildHasher,
{
    fn except<'a>(
        &'a mut self,
        key: &'a K,
    ) -> Result<Except<'a, Self, K>, KeyNotFoundError<&'a K>> {
        self.contains_key(key)
            .then_some(Except::unchecked(&mut self.items, key))
            .ok_or_else(|| KeyNotFoundError::from_key(key))
    }
}

impl<K, V, S, R> ByRange<usize, R> for IndexMap1<K, V, S>
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

impl<K, V, S> ByTail for IndexMap1<K, V, S> {
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self> {
        self.items.tail().rekind()
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        self.items.rtail().rekind()
    }
}

impl<K, V, S> ClosedIndexMap for IndexMap1<K, V, S> {
    type Key = K;
    type Value = V;
    type State = S;

    fn as_index_map(&self) -> &IndexMap<Self::Key, Self::Value, Self::State> {
        self.as_ref()
    }
}

impl<K, V, S> Debug for IndexMap1<K, V, S>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<K, V, S> Exception for IndexMap1<K, V, S> {
    type Kind = Self;
    type Target = IndexMap<K, V, S>;
}

impl<K, V, S> Extend<(K, V)> for IndexMap1<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        self.items.extend(extension)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<K, V, const N: usize> From<[(K, V); N]> for IndexMap1<K, V>
where
    [(K, V); N]: Array1,
    K: Eq + Hash,
{
    fn from(items: [(K, V); N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { IndexMap1::from_index_map_unchecked(IndexMap::from(items)) }
    }
}

impl<K, V, S> From<IndexMap1<K, V, S>> for IndexMap<K, V, S> {
    fn from(items: IndexMap1<K, V, S>) -> Self {
        items.items
    }
}

impl<K, V, S> FromIterator1<(K, V)> for IndexMap1<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = (K, V)>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { IndexMap1::from_index_map_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> FromParallelIterator1<(K, V)> for IndexMap1<K, V, S>
where
    K: Eq + Hash + Send,
    V: Send,
    S: BuildHasher + Default + Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = (K, V)>,
    {
        // SAFETY: `items` is non-empty.
        unsafe {
            IndexMap1::from_index_map_unchecked(items.into_par_iter1().into_par_iter().collect())
        }
    }
}

impl<K, V, S> IntoIterator for IndexMap1<K, V, S> {
    type Item = (K, V);
    type IntoIter = index_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a IndexMap1<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = index_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut IndexMap1<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = index_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<K, V, S> IntoIterator1 for IndexMap1<K, V, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<K, V, S> IntoIterator1 for &'_ IndexMap1<K, V, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<K, V, S> IntoIterator1 for &'_ mut IndexMap1<K, V, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IntoParallelIterator for IndexMap1<K, V, S>
where
    K: Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = <IndexMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V, S> IntoParallelIterator for &'a IndexMap1<K, V, S>
where
    K: Sync,
    V: Sync,
{
    type Item = (&'a K, &'a V);
    type Iter = <&'a IndexMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V, S> IntoParallelIterator for &'a mut IndexMap1<K, V, S>
where
    K: Send + Sync,
    V: Send,
{
    type Item = (&'a K, &'a mut V);
    type Iter = <&'a mut IndexMap<K, V> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IntoParallelIterator1 for IndexMap1<K, V, S>
where
    K: Send,
    V: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IntoParallelIterator1 for &'_ IndexMap1<K, V, S>
where
    K: Sync,
    V: Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IntoParallelIterator1 for &'_ mut IndexMap1<K, V, S>
where
    K: Send + Sync,
    V: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&mut self.items) }
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<K, V, S> JsonSchema for IndexMap1<K, V, S>
where
    K: JsonSchema,
    V: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        IndexMap::<K, V, S>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<IndexMap<K, V, S>>(
            schemars::NON_EMPTY_KEY_OBJECT,
            generator,
        )
    }

    fn inline_schema() -> bool {
        IndexMap::<K, V, S>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        IndexMap::<K, V, S>::schema_id()
    }
}

impl<K, V, S> Segmentation for IndexMap1<K, V, S> {
    type Kind = Self;
    type Target = IndexMap<K, V, S>;
}

impl<K, V, S> TryFrom<IndexMap<K, V, S>> for IndexMap1<K, V, S> {
    type Error = EmptyError<IndexMap<K, V, S>>;

    fn try_from(items: IndexMap<K, V, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, K, V, S> TryFrom<&'a IndexMap<K, V, S>> for &'a IndexMap1<K, V, S> {
    type Error = EmptyError<&'a IndexMap<K, V, S>>;

    fn try_from(items: &'a IndexMap<K, V, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, K, V, S> TryFrom<&'a mut IndexMap<K, V, S>> for &'a mut IndexMap1<K, V, S> {
    type Error = EmptyError<&'a mut IndexMap<K, V, S>>;

    fn try_from(items: &'a mut IndexMap<K, V, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

// Unfortunately, the type of the `ExtractIf` predicate `F` cannot be named in `Except::drain` and
// so prevents returning a complete type.
struct DrainExcept<'a, K, V, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    input: index_map::ExtractIf<'a, K, V, F>,
}

impl<K, V, F> Debug for DrainExcept<'_, K, V, F>
where
    K: Debug,
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
    F: FnMut(&K, &mut V) -> bool,
{
    fn drop(&mut self) {
        self.input.by_ref().for_each(|_| {});
    }
}

impl<K, V, F> FusedIterator for DrainExcept<'_, K, V, F> where F: FnMut(&K, &mut V) -> bool {}

impl<K, V, F> Iterator for DrainExcept<'_, K, V, F>
where
    F: FnMut(&K, &mut V) -> bool,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next()
    }
}

pub type Except<'a, T, Q> = except::Except<'a, T, IndexMap<KeyFor<T>, ValueFor<T>, StateFor<T>>, Q>;

impl<T, K, V, S, Q> Except<'_, T, Q>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + Exception<Target = IndexMap<K, V, S>>,
    Q: Equivalent<K> + Hash + ?Sized,
{
    pub fn drain(&mut self) -> impl '_ + Drop + Iterator<Item = (K, V)> {
        DrainExcept {
            input: self
                .items
                .extract_if(.., |key, _| !self.key.equivalent(key)),
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.items.retain(|key, value| {
            let is_retained = self.key.equivalent(key);
            is_retained || f(key, value)
        });
    }

    pub fn clear(&mut self) {
        self.retain(|_, _| false)
    }

    pub fn iter(&self) -> impl '_ + Clone + Iterator<Item = (&'_ K, &'_ V)> {
        self.items
            .iter()
            .filter(|&(key, _)| !self.key.equivalent(key))
    }
}

pub type Segment<'a, T> =
    segment::Segment<'a, T, IndexMap<KeyFor<T>, ValueFor<T>, StateFor<T>>, IndexRange>;

impl<T, K, V, S> Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + Segmentation<Target = IndexMap<K, V, S>>,
{
    pub fn truncate(&mut self, len: usize) {
        if let Some(range) = self.range.truncate_from_end(len) {
            self.items.drain(range);
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.items.retain(self.range.retain_key_value_from_end(f))
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

    pub fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        let index = self.range.project(index).ok()?;
        self.items
            .shift_remove_index(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
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

    pub fn iter(&self) -> Take<Skip<index_map::Iter<'_, K, V>>> {
        self.items.iter().skip(self.range.start()).take(self.len())
    }

    pub fn iter_mut(&mut self) -> Take<Skip<index_map::IterMut<'_, K, V>>> {
        let body = self.len();
        self.items.iter_mut().skip(self.range.start()).take(body)
    }
}

impl<T, K, V, S> Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + Segmentation<Target = IndexMap<K, V, S>>,
    S: BuildHasher,
{
    pub fn contains_key<Q>(&self, query: &Q) -> bool
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items
            .get_index_of(query)
            .is_some_and(|index| self.range.contains(index))
    }
}

impl<T, K, V, S, R> ByRange<usize, R> for Segment<'_, T>
where
    IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
    T: ClosedIndexMap<Key = K, Value = V, State = S> + Segmentation<Target = IndexMap<K, V, S>>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, T>, Self::Error> {
        self.project_and_intersect(range)
    }
}

impl<T, K, V, S> ByTail for Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + Segmentation<Target = IndexMap<K, V, S>>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, T> {
        self.project_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, T> {
        let n = self.len();
        self.project_rtail_range(n)
    }
}

#[cfg(all(test, feature = "std"))]
pub mod harness {
    use rstest::fixture;

    use crate::index_map1::IndexMap1;
    use crate::iter1::{self, FromIterator1};

    pub const VALUE: char = 'x';

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> IndexMap1<u8, char> {
        IndexMap1::from_iter1(iter1::harness::xs1(end).map(|x| (x, VALUE)))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec::Vec;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::except::ByKey;
    use crate::harness::KeyValueRef;
    use crate::index_map1::IndexMap1;
    use crate::index_map1::harness::{self, VALUE, xs1};
    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::segment::ByTail;
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::map};

    #[rstest]
    #[case(0, &[(1, VALUE), (2, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(1, &[(0, VALUE), (2, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(2, &[(0, VALUE), (1, VALUE), (3, VALUE), (4, VALUE)])]
    #[case(3, &[(0, VALUE), (1, VALUE), (2, VALUE), (4, VALUE)])]
    #[case(4, &[(0, VALUE), (1, VALUE), (2, VALUE), (3, VALUE)])]
    fn drain_except_of_index_map1_then_drained_eq(
        mut xs1: IndexMap1<u8, char>,
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
    fn clear_except_of_index_map1_then_index_map1_eq_key_value(
        mut xs1: IndexMap1<u8, char>,
        #[case] entry: (u8, char),
    ) {
        let (key, value) = entry;
        xs1.except(&key).unwrap().clear();
        assert_eq!(xs1, IndexMap1::<_, _>::from_one((key, value)));
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    fn iter_except_of_index_map1_then_iter_does_not_contain_key(
        mut xs1: IndexMap1<u8, char>,
        #[case] key: u8,
    ) {
        let xs = xs1.except(&key).unwrap();
        assert!(!xs.iter().any(|(&x, _)| x == key));
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_index_map1_then_index_map1_eq_head(#[case] mut xs1: IndexMap1<u8, char>) {
        xs1.tail().clear();
        assert_eq!(xs1, IndexMap1::<_, _>::from_one((0, VALUE)));
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_index_map1_then_index_map1_eq_tail(#[case] mut xs1: IndexMap1<u8, char>) {
        let tail = xs1.last().cloned();
        xs1.rtail().clear();
        assert_eq!(xs1, IndexMap1::<_, _>::from_one(tail));
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_index_map1_then_index_map1_eq_head_and_tail(
        #[case] mut xs1: IndexMap1<u8, char>,
    ) {
        let n = xs1.len().get();
        let head_and_tail = [(0, VALUE), xs1.last().cloned()];
        xs1.tail().rtail().clear();
        assert_eq!(
            xs1,
            IndexMap1::<_, _>::try_from_iter(if n > 1 {
                head_and_tail[..].iter().copied()
            }
            else {
                head_and_tail[..1].iter().copied()
            })
            .unwrap(),
        );
    }

    #[cfg(feature = "schemars")]
    #[rstest]
    fn index_map1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<IndexMap1<u8, char>>(
            schemars::NON_EMPTY_KEY_OBJECT,
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_index_map1_into_and_from_tokens_eq(
        xs1: IndexMap1<u8, char>,
        map: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, map)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_index_map1_from_empty_tokens_then_empty_error(
        #[with(0)] map: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<IndexMap1<u8, char>, Vec<_>>(map)
    }
}

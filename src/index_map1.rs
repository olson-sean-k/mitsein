//! A non-empty [`IndexMap`].
//!
//! [`IndexMap`]: indexmap::map

#![cfg(feature = "indexmap")]
#![cfg_attr(docsrs, doc(cfg(feature = "indexmap")))]

use alloc::boxed::Box;
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::hash::{BuildHasher, Hash};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, RangeBounds};
use indexmap::map::{self as index_map, IndexMap, Slice, VacantEntry};
use indexmap::Equivalent;
#[cfg(feature = "rayon")]
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "std")]
use std::hash::RandomState;

use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{self, NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::take;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};

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

impl<K, V, S> Ranged for IndexMap<K, V, S> {
    type Range = PositionalRange;

    fn range(&self) -> Self::Range {
        From::from(0..self.len())
    }

    fn tail(&self) -> Self::Range {
        From::from(1..self.len())
    }

    fn rtail(&self) -> Self::Range {
        From::from(0..self.len().saturating_sub(1))
    }
}

impl<K, V, S> Segmentation for IndexMap<K, V, S> {
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(self))
    }
}

impl<K, V, S, R> SegmentedBy<R> for IndexMap<K, V, S>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<K, V, S> SegmentedOver for IndexMap<K, V, S> {
    type Target = Self;
    type Kind = Self;
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

    pub fn index(&self) -> usize {
        match self {
            OccupiedEntry::Many(ref many) => many.index(),
            OccupiedEntry::One(ref only) => only.index(),
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

impl<'a, K, V> OrOnlyEntryExt<'a, K, V> for OrOnlyEntry<'a, (K, V), K, V> {
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
            IndexedEntry::Many(ref mut many) => many.insert(value),
            IndexedEntry::One(ref mut only) => only.insert(value),
        }
    }

    pub fn get(&self) -> &V {
        match self {
            IndexedEntry::Many(ref many) => many.get(),
            IndexedEntry::One(ref only) => only.get(),
        }
    }

    pub fn get_mut(&mut self) -> &mut V {
        match self {
            IndexedEntry::Many(ref mut many) => many.get_mut(),
            IndexedEntry::One(ref mut only) => only.get_mut(),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            IndexedEntry::Many(ref many) => many.key(),
            IndexedEntry::One(ref only) => only.key(),
        }
    }

    pub fn index(&self) -> usize {
        match self {
            IndexedEntry::Many(ref many) => many.index(),
            IndexedEntry::One(ref only) => only.index(),
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

type TakeOr<'a, K, V, S, U, N = ()> = take::TakeOr<'a, IndexMap<K, V, S>, U, N>;

pub type PopOr<'a, T> = TakeOr<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, EntryFor<T>>;

pub type RemoveOr<'a, 'q, T, Q> =
    TakeOr<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, Option<ValueFor<T>>, &'q Q>;

pub type RemoveEntryOr<'a, 'q, T, Q> =
    TakeOr<'a, KeyFor<T>, ValueFor<T>, StateFor<T>, Option<EntryFor<T>>, &'q Q>;

impl<'a, K, V, S, U, N> TakeOr<'a, K, V, S, U, N>
where
    S: BuildHasher,
{
    pub fn get_only(self) -> Result<U, IndexedOnlyEntry<'a, K, V>> {
        self.take_or_else(|items, _| items.first_entry_as_only())
    }

    pub fn replace_only(self, value: V) -> Result<U, V> {
        self.else_replace_only(move || value)
    }

    pub fn else_replace_only<F>(self, f: F) -> Result<U, V>
    where
        F: FnOnce() -> V,
    {
        self.take_or_else(move |items, _| mem::replace(items.first_entry().get_mut(), f()))
    }
}

impl<'a, K, V, S, U, Q> TakeOr<'a, K, V, S, Option<U>, &'_ Q>
where
    K: Borrow<Q> + Eq + Hash,
    S: BuildHasher,
    Q: Equivalent<K> + Hash + ?Sized,
{
    pub fn get(self) -> Option<Result<U, OnlyEntry<'a, K, V>>> {
        self.try_take_or_else(|items, query| {
            items
                .items
                .contains_key(query)
                .then(|| items.first_entry_as_only().into())
        })
    }

    pub fn replace(self, value: V) -> Option<Result<U, V>> {
        self.else_replace(move || value)
    }

    pub fn else_replace<F>(self, f: F) -> Option<Result<U, V>>
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
        mem::transmute::<&'_ Slice<K, V>, &'_ Slice1<K, V>>(items)
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
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
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

    pub fn split_off_tail(&mut self) -> IndexMap<K, V, S>
    where
        S: Clone,
    {
        self.items.split_off(1)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub const fn as_index_map(&self) -> &IndexMap<K, V, S> {
        &self.items
    }
}

impl<K, V, S> IndexMap1<K, V, S>
where
    S: BuildHasher,
{
    pub fn pop_or(&mut self) -> PopOr<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn shift_remove_or<'a, 'q, Q>(&'a mut self, query: &'q Q) -> RemoveOr<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| items.items.shift_remove(query))
    }

    pub fn swap_remove_or<'a, 'q, Q>(&'a mut self, query: &'q Q) -> RemoveOr<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| items.items.swap_remove(query))
    }

    pub fn shift_remove_entry_or<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveEntryOr<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| {
            items.items.shift_remove_entry(query)
        })
    }

    pub fn swap_remove_entry_or<'a, 'q, Q>(
        &'a mut self,
        query: &'q Q,
    ) -> RemoveEntryOr<'a, 'q, Self, Q>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| {
            items.items.swap_remove_entry(query)
        })
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get(query)
    }

    pub fn get_key_value<Q>(&self, query: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_key_value(query)
    }

    pub fn get_mut<Q>(&mut self, query: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items.get_mut(query)
    }

    pub fn first(&self) -> (&K, &V) {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
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

    pub fn last_entry(&mut self) -> IndexedEntry<'_, K, V> {
        self.as_cardinality_items_mut()
            // SAFETY: `self` must be non-empty.
            .map(|items| unsafe { items.last_entry().unwrap_maybe_unchecked() })
            .map_one(IndexedOnlyEntry::from_indexed_entry)
            .map_one(From::from)
            .map_many(From::from)
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

    pub fn contains_key<Q>(&self, query: &Q) -> bool
    where
        K: Borrow<Q> + Eq + Hash,
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

    pub fn from_head_and_tail<I>(head: (K, V), tail: I) -> Self
    where
        S: Default,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: (K, V)) -> Self
    where
        S: Default,
        I: IntoIterator<Item = (K, V)>,
    {
        iter1::tail_and_head(tail, head).collect1()
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

    pub fn get_index_entry(&mut self, index: usize) -> Option<IndexedEntry<'_, K, V>> {
        self.items.get_index_entry(index).map(From::from)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.items.insert(key, value)
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> IndexMap1<K, V, S> {
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ IndexMap<K, V> as IntoParallelIterator>::Iter>
    where
        K: Sync,
        V: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }

    pub fn par_iter1_mut(
        &mut self,
    ) -> ParallelIterator1<<&'_ mut IndexMap<K, V> as IntoParallelIterator>::Iter>
    where
        K: Send + Sync,
        V: Send,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter_mut()) }
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

impl<K, V, S> IntoIterator1 for IndexMap1<K, V, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
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

impl<K, V, S> Segmentation for IndexMap1<K, V, S> {
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(&self.items))
    }
}

impl<K, V, S, R> SegmentedBy<R> for IndexMap1<K, V, S>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<K, V, S> SegmentedOver for IndexMap1<K, V, S> {
    type Target = IndexMap<K, V, S>;
    type Kind = Self;
}

impl<K, V, S> TryFrom<IndexMap<K, V, S>> for IndexMap1<K, V, S> {
    type Error = IndexMap<K, V, S>;

    fn try_from(items: IndexMap<K, V, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

pub type Segment<'a, T> = segment::Segment<'a, T, IndexMap<KeyFor<T>, ValueFor<T>, StateFor<T>>>;

impl<T, K, V, S> Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + SegmentedOver<Target = IndexMap<K, V, S>>,
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
        let from = self.range.project(&from).expect_in_bounds();
        let to = self.range.project(&to).expect_in_bounds();
        self.items.move_index(from, to)
    }

    pub fn swap_indices(&mut self, a: usize, b: usize) {
        let a = self.range.project(&a).expect_in_bounds();
        let b = self.range.project(&b).expect_in_bounds();
        self.items.swap_indices(a, b)
    }

    pub fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        let index = self.range.project(&index).ok()?;
        self.items
            .shift_remove_index(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        let index = self.range.project(&index).ok()?;
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
}

impl<T, K, V, S> Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + SegmentedOver<Target = IndexMap<K, V, S>>,
    S: BuildHasher,
{
    pub fn contains_key<Q>(&self, query: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Equivalent<K> + Hash + ?Sized,
    {
        self.items
            .get_index_of(query)
            .is_some_and(|index| self.range.contains(index))
    }
}

impl<T, K, V, S> Segmentation for Segment<'_, T>
where
    T: ClosedIndexMap<Key = K, Value = V, State = S> + SegmentedOver<Target = IndexMap<K, V, S>>,
{
    fn tail(&mut self) -> Segment<'_, T> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> Segment<'_, T> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<T, K, V, S, R> SegmentedBy<R> for Segment<'_, T>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    T: ClosedIndexMap<Key = K, Value = V, State = S>
        + SegmentedBy<R>
        + SegmentedOver<Target = IndexMap<K, V, S>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, T> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::index_map1::IndexMap1;
    use crate::iter1::{self, FromIterator1};

    pub const VALUE: char = 'x';

    pub trait KeyValueRef {
        type Cloned;

        fn cloned(&self) -> Self::Cloned;
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a mut V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> IndexMap1<u8, char> {
        IndexMap1::from_iter1(iter1::harness::xs1(end).map(|x| (x, VALUE)))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::index_map1::harness::{self, KeyValueRef, VALUE};
    use crate::index_map1::IndexMap1;
    use crate::iter1::FromIterator1;
    use crate::Segmentation;
    #[cfg(feature = "serde")]
    use crate::{
        index_map1::harness::xs1,
        serde::{self, harness::map},
    };

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_index_map1_then_index_map1_eq_head(#[case] mut xs1: IndexMap1<u8, char>) {
        xs1.tail().clear();
        assert_eq!(xs1, IndexMap1::from_one((0, VALUE)));
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_index_map1_then_index_map1_eq_tail(#[case] mut xs1: IndexMap1<u8, char>) {
        let tail = xs1.last().cloned();
        xs1.rtail().clear();
        assert_eq!(xs1, IndexMap1::from_one(tail));
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
            IndexMap1::try_from_iter(if n > 1 {
                head_and_tail[..].iter().copied()
            }
            else {
                head_and_tail[..1].iter().copied()
            })
            .unwrap(),
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

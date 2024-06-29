#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_map::{self, BTreeMap, VacantEntry};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::mem;
use core::num::NonZeroUsize;

use crate::array1::Array1;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::{Arity, NonEmpty};

type BTreeMapArity<'a, K, V> = Arity<&'a mut BTreeMap<K, V>, &'a mut BTreeMap<K, V>>;

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
    pub(crate) fn from_occupied_entry(entry: btree_map::OccupiedEntry<'a, K, V>) -> Self {
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

pub type OccupiedEntry<'a, K, V> = Arity<OnlyEntry<'a, K, V>, ManyEntry<'a, K, V>>;

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

    pub fn remove_key_value_or_only(self) -> KeyValueOrOnly<'a, K, V> {
        match self {
            OccupiedEntry::Many(many) => Ok(many.remove_entry()),
            OccupiedEntry::One(only) => Err(only),
        }
    }

    pub fn remove_or_only(self) -> ValueOrOnly<'a, K, V> {
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
    fn from_entry_many(entry: btree_map::Entry<'a, K, V>) -> Self {
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
        // CLIPPY: This is a false positive. `Entry::or_insert_with` implements `Entry::or_default`
        //         here.
        #[allow(clippy::unwrap_or_default)]
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

pub type OrOnly<'a, T, K, V> = Result<T, OnlyEntry<'a, K, V>>;

pub type ValueOrOnly<'a, K, V> = OrOnly<'a, V, K, V>;

pub type KeyValueOrOnly<'a, K, V> = OrOnly<'a, (K, V), K, V>;

pub trait OrOnlyExt<'a, K, V>
where
    K: Ord,
{
    fn get(&self) -> &V;

    fn get_mut(&mut self) -> &mut V;
}

impl<'a, K, V> OrOnlyExt<'a, K, V> for ValueOrOnly<'a, K, V>
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

impl<'a, K, V> OrOnlyExt<'a, K, V> for KeyValueOrOnly<'a, K, V>
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

pub type BTreeMap1<K, V> = NonEmpty<BTreeMap<K, V>>;

impl<K, V> BTreeMap1<K, V> {
    /// # Safety
    pub const unsafe fn from_btree_map_unchecked(items: BTreeMap<K, V>) -> Self {
        BTreeMap1 { items }
    }

    pub fn from_one(key: K, value: V) -> Self
    where
        K: Ord,
    {
        iter1::from_one((key, value)).collect()
    }

    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        K: Ord,
        I: IntoIterator<Item = (K, V)>,
    {
        Iterator1::try_from_iter(items).map(BTreeMap1::from_iter1)
    }

    pub fn into_btree_map(self) -> BTreeMap<K, V> {
        self.items
    }

    pub fn into_keys1(self) -> Iterator1<btree_map::IntoKeys<K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.into_keys()) }
    }

    pub fn into_values1(self) -> Iterator1<btree_map::IntoValues<K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.into_values()) }
    }

    fn arity(&mut self) -> BTreeMapArity<'_, K, V> {
        match self.items.len() {
            0 => unreachable!(),
            1 => Arity::One(&mut self.items),
            _ => Arity::Many(&mut self.items),
        }
    }

    fn many_or_only<'a, T, F>(&'a mut self, f: F) -> Result<T, OnlyEntry<'a, K, V>>
    where
        K: Ord,
        F: FnOnce(&'a mut BTreeMap<K, V>) -> T,
    {
        match self.arity() {
            // SAFETY:
            Arity::One(one) => Err(OnlyEntry::from_occupied_entry(unsafe {
                one.first_entry().unwrap_unchecked()
            })),
            Arity::Many(many) => Ok(f(many)),
        }
    }

    fn many_or_get<'a, Q, T, F>(
        &'a mut self,
        query: &Q,
        f: F,
    ) -> Option<Result<T, OnlyEntry<'a, K, V>>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        F: FnOnce(&'a mut BTreeMap<K, V>) -> Option<T>,
    {
        let result = match self.arity() {
            // SAFETY:
            Arity::One(one) => Err(one.contains_key(query).then(|| {
                OnlyEntry::from_occupied_entry(unsafe { one.first_entry().unwrap_unchecked() })
            })),
            Arity::Many(many) => Ok(f(many)),
        };
        match result {
            Err(one) => one.map(Err),
            Ok(many) => many.map(Ok),
        }
    }

    pub fn split_off_first(&mut self) -> BTreeMap<K, V>
    where
        K: Clone + Ord,
    {
        match self.items.keys().nth(1).cloned() {
            Some(key) => self.items.split_off(&key),
            _ => BTreeMap::new(),
        }
    }

    pub fn split_off_last(&mut self) -> BTreeMap<K, V>
    where
        K: Clone + Ord,
    {
        let key = self.keys1().rev().first().clone();
        match self.arity() {
            Arity::One(_) => BTreeMap::new(),
            Arity::Many(items) => {
                let mut last = items.split_off(&key);
                mem::swap(items, &mut last);
                last
            },
        }
    }

    pub fn append<R>(&mut self, items: R)
    where
        K: Ord,
        R: Into<BTreeMap<K, V>>,
    {
        self.items.append(&mut items.into())
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V>
    where
        K: Ord,
    {
        match self.arity() {
            Arity::One(one) => Entry::from_entry_only(one.entry(key)),
            Arity::Many(many) => Entry::from_entry_many(many.entry(key)),
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V>
    where
        K: Ord,
    {
        self.items.insert(key, value)
    }

    pub fn pop_first_key_value_or_only(&mut self) -> KeyValueOrOnly<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_first().unwrap_unchecked() })
    }

    pub fn pop_first_or_only(&mut self) -> ValueOrOnly<'_, K, V>
    where
        K: Ord,
    {
        self.pop_first_key_value_or_only()
            .map(|(_key, value)| value)
    }

    pub fn pop_last_key_value_or_only(&mut self) -> KeyValueOrOnly<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_last().unwrap_unchecked() })
    }

    pub fn pop_last_or_only(&mut self) -> ValueOrOnly<'_, K, V>
    where
        K: Ord,
    {
        self.pop_last_key_value_or_only().map(|(_key, value)| value)
    }

    pub fn remove_key_value_or_only<'a, Q>(
        &'a mut self,
        query: &Q,
    ) -> Option<KeyValueOrOnly<'a, K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.many_or_get(query, move |items| items.remove_entry(query))
    }

    pub fn remove_or_only<'a, Q>(&'a mut self, query: &Q) -> Option<ValueOrOnly<'a, K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.many_or_get(query, move |items| items.remove(query))
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get(query)
    }

    pub fn get_mut<Q>(&mut self, query: &Q) -> Option<&mut V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get_mut(query)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub fn first_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY:
        unsafe { self.items.first_key_value().unwrap_unchecked() }
    }

    pub fn first_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY:
        match self.many_or_only(|items| unsafe { items.first_entry().unwrap_unchecked() }) {
            Ok(many) => many.into(),
            Err(only) => only.into(),
        }
    }

    pub fn last_key_value(&self) -> (&K, &V)
    where
        K: Ord,
    {
        // SAFETY:
        unsafe { self.items.last_key_value().unwrap_unchecked() }
    }

    pub fn last_entry(&mut self) -> OccupiedEntry<'_, K, V>
    where
        K: Ord,
    {
        // SAFETY:
        match self.many_or_only(|items| unsafe { items.last_entry().unwrap_unchecked() }) {
            Ok(many) => many.into(),
            Err(only) => only.into(),
        }
    }

    pub fn iter1(&self) -> Iterator1<btree_map::Iter<'_, K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<btree_map::IterMut<'_, K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub fn keys1(&self) -> Iterator1<btree_map::Keys<'_, K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.keys()) }
    }

    pub fn values1(&self) -> Iterator1<btree_map::Values<'_, K, V>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.values()) }
    }

    pub fn values1_mut(&mut self) -> Iterator1<btree_map::ValuesMut<'_, K, V>> {
        // SAFETY:
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
        // SAFETY:
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
        BTreeMap1 {
            //items: items.into_iter1().collect(),
            items: items.into_iter1().into_iter().collect(),
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

impl<K, V> IntoIterator1 for BTreeMap1<K, V> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<K, V> TryFrom<Serde<BTreeMap<K, V>>> for BTreeMap1<K, V> {
    type Error = EmptyError;

    fn try_from(serde: Serde<BTreeMap<K, V>>) -> Result<Self, Self::Error> {
        BTreeMap1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<K, V> TryFrom<BTreeMap<K, V>> for BTreeMap1<K, V> {
    type Error = BTreeMap<K, V>;

    fn try_from(items: BTreeMap<K, V>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY:
            _ => Ok(unsafe { BTreeMap1::from_btree_map_unchecked(items) }),
        }
    }
}

#[cfg(test)]
mod tests {}

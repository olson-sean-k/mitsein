#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::btree_set::{self, BTreeSet};
use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};

use crate::array1::Array1;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::{Arity, NonEmpty};

type BTreeSetArity<'a, T> = Arity<&'a mut BTreeSet<T>, &'a mut BTreeSet<T>>;

pub type BTreeSet1<T> = NonEmpty<BTreeSet<T>>;

impl<T> BTreeSet1<T> {
    pub(crate) fn from_btree_set_unchecked(items: BTreeSet<T>) -> Self {
        BTreeSet1 { items }
    }

    pub fn from_one(item: T) -> Self
    where
        T: Ord,
    {
        iter1::from_one(item).collect()
    }

    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        T: Ord,
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(BTreeSet1::from_iter1)
    }

    pub fn into_btree_set(self) -> BTreeSet<T> {
        self.items
    }

    fn arity(&mut self) -> BTreeSetArity<'_, T> {
        match self.items.len() {
            0 => unreachable!(),
            1 => Arity::One(&mut self.items),
            _ => Arity::Many(&mut self.items),
        }
    }

    fn many_or_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        T: Ord,
        F: FnOnce(&mut BTreeSet<T>) -> T,
    {
        match self.arity() {
            // SAFETY:
            Arity::One(one) => Err(unsafe { one.first().unwrap_unchecked() }),
            Arity::Many(many) => Ok(f(many)),
        }
    }

    fn many_or_get<Q, F>(&mut self, query: &Q, f: F) -> Option<Result<T, &T>>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        F: FnOnce(&mut BTreeSet<T>) -> Option<T>,
    {
        let result = match self.arity() {
            Arity::One(one) => Err(one.get(query)),
            Arity::Many(many) => Ok(f(many)),
        };
        match result {
            Err(one) => one.map(Err),
            Ok(many) => many.map(Ok),
        }
    }

    pub fn split_off_first(&mut self) -> BTreeSet<T>
    where
        T: Clone + Ord,
    {
        match self.items.iter().nth(1).cloned() {
            Some(item) => self.items.split_off(&item),
            _ => BTreeSet::new(),
        }
    }

    pub fn split_off_last(&mut self) -> BTreeSet<T>
    where
        T: Clone + Ord,
    {
        let item = self.iter1().rev().first().clone();
        match self.arity() {
            Arity::One(_) => BTreeSet::new(),
            Arity::Many(items) => {
                let mut last = items.split_off(&item);
                mem::swap(items, &mut last);
                last
            },
        }
    }

    pub fn append<R>(&mut self, items: R)
    where
        T: Ord,
        R: Into<BTreeSet<T>>,
    {
        self.items.append(&mut items.into())
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

    pub fn pop_first_or_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_first().unwrap_unchecked() })
    }

    pub fn pop_last_or_only(&mut self) -> Result<T, &T>
    where
        T: Ord,
    {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_last().unwrap_unchecked() })
    }

    pub fn remove_or_only<Q>(&mut self, query: &Q) -> Result<bool, &T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        match self
            .many_or_get(query, move |items| items.take(query))
            .transpose()
        {
            Ok(item) => Ok(item.is_some()),
            Err(only) => Err(only),
        }
    }

    pub fn take_or_only<Q>(&mut self, query: &Q) -> Option<Result<T, &T>>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.many_or_get(query, move |items| items.take(query))
    }

    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.get(query)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub fn first(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.first().unwrap_unchecked() }
    }

    pub fn last(&self) -> &T
    where
        T: Ord,
    {
        // SAFETY:
        unsafe { self.items.last().unwrap_unchecked() }
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
        R: AsRef<BTreeSet<T>>,
    {
        self.items.difference(other.as_ref())
    }

    pub fn symmetric_difference<'a, R>(
        &'a self,
        other: &'a R,
    ) -> btree_set::SymmetricDifference<'a, T>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.symmetric_difference(other.as_ref())
    }

    pub fn intersection<'a, R>(&'a self, other: &'a R) -> btree_set::Intersection<'a, T>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.intersection(other.as_ref())
    }

    pub fn union<'a, R>(&'a self, other: &'a R) -> Iterator1<btree_set::Union<'a, T>>
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        Iterator1::from_iter_unchecked(self.items.union(other.as_ref()))
    }

    pub fn iter1(&self) -> Iterator1<btree_set::Iter<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter())
    }

    pub fn is_disjoint<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_disjoint(other.as_ref())
    }

    pub fn is_subset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_subset(other.as_ref())
    }

    pub fn is_superset<R>(&self, other: &R) -> bool
    where
        T: Ord,
        R: AsRef<BTreeSet<T>>,
    {
        self.items.is_superset(other.as_ref())
    }

    pub fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items.contains(item)
    }

    pub fn as_btree_set(&self) -> &BTreeSet<T> {
        &self.items
    }
}

impl<R, T> BitAnd<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() & rhs.as_ref()
    }
}

impl<R, T> BitOr<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet1<T>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        BTreeSet1::from_btree_set_unchecked(self.as_btree_set() | rhs.as_ref())
    }
}

impl<R, T> BitXor<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() ^ rhs.as_ref()
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
        BTreeSet1::from_btree_set_unchecked(BTreeSet::from(items))
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
        BTreeSet1 {
            //items: items.into_iter1().collect(),
            items: items.into_iter1().into_iter().collect(),
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

impl<T> IntoIterator1 for BTreeSet1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        Iterator1::from_iter_unchecked(self.items)
    }
}

impl<R, T> Sub<&'_ R> for &'_ BTreeSet1<T>
where
    R: AsRef<BTreeSet<T>>,
    T: Clone + Ord,
{
    type Output = BTreeSet<T>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_btree_set() - rhs.as_ref()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<BTreeSet<T>>> for BTreeSet1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<BTreeSet<T>>) -> Result<Self, Self::Error> {
        BTreeSet1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T> TryFrom<BTreeSet<T>> for BTreeSet1<T> {
    type Error = BTreeSet<T>;

    fn try_from(items: BTreeSet<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            _ => Ok(BTreeSet1::from_btree_set_unchecked(items)),
        }
    }
}

#[cfg(test)]
mod tests {}

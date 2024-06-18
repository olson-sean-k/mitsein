#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, Cow};
use alloc::vec::{self, Splice, Vec};
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::slice;

use crate::boxed1::BoxedSlice1;
use crate::iter1::{FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{NonEmpty, NonZeroUsizeExt as _};

pub type Cow1<'a, T> = Cow<'a, Slice1<T>>;

pub type Vec1<T> = NonEmpty<Vec<T>>;

impl<T> Vec1<T> {
    pub(crate) fn from_vec_unchecked(items: Vec<T>) -> Self {
        Vec1 { items }
    }

    pub fn from_item(item: T) -> Self {
        Vec1::from_vec_unchecked(alloc::vec![item])
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Vec1::from_vec_unchecked(Some(head).into_iter().chain(tail).collect())
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Vec1::from_vec_unchecked(tail.into_iter().chain(Some(head)).collect())
    }

    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(Vec1::from_iter1)
    }

    pub fn from_item_with_capacity(item: T, capacity: usize) -> Self {
        let mut items = Vec::with_capacity(capacity);
        items.push(item);
        Vec1::from_vec_unchecked(items)
    }

    pub fn into_head_and_tail(mut self) -> (T, Vec<T>) {
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_tail_and_head(mut self) -> (Vec<T>, T) {
        // SAFETY:
        let head = unsafe { self.items.pop().unwrap_unchecked() };
        (self.items, head)
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items
    }

    pub fn into_boxed_slice1(self) -> BoxedSlice1<T> {
        BoxedSlice1::from_boxed_slice_unchecked(self.items.into_boxed_slice())
    }

    fn many_or_else<M, O>(&mut self, many: M, one: O) -> Result<T, &T>
    where
        M: FnOnce(&mut Vec<T>) -> T,
        O: FnOnce(&mut Vec<T>) -> &T,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
    }

    fn many_or_first<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut Vec<T>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, |items| unsafe { items.get_unchecked(0) })
    }

    fn many_or_index<F>(&mut self, index: usize, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut Vec<T>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, move |items| &items[index])
    }

    pub fn resize(&mut self, len: NonZeroUsize, fill: T)
    where
        T: Clone,
    {
        self.resize_with(len, move || fill.clone())
    }

    pub fn resize_with<F>(&mut self, len: NonZeroUsize, f: F)
    where
        F: FnMut() -> T,
    {
        self.items.resize_with(len.into(), f)
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn split_off(&mut self, at: NonZeroUsize) -> Vec<T> {
        self.items.split_off(at.into())
    }

    // NOTE: This is as similar to `Vec::clear` as `Vec1` can afford.
    pub fn split_off_first(&mut self) -> Vec<T> {
        self.split_off(NonZeroUsize::ONE)
    }

    pub fn append(&mut self, mut items: Self) {
        self.items.append(&mut items.items)
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn pop(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_first(|items| unsafe { items.pop().unwrap_unchecked() })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_index(index, move |items| items.remove(index))
    }

    pub fn swap_remove(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_index(index, move |items| items.swap_remove(index))
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.capacity()) }
    }

    pub fn first(&self) -> &T {
        // SAFETY:
        unsafe { self.items.first().unwrap_unchecked() }
    }

    pub fn first_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.first_mut().unwrap_unchecked() }
    }

    pub fn last(&self) -> &T {
        // SAFETY:
        unsafe { self.items.last().unwrap_unchecked() }
    }

    pub fn last_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.last_mut().unwrap_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<slice::Iter<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter())
    }

    pub fn iter1_mut(&mut self) -> Iterator1<slice::IterMut<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter_mut())
    }

    pub fn splice<R, I>(&mut self, range: R, replacement: I) -> Splice<'_, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator1<Item = T>,
    {
        self.items.splice(range, replacement.into_iter1())
    }

    pub fn as_vec(&self) -> &Vec<T> {
        &self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        Slice1::from_slice_unchecked(self.items.as_slice())
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        Slice1::from_mut_slice_unchecked(self.items.as_mut_slice())
    }

    pub fn as_ptr(&self) -> *const T {
        self.items.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.items.as_mut_ptr()
    }
}

impl<T> AsMut<[T]> for Vec1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T> AsMut<Slice1<T>> for Vec1<T> {
    fn as_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T> AsRef<[T]> for Vec1<T> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T> AsRef<Slice1<T>> for Vec1<T> {
    fn as_ref(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T> AsRef<Vec<T>> for Vec1<T> {
    fn as_ref(&self) -> &Vec<T> {
        &self.items
    }
}

impl<T> Borrow<[T]> for Vec1<T> {
    fn borrow(&self) -> &[T] {
        self.items.borrow()
    }
}

impl<T> Borrow<Slice1<T>> for Vec1<T> {
    fn borrow(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T> BorrowMut<[T]> for Vec1<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.items.borrow_mut()
    }
}

impl<T> BorrowMut<Slice1<T>> for Vec1<T> {
    fn borrow_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T> Debug for Vec1<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T> Deref for Vec1<T> {
    type Target = Slice1<T>;

    fn deref(&self) -> &Self::Target {
        self.as_slice1()
    }
}

impl<T> DerefMut for Vec1<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice1()
    }
}

impl<T> Extend<T> for Vec1<T> {
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<'a, T> From<&'a Slice1<T>> for Vec1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        Vec1::from_vec_unchecked(Vec::from(items.as_slice()))
    }
}

impl<T> From<BoxedSlice1<T>> for Vec1<T> {
    fn from(items: BoxedSlice1<T>) -> Self {
        Vec1::from_vec_unchecked(Vec::from(items.into_boxed_slice()))
    }
}

impl<T> From<Vec1<T>> for Vec<T> {
    fn from(items: Vec1<T>) -> Self {
        items.items
    }
}

impl<T> FromIterator1<T> for Vec1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        Vec1 {
            // TODO: This should work, but `rustc` incorrectly rejects a blanket implementation for
            //       `FromIterator` types. See the `iter1` module.
            //items: items.into_iter1().collect(),
            items: items.into_iter1().into_iter().collect(),
        }
    }
}

impl<T> Index<usize> for Vec1<T> {
    type Output = <Vec<T> as Index<usize>>::Output;

    fn index(&self, at: usize) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T> IndexMut<usize> for Vec1<T> {
    fn index_mut(&mut self, at: usize) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<T> IntoIterator for Vec1<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for Vec1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        Iterator1::from_iter_unchecked(self.items)
    }
}

impl<'a, T> TryFrom<&'a [T]> for Vec1<T>
where
    T: Clone,
{
    type Error = &'a [T];

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            _ => Ok(Vec1::from_vec_unchecked(Vec::from(items))),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<Vec<T>>> for Vec1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<Vec<T>>) -> Result<Self, Self::Error> {
        Vec1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T> TryFrom<Vec<T>> for Vec1<T> {
    type Error = Vec<T>;

    fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            _ => Ok(Vec1::from_vec_unchecked(items)),
        }
    }
}

#[macro_export]
macro_rules! vec1 {
    ($($item:expr $(,)?)+) => {
        $crate::vec1::Vec1::from_vec_unchecked(alloc::vec![$($item,)+])
    };
}
pub use vec1;

macro_rules! impl_from_array_for_vec1 {
    ($N:literal) => {
        impl<T> From<[T; $N]> for $crate::vec1::Vec1<T> {
            fn from(items: [T; $N]) -> Self {
                $crate::vec1::Vec1::from_vec_unchecked(alloc::vec::Vec::from(items))
            }
        }

        impl<'a, T> From<&'a [T; $N]> for $crate::vec1::Vec1<T>
        where
            T: Copy,
        {
            fn from(items: &'a [T; $N]) -> Self {
                $crate::vec1::Vec1::from_vec_unchecked(items.iter().copied().collect())
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_from_array_for_vec1);

#[cfg(test)]
mod tests {}

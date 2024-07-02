#![cfg(feature = "arrayvec")]
#![cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]

use alloc::borrow::{Borrow, BorrowMut};
use arrayvec::ArrayVec;
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut};

use crate::array1::Array1;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1, IteratorExt as _};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{NonEmpty, Saturated, Vacancy};

pub type ArrayVec1<T, const N: usize> = NonEmpty<ArrayVec<T, N>>;

impl<T, const N: usize> ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    /// # Safety
    pub const unsafe fn from_array_vec_unchecked(items: ArrayVec<T, N>) -> Self {
        ArrayVec1 { items }
    }

    pub fn from_one(item: T) -> Self {
        unsafe {
            // SAFETY:
            ArrayVec1::from_array_vec_unchecked({
                let mut items = ArrayVec::new();
                // SAFETY:
                items.push_unchecked(item);
                items
            })
        }
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::from_head_and_tail(head, tail).collect()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::from_tail_and_head(tail, head).collect()
    }

    // NOTE: Panics on overflow.
    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(ArrayVec1::from_iter1)
    }

    pub fn into_head_and_tail(mut self) -> (T, ArrayVec<T, N>) {
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_tail_and_head(mut self) -> (ArrayVec<T, N>, T) {
        // SAFETY:
        let head = unsafe { self.items.pop().unwrap_unchecked() };
        (self.items, head)
    }

    pub fn into_array_vec(self) -> ArrayVec<T, N> {
        self.items
    }

    pub fn try_into_array(self) -> Result<[T; N], Self> {
        self.items
            .into_inner()
            // SAFETY:
            .map_err(|items| unsafe { ArrayVec1::from_array_vec_unchecked(items) })
    }

    /// # Safety
    pub unsafe fn into_array_unchecked(self) -> [T; N] {
        self.items.into_inner_unchecked()
    }

    fn many_or_else<M, O>(&mut self, many: M, one: O) -> Result<T, &T>
    where
        M: FnOnce(&mut ArrayVec<T, N>) -> T,
        O: FnOnce(&mut ArrayVec<T, N>) -> &T,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
    }

    fn many_or_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut ArrayVec<T, N>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, |items| unsafe { items.get_unchecked(0) })
    }

    fn many_or_get<F>(&mut self, index: usize, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut ArrayVec<T, N>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, move |items| {
            items.get(index).expect("index out of bounds")
        })
    }

    fn vacant_or_else<U, V, F>(&mut self, item: T, vacant: V, full: F) -> Result<U, (T, &T)>
    where
        V: FnOnce(T, &mut ArrayVec<T, N>) -> U,
        F: FnOnce(&mut ArrayVec<T, N>) -> &T,
    {
        match self.items.len() {
            0 => unreachable!(),
            n if n == N => Err((item, full(&mut self.items))),
            _ => Ok(vacant(item, &mut self.items)),
        }
    }

    fn vacant_or_last<U, F>(&mut self, item: T, f: F) -> Result<U, (T, &T)>
    where
        F: FnOnce(T, &mut ArrayVec<T, N>) -> U,
    {
        // SAFETY:
        self.vacant_or_else(item, f, |items| unsafe { items.last().unwrap_unchecked() })
    }

    pub fn push_or_last(&mut self, item: T) -> Result<(), (T, &T)> {
        self.vacant_or_last(item, |item, items| items.push(item))
    }

    pub fn pop_or_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop().unwrap_unchecked() })
    }

    pub fn insert_or_last(&mut self, index: usize, item: T) -> Result<(), (T, &T)> {
        self.vacant_or_last(item, move |item, items| items.insert(index, item))
    }

    pub fn remove_or_only(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_get(index, move |items| items.remove(index))
    }

    pub fn swap_remove_or_only(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_get(index, move |items| items.swap_remove(index))
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub const fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.capacity()) }
    }

    pub const fn as_array_vec(&self) -> &ArrayVec<T, N> {
        &self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        // SAFETY:
        unsafe { Slice1::from_slice_unchecked(self.items.as_slice()) }
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        // SAFETY:
        unsafe { Slice1::from_mut_slice_unchecked(self.items.as_mut_slice()) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.items.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.items.as_mut_ptr()
    }
}

impl<T, const N: usize> AsMut<[T]> for ArrayVec1<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T, const N: usize> AsMut<Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn as_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T, const N: usize> AsRef<[T]> for ArrayVec1<T, N> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T, const N: usize> AsRef<Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn as_ref(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T, const N: usize> Borrow<[T]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn borrow(&self) -> &[T] {
        self.items.borrow()
    }
}

impl<T, const N: usize> Borrow<Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn borrow(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T, const N: usize> BorrowMut<[T]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.items.borrow_mut()
    }
}

impl<T, const N: usize> BorrowMut<Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn borrow_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T, const N: usize> Debug for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T, const N: usize> Deref for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Target = Slice1<T>;

    fn deref(&self) -> &Self::Target {
        self.as_slice1()
    }
}

impl<T, const N: usize> DerefMut for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice1()
    }
}

// NOTE: Panics on overflow.
impl<T, const N: usize> Extend<T> for ArrayVec1<T, N> {
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<T, const N: usize> From<[T; N]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY:
        unsafe { ArrayVec1::from_array_vec_unchecked(ArrayVec::from(items)) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Copy,
{
    fn from(items: &'a [T; N]) -> Self {
        // SAFETY:
        unsafe { ArrayVec1::from_array_vec_unchecked(items.iter().copied().collect()) }
    }
}

impl<T, const N: usize> From<ArrayVec1<T, N>> for ArrayVec<T, N> {
    fn from(items: ArrayVec1<T, N>) -> Self {
        items.items
    }
}

// NOTE: Panics on overflow.
impl<T, const N: usize> FromIterator1<T> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY:
        unsafe { ArrayVec1::from_array_vec_unchecked(items.into_iter1().into_iter().collect()) }
    }
}

impl<T, const N: usize> IntoIterator for ArrayVec1<T, N> {
    type Item = T;
    type IntoIter = arrayvec::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T, const N: usize> IntoIterator1 for ArrayVec1<T, N> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, I, const N: usize> Saturated<I> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    I: IntoIterator1<Item = T>,
{
    type Remainder = iter1::Remainder<I::IntoIter>;

    fn saturated(items: I) -> (Self, Self::Remainder) {
        let mut remainder = items.into_iter1().into_iter();
        // SAFETY:
        let items =
            unsafe { ArrayVec1::from_array_vec_unchecked(remainder.by_ref().take(N).collect()) };
        (items, remainder.try_into_iter1())
    }
}

impl<'a, T, const N: usize> TryFrom<&'a [T]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Clone,
{
    type Error = &'a [T];

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        Slice1::try_from_slice(items)
            .and_then(|items1| ArrayVec1::try_from(items1).map_err(|_| items))
    }
}

impl<'a, T, const N: usize> TryFrom<&'a Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Clone,
{
    type Error = &'a Slice1<T>;

    fn try_from(items: &'a Slice1<T>) -> Result<Self, Self::Error> {
        ArrayVec::try_from(items.as_slice())
            // SAFETY:
            .map(|items| unsafe { ArrayVec1::from_array_vec_unchecked(items) })
            .map_err(|_| items)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T, const N: usize> TryFrom<Serde<ArrayVec<T, N>>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = EmptyError;

    fn try_from(serde: Serde<ArrayVec<T, N>>) -> Result<Self, Self::Error> {
        ArrayVec1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T, const N: usize> TryFrom<ArrayVec<T, N>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = ArrayVec<T, N>;

    fn try_from(items: ArrayVec<T, N>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY:
            _ => Ok(unsafe { ArrayVec1::from_array_vec_unchecked(items) }),
        }
    }
}

impl<T, const N: usize> Vacancy for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn vacancy(&self) -> usize {
        self.capacity().get() - self.len().get()
    }
}

#[cfg(test)]
mod tests {}

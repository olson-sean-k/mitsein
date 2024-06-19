#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::vec_deque::{self, VecDeque};
use core::fmt::{self, Debug, Formatter};
use core::iter::Peekable;
use core::num::NonZeroUsize;
use core::ops::{Index, IndexMut};

use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{NonEmpty, NonZeroUsizeExt as _};

pub type VecDeque1<T> = NonEmpty<VecDeque<T>>;

impl<T> VecDeque1<T> {
    pub(crate) fn from_vec_deque_unchecked(items: VecDeque<T>) -> Self {
        VecDeque1 { items }
    }

    pub fn from_item(item: T) -> Self {
        iter1::from_item(item).collect()
    }

    pub fn from_item_with_capacity(item: T, capacity: usize) -> Self {
        let mut items = VecDeque::with_capacity(capacity);
        items.push_back(item);
        VecDeque1::from_vec_deque_unchecked(items)
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        VecDeque1::from_vec_deque_unchecked(Some(head).into_iter().chain(tail).collect())
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        VecDeque1::from_vec_deque_unchecked(tail.into_iter().chain(Some(head)).collect())
    }

    pub fn try_from_iter<I>(items: I) -> Result<Self, Peekable<I::IntoIter>>
    where
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(VecDeque1::from_iter1)
    }

    pub fn into_vec_deque(self) -> VecDeque<T> {
        self.items
    }

    fn many_or_else<M, O>(&mut self, many: M, one: O) -> Result<T, &T>
    where
        M: FnOnce(&mut VecDeque<T>) -> T,
        O: FnOnce(&mut VecDeque<T>) -> &T,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
    }

    fn try_many_or_else<M, O>(&mut self, many: M, one: O) -> Option<Result<T, &T>>
    where
        M: FnOnce(&mut VecDeque<T>) -> Option<T>,
        O: FnOnce(&mut VecDeque<T>) -> Option<&T>,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => one(&mut self.items).map(Err),
            _ => many(&mut self.items).map(Ok),
        }
    }

    fn many_or_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut VecDeque<T>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, |items| unsafe { items.front().unwrap_unchecked() })
    }

    fn try_many_or_index<F>(&mut self, index: usize, f: F) -> Option<Result<T, &T>>
    where
        F: FnOnce(&mut VecDeque<T>) -> Option<T>,
    {
        // SAFETY:
        self.try_many_or_else(f, move |items| items.get(index))
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

    pub fn make_contiguous(&mut self) -> &mut Slice1<T> {
        Slice1::from_mut_slice_unchecked(self.items.make_contiguous())
    }

    pub fn rotate_left(&mut self, n: usize) {
        self.items.rotate_left(n)
    }

    pub fn rotate_right(&mut self, n: usize) {
        self.items.rotate_right(n)
    }

    pub fn truncate(&mut self, len: NonZeroUsize) {
        self.items.truncate(len.into())
    }

    pub fn split_off(&mut self, at: NonZeroUsize) -> VecDeque<T> {
        self.items.split_off(at.into())
    }

    // NOTE: This is as similar to `VecDeque::clear` as `VecDeque1` can afford.
    pub fn split_off_front(&mut self) -> VecDeque<T> {
        self.split_off(NonZeroUsize::ONE)
    }

    pub fn append(&mut self, mut items: Self) {
        self.items.append(&mut items.items)
    }

    pub fn push_front(&mut self, item: T) {
        self.items.push_front(item)
    }

    pub fn push_back(&mut self, item: T) {
        self.items.push_back(item)
    }

    pub fn pop_front_or_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_front().unwrap_unchecked() })
    }

    pub fn pop_back_or_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_only(|items| unsafe { items.pop_back().unwrap_unchecked() })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_or_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_index(index, move |items| items.remove(index))
    }

    pub fn swap_remove_front_or_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_index(index, move |items| items.swap_remove_front(index))
    }

    pub fn swap_remove_back_or_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_index(index, move |items| items.swap_remove_back(index))
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.capacity()) }
    }

    pub fn front(&self) -> &T {
        // SAFETY:
        unsafe { self.items.front().unwrap_unchecked() }
    }

    pub fn front_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.front_mut().unwrap_unchecked() }
    }

    pub fn back(&self) -> &T {
        // SAFETY:
        unsafe { self.items.back().unwrap_unchecked() }
    }

    pub fn back_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.back_mut().unwrap_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<vec_deque::Iter<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter())
    }

    pub fn iter1_mut(&mut self) -> Iterator1<vec_deque::IterMut<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter_mut())
    }

    pub fn as_vec_deque(&self) -> &VecDeque<T> {
        &self.items
    }
}

impl<T> AsRef<VecDeque<T>> for VecDeque1<T> {
    fn as_ref(&self) -> &VecDeque<T> {
        &self.items
    }
}

impl<T> Debug for VecDeque1<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T> Extend<T> for VecDeque1<T> {
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<T> From<VecDeque1<T>> for VecDeque<T> {
    fn from(items: VecDeque1<T>) -> Self {
        items.items
    }
}

impl<T> FromIterator1<T> for VecDeque1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        VecDeque1 {
            //items: items.into_iter1().collect(),
            items: items.into_iter1().into_iter().collect(),
        }
    }
}

impl<T> Index<usize> for VecDeque1<T> {
    type Output = <VecDeque<T> as Index<usize>>::Output;

    fn index(&self, at: usize) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T> IndexMut<usize> for VecDeque1<T> {
    fn index_mut(&mut self, at: usize) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<T> IntoIterator for VecDeque1<T> {
    type Item = T;
    type IntoIter = vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for VecDeque1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        Iterator1::from_iter_unchecked(self.items)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<VecDeque<T>>> for VecDeque1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<VecDeque<T>>) -> Result<Self, Self::Error> {
        VecDeque1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T> TryFrom<VecDeque<T>> for VecDeque1<T> {
    type Error = VecDeque<T>;

    fn try_from(items: VecDeque<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            _ => Ok(VecDeque1::from_vec_deque_unchecked(items)),
        }
    }
}

macro_rules! impl_from_array_for_vec_deque1 {
    ($N:literal) => {
        impl<T> From<[T; $N]> for $crate::vec_deque1::VecDeque1<T> {
            fn from(items: [T; $N]) -> Self {
                $crate::vec_deque1::VecDeque1::from_vec_deque_unchecked(
                    alloc::collections::vec_deque::VecDeque::from(items),
                )
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_from_array_for_vec_deque1);

#[cfg(test)]
mod tests {}

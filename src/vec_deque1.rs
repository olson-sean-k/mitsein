//! A non-empty [`VecDeque`][`vec_deque`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::vec_deque::{self, VecDeque};
use alloc::vec::Vec;
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter;
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Index, IndexMut, RangeBounds};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "std")]
use std::io::{self, IoSlice, Write};

use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::slice1::Slice1;
use crate::take;
use crate::vec1::Vec1;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedVecDeque>::Item;

pub trait ClosedVecDeque {
    type Item;

    fn as_vec_deque(&self) -> &VecDeque<Self::Item>;
}

impl<T> ClosedVecDeque for VecDeque<T> {
    type Item = T;

    fn as_vec_deque(&self) -> &VecDeque<Self::Item> {
        self
    }
}

impl<T> Extend1<T> for VecDeque<T> {
    fn extend_non_empty<I>(mut self, items: I) -> VecDeque1<T>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { VecDeque1::from_vec_deque_unchecked(self) }
    }
}

unsafe impl<T> MaybeEmpty for VecDeque<T> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

impl<T> Ranged for VecDeque<T> {
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

impl<T> Segmentation for VecDeque<T> {
    fn tail(&mut self) -> Segment<'_, Self> {
        Segment::intersect(self, &Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segment::intersect(self, &Ranged::rtail(self))
    }
}

impl<T, R> SegmentedBy<R> for VecDeque<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for VecDeque<T> {
    type Target = Self;
    type Kind = Self;
}

type TakeOr<'a, T, U, N = ()> = take::TakeOr<'a, VecDeque<T>, U, N>;

pub type PopOr<'a, K> = TakeOr<'a, ItemFor<K>, ItemFor<K>, ()>;

pub type RemoveOr<'a, K> = TakeOr<'a, ItemFor<K>, Option<ItemFor<K>>, usize>;

impl<'a, T, N> TakeOr<'a, T, T, N> {
    pub fn get_only(self) -> Result<T, &'a T> {
        self.take_or_else(|items, _| items.front())
    }

    pub fn replace_only(self, replacement: T) -> Result<T, T> {
        self.else_replace_only(move || replacement)
    }

    pub fn else_replace_only<F>(self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.take_or_else(move |items, _| mem::replace(items.front_mut(), f()))
    }
}

impl<'a, T> TakeOr<'a, T, Option<T>, usize> {
    pub fn get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, index| items.get(index))
    }

    pub fn replace(self, replacement: T) -> Option<Result<T, T>> {
        self.else_replace(move || replacement)
    }

    pub fn else_replace<F>(self, f: F) -> Option<Result<T, T>>
    where
        F: FnOnce() -> T,
    {
        self.try_take_or_else(move |items, index| {
            items
                .get_mut(index)
                .map(move |item| mem::replace(item, f()))
        })
    }
}

pub type VecDeque1<T> = NonEmpty<VecDeque<T>>;

impl<T> VecDeque1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`VecDeque::new()`][`VecDeque::new`].
    ///
    /// [`VecDeque::new`]: alloc::collections::vec_deque::VecDeque::new
    pub unsafe fn from_vec_deque_unchecked(items: VecDeque<T>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        VecDeque1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = VecDeque::with_capacity(capacity);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { VecDeque1::from_vec_deque_unchecked(items) }
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn into_vec_deque(self) -> VecDeque<T> {
        self.items
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn make_contiguous(&mut self) -> &mut Slice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.items.make_contiguous()) }
    }

    pub fn rotate_left(&mut self, n: usize) {
        self.items.rotate_left(n)
    }

    pub fn rotate_right(&mut self, n: usize) {
        self.items.rotate_right(n)
    }

    pub fn split_off_tail(&mut self) -> VecDeque<T> {
        self.items.split_off(1)
    }

    pub fn append(&mut self, items: &mut VecDeque<T>) {
        self.items.append(items)
    }

    pub fn push_front(&mut self, item: T) {
        self.items.push_front(item)
    }

    pub fn push_back(&mut self, item: T) {
        self.items.push_back(item)
    }

    pub fn pop_front_or(&mut self) -> PopOr<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, ()| unsafe {
            items.items.pop_front().unwrap_maybe_unchecked()
        })
    }

    pub fn pop_back_or(&mut self) -> PopOr<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, ()| unsafe {
            items.items.pop_back().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_or(&mut self, index: usize) -> RemoveOr<'_, Self> {
        TakeOr::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove_front_or(&mut self, index: usize) -> RemoveOr<'_, Self> {
        TakeOr::with(self, index, |items, index| {
            items.items.swap_remove_front(index)
        })
    }

    pub fn swap_remove_back_or(&mut self, index: usize) -> RemoveOr<'_, Self> {
        TakeOr::with(self, index, |items, index| {
            items.items.swap_remove_back(index)
        })
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn front(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.front().unwrap_maybe_unchecked() }
    }

    pub fn front_mut(&mut self) -> &mut T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.front_mut().unwrap_maybe_unchecked() }
    }

    pub fn back(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.back().unwrap_maybe_unchecked() }
    }

    pub fn back_mut(&mut self) -> &mut T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.back_mut().unwrap_maybe_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<vec_deque::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<vec_deque::IterMut<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub const fn as_vec_deque(&self) -> &VecDeque<T> {
        &self.items
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> VecDeque1<T> {
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ Self as IntoParallelIterator>::Iter>
    where
        T: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }

    pub fn par_iter1_mut(
        &mut self,
    ) -> ParallelIterator1<<&'_ mut Self as IntoParallelIterator>::Iter>
    where
        T: Send,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter_mut()) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T> Arbitrary<'a> for VecDeque1<T>
where
    T: Arbitrary<'a>,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(T::arbitrary(unstructured), unstructured.arbitrary_iter()?).collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (T::size_hint(depth).0, None)
    }
}

impl<T> ClosedVecDeque for VecDeque1<T> {
    type Item = T;

    fn as_vec_deque(&self) -> &VecDeque<Self::Item> {
        self.as_ref()
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

impl<T, const N: usize> From<[T; N]> for VecDeque1<T>
where
    [T; N]: Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { VecDeque1::from_vec_deque_unchecked(VecDeque::from(items)) }
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
        // SAFETY: `items` must be non-empty.
        unsafe { VecDeque1::from_vec_deque_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> FromParallelIterator1<T> for VecDeque1<T>
where
    T: Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = T>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { VecDeque1::from_vec_deque_unchecked(items.into_par_iter().collect()) }
    }
}

impl<T, I> Index<I> for VecDeque1<T>
where
    VecDeque<T>: Index<I>,
{
    type Output = <VecDeque<T> as Index<I>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T, I> IndexMut<I> for VecDeque1<T>
where
    VecDeque<T>: IndexMut<I>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
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

impl<'a, T> IntoIterator for &'a VecDeque1<T> {
    type Item = &'a T;
    type IntoIter = vec_deque::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut VecDeque1<T> {
    type Item = &'a mut T;
    type IntoIter = vec_deque::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T> IntoIterator1 for VecDeque1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> IntoIterator1 for &'_ VecDeque1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<T> IntoIterator1 for &'_ mut VecDeque1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator for VecDeque1<T>
where
    T: Send,
{
    type Item = T;
    type Iter = <VecDeque<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a VecDeque1<T>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = <&'a VecDeque<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a mut VecDeque1<T>
where
    T: Send,
{
    type Item = &'a mut T;
    type Iter = <&'a mut VecDeque<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for VecDeque1<T>
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
impl<T> IntoParallelIterator1 for &'_ VecDeque1<T>
where
    T: Sync,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&self.items) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for &'_ mut VecDeque1<T>
where
    T: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&mut self.items) }
    }
}

crate::impl_partial_eq_for_non_empty!([for U, const N: usize in [U; N]] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U, const N: usize in &[U; N]] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U, const N: usize in &mut [U; N]] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &[U]] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &mut [U]] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in Vec<U>] <= [for T in VecDeque1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] => [for T in VecDeque<T>]);

impl<T> Segmentation for VecDeque1<T> {
    fn tail(&mut self) -> Segment<'_, Self> {
        let range = Ranged::tail(&self.items);
        Segment::intersect_strict_subset(&mut self.items, &range)
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        let range = Ranged::rtail(&self.items);
        Segment::intersect_strict_subset(&mut self.items, &range)
    }
}

impl<T, R> SegmentedBy<R> for VecDeque1<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for VecDeque1<T> {
    type Target = VecDeque<T>;
    type Kind = Self;
}

impl<T> TryFrom<VecDeque<T>> for VecDeque1<T> {
    type Error = VecDeque<T>;

    fn try_from(items: VecDeque<T>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl Write for VecDeque1<u8> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.items.extend(buffer);
        Ok(buffer.len())
    }

    fn write_vectored(&mut self, buffers: &[IoSlice<'_>]) -> io::Result<usize> {
        let len = buffers.iter().map(|buffer| buffer.len()).sum();
        self.items.reserve(len);
        for buffer in buffers {
            self.items.extend(&**buffer);
        }
        Ok(len)
    }

    fn write_all(&mut self, buffer: &[u8]) -> io::Result<()> {
        self.items.extend(buffer);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub type Segment<'a, K> = segment::Segment<'a, K, VecDeque<ItemFor<K>>>;

impl<K, T> Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
{
    pub fn split_off(&mut self, at: usize) -> VecDeque<T> {
        let at = self.range.project(&at).expect_in_bounds();
        let range = From::from(at..self.range.end);
        let items = self.items.drain(range).collect();
        self.range = range;
        items
    }

    pub fn resize(&mut self, len: usize, fill: T)
    where
        T: Clone,
    {
        self.resize_with(len, move || fill.clone())
    }

    pub fn resize_with<F>(&mut self, len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let to = len;
        let from = self.len();
        if to > from {
            let n = to - from;
            self.extend(iter::repeat_with(f).take(n))
        }
        else {
            self.truncate(to)
        }
    }

    pub fn truncate(&mut self, len: usize) {
        if let Some(range) = self.range.truncate_from_end(len) {
            self.items.drain(range);
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(move |item| f(&*item))
    }

    pub fn retain_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.items.retain_mut(self.range.retain_mut_from_end(f))
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let index = self.range.project(&index).expect_in_bounds();
        self.items.insert(index, item);
        self.range.put_from_end(1);
    }

    pub fn insert_front(&mut self, item: T) {
        self.items.insert(self.range.start, item);
        self.range.put_from_end(1);
    }

    pub fn insert_back(&mut self, item: T) {
        self.items.insert(self.range.end, item);
        self.range.put_from_end(1);
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        self.items
            .remove(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn remove_front(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            self.items
                .remove(self.range.start)
                .inspect(|_| self.range.take_from_end(1))
        }
    }

    pub fn remove_back(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            self.items
                .remove(self.range.end - 1)
                .inspect(|_| self.range.take_from_end(1))
        }
    }

    pub fn swap_remove_front(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        self.items.swap(index, self.range.start);
        self.items
            .remove(self.range.start)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_back(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        let swapped = self.range.end - 1;
        self.items.swap(index, swapped);
        self.items
            .remove(swapped)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn clear(&mut self) {
        self.items.drain(self.range.get_and_clear_from_end());
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn iter(&self) -> vec_deque::Iter<'_, T> {
        VecDeque::range(self.items, self.range)
    }

    pub fn iter_mut(&mut self) -> vec_deque::IterMut<'_, T> {
        self.items.range_mut(self.range)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, T> Eq for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
    T: Eq,
{
}

impl<K, T> Extend<T> for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
{
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let n = self.items.len();
        // Split off the remainder beyond the segment to avoid spurious inserts and copying. This
        // comes at the cost of a necessary allocation and bulk copy, which isn't great when
        // extending from a small number of items with a small remainder.
        let tail = self.items.split_off(self.range.end);
        self.items.extend(items);
        self.items.extend(tail);
        let n = self.items.len() - n;
        self.range.put_from_end(n);
    }
}

impl<K, T> Ord for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, T> PartialEq<Self> for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<K, T> PartialOrd<Self> for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, T> Segmentation for Segment<'_, K>
where
    K: ClosedVecDeque<Item = T> + SegmentedOver<Target = VecDeque<T>>,
{
    fn tail(&mut self) -> Segment<'_, K> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> Segment<'_, K> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<K, T, R> SegmentedBy<R> for Segment<'_, K>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: ClosedVecDeque<Item = T> + SegmentedBy<R> + SegmentedOver<Target = VecDeque<T>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, K> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::iter1::{self, FromIterator1};
    use crate::vec_deque1::VecDeque1;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> VecDeque1<u8> {
        VecDeque1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    use core::iter;
    use core::ops::RangeBounds;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::iter1::IntoIterator1;
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};
    use crate::slice1::{slice1, Slice1};
    use crate::vec_deque1::harness;
    use crate::vec_deque1::harness::xs1;
    use crate::vec_deque1::VecDeque1;
    use crate::Segmentation;

    #[rstest]
    #[case::one_into_empty_front(0..0, [42], slice1![42, 0, 1, 2, 3, 4])]
    #[case::many_into_empty_front(0..0, [42, 88], slice1![88, 42, 0, 1, 2, 3, 4])]
    #[case::one_into_empty_back(5..5, [42], slice1![0, 1, 2, 3, 4, 42])]
    #[case::many_into_empty_back(5..5, [42, 88], slice1![0, 1, 2, 3, 4, 88, 42])]
    #[case::one_into_empty_middle(2..2, [42], slice1![0, 1, 42, 2, 3, 4])]
    #[case::many_into_empty_middle(2..2, [42, 88], slice1![0, 1, 88, 42, 2, 3, 4])]
    #[case::one_into_non_empty(1..2, [42], slice1![0, 42, 1, 2, 3, 4])]
    #[case::many_into_non_empty(1..2, [42, 88], slice1![0, 88, 42, 1, 2, 3, 4])]
    fn insert_front_into_vec_deque1_segment_then_vec_deque1_eq<S, T>(
        mut xs1: VecDeque1<u8>,
        #[case] segment: S,
        #[case] items: T,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
        T: IntoIterator1<Item = u8>,
    {
        let mut segment = xs1.segment(segment);
        for item in items {
            segment.insert_front(item);
        }
        assert_eq!(xs1.make_contiguous(), expected);
    }

    #[rstest]
    #[case::one_into_empty_front(0..0, [42], slice1![42, 0, 1, 2, 3, 4])]
    #[case::many_into_empty_front(0..0, [42, 88], slice1![42, 88, 0, 1, 2, 3, 4])]
    #[case::one_into_empty_back(5..5, [42], slice1![0, 1, 2, 3, 4, 42])]
    #[case::many_into_empty_back(5..5, [42, 88], slice1![0, 1, 2, 3, 4, 42, 88])]
    #[case::one_into_empty_middle(2..2, [42], slice1![0, 1, 42, 2, 3, 4])]
    #[case::many_into_empty_middle(2..2, [42, 88], slice1![0, 1, 42, 88, 2, 3, 4])]
    #[case::one_into_non_empty(0..2, [42], slice1![0, 1, 42, 2, 3, 4])]
    #[case::many_into_non_empty(0..2, [42, 88], slice1![0, 1, 42, 88, 2, 3, 4])]
    fn insert_back_into_vec_deque1_segment_then_vec_deque1_eq<S, T>(
        mut xs1: VecDeque1<u8>,
        #[case] segment: S,
        #[case] items: T,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
        T: IntoIterator1<Item = u8>,
    {
        let mut segment = xs1.segment(segment);
        for item in items {
            segment.insert_back(item);
        }
        assert_eq!(xs1.make_contiguous(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn remove_front_all_from_rtail_of_vec_deque1_then_vec_deque1_eq_rhead(
        #[case] mut xs1: VecDeque1<u8>,
    ) {
        let n = xs1.len().get();
        let rhead = *xs1.back();
        let mut rtail = xs1.rtail();
        iter::from_fn(|| rtail.remove_front())
            .take(n)
            .for_each(|_| {});
        assert!(rtail.is_empty());
        assert_eq!(xs1.make_contiguous().as_slice(), &[rhead]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn remove_back_all_from_tail_of_vec_deque1_then_vec_deque1_eq_head(
        #[case] mut xs1: VecDeque1<u8>,
    ) {
        let n = xs1.len().get();
        let mut tail = xs1.tail();
        iter::from_fn(|| tail.remove_back())
            .take(n)
            .for_each(|_| {});
        assert!(tail.is_empty());
        assert_eq!(xs1.make_contiguous().as_slice(), &[0]);
    }

    #[rstest]
    #[case::tail(harness::xs1(3), 1.., slice1![0])]
    #[case::rtail(harness::xs1(3), ..3, slice1![3])]
    #[case::middle(harness::xs1(9), 4..8, slice1![0, 1, 2, 3, 8, 9])]
    fn retain_none_from_vec_deque1_segment_then_vec_deque1_eq<S>(
        #[case] mut xs1: VecDeque1<u8>,
        #[case] segment: S,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
    {
        xs1.segment(segment).retain(|_| false);
        assert_eq!(xs1.make_contiguous(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_vec_deque1_then_vec_deque1_eq_head(#[case] mut xs1: VecDeque1<u8>) {
        xs1.tail().clear();
        assert_eq!(xs1.make_contiguous().as_slice(), &[0]);
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_vec_deque1_then_vec_deque1_eq_tail(#[case] mut xs1: VecDeque1<u8>) {
        let tail = *xs1.back();
        xs1.rtail().clear();
        assert_eq!(xs1.make_contiguous().as_slice(), &[tail]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_vec_deque1_then_vec_deque1_eq_head_and_tail(
        #[case] mut xs1: VecDeque1<u8>,
    ) {
        let n = xs1.len().get();
        let head_and_tail = [0, *xs1.back()];
        xs1.tail().rtail().clear();
        assert_eq!(
            xs1.make_contiguous().as_slice(),
            if n > 1 {
                &head_and_tail[..]
            }
            else {
                &head_and_tail[..1]
            }
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_vec_deque1_into_and_from_tokens_eq(
        xs1: VecDeque1<u8>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_vec_deque1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<VecDeque1<u8>, Vec<_>>(sequence)
    }
}

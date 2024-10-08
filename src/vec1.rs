#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, Cow};
use alloc::vec::{self, Drain, Splice, Vec};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter::{self, FusedIterator};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
#[cfg(feature = "std")]
use std::io::{self, IoSlice, Write};

use crate::array1::Array1;
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _};
use crate::iter1::{self, ExtendUntil, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _, SliceExt as _};
use crate::segment::range::{
    self, Intersect, IntersectionExt as _, PositionalRange, Project, ProjectionExt as _,
};
use crate::segment::{self, Ranged, Segment, Segmentation, SegmentedOver};
use crate::slice1::Slice1;
#[cfg(target_has_atomic = "ptr")]
use crate::sync1::{ArcSlice1, ArcSlice1Ext as _};
use crate::{NonEmpty, Vacancy};

segment::impl_target_forward_type_and_definition!(
    for <T> => Vec,
    VecTarget,
    VecSegment,
);

impl<T, I> ExtendUntil<I> for Vec<T>
where
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<T> Ranged for Vec<T> {
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

impl<T> Segmentation for Vec<T> {
    fn tail(&mut self) -> VecSegment<'_, Self> {
        self.segment(Ranged::tail(self))
    }

    fn rtail(&mut self) -> VecSegment<'_, Self> {
        self.segment(Ranged::rtail(self))
    }
}

impl<T, R> segment::SegmentedBy<R> for Vec<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecSegment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for Vec<T> {
    type Kind = VecTarget<Self>;
    type Target = Self;
}

impl<T> Vacancy for Vec<T> {
    fn vacancy(&self) -> usize {
        self.capacity() - self.len()
    }
}

pub type CowSlice1<'a, T> = Cow<'a, Slice1<T>>;

pub trait CowSlice1Ext<'a, T>
where
    T: Clone,
{
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T>;

    fn into_cow_slice(self) -> Cow<'a, [T]>;
}

impl<'a, T> CowSlice1Ext<'a, T> for CowSlice1<'a, T>
where
    T: Clone,
{
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T> {
        ArcSlice1::from_cow_slice1(self)
    }

    fn into_cow_slice(self) -> Cow<'a, [T]> {
        match self {
            Cow::Borrowed(borrowed) => Cow::Borrowed(borrowed),
            Cow::Owned(owned) => Cow::Owned(owned.into_vec()),
        }
    }
}

pub type Vec1<T> = NonEmpty<Vec<T>>;

impl<T> Vec1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`Vec::new()`][`Vec::new`].
    ///
    /// [`Vec::new`]: alloc::vec::Vec::new
    pub const unsafe fn from_vec_unchecked(items: Vec<T>) -> Self {
        Vec1 { items }
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        let mut items = Vec::with_capacity(capacity);
        items.push(item);
        // SAFETY: `items` must contain `item` and therefore is non-empty here, because `push`
        //         either pushes the item or panics.
        unsafe { Vec1::from_vec_unchecked(items) }
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

    pub fn into_head_and_tail(mut self) -> (T, Vec<T>) {
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_tail_and_head(mut self) -> (Vec<T>, T) {
        // SAFETY: `self` must be non-empty.
        let head = unsafe { self.items.pop().unwrap_maybe_unchecked() };
        (self.items, head)
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items
    }

    pub fn into_boxed_slice1(self) -> BoxedSlice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { BoxedSlice1::from_boxed_slice_unchecked(self.items.into_boxed_slice()) }
    }

    fn many_or_else<'a, U, M, O>(&'a mut self, many: M, one: O) -> Result<T, U>
    where
        M: FnOnce(&'a mut Vec<T>) -> T,
        O: FnOnce(&'a mut Vec<T>) -> U,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
    }

    fn many_or_get<F>(&mut self, index: usize, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut Vec<T>) -> T,
    {
        self.many_or_else(f, move |items| &items[index])
    }

    fn many_or_get_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut Vec<T>) -> T,
    {
        // SAFETY: `self` must be non-empty.
        self.many_or_else(f, |items| unsafe { items.get_maybe_unchecked(0) })
    }

    fn many_or_replace_only_with<F, R>(&mut self, f: F, replace: R) -> Result<T, T>
    where
        F: FnOnce(&mut Vec<T>) -> T,
        R: FnOnce() -> T,
    {
        // SAFETY: `self` must be non-empty.
        self.many_or_else(f, move |items| unsafe {
            mem::replace(items.get_maybe_unchecked_mut(0), replace())
        })
    }

    pub fn leak<'a>(self) -> &'a mut Slice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.items.leak()) }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.items.reserve_exact(additional)
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn split_off_tail(&mut self) -> Vec<T> {
        self.items.split_off(1)
    }

    pub fn append<R>(&mut self, items: R)
    where
        R: Into<Vec<T>>,
    {
        self.items.append(&mut items.into())
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn pop_or_get_only(&mut self) -> Result<T, &T> {
        // SAFETY: `self` must be non-empty.
        self.many_or_get_only(|items| unsafe { items.pop().unwrap_maybe_unchecked() })
    }

    pub fn pop_or_replace_only(&mut self, replacement: T) -> Result<T, T> {
        self.pop_or_replace_only_with(move || replacement)
    }

    pub fn pop_or_replace_only_with<F>(&mut self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        // SAFETY: `self` must be non-empty.
        self.many_or_replace_only_with(|items| unsafe { items.pop().unwrap_maybe_unchecked() }, f)
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_or_get_only(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_get(index, move |items| items.remove(index))
    }

    pub fn remove_or_replace_only(&mut self, index: usize, replacement: T) -> Result<T, T> {
        self.remove_or_replace_only_with(index, move || replacement)
    }

    pub fn remove_or_replace_only_with<F>(&mut self, index: usize, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.many_or_replace_only_with(
            |items| items.remove(index),
            move || {
                assert!(index == 0, "index out of bounds");
                f()
            },
        )
    }

    pub fn swap_remove_or_get_only(&mut self, index: usize) -> Result<T, &T> {
        self.many_or_get(index, move |items| items.swap_remove(index))
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn splice<R, I>(&mut self, range: R, replacement: I) -> Splice<'_, I::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator1<Item = T>,
    {
        self.items.splice(range, replacement.into_iter1())
    }

    pub const fn as_vec(&self) -> &Vec<T> {
        &self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_slice_unchecked(self.items.as_slice()) }
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.items.as_mut_slice()) }
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

impl<T, I> ExtendUntil<I> for Vec1<T>
where
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<T, const N: usize> From<[T; N]> for Vec1<T>
where
    [T; N]: Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items)) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for Vec1<T>
where
    [T; N]: Array1,
    T: Copy,
{
    fn from(items: &'a [T; N]) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(items.iter().copied().collect()) }
    }
}

impl<'a, T> From<&'a Slice1<T>> for Vec1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.as_slice())) }
    }
}

impl<T> From<BoxedSlice1<T>> for Vec1<T> {
    fn from(items: BoxedSlice1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.into_boxed_slice())) }
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
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(items.into_iter1().collect()) }
    }
}

impl<T, I> Index<I> for Vec1<T>
where
    Vec<T>: Index<I>,
{
    type Output = <Vec<T> as Index<I>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T, I> IndexMut<I> for Vec1<T>
where
    Vec<T>: IndexMut<I>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
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
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> Segmentation for Vec1<T> {
    fn tail(&mut self) -> VecSegment<'_, Self> {
        self.segment(Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> VecSegment<'_, Self> {
        self.segment(Ranged::rtail(&self.items))
    }
}

impl<T, R> segment::SegmentedBy<R> for Vec1<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecSegment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for Vec1<T> {
    type Kind = VecTarget<Self>;
    type Target = Vec<T>;
}

impl<'a, T> TryFrom<&'a [T]> for Vec1<T>
where
    T: Clone,
{
    type Error = &'a [T];

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { Vec1::from_vec_unchecked(Vec::from(items)) }),
        }
    }
}

impl<T> TryFrom<Vec<T>> for Vec1<T> {
    type Error = Vec<T>;

    fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { Vec1::from_vec_unchecked(items) }),
        }
    }
}

impl<T> Vacancy for Vec1<T> {
    fn vacancy(&self) -> usize {
        self.items.vacancy()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl Write for Vec1<u8> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.items.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn write_vectored(&mut self, buffers: &[IoSlice<'_>]) -> io::Result<usize> {
        let len = buffers.iter().map(|buffer| buffer.len()).sum();
        self.items.reserve(len);
        for buffer in buffers {
            self.items.extend_from_slice(buffer);
        }
        Ok(len)
    }

    fn write_all(&mut self, buffer: &[u8]) -> io::Result<()> {
        self.items.extend_from_slice(buffer);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct DrainSegment<'a, T> {
    drain: Drain<'a, T>,
    range: &'a mut PositionalRange,
    after: PositionalRange,
}

impl<'a, T> AsRef<[T]> for DrainSegment<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.drain.as_ref()
    }
}

impl<'a, T> DoubleEndedIterator for DrainSegment<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<'a, T> Drop for DrainSegment<'a, T> {
    fn drop(&mut self) {
        *self.range = self.after;
    }
}

impl<'a, T> ExactSizeIterator for DrainSegment<'a, T> {
    fn len(&self) -> usize {
        self.drain.len()
    }
}

impl<'a, T> FusedIterator for DrainSegment<'a, T> {}

impl<'a, T> Iterator for DrainSegment<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }
}

#[derive(Debug)]
pub struct SwapDrainSegment<'a, T> {
    drain: DrainSegment<'a, T>,
    swapped: Option<T>,
}

impl<'a, T> DoubleEndedIterator for SwapDrainSegment<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.drain.next();
        next.or_else(|| self.swapped.take())
    }
}

impl<'a, T> ExactSizeIterator for SwapDrainSegment<'a, T> {
    fn len(&self) -> usize {
        self.drain
            .len()
            .checked_add(if self.swapped.is_some() { 1 } else { 0 })
            .expect("overflow in iterator length")
    }
}

impl<'a, T> FusedIterator for SwapDrainSegment<'a, T> {}

impl<'a, T> Iterator for SwapDrainSegment<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let swapped = self.swapped.take();
        swapped.or_else(|| self.drain.next())
    }
}

impl<'a, T> VecSegment<'a, Vec<T>> {
    pub fn drain<R>(&mut self, range: R) -> DrainSegment<'_, T>
    where
        PositionalRange: Project<R, Output = PositionalRange>,
        R: RangeBounds<usize>,
    {
        let DrainRange {
            intersection,
            before,
            after,
        } = DrainRange::project_and_intersect(&self.range, &range::ordered_range_offsets(range));
        self.range = before;
        DrainSegment {
            drain: self.items.drain(intersection),
            range: &mut self.range,
            after,
        }
    }
}

impl<'a, T> VecSegment<'a, Vec1<T>> {
    // This implementation, like `DrainSegment`, assumes that no items before the start of the
    // drain range are ever forgotten in the target `Vec`. The `Vec` documentation does not specify
    // this, but the implementation behaves this way and it is very reasonable behavior that is
    // very unlikely to change. This API is unsound if this assumption does not hold.
    pub fn drain<R>(&mut self, range: R) -> SwapDrainSegment<'_, T>
    where
        PositionalRange: Project<R, Output = PositionalRange>,
        R: RangeBounds<usize>,
    {
        let DrainRange {
            mut intersection,
            before,
            after,
        } = DrainRange::project_and_intersect(&self.range, &range::ordered_range_offsets(range));
        if self.range.is_prefix() && intersection.is_prefix() {
            // If both the segment and drain ranges are prefixes, then the target `Vec` may be left
            // empty if the drain iterator leaks (e.g., via `mem::forget`). Before the drain
            // operation begins and data beyond the start of the drain range may be left
            // uninitialized, the target `Vec` and drain must be configured such that the non-empty
            // invariant cannot be compromised.
            //
            // Because the segment is a prefix and strict subset of a `Vec1`, this code can assume
            // that there is at least one item beyond the end of the drain range. This item is
            // swapped with the first in the target `Vec` and the drain range is advanced by one.
            // This guarantees that the target `Vec` is not empty if the drain iterator leaks. If
            // the drain iterator is dropped, its range is removed from the target `Vec` and its
            // remainder is restored with the swapped item in the correct position.
            self.items
                .as_mut_slice()
                .swap(intersection.start, intersection.end);
            intersection.advance_by(1);
            self.range = before;
            let mut drain = DrainSegment {
                drain: self.items.drain(intersection),
                range: &mut self.range,
                after,
            };
            let swapped = drain.next_back();
            SwapDrainSegment { drain, swapped }
        }
        else {
            self.range = before;
            SwapDrainSegment {
                drain: DrainSegment {
                    drain: self.items.drain(intersection),
                    range: &mut self.range,
                    after,
                },
                swapped: None,
            }
        }
    }
}

impl<'a, K, T> VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    pub fn split_off(&mut self, at: usize) -> Vec<T> {
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
        let basis = self.len();
        if len > basis {
            let n = len - basis;
            self.extend(iter::repeat_with(f).take(n))
        }
        else {
            self.truncate(len)
        }
    }

    pub fn truncate(&mut self, len: usize) {
        let basis = self.len();
        if len < basis {
            let n = basis - len;
            self.items.drain((self.range.end - n)..self.range.end);
            self.range.take_from_end(n);
        }
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let index = self.range.project(&index).expect_in_bounds();
        self.items.insert(index, item);
        self.range.put_from_end(1);
    }

    pub fn remove(&mut self, index: usize) -> T {
        let index = self.range.project(&index).expect_in_bounds();
        let item = self.items.remove(index);
        self.range.take_from_end(1);
        item
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        if self.range.is_empty() {
            panic!("index out of bounds")
        }
        else {
            let index = self.range.project(&index).expect_in_bounds();
            let swapped = self.range.end - 1;
            self.items.as_mut_slice().swap(index, swapped);
            let item = self.items.remove(swapped);
            self.range.take_from_end(1);
            item
        }
    }

    pub fn clear(&mut self) {
        self.items.drain(self.range.get_and_clear_from_end());
    }

    pub fn splice<R, I>(&mut self, range: R, replacement: I) -> Splice<'_, I::IntoIter>
    where
        PositionalRange: Project<R, Output = PositionalRange>,
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        let range = self.range.project(&range).expect_in_bounds();
        self.items.splice(range, replacement)
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items.as_slice()[self.range.start..self.range.end]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items.as_mut_slice()[self.range.start..self.range.end]
    }

    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, K, T> AsMut<[T]> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, K, T> AsRef<[T]> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, K, T> Borrow<[T]> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, K, T> BorrowMut<[T]> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, K, T> Deref for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, K, T> DerefMut for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, K, T> Eq for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
    T: Eq,
{
}

impl<'a, K, T> Extend<T> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
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

// TODO: At time of writing, this implementation conflicts with the `Extend` implementation above
//       (E0119). However, `T` does not generalize `&'i T` here, because the associated `Target`
//       type is the same (`Vec<T>`) in both implementations (and a reference would be added to all
//       `T`)! This appears to be a limitation rather than a true conflict.
//
// impl<'a, 'i, K, T> Extend<&'i T> for VecSegment<'a, K>
// where
//     K: Segmentation<Target = Vec<T>>,
//     T: 'i + Copy,
// {
//     fn extend<I>(&mut self, items: I)
//     where
//         I: IntoIterator<Item = &'i T>,
//     {
//         self.extend(items.into_iter().copied())
//     }
// }

impl<'a, K, T, I> ExtendUntil<I> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<'a, K, T> Ord for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, KT, KU, T, U> PartialEq<VecSegment<'a, KU>> for VecSegment<'a, KT>
where
    KT: SegmentedOver<Target = Vec<T>>,
    KU: SegmentedOver<Target = Vec<U>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &VecSegment<'a, KU>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, K, T> PartialOrd<Self> for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, K, T> Segmentation for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn tail(&mut self) -> VecSegment<'_, K> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> VecSegment<'_, K> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T, R> segment::SegmentedBy<R> for VecSegment<'a, K>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: segment::SegmentedBy<R> + SegmentedOver<Target = Vec<T>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecSegment<'_, K> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T> Vacancy for VecSegment<'a, K>
where
    K: SegmentedOver<Target = Vec<T>>,
{
    fn vacancy(&self) -> usize {
        self.items.vacancy()
    }
}

#[derive(Debug)]
struct DrainRange {
    intersection: PositionalRange,
    before: PositionalRange,
    after: PositionalRange,
}

impl DrainRange {
    fn project_and_intersect<R>(segment: &PositionalRange, range: &R) -> Self
    where
        PositionalRange: Project<R, Output = PositionalRange>,
        R: RangeBounds<usize>,
    {
        let intersection = segment
            .intersect(&segment.project(range).expect_in_bounds())
            .expect_in_bounds();
        let before = From::from(
            segment.start
                ..intersection
                    .start
                    .checked_add(1)
                    .expect("overflow in segment end"),
        );
        let after = From::from(segment.start..(segment.end - intersection.len()));
        DrainRange {
            intersection,
            before,
            after,
        }
    }
}

#[macro_export]
macro_rules! vec1 {
    ($($item:expr $(,)?)+) => {{
        extern crate alloc;

        // SAFETY: There must be one or more `item` metavariables in the repetition.
        unsafe { $crate::vec1::Vec1::from_vec_unchecked(alloc::vec![$($item,)+]) }
    }};
}
pub use vec1;

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::iter1::{self, FromIterator1};
    use crate::vec1::Vec1;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> Vec1<u8> {
        Vec1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    use core::mem;
    use core::ops::RangeBounds;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::segment::range::{PositionalRange, Project};
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};
    use crate::slice1::{slice1, Slice1};
    use crate::vec1::harness::{self, xs1};
    use crate::vec1::Vec1;
    use crate::Segmentation;

    #[rstest]
    fn pop_from_vec1_until_and_after_only_then_vec1_eq_first(mut xs1: Vec1<u8>) {
        let first = *xs1.first();
        let mut tail = xs1.as_slice()[1..].to_vec();
        while let Ok(item) = xs1.pop_or_get_only() {
            assert_eq!(tail.pop().unwrap(), item);
        }
        for _ in 0..3 {
            assert_eq!(xs1.pop_or_get_only(), Err(&first));
        }
        assert_eq!(xs1.as_slice(), &[first]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_vec1_then_vec1_eq_head(#[case] mut xs1: Vec1<u8>) {
        xs1.tail().clear();
        assert_eq!(xs1.as_slice(), &[0]);
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_vec1_then_vec1_eq_tail(#[case] mut xs1: Vec1<u8>) {
        let tail = *xs1.last();
        xs1.rtail().clear();
        assert_eq!(xs1.as_slice(), &[tail]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_vec1_then_vec1_eq_head_and_tail(#[case] mut xs1: Vec1<u8>) {
        let n = xs1.len().get();
        let head_and_tail = [0, *xs1.last()];
        xs1.tail().rtail().clear();
        assert_eq!(
            xs1.as_slice(),
            if n > 1 {
                &head_and_tail[..]
            }
            else {
                &head_and_tail[..1]
            }
        );
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0), 1.., .., slice1![0])]
    #[case::one_tail(harness::xs1(1), 1.., .., slice1![0])]
    #[case::many_tail(harness::xs1(2), 1.., .., slice1![0])]
    #[case::many_tail(harness::xs1(2), 1.., 1.., slice1![0, 1])]
    #[case::many_tail(harness::xs1(2), 1.., ..1, slice1![0, 2])]
    #[case::empty_rtail(harness::xs1(0), ..0, .., slice1![0])]
    #[case::one_rtail(harness::xs1(1), ..1, .., slice1![1])]
    #[case::many_rtail(harness::xs1(2), ..2, .., slice1![2])]
    #[case::many_rtail(harness::xs1(2), ..2, 1.., slice1![0, 2])]
    fn drain_vec1_segment_then_vec1_eq<S, D>(
        #[case] mut xs1: Vec1<u8>,
        #[case] segment: S,
        #[case] drain: D,
        #[case] expected: &Slice1<u8>,
    ) where
        PositionalRange: Project<D, Output = PositionalRange>,
        S: RangeBounds<usize>,
        D: RangeBounds<usize>,
    {
        xs1.segment(segment).drain(drain);
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0), 1.., .., slice1![0])]
    #[case::one_tail(harness::xs1(1), 1.., .., slice1![0])]
    #[case::many_tail(harness::xs1(2), 1.., .., slice1![0])]
    #[case::many_tail(harness::xs1(2), 1.., 1.., slice1![0, 1])]
    #[case::empty_rtail(harness::xs1(0), ..0, .., slice1![0])]
    #[case::one_rtail(harness::xs1(1), ..1, .., slice1![1])]
    #[case::many_rtail(harness::xs1(2), ..2, .., slice1![2])]
    #[case::many_rtail(harness::xs1(2), ..2, 1.., slice1![0])]
    fn leak_drain_of_vec1_segment_then_vec1_eq<S, D>(
        #[case] mut xs1: Vec1<u8>,
        #[case] segment: S,
        #[case] drain: D,
        #[case] expected: &Slice1<u8>,
    ) where
        PositionalRange: Project<D, Output = PositionalRange>,
        S: RangeBounds<usize>,
        D: RangeBounds<usize>,
    {
        let mut segment = xs1.segment(segment);
        mem::forget(segment.drain(drain));
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_vec1_into_and_from_tokens_eq(
        xs1: Vec1<u8>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_vec1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<Vec1<u8>, Vec<_>>(sequence)
    }
}

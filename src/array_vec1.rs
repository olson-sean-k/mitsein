#![cfg(feature = "arrayvec")]
#![cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]

use arrayvec::ArrayVec;
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, RangeBounds};

use crate::array1::Array1;
use crate::iter1::{self, AndRemainder, FromIterator1, IntoIterator1, Iterator1};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segment, Segmentation, SegmentedOver};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{NonEmpty, OptionExt as _, Saturate, Saturated, Vacancy};

segment::impl_target_forward_type_and_definition!(
    for <T, [const N: usize]> => ArrayVec,
    ArrayVecTarget,
    ArrayVecSegment,
);

pub trait OrSaturated<T> {
    fn push_or_get_last(&mut self, item: T) -> Result<(), (T, &T)>;

    fn push_or_replace_last(&mut self, item: T) -> Result<(), T>;

    fn insert_or_get(&mut self, index: usize, item: T) -> Result<(), (T, &T)>;
}

impl<T, const N: usize> OrSaturated<T> for ArrayVec<T, N>
where
    [T; N]: Array1,
{
    fn push_or_get_last(&mut self, item: T) -> Result<(), (T, &T)> {
        self::push_or_get_last(self, item)
    }

    fn push_or_replace_last(&mut self, item: T) -> Result<(), T> {
        self::push_or_replace_last(self, item)
    }

    fn insert_or_get(&mut self, index: usize, item: T) -> Result<(), (T, &T)> {
        self::insert_or_get(self, index, item)
    }
}

impl<T, const N: usize> Ranged for ArrayVec<T, N> {
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

impl<T, I, const N: usize> Saturate<I> for ArrayVec<T, N>
where
    I: IntoIterator<Item = T>,
{
    type Remainder = I::IntoIter;

    fn saturate(&mut self, items: I) -> Self::Remainder {
        crate::saturate_positional_vacancy(self, items)
    }
}

impl<T, I, const N: usize> Saturated<I> for ArrayVec<T, N>
where
    I: IntoIterator<Item = T>,
{
    type Remainder = I::IntoIter;

    fn saturated(items: I) -> AndRemainder<Self, Self::Remainder> {
        let mut remainder = items.into_iter();
        let items: ArrayVec<_, N> = remainder.by_ref().take(N).collect();
        AndRemainder {
            output: items,
            remainder,
        }
    }
}

impl<T, const N: usize> Segmentation for ArrayVec<T, N> {
    fn tail(&mut self) -> ArrayVecSegment<'_, Self> {
        self.segment(Ranged::tail(self))
    }

    fn rtail(&mut self) -> ArrayVecSegment<'_, Self> {
        self.segment(Ranged::rtail(self))
    }
}

impl<T, R, const N: usize> segment::SegmentedBy<R> for ArrayVec<T, N>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> ArrayVecSegment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T, const N: usize> SegmentedOver for ArrayVec<T, N> {
    type Kind = ArrayVecTarget<Self>;
    type Target = Self;
}

impl<T, const N: usize> Vacancy for ArrayVec<T, N> {
    fn vacancy(&self) -> usize {
        self.capacity() - self.len()
    }
}

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
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn into_head_and_tail(mut self) -> (T, ArrayVec<T, N>) {
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_tail_and_head(mut self) -> (ArrayVec<T, N>, T) {
        // SAFETY:
        let head = unsafe { self.items.pop().unwrap_maybe_unchecked() };
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

    fn many_or_else<'a, U, M, O>(&'a mut self, many: M, one: O) -> Result<T, U>
    where
        M: FnOnce(&'a mut ArrayVec<T, N>) -> T,
        O: FnOnce(&'a mut ArrayVec<T, N>) -> U,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
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

    fn many_or_get_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut ArrayVec<T, N>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, |items| unsafe { items.get_unchecked(0) })
    }

    fn many_or_replace_only_with<F, R>(&mut self, f: F, replace: R) -> Result<T, T>
    where
        F: FnOnce(&mut ArrayVec<T, N>) -> T,
        R: FnOnce() -> T,
    {
        // SAFETY:
        self.many_or_else(f, move |items| unsafe {
            mem::replace(items.get_unchecked_mut(0), replace())
        })
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn pop_or_get_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_get_only(|items| unsafe { items.pop().unwrap_maybe_unchecked() })
    }

    pub fn pop_or_replace_only(&mut self, replacement: T) -> Result<T, T> {
        self.pop_or_replace_only_with(move || replacement)
    }

    pub fn pop_or_replace_only_with<F>(&mut self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        // SAFETY:
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

    // This function does not use `NonZeroExt`, because at time of writing it is not possible to
    // implement constant functions in traits.
    pub const fn len(&self) -> NonZeroUsize {
        #[cfg(all(not(miri), test))]
        {
            match NonZeroUsize::new(self.items.len()) {
                Some(len) => len,
                _ => panic!(),
            }
        }
        // SAFETY:
        #[cfg(not(all(not(miri), test)))]
        unsafe {
            NonZeroUsize::new_unchecked(self.items.len())
        }
    }

    // This function does not use `NonZeroExt`, because at time of writing it is not possible to
    // implement constant functions in traits.
    pub const fn capacity(&self) -> NonZeroUsize {
        #[cfg(all(not(miri), test))]
        {
            match NonZeroUsize::new(self.items.capacity()) {
                Some(capacity) => capacity,
                _ => panic!(),
            }
        }
        // SAFETY:
        #[cfg(not(all(not(miri), test)))]
        unsafe {
            NonZeroUsize::new_unchecked(self.items.capacity())
        }
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
        unsafe { ArrayVec1::from_array_vec_unchecked(items.into_iter1().collect()) }
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

impl<T, const N: usize> OrSaturated<T> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn push_or_get_last(&mut self, item: T) -> Result<(), (T, &T)> {
        self::push_or_get_last(&mut self.items, item)
    }

    fn push_or_replace_last(&mut self, item: T) -> Result<(), T> {
        self::push_or_replace_last(&mut self.items, item)
    }

    fn insert_or_get(&mut self, index: usize, item: T) -> Result<(), (T, &T)> {
        self::insert_or_get(&mut self.items, index, item)
    }
}

impl<T, I, const N: usize> Saturate<I> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    I: IntoIterator<Item = T>,
{
    type Remainder = I::IntoIter;

    fn saturate(&mut self, items: I) -> Self::Remainder {
        crate::saturate_positional_vacancy(self, items)
    }
}

impl<T, I, const N: usize> Saturated<I> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    I: IntoIterator1<Item = T>,
{
    type Remainder = I::IntoIter;

    fn saturated(items: I) -> AndRemainder<Self, Self::Remainder> {
        let mut remainder = items.into_iter1().into_iter();
        // SAFETY:
        let items =
            unsafe { ArrayVec1::from_array_vec_unchecked(remainder.by_ref().take(N).collect()) };
        AndRemainder {
            output: items,
            remainder,
        }
    }
}

impl<T, const N: usize> Segmentation for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn tail(&mut self) -> ArrayVecSegment<'_, Self> {
        self.segment(Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> ArrayVecSegment<'_, Self> {
        self.segment(Ranged::rtail(&self.items))
    }
}

impl<T, R, const N: usize> segment::SegmentedBy<R> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> ArrayVecSegment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T, const N: usize> SegmentedOver for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Kind = ArrayVecTarget<Self>;
    type Target = ArrayVec<T, N>;
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

impl<'a, K, T, const N: usize> ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    pub fn truncate(&mut self, len: usize) {
        let basis = self.len();
        if len < basis {
            let n = basis - len;
            self.items.drain((self.range.end - n)..self.range.end);
            self.range.take_from_end(n);
        }
    }

    pub fn push(&mut self, item: T) {
        self.items.insert(self.range.end, item);
        self.range.put_from_end(1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            let item = self.items.remove(self.range.end - 1);
            self.range.take_from_end(1);
            Some(item)
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

impl<'a, K, T, const N: usize> AsMut<[T]> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, K, T, const N: usize> AsRef<[T]> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, K, T, const N: usize> Borrow<[T]> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, K, T, const N: usize> BorrowMut<[T]> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, K, T, const N: usize> Deref for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, K, T, const N: usize> DerefMut for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, K, T, const N: usize> Eq for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
    T: Eq,
{
}

impl<'a, K, T, const N: usize> Extend<T> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let n = self.items.len();
        // Split off the remainder beyond the segment to avoid spurious inserts and copying. This
        // comes at the cost of a necessary array on the stack and bulk copy.
        let tail: ArrayVec<_, N> = self.items.drain(self.range.end..).collect();
        self.items.extend(items);
        self.items.extend(tail);
        let n = self.items.len() - n;
        self.range.put_from_end(n);
    }
}

impl<'a, K, T, I, const N: usize> Saturate<I> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
    I: IntoIterator<Item = T>,
{
    type Remainder = I::IntoIter;

    fn saturate(&mut self, items: I) -> Self::Remainder {
        crate::saturate_positional_vacancy(self, items)
    }
}

impl<'a, K, T, const N: usize> Ord for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, K, T, const N: usize> PartialEq<Self> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, K, T, const N: usize> PartialOrd<Self> for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<'a, K, T, const N: usize> Segmentation for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn tail(&mut self) -> ArrayVecSegment<'_, K> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> ArrayVecSegment<'_, K> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T, R, const N: usize> segment::SegmentedBy<R> for ArrayVecSegment<'a, K>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: segment::SegmentedBy<R> + SegmentedOver<Target = ArrayVec<T, N>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> ArrayVecSegment<'_, K> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T, const N: usize> Vacancy for ArrayVecSegment<'a, K>
where
    K: SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn vacancy(&self) -> usize {
        self.items.vacancy()
    }
}

fn vacancy_or_else<'a, T, U, E, V, S, const N: usize>(
    items: &'a mut ArrayVec<T, N>,
    item: T,
    vacant: V,
    saturated: S,
) -> Result<U, E>
where
    V: FnOnce(T, &'a mut ArrayVec<T, N>) -> U,
    S: FnOnce(T, &'a mut ArrayVec<T, N>) -> E,
{
    if items.len() < N {
        Ok(vacant(item, items))
    }
    else {
        Err(saturated(item, items))
    }
}

fn push_or_else<'a, T, U, F, const N: usize>(
    items: &'a mut ArrayVec<T, N>,
    item: T,
    f: F,
) -> Result<(), U>
where
    F: FnOnce(T, &'a mut ArrayVec<T, N>) -> U,
{
    self::vacancy_or_else(items, item, |item, items| items.push(item), f)
}

fn push_or_get_last<T, const N: usize>(items: &mut ArrayVec<T, N>, item: T) -> Result<(), (T, &T)>
where
    [T; N]: Array1,
{
    self::push_or_else(
        items,
        item,
        // SAFETY:
        |item, items| unsafe { (item, items.last().unwrap_maybe_unchecked()) },
    )
}

fn push_or_replace_last<T, const N: usize>(items: &mut ArrayVec<T, N>, item: T) -> Result<(), T>
where
    [T; N]: Array1,
{
    self::push_or_else(
        items,
        item,
        // SAFETY:
        |item, items| unsafe { mem::replace(items.last_mut().unwrap_maybe_unchecked(), item) },
    )
}

fn insert_or_else<'a, T, U, F, const N: usize>(
    items: &'a mut ArrayVec<T, N>,
    index: usize,
    item: T,
    f: F,
) -> Result<(), U>
where
    F: FnOnce(T, &'a mut ArrayVec<T, N>) -> U,
{
    self::vacancy_or_else(items, item, move |item, items| items.insert(index, item), f)
}

fn insert_or_get<T, const N: usize>(
    items: &mut ArrayVec<T, N>,
    index: usize,
    item: T,
) -> Result<(), (T, &T)>
where
    [T; N]: Array1,
{
    self::insert_or_else(items, index, item, move |item, items| (item, &items[index]))
}

#[cfg(test)]
mod tests {
    use arrayvec::ArrayVec;

    use crate::array_vec1::ArrayVec1;
    use crate::iter1::{IntoIterator1, IteratorExt as _};
    use crate::Segmentation;

    #[test]
    fn saturation() {
        let (xs, remainder): (ArrayVec<_, 3>, _) = [0i32, 1, 2, 3].into_iter().saturate().into();
        assert_eq!(xs.as_slice(), &[0, 1, 2]);
        assert!(remainder.eq([3]));

        let (xs, remainder): (ArrayVec<_, 4>, _) = [0i32, 1].into_iter1().saturate().into();
        assert_eq!(xs.as_slice(), &[0, 1]);
        assert!(remainder.eq([]));

        let (xs, remainder): (ArrayVec1<_, 3>, _) = [0i32, 1, 2, 3].into_iter1().saturate().into();
        assert_eq!(xs.as_slice(), &[0, 1, 2]);
        assert!(remainder.eq([3]));
    }

    #[test]
    fn segmentation() {
        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.tail().clear();
        assert_eq!(xs.as_slice(), &[0]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.tail().tail().clear();
        assert_eq!(xs.as_slice(), &[0, 1]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.rtail().clear();
        assert_eq!(xs.as_slice(), &[3]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.tail().rtail().clear();
        assert_eq!(xs.as_slice(), &[0, 3]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.tail().rtail().truncate(1);
        assert_eq!(xs.as_slice(), &[0, 1, 3]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.tail().clear();
        xs.tail().extend([4, 5, 6]);
        assert_eq!(xs.as_slice(), &[0, 4, 5, 6]);

        let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
        xs.rtail().clear();
        xs.rtail().extend([4, 5, 6]);
        assert_eq!(xs.as_slice(), &[4, 5, 6, 3]);
    }
}

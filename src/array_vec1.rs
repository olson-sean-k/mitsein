//! A non-empty [`ArrayVec`].

#![cfg(feature = "arrayvec")]
#![cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]

#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use arrayvec::{ArrayVec, CapacityError};
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, RangeBounds};
#[cfg(feature = "std")]
use {
    core::cmp,
    std::io::{self, Write},
};

use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{self, ArrayVecExt as _, OptionExt as _};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::slice1::Slice1;
use crate::take;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K, const N: usize> = <K as ClosedArrayVec<N>>::Item;

// TODO: At time of writing, Rust does not support generic parameters in `const` expressions and
//       operatations, so `N` is an input of this trait. When support lands, factor `N` into the
//       trait:
//
//       pub trait ClosedArrayVec {
//           ...
//           const N: usize;
//
//           fn as_array_vec(&self) -> &ArrayVec<Self::Item, { Self::N }>;
//       }
//
//       This factorization applies to many types and implementations below as well.
pub trait ClosedArrayVec<const N: usize> {
    type Item;

    fn as_array_vec(&self) -> &ArrayVec<Self::Item, N>;
}

impl<T, const N: usize> ClosedArrayVec<N> for ArrayVec<T, N> {
    type Item = T;

    fn as_array_vec(&self) -> &ArrayVec<Self::Item, N> {
        self
    }
}

impl<T, const N: usize> Extend1<T> for ArrayVec<T, N>
where
    // This bound isn't necessary for memory safety here, because an `ArrayVec` with no capacity
    // panics when any item is inserted, so `extend_non_empty` panics. However, this bound is
    // logically appropriate and prevents the definition of a function that always panics and has a
    // nonsense output type.
    [T; N]: Array1,
{
    fn extend_non_empty<I>(mut self, items: I) -> ArrayVec1<T, N>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The bound `[T; N]: Array1` guarantees that capacity is non-zero, input iterator
        //         `items` is non-empty, and `extend` either pushes one or more items or panics, so
        //         `self` must be non-empty here.
        unsafe { ArrayVec1::from_array_vec_unchecked(self) }
    }
}

unsafe impl<T, const N: usize> MaybeEmpty for ArrayVec<T, N> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
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

impl<T, const N: usize> Segmentation for ArrayVec<T, N> {
    fn tail(&mut self) -> Segment<'_, Self, N> {
        Segmentation::segment(self, Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self, N> {
        Segmentation::segment(self, Ranged::rtail(self))
    }
}

impl<T, R, const N: usize> SegmentedBy<R> for ArrayVec<T, N>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self, N> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T, const N: usize> SegmentedOver for ArrayVec<T, N> {
    type Target = Self;
    type Kind = Self;
}

type TakeOr<'a, T, U, M, const N: usize> = take::TakeOr<'a, ArrayVec<T, N>, U, M>;

pub type PopOr<'a, K, const N: usize> = TakeOr<'a, ItemFor<K, N>, ItemFor<K, N>, (), N>;

pub type SwapPopOr<'a, K, const N: usize> =
    TakeOr<'a, ItemFor<K, N>, Option<ItemFor<K, N>>, usize, N>;

pub type RemoveOr<'a, K, const N: usize> = TakeOr<'a, ItemFor<K, N>, ItemFor<K, N>, usize, N>;

impl<'a, T, M, const N: usize> TakeOr<'a, T, T, M, N>
where
    [T; N]: Array1,
{
    pub fn get_only(self) -> Result<T, &'a T> {
        self.take_or_else(|items, _| items.first())
    }

    pub fn replace_only(self, replacement: T) -> Result<T, T> {
        self.else_replace_only(move || replacement)
    }

    pub fn else_replace_only<F>(self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.take_or_else(move |items, _| mem::replace(items.first_mut(), f()))
    }
}

impl<'a, T, const N: usize> TakeOr<'a, T, T, usize, N>
where
    [T; N]: Array1,
{
    pub fn get(self) -> Result<T, &'a T> {
        self.take_or_else(|items, index| &items[index])
    }

    pub fn replace(self, replacement: T) -> Result<T, T> {
        self.else_replace(move || replacement)
    }

    pub fn else_replace<F>(self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.take_or_else(move |items, index| mem::replace(&mut items[index], f()))
    }
}

impl<'a, T, const N: usize> TakeOr<'a, T, Option<T>, usize, N>
where
    [T; N]: Array1,
{
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
            items.get_mut(index).map(|item| mem::replace(item, f()))
        })
    }
}

pub type ArrayVec1<T, const N: usize> = NonEmpty<ArrayVec<T, N>>;

impl<T, const N: usize> ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`ArrayVec::new()`][`ArrayVec::new`].
    ///
    /// [`ArrayVec::new`]: arrayvec::ArrayVec::new
    pub unsafe fn from_array_vec_unchecked(items: ArrayVec<T, N>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn from_one(item: T) -> Self {
        unsafe {
            // SAFETY: `items` must contain `item` and therefore is non-empty here.
            ArrayVec1::from_array_vec_unchecked({
                let mut items = ArrayVec::new();
                // SAFETY: The bound on `[T; N]: Array1` guarantees that `items` has vacancy for a
                //         first item here.
                items.push_maybe_unchecked(item);
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
        // SAFETY: `self` must be non-empty.
        let head = unsafe { self.items.pop().unwrap_maybe_unchecked() };
        (self.items, head)
    }

    pub fn into_array_vec(self) -> ArrayVec<T, N> {
        self.items
    }

    pub fn try_into_array(self) -> Result<[T; N], Self> {
        self.items
            .into_inner()
            // SAFETY: `self` must be non-empty.
            .map_err(|items| unsafe { ArrayVec1::from_array_vec_unchecked(items) })
    }

    /// # Safety
    ///
    /// The `ArrayVec1` must be saturated (length equals capacity), otherwise the output is
    /// uninitialized and unsound.
    pub unsafe fn into_array_unchecked(self) -> [T; N] {
        self.items.into_inner_unchecked()
    }

    pub fn try_extend_from_slice(&mut self, items: &[T]) -> Result<(), CapacityError>
    where
        T: Copy,
    {
        self.items.try_extend_from_slice(items)
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn try_push(&mut self, item: T) -> Result<(), CapacityError<T>> {
        self.items.try_push(item)
    }

    /// # Safety
    ///
    /// The `ArrayVec1` must have vacancy (available capacity) for the given item. Calling this
    /// function against a saturated `ArrayVec1` is undefined behavior.
    pub unsafe fn push_unchecked(&mut self, item: T) {
        self.items.push_unchecked(item)
    }

    pub fn pop_or(&mut self) -> PopOr<'_, Self, N> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn swap_pop_or(&mut self, index: usize) -> SwapPopOr<'_, Self, N> {
        TakeOr::with(self, index, |items, index| items.items.swap_pop(index))
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn try_insert(&mut self, index: usize, item: T) -> Result<(), CapacityError<T>> {
        self.items.try_insert(index, item)
    }

    pub fn remove_or(&mut self, index: usize) -> RemoveOr<'_, Self, N> {
        TakeOr::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove_or(&mut self, index: usize) -> RemoveOr<'_, Self, N> {
        TakeOr::with(self, index, |items, index| items.items.swap_remove(index))
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.len()) }
    }

    pub const fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.capacity()) }
    }

    pub const fn as_array_vec(&self) -> &ArrayVec<T, N> {
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

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T, const N: usize> Arbitrary<'a> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Arbitrary<'a>,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(
            T::arbitrary(unstructured),
            unstructured.arbitrary_iter()?.take(N - 1),
        )
        .collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        let item = T::size_hint(depth).0;
        (item, Some(item.saturating_mul(N)))
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

impl<T, const N: usize> ClosedArrayVec<N> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Item = T;

    fn as_array_vec(&self) -> &ArrayVec<Self::Item, N> {
        self.as_ref()
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
        // SAFETY: `items` is non-empty.
        unsafe { ArrayVec1::from_array_vec_unchecked(ArrayVec::from(items)) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Copy,
{
    fn from(items: &'a [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
        unsafe { ArrayVec1::from_array_vec_unchecked(items.into_iter().collect()) }
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
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, const N: usize> Segmentation for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn tail(&mut self) -> Segment<'_, Self, N> {
        Segmentation::segment(self, Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> Segment<'_, Self, N> {
        Segmentation::segment(self, Ranged::rtail(&self.items))
    }
}

impl<T, R, const N: usize> SegmentedBy<R> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self, N> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T, const N: usize> SegmentedOver for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Target = ArrayVec<T, N>;
    type Kind = Self;
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
            // SAFETY: `items` is non-empty.
            .map(|items| unsafe { ArrayVec1::from_array_vec_unchecked(items) })
            .map_err(|_| items)
    }
}

impl<T, const N: usize> TryFrom<ArrayVec<T, N>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = ArrayVec<T, N>;

    fn try_from(items: ArrayVec<T, N>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<const N: usize> Write for ArrayVec1<u8, N>
where
    [u8; N]: Array1,
{
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let len = cmp::min(self.items.capacity() - self.items.len(), buffer.len());
        self.items.extend(buffer.iter().take(len).copied());
        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub type Segment<'a, K, const N: usize> = segment::Segment<'a, K, ArrayVec<ItemFor<K, N>, N>>;

impl<K, T, const N: usize> Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    pub fn truncate(&mut self, len: usize) {
        let basis = self.len();
        if len < basis {
            let n = basis - len;
            self.items.drain((self.range.end - n)..self.range.end);
            self.range.take_from_end(n);
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.items.retain(self.range.retain_in_bounds(f))
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let index = self.range.project(&index).expect_in_bounds();
        self.items.insert(index, item);
        self.range.put_from_end(1);
    }

    pub fn insert_back(&mut self, item: T) {
        self.items.insert(self.range.end, item);
        self.range.put_from_end(1);
    }

    pub fn remove(&mut self, index: usize) -> T {
        let index = self.range.project(&index).expect_in_bounds();
        let item = self.items.remove(index);
        self.range.take_from_end(1);
        item
    }

    pub fn remove_back(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            let item = self.items.remove(self.range.end - 1);
            self.range.take_from_end(1);
            Some(item)
        }
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

impl<K, T, const N: usize> AsMut<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> AsRef<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, const N: usize> Borrow<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, const N: usize> BorrowMut<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> Deref for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<K, T, const N: usize> DerefMut for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> Eq for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
    T: Eq,
{
}

impl<K, T, const N: usize> Extend<T> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
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

impl<K, T, const N: usize> Ord for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<K, T, const N: usize> PartialEq<Self> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<K, T, const N: usize> PartialOrd<Self> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<K, T, const N: usize> Segmentation for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + SegmentedOver<Target = ArrayVec<T, N>>,
{
    fn tail(&mut self) -> Segment<'_, K, N> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> Segment<'_, K, N> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<K, T, R, const N: usize> SegmentedBy<R> for Segment<'_, K, N>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: ClosedArrayVec<N, Item = T> + SegmentedBy<R> + SegmentedOver<Target = ArrayVec<T, N>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, K, N> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::array_vec1::ArrayVec1;
    use crate::iter1::{self, FromIterator1};

    pub const CAPACITY: usize = 10;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> ArrayVec1<u8, CAPACITY> {
        ArrayVec1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {arrayvec::ArrayVec, serde_test::Token};

    use crate::array_vec1::harness::{self, CAPACITY};
    use crate::array_vec1::ArrayVec1;
    use crate::Segmentation;
    #[cfg(feature = "serde")]
    use crate::{
        array_vec1::harness::xs1,
        serde::{self, harness::sequence},
    };

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn clear_tail_of_array_vec1_then_array_vec1_eq_head(#[case] mut xs1: ArrayVec1<u8, CAPACITY>) {
        xs1.tail().clear();
        assert_eq!(xs1.as_slice(), &[0]);
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_array_vec1_then_array_vec1_eq_tail(#[case] mut xs1: ArrayVec1<u8, CAPACITY>) {
        let tail = *xs1.last();
        xs1.rtail().clear();
        assert_eq!(xs1.as_slice(), &[tail]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_array_vec1_then_array_vec1_eq_head_and_tail(
        #[case] mut xs1: ArrayVec1<u8, CAPACITY>,
    ) {
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

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_array_vec1_into_and_from_tokens_eq(
        xs1: ArrayVec1<u8, CAPACITY>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, ArrayVec<_, 16>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_array_vec1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<ArrayVec1<u8, 1>, ArrayVec<_, 16>>(
            sequence,
        )
    }
}

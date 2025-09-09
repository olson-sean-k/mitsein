//! A non-empty [`SmallVec`].

#![cfg(feature = "smallvec")]
#![cfg_attr(docsrs, doc(cfg(feature = "smallvec")))]

use alloc::borrow::{Borrow, BorrowMut};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter;
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::slice;
use smallvec::{Array, SmallVec};
#[cfg(feature = "std")]
use std::io::{self, IoSlice, Write};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _};
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::slice1::Slice1;
use crate::take;
use crate::vec1::Vec1;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ArrayFor<K> = <K as ClosedSmallVec>::Array;
type ItemFor<K> = <K as ClosedSmallVec>::Item;

pub trait ClosedSmallVec {
    type Array: Array<Item = Self::Item>;
    type Item;

    fn as_small_vec(&self) -> &SmallVec<Self::Array>;
}

impl<A> ClosedSmallVec for SmallVec<A>
where
    A: Array,
{
    type Array = A;
    type Item = A::Item;

    fn as_small_vec(&self) -> &SmallVec<Self::Array> {
        self
    }
}

impl<A> Extend1<A::Item> for SmallVec<A>
where
    A: Array,
{
    fn extend_non_empty<I>(mut self, items: I) -> SmallVec1<A>
    where
        I: IntoIterator1<Item = A::Item>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { SmallVec1::from_maybe_empty_unchecked(self) }
    }
}

unsafe impl<A> MaybeEmpty for SmallVec<A>
where
    A: Array,
{
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_slice().cardinality()
    }
}

impl<A> Ranged for SmallVec<A>
where
    A: Array,
{
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

impl<A> Segmentation for SmallVec<A>
where
    A: Array,
{
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(self))
    }
}

impl<A, R> SegmentedBy<R> for SmallVec<A>
where
    A: Array,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<A> SegmentedOver for SmallVec<A>
where
    A: Array,
{
    type Target = Self;
    type Kind = Self;
}

type Take<'a, A, T, N = ()> = take::Take<'a, SmallVec<A>, T, N>;

pub type Pop<'a, K> = Take<'a, ArrayFor<K>, ItemFor<K>, ()>;

pub type Remove<'a, K> = Take<'a, ArrayFor<K>, ItemFor<K>, usize>;

impl<'a, A, T, N> Take<'a, A, T, N>
where
    A: Array<Item = T>,
{
    pub fn or_get_only(self) -> Result<T, &'a T> {
        self.take_or_else(|items, _| items.first())
    }

    pub fn or_replace_only(self, replacement: T) -> Result<T, T> {
        self.or_else_replace_only(move || replacement)
    }

    pub fn or_else_replace_only<F>(self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.take_or_else(move |items, _| mem::replace(items.first_mut(), f()))
    }
}

impl<'a, A, T> Take<'a, A, T, usize>
where
    A: Array<Item = T>,
{
    pub fn or_get(self) -> Result<T, &'a T> {
        self.take_or_else(|items, index| &items[index])
    }

    pub fn or_replace(self, replacement: T) -> Result<T, T> {
        self.or_else_replace(move || replacement)
    }

    pub fn or_else_replace<F>(self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        self.take_or_else(move |items, index| mem::replace(&mut items[index], f()))
    }
}

pub type SmallVec1<A> = NonEmpty<SmallVec<A>>;

impl<A, T> SmallVec1<A>
where
    A: Array<Item = T>,
{
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`SmallVec::new()`][`SmallVec::new`].
    ///
    /// [`SmallVec::new`]: smallvec::SmallVec::new
    pub unsafe fn from_small_vec_unchecked(items: SmallVec<A>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        SmallVec1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = SmallVec::with_capacity(capacity);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
        unsafe { SmallVec1::from_small_vec_unchecked(items) }
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

    pub fn into_head_and_tail(mut self) -> (T, SmallVec<A>) {
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_tail_and_head(mut self) -> (SmallVec<A>, T) {
        // SAFETY: `self` must be non-empty.
        let head = unsafe { self.items.pop().unwrap_maybe_unchecked() };
        (self.items, head)
    }

    pub fn into_small_vec(self) -> SmallVec<A> {
        self.items
    }

    pub fn try_into_array(self) -> Result<A, Self> {
        self.items
            .into_inner()
            // SAFETY: `self` must be non-empty.
            .map_err(|items| unsafe { SmallVec1::from_small_vec_unchecked(items) })
    }

    pub fn into_boxed_slice1(self) -> BoxedSlice1<T> {
        // SAFETY: `self` must be non-empty.
        unsafe { BoxedSlice1::from_boxed_slice_unchecked(self.items.into_boxed_slice()) }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.items.reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn append(&mut self, items: &mut SmallVec<A>) {
        self.items.append(items)
    }

    pub fn extend_from_slice(&mut self, items: &[T])
    where
        T: Copy,
    {
        self.items.extend_from_slice(items)
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn pop(&mut self) -> Pop<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        Take::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn insert_from_slice(&mut self, index: usize, items: &[T])
    where
        T: Copy,
    {
        self.items.insert_from_slice(index, items)
    }

    pub fn remove(&mut self, index: usize) -> Remove<'_, Self> {
        Take::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove(&mut self, index: usize) -> Remove<'_, Self> {
        Take::with(self, index, |items, index| items.items.swap_remove(index))
    }

    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.items.dedup()
    }

    pub fn dedup_by<F>(&mut self, f: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        self.items.dedup_by(f)
    }

    pub fn dedup_by_key<K, F>(&mut self, f: F)
    where
        K: PartialEq,
        F: FnMut(&mut T) -> K,
    {
        self.items.dedup_by_key(f)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn inline_size(&self) -> usize {
        self.items.inline_size()
    }

    pub const fn as_small_vec(&self) -> &SmallVec<A> {
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

    pub fn spilled(&self) -> bool {
        self.items.spilled()
    }
}

impl<A> SmallVec1<A>
where
    A: Array + Array1,
{
    pub fn from_buf(buf: A) -> Self {
        SmallVec1::from_buf_and_tail_len(buf, A::N.get() - 1)
    }

    pub fn from_buf_and_tail_len(buf: A, len: usize) -> Self {
        // A saturating increment is sufficient here, because the buffer can never contain
        // `usize::MAX + 1` items.
        // SAFETY: `A` is bound on `Array1`, so `buf` is non-empty. `len` is the length of the tail
        //         and so is incremented by one to include the head, so the vector is non-empty.
        unsafe {
            SmallVec1::from_small_vec_unchecked(SmallVec::from_buf_and_len(
                buf,
                len.saturating_add(1),
            ))
        }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, A, T> Arbitrary<'a> for SmallVec1<A>
where
    A: Array<Item = T>,
    T: Arbitrary<'a>,
{
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        iter1::head_and_tail(T::arbitrary(unstructured), unstructured.arbitrary_iter()?).collect1()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (T::size_hint(depth).0, None)
    }
}

impl<A, T> AsMut<[T]> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<A, T> AsMut<Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn as_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<A, T> AsRef<[T]> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<A, T> AsRef<Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn as_ref(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<A, T> Borrow<[T]> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn borrow(&self) -> &[T] {
        self.items.borrow()
    }
}

impl<A, T> Borrow<Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn borrow(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<A, T> BorrowMut<[T]> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.items.borrow_mut()
    }
}

impl<A, T> BorrowMut<Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn borrow_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<A, T> ClosedSmallVec for SmallVec1<A>
where
    A: Array<Item = T>,
{
    type Array = A;
    type Item = T;

    fn as_small_vec(&self) -> &SmallVec<Self::Array> {
        self.as_ref()
    }
}

impl<A, T> Debug for SmallVec1<A>
where
    A: Array<Item = T>,
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<A, T> Deref for SmallVec1<A>
where
    A: Array<Item = T>,
{
    type Target = Slice1<T>;

    fn deref(&self) -> &Self::Target {
        self.as_slice1()
    }
}

impl<A, T> DerefMut for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice1()
    }
}

impl<A, T> Extend<T> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<T, const N: usize> From<[T; N]> for SmallVec1<[T; N]>
where
    [T; N]: Array<Item = T> + Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(SmallVec::from(items)) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for SmallVec1<[T; N]>
where
    [T; N]: Array<Item = T> + Array1,
    T: Copy,
{
    fn from(items: &'a [T; N]) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(items.iter().copied().collect()) }
    }
}

impl<'a, T, const N: usize> From<&'a mut [T; N]> for SmallVec1<[T; N]>
where
    [T; N]: Array<Item = T> + Array1,
    T: Copy,
{
    fn from(items: &'a mut [T; N]) -> Self {
        SmallVec1::from(&*items)
    }
}

impl<'a, A, T> From<&'a Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(SmallVec::from(items.as_slice())) }
    }
}

impl<'a, A, T> From<&'a mut Slice1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
    T: Clone,
{
    fn from(items: &'a mut Slice1<T>) -> Self {
        SmallVec1::from(&*items)
    }
}

impl<A> From<SmallVec1<A>> for SmallVec<A>
where
    A: Array,
{
    fn from(items: SmallVec1<A>) -> Self {
        items.items
    }
}

impl<A, T> From<Vec1<T>> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn from(items: Vec1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(SmallVec::from(items.into_vec())) }
    }
}

impl<A, T> FromIterator1<T> for SmallVec1<A>
where
    A: Array<Item = T>,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(items.into_iter().collect()) }
    }
}

impl<A, I> Index<I> for SmallVec1<A>
where
    SmallVec<A>: Index<I>,
    A: Array,
{
    type Output = <SmallVec<A> as Index<I>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<A, I> IndexMut<I> for SmallVec1<A>
where
    SmallVec<A>: IndexMut<I>,
    A: Array,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<A, T> IntoIterator for SmallVec1<A>
where
    A: Array<Item = T>,
{
    type Item = T;
    type IntoIter = smallvec::IntoIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, A, T> IntoIterator for &'a SmallVec1<A>
where
    A: Array<Item = T>,
    T: 'a,
{
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, A, T> IntoIterator for &'a mut SmallVec1<A>
where
    A: Array<Item = T>,
    T: 'a,
{
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<A> IntoIterator1 for SmallVec1<A>
where
    A: Array,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<A> IntoIterator1 for &'_ SmallVec1<A>
where
    A: Array,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<A> IntoIterator1 for &'_ mut SmallVec1<A>
where
    A: Array,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<A, T> JsonSchema for SmallVec1<A>
where
    A: Array<Item = T>,
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        SmallVec::<A>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<SmallVec<A>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        SmallVec::<A>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        SmallVec::<A>::schema_id()
    }
}

impl<A> Segmentation for SmallVec1<A>
where
    A: Array,
{
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(&self.items))
    }
}

impl<A, R> SegmentedBy<R> for SmallVec1<A>
where
    A: Array,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<A> SegmentedOver for SmallVec1<A>
where
    A: Array,
{
    type Target = SmallVec<A>;
    type Kind = Self;
}

impl<'a, A, T> TryFrom<&'a [T]> for SmallVec1<A>
where
    A: Array<Item = T>,
    T: Clone,
{
    type Error = EmptyError<&'a [T]>;

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        Slice1::try_from_slice(items).map(SmallVec1::from)
    }
}

impl<A> TryFrom<SmallVec<A>> for SmallVec1<A>
where
    A: Array,
{
    type Error = EmptyError<SmallVec<A>>;

    fn try_from(items: SmallVec<A>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<A> Write for SmallVec1<A>
where
    A: Array<Item = u8>,
{
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

pub type Segment<'a, K> = segment::Segment<'a, K, SmallVec<ArrayFor<K>>>;

impl<K, A, T> Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
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
        let from = self.len();
        let to = len;
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

impl<K, A, T> AsMut<[T]> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, A, T> AsRef<[T]> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, A, T> Borrow<[T]> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, A, T> BorrowMut<[T]> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, A, T> Deref for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<K, A, T> DerefMut for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<K, A, T> Eq for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
    T: Eq,
{
}

impl<K, A, T> Extend<T> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
{
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let n = self.items.len();
        // TODO: This can be a very expensive operation. Implementations for other non-empty
        //       collections split the collection and then extend onto the end, though even this is
        //       inefficient. Use a faster implementation, though it will likely require unsafe
        //       code.
        self.items.insert_many(self.range.end, items);
        let n = self.items.len() - n;
        self.range.put_from_end(n);
    }
}

impl<K, A, T> Ord for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, KT, KU, AT, AU, T, U> PartialEq<Segment<'a, KU>> for Segment<'a, KT>
where
    KT: ClosedSmallVec<Array = AT> + SegmentedOver<Target = SmallVec<AT>>,
    KU: ClosedSmallVec<Array = AU> + SegmentedOver<Target = SmallVec<AU>>,
    AT: Array<Item = T>,
    AU: Array<Item = U>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &Segment<'a, KU>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<K, A, T> PartialOrd<Self> for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<K, A, T> Segmentation for Segment<'_, K>
where
    K: ClosedSmallVec<Array = A> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
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

impl<K, A, T, R> SegmentedBy<R> for Segment<'_, K>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: ClosedSmallVec<Array = A> + SegmentedBy<R> + SegmentedOver<Target = SmallVec<A>>,
    A: Array<Item = T>,
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
    use crate::small_vec1::SmallVec1;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> SmallVec1<[u8; 5]> {
        SmallVec1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::iter1::FromIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::small_vec1::SmallVec1;

    #[rstest]
    #[case::from_iter1(SmallVec1::from_iter1([0, 1, 2]))]
    #[case::from_iter1_with_capacity(SmallVec1::from_iter1_with_capacity([0, 1, 2], 1))]
    #[case::from_one(SmallVec1::from_one(0))]
    #[case::from_one_with_capacity(SmallVec1::from_one_with_capacity(0, 1))]
    fn small_vec1_with_zero_sized_buf_spills_and_is_non_empty(#[case] xs1: SmallVec1<[u8; 0]>) {
        assert_eq!(xs1.inline_size(), 0);
        assert!(!xs1.as_small_vec().is_empty());
    }

    #[cfg(feature = "schemars")]
    #[rstest]
    fn small_vec1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<SmallVec1<[u8; 5]>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
    }
}

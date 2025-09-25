//! A non-empty [`Vec`][`vec`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, ToOwned};
use alloc::vec::{self, Drain, Vec};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter::{self, FusedIterator};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::slice;
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "std")]
use std::io::{self, IoSlice, Write};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};

use crate::array1::Array1;
use crate::borrow1::CowSlice1;
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _};
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{FromParallelIterator1, IntoParallelIterator1, ParallelIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{
    self, Intersect, IntersectionExt as _, PositionalRange, Project, ProjectionExt as _,
};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::string1::String1;
use crate::take;
use crate::vec_deque1::VecDeque1;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedVec>::Item;

pub trait ClosedVec {
    type Item;

    fn as_vec(&self) -> &Vec<Self::Item>;
}

impl<T> ClosedVec for Vec<T> {
    type Item = T;

    fn as_vec(&self) -> &Vec<Self::Item> {
        self
    }
}

impl<T> Extend1<T> for Vec<T> {
    fn extend_non_empty<I>(mut self, items: I) -> Vec1<T>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { Vec1::from_maybe_empty_unchecked(self) }
    }
}

unsafe impl<T> MaybeEmpty for Vec<T> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_slice().cardinality()
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
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(self))
    }
}

impl<T, R> SegmentedBy<R> for Vec<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for Vec<T> {
    type Target = Self;
    type Kind = Self;
}

type TakeIfMany<'a, T, N = ()> = take::TakeIfMany<'a, Vec<T>, T, N>;

pub type PopIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, ()>;

pub type RemoveIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, usize>;

impl<'a, T, N> TakeIfMany<'a, T, N> {
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

impl<'a, T> TakeIfMany<'a, T, usize> {
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

pub type Vec1<T> = NonEmpty<Vec<T>>;

impl<T> Vec1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`Vec::new()`][`Vec::new`].
    ///
    /// [`Vec::new`]: alloc::vec::Vec::new
    pub unsafe fn from_vec_unchecked(items: Vec<T>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        Vec1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        U: IntoIterator1<Item = T>,
    {
        let items = {
            let mut xs = Vec::with_capacity(capacity);
            xs.extend(items);
            xs
        };
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `items` must be non-empty here.
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

    pub fn append(&mut self, items: &mut Vec<T>) {
        self.items.append(items)
    }

    pub fn extend_from_slice(&mut self, items: &[T])
    where
        T: Clone,
    {
        self.items.extend_from_slice(items)
    }

    pub fn extend_from_within<R>(&mut self, range: R)
    where
        T: Clone,
        R: RangeBounds<usize>,
    {
        self.items.extend_from_within(range)
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item)
    }

    pub fn pop_if_many(&mut self) -> PopIfMany<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| items.items.swap_remove(index))
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

// A bound `[T; N]: Array1` is not necessary here, because `Vec::into_flattened` panics when `N` is
// zero. See below.
impl<T, const N: usize> Vec1<[T; N]> {
    pub fn into_flattened(self) -> Vec1<T> {
        // SAFETY: `self` must be non-empty and `Vec::into_flattened` panics if `N` is zero, so the
        //         flattened `Vec` cannot be empty.
        unsafe { Vec1::from_vec_unchecked(self.items.into_flattened()) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a, T> Arbitrary<'a> for Vec1<T>
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

impl<T> ClosedVec for Vec1<T> {
    type Item = T;

    fn as_vec(&self) -> &Vec<Self::Item> {
        self.as_ref()
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

impl<'a, T> Extend<&'a T> for Vec1<T>
where
    T: 'a + Copy,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        self.items.extend(extension)
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

impl<'a, T, const N: usize> From<&'a mut [T; N]> for Vec1<T>
where
    [T; N]: Array1,
    T: Copy,
{
    fn from(items: &'a mut [T; N]) -> Self {
        Vec1::from(&*items)
    }
}

impl<T> From<BoxedSlice1<T>> for Vec1<T> {
    fn from(items: BoxedSlice1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.into_boxed_slice())) }
    }
}

impl<'a, T> From<CowSlice1<'a, T>> for Vec1<T>
where
    Slice1<T>: ToOwned<Owned = Vec1<T>>,
{
    fn from(items: CowSlice1<'a, T>) -> Self {
        items.into_owned()
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

impl<'a, T> From<&'a mut Slice1<T>> for Vec1<T>
where
    T: Clone,
{
    fn from(items: &'a mut Slice1<T>) -> Self {
        Vec1::from(&*items)
    }
}

impl<'a> From<&'a Str1> for Vec1<u8> {
    fn from(items: &'a Str1) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.as_str())) }
    }
}

impl From<String1> for Vec1<u8> {
    fn from(items: String1) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.into_string())) }
    }
}

impl<T> From<Vec1<T>> for Vec<T> {
    fn from(items: Vec1<T>) -> Self {
        items.items
    }
}

impl<T> From<VecDeque1<T>> for Vec1<T> {
    fn from(items: VecDeque1<T>) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.into_vec_deque())) }
    }
}

impl<T> FromIterator1<T> for Vec1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(items.into_iter().collect()) }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> FromParallelIterator1<T> for Vec1<T>
where
    T: Send,
{
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = T>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(items.into_par_iter().collect()) }
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

impl<'a, T> IntoIterator for &'a Vec1<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec1<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T> IntoIterator1 for Vec1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> IntoIterator1 for &'_ Vec1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<T> IntoIterator1 for &'_ mut Vec1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator for Vec1<T>
where
    T: Send,
{
    type Item = T;
    type Iter = <Vec<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.items.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a Vec1<T>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = <&'a Vec<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a mut Vec1<T>
where
    T: Send,
{
    type Item = &'a mut T;
    type Iter = <&'a mut Vec<T> as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for Vec1<T>
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
impl<T> IntoParallelIterator1 for &'_ Vec1<T>
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
impl<T> IntoParallelIterator1 for &'_ mut Vec1<T>
where
    T: Send,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(&mut self.items) }
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<T> JsonSchema for Vec1<T>
where
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        Vec::<T>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<Vec<T>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        Vec::<T>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        Vec::<T>::schema_id()
    }
}

crate::impl_partial_eq_for_non_empty!([for U, const N: usize in [U; N]] <= [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U, const N: usize in &[U; N]] <= [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in [U]] <= [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &[U]] <= [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &mut [U]] <= [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &Slice1<U>] == [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in &mut Slice1<U>] == [for T in Vec1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] => [for T in [T]]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] => [for T in &[T]]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] => [for T in &mut [T]]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] == [for T in &Slice1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in Vec1<U>] == [for T in &mut Slice1<T>]);

impl<T> Segmentation for Vec1<T> {
    fn tail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        Segmentation::segment(self, Ranged::rtail(&self.items))
    }
}

impl<T, R> SegmentedBy<R> for Vec1<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for Vec1<T> {
    type Target = Vec<T>;
    type Kind = Self;
}

impl<'a, T> TryFrom<&'a [T]> for Vec1<T>
where
    T: Clone,
{
    type Error = EmptyError<&'a [T]>;

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        Slice1::try_from_slice(items).map(Vec1::from)
    }
}

impl<'a, T> TryFrom<&'a mut [T]> for Vec1<T>
where
    T: Clone,
{
    type Error = EmptyError<&'a mut [T]>;

    fn try_from(items: &'a mut [T]) -> Result<Self, Self::Error> {
        Slice1::try_from_mut_slice(items).map(Vec1::from)
    }
}

impl<T> TryFrom<Vec<T>> for Vec1<T> {
    type Error = EmptyError<Vec<T>>;

    fn try_from(items: Vec<T>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
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

impl<T> AsRef<[T]> for DrainSegment<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.drain.as_ref()
    }
}

impl<T> DoubleEndedIterator for DrainSegment<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<T> Drop for DrainSegment<'_, T> {
    fn drop(&mut self) {
        *self.range = self.after;
    }
}

impl<T> ExactSizeIterator for DrainSegment<'_, T> {
    fn len(&self) -> usize {
        self.drain.len()
    }
}

impl<T> FusedIterator for DrainSegment<'_, T> {}

impl<T> Iterator for DrainSegment<'_, T> {
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

impl<T> DoubleEndedIterator for SwapDrainSegment<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.drain.next();
        next.or_else(|| self.swapped.take())
    }
}

impl<T> ExactSizeIterator for SwapDrainSegment<'_, T> {
    fn len(&self) -> usize {
        self.drain
            .len()
            .checked_add(if self.swapped.is_some() { 1 } else { 0 })
            .expect("overflow in iterator length")
    }
}

impl<T> FusedIterator for SwapDrainSegment<'_, T> {}

impl<T> Iterator for SwapDrainSegment<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let swapped = self.swapped.take();
        swapped.or_else(|| self.drain.next())
    }
}

pub type Segment<'a, K> = segment::Segment<'a, K, Vec<ItemFor<K>>>;

impl<T> Segment<'_, Vec<T>> {
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

impl<T> Segment<'_, Vec1<T>> {
    // This implementation, like `DrainSegment`, assumes that no items before the start of the
    // drain range are ever forgotten in the target `Vec`. The `Vec` documentation does not specify
    // this, but the implementation behaves this way and it is very reasonable behavior that is
    // very unlikely to change. This API is unsound if this assumption does not hold.
    pub fn swap_drain<R>(&mut self, range: R) -> SwapDrainSegment<'_, T>
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

impl<K, T> Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
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

impl<K, T> AsMut<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T> AsRef<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T> Borrow<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T> BorrowMut<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T> Deref for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<K, T> DerefMut for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<K, T> Eq for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
    T: Eq,
{
}

impl<K, T> Extend<T> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
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
//       `T`)! This appears to be a limitation rather than a true conflict. See other segment
//       implementations as well.
//
// impl<'i, K, T> Extend<&'i T> for Segment<'_, K>
// where
//     K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
//     T: 'i + Copy,
// {
//     fn extend<I>(&mut self, items: I)
//     where
//         I: IntoIterator<Item = &'i T>,
//     {
//         self.extend(items.into_iter().copied())
//     }
// }

impl<K, T> Ord for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, KT, KU, T, U> PartialEq<Segment<'a, KU>> for Segment<'a, KT>
where
    KT: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
    KU: ClosedVec<Item = U> + SegmentedOver<Target = Vec<U>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &Segment<'a, KU>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<K, T> PartialOrd<Self> for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<K, T> Segmentation for Segment<'_, K>
where
    K: ClosedVec<Item = T> + SegmentedOver<Target = Vec<T>>,
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
    K: ClosedVec<Item = T> + SegmentedBy<R> + SegmentedOver<Target = Vec<T>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, K> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
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

        let items = alloc::vec![$($item,)+];
        // SAFETY: There must be one or more `item` metavariables in the repetition.
        unsafe { $crate::vec1::Vec1::from_vec_unchecked(items) }
    }};
    ($item:expr ; $N:literal) => {{
        extern crate alloc;

        const fn non_zero_usize_capacity<const N: usize>()
        where
            [(); N]: $crate::array1::Array1,
        {}
        non_zero_usize_capacity::<$N>();

        let items = alloc::vec![$item; $N];
        // SAFETY: The literal `$N` is non-zero.
        unsafe { $crate::vec1::Vec1::from_vec_unchecked(items) }
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
    use alloc::vec;
    use core::iter;
    use core::mem;
    use core::ops::RangeBounds;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    use crate::iter1::IntoIterator1;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::segment::range::{PositionalRange, Project};
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};
    use crate::slice1::{slice1, Slice1};
    use crate::vec1::harness::{self, xs1};
    use crate::vec1::Vec1;
    use crate::Segmentation;

    #[rstest]
    fn vec1_from_vec_macro_eq_vec1_from_vec1_macro_by_rep_expr() {
        assert_eq!(
            Vec1::try_from(vec![0u8, 1, 2, 3]).unwrap(),
            vec1![0u8, 1, 2, 3],
        );
    }

    #[rstest]
    fn vec1_from_vec_macro_eq_vec1_from_vec1_macro_by_expr_literal() {
        assert_eq!(Vec1::try_from(vec![0u8; 4]).unwrap(), vec1![0u8; 4]);
    }

    #[rstest]
    fn pop_if_many_from_vec1_until_and_after_only_then_vec1_eq_first(mut xs1: Vec1<u8>) {
        let first = *xs1.first();
        let mut tail = xs1.as_slice()[1..].to_vec();
        while let Ok(item) = xs1.pop_if_many().or_get_only() {
            assert_eq!(tail.pop().unwrap(), item);
        }
        for _ in 0..3 {
            assert_eq!(xs1.pop_if_many().or_get_only(), Err(&first));
        }
        assert_eq!(xs1.as_slice(), &[first]);
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
    fn insert_back_into_vec1_segment_then_vec1_eq<S, T>(
        mut xs1: Vec1<u8>,
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
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn remove_back_all_from_tail_of_vec1_then_vec1_eq_head(#[case] mut xs1: Vec1<u8>) {
        let n = xs1.len().get();
        let mut tail = xs1.tail();
        iter::from_fn(|| tail.remove_back())
            .take(n)
            .for_each(|_| {});
        assert!(tail.is_empty());
        assert_eq!(xs1.as_slice(), &[0]);
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
    #[case::tail(harness::xs1(3), 1..)]
    #[case::rtail(harness::xs1(3), ..3)]
    #[case::middle(harness::xs1(9), 4..8)]
    fn retain_none_from_vec1_segment_then_segment_is_empty<S>(
        #[case] mut xs1: Vec1<u8>,
        #[case] segment: S,
    ) where
        S: RangeBounds<usize>,
    {
        let mut xss = xs1.segment(segment);
        xss.retain(|_| false);
        assert!(xss.is_empty());
    }

    #[rstest]
    #[case::tail(harness::xs1(3), 1.., slice1![0])]
    #[case::rtail(harness::xs1(3), ..3, slice1![3])]
    #[case::middle(harness::xs1(9), 4..8, slice1![0, 1, 2, 3, 8, 9])]
    fn retain_none_from_vec1_segment_then_vec1_eq<S>(
        #[case] mut xs1: Vec1<u8>,
        #[case] segment: S,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
    {
        xs1.segment(segment).retain(|_| false);
        assert_eq!(xs1.as_slice1(), expected);
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
    fn swap_drain_from_vec1_segment_then_vec1_eq<S, D>(
        #[case] mut xs1: Vec1<u8>,
        #[case] segment: S,
        #[case] drain: D,
        #[case] expected: &Slice1<u8>,
    ) where
        PositionalRange: Project<D, Output = PositionalRange>,
        S: RangeBounds<usize>,
        D: RangeBounds<usize>,
    {
        xs1.segment(segment).swap_drain(drain);
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
    fn leak_swap_drain_of_vec1_segment_then_vec1_eq<S, D>(
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
        mem::forget(segment.swap_drain(drain));
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[cfg(feature = "schemars")]
    #[rstest]
    fn vec1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<Vec1<u8>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
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

mod _compile_fail_tests {
    /// ```compile_fail
    /// let xs = mitsein::vec1![];
    /// ```
    #[doc(hidden)]
    const fn _empty_rep_expr_metaparameters_then_vec1_compilation_fails() {}

    /// ```compile_fail,E0277
    /// let xs = mitsein::vec1![0u8; 0];
    /// ```
    #[doc(hidden)]
    const fn _empty_expr_literal_metaparameters_then_vec1_compilation_fails() {}
}

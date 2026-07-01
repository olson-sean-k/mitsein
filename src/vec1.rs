//! A non-empty [`Vec`][`vec`].

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, ToOwned};
use alloc::vec::{self, Drain, Vec};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter::{FusedIterator, Skip, Take};
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
use crate::range1::{Range1, RangeInclusive1};
use crate::safety::{self, NonZeroExt as _, OptionExt as _};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::string1::String1;
use crate::subset;
use crate::subset::range::{self, IndexRange, RangeError};
use crate::vec_deque1::VecDeque1;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, Many, MaybeEmpty, NonEmpty, One};

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

impl<T> Index<Range1<usize>> for Vec<T> {
    type Output = Slice1<T>;

    fn index(&self, at: Range1<usize>) -> &Self::Output {
        self.as_slice().index(at)
    }
}

impl<T> Index<RangeInclusive1<usize>> for Vec<T> {
    type Output = Slice1<T>;

    fn index(&self, at: RangeInclusive1<usize>) -> &Self::Output {
        self.as_slice().index(at)
    }
}

impl<T> IndexMut<Range1<usize>> for Vec<T> {
    fn index_mut(&mut self, at: Range1<usize>) -> &mut Self::Output {
        self.as_mut_slice().index_mut(at)
    }
}

impl<T> IndexMut<RangeInclusive1<usize>> for Vec<T> {
    fn index_mut(&mut self, at: RangeInclusive1<usize>) -> &mut Self::Output {
        self.as_mut_slice().index_mut(at)
    }
}

unsafe impl<T> MaybeEmpty for Vec<T> {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_slice().cardinality()
    }
}

pub type OrOnly<'a, O, M> = Cardinality<&'a mut O, M>;

pub type Vec1<T> = NonEmpty<Vec<T>>;

impl<T> Vec1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`Vec::new()`][`Vec::new`].
    ///
    /// [`Vec::new`]: alloc::vec::Vec::new
    pub unsafe fn from_vec_unchecked(items: Vec<T>) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    pub fn from_one(item: T) -> Self {
        iter1::once(item).collect1()
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

    pub fn from_rtail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::rtail_and_head(tail, head).collect1()
    }

    pub fn try_from_ref(items: &Vec<T>) -> Result<&'_ Self, EmptyError<&'_ Vec<T>>> {
        items.try_into()
    }

    pub fn try_from_mut(items: &mut Vec<T>) -> Result<&'_ mut Self, EmptyError<&'_ mut Vec<T>>> {
        items.try_into()
    }

    pub fn into_head_and_tail(mut self) -> (T, Vec<T>) {
        // SAFETY: `self` is non-empty.
        unsafe { safety::assume_is_non_empty_maybe_unchecked(&self.items) };
        let head = self.items.remove(0);
        (head, self.items)
    }

    pub fn into_rtail_and_head(mut self) -> (Vec<T>, T) {
        // SAFETY: `self` is non-empty.
        unsafe { safety::assume_is_non_empty_maybe_unchecked(&self.items) };
        // SAFETY: `self` is non-empty.
        let head = unsafe { self.items.pop().unwrap_maybe_unchecked() };
        (self.items, head)
    }

    pub fn into_vec(self) -> Vec<T> {
        self.items
    }

    pub fn into_boxed_slice1(self) -> BoxedSlice1<T> {
        // SAFETY: `self` is non-empty.
        unsafe { BoxedSlice1::from_boxed_slice_unchecked(self.items.into_boxed_slice()) }
    }

    pub fn leak<'a>(self) -> &'a mut Slice1<T> {
        // SAFETY: `self` is non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.items.leak()) }
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<Vec<T>>>
    where
        F: FnMut(&T) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }

    pub fn retain_until_only<F>(&mut self, mut f: F) -> Option<&'_ T>
    where
        F: FnMut(&T) -> bool,
    {
        self.rtail().retain(|item| f(item));
        if self.len().get() == 1 {
            let last = self.last();
            if f(last) { None } else { Some(last) }
        }
        else {
            if !f(self.last()) {
                // The last item is **not** retained and there is more than one item.
                self.pop_if_many();
            }
            None
        }
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

    fn get_only_or_else<U, F>(&mut self, f: F) -> OrOnly<'_, T, U>
    where
        F: FnOnce(&mut Self) -> U,
    {
        match self.cardinality() {
            One(_) => One(self.first_mut()),
            Many(_) => Many(f(self)),
        }
    }

    pub fn pop_if_many(&mut self) -> OrOnly<'_, T, T> {
        // SAFETY: `get_only_or_else` only calls the given function if `items` has many items.
        self.get_only_or_else(|items| unsafe { items.items.pop().unwrap_maybe_unchecked() })
    }

    pub fn pop_if_many_and<F>(&mut self, f: F) -> OrOnly<'_, T, Option<T>>
    where
        F: FnOnce(&mut T) -> bool,
    {
        self.get_only_or_else(|items| {
            if f(items.last_mut()) {
                items.items.pop()
            }
            else {
                None
            }
        })
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_if_many(&mut self, index: usize) -> OrOnly<'_, T, T> {
        self.get_only_or_else(|items| items.items.remove(index))
    }

    pub fn swap_remove_if_many(&mut self, index: usize) -> OrOnly<'_, T, T> {
        self.get_only_or_else(|items| items.items.swap_remove(index))
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
        // SAFETY: `self` is non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` is non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub const fn as_vec(&self) -> &Vec<T> {
        &self.items
    }

    /// # Safety
    ///
    /// The [`Vec`] behind the returned mutable reference **must not** be empty when the reference
    /// is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::prelude::*;
    ///
    /// let mut xs = vec1![0i32, 1, 2, 3];
    /// // This block is unsound. The `&mut Vec<_>` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_vec().clear();
    /// }
    /// let x = xs.first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        // SAFETY: `self` is non-empty.
        unsafe { Slice1::from_slice_unchecked(self.items.as_slice()) }
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        // SAFETY: `self` is non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.items.as_mut_slice()) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.items.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.items.as_mut_ptr()
    }
}

impl<T> Vec1<T> {
    pub fn only<R>(&mut self, range: R) -> Result<OnlyRangeSubset<'_, T>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        let n = self.items.len();
        OnlyRangeSubset::contacted_strict_subset(&mut self.items, n, range)
    }

    pub fn tail(&mut self) -> OnlyRangeSubset<'_, T> {
        let n = self.items.len();
        OnlyRangeSubset::from_tail_range(&mut self.items, n)
    }

    pub fn rtail(&mut self) -> OnlyRangeSubset<'_, T> {
        let n = self.items.len();
        OnlyRangeSubset::from_rtail_range(&mut self.items, n)
    }
}

// A bound `[T; N]: Array1` is not necessary here, because `Vec::into_flattened` panics when `N` is
// zero. See below.
impl<T, const N: usize> Vec1<[T; N]> {
    pub fn into_flattened(self) -> Vec1<T> {
        // SAFETY: `self` is non-empty and `Vec::into_flattened` panics if `N` is zero, so the
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
        // SAFETY: `items` is non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items)) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for Vec1<T>
where
    [T; N]: Array1,
    T: Copy,
{
    fn from(items: &'a [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.as_str())) }
    }
}

impl From<String1> for Vec1<u8> {
    fn from(items: String1) -> Self {
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from(items.into_vec_deque())) }
    }
}

impl<T> FromIterator1<T> for Vec1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `items` is non-empty.
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
        // SAFETY: `self` is non-empty.
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
        // SAFETY: `self` is non-empty.
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
        // SAFETY: `self` is non-empty.
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
        // SAFETY: `self` is non-empty.
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

impl<'a, T> TryFrom<&'a Vec<T>> for &'a Vec1<T> {
    type Error = EmptyError<&'a Vec<T>>;

    fn try_from(items: &'a Vec<T>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T> TryFrom<&'a mut Vec<T>> for &'a mut Vec1<T> {
    type Error = EmptyError<&'a mut Vec<T>>;

    fn try_from(items: &'a mut Vec<T>) -> Result<Self, Self::Error> {
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
struct DrainRange {
    contact: IndexRange,
    before: IndexRange,
    after: IndexRange,
}

impl DrainRange {
    fn project_and_contact<R>(subset: IndexRange, range: R) -> Result<Self, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        let contact = subset.contact(subset.project_range_bounds(range)?)?;
        let before = IndexRange::unchecked(
            subset.start(),
            contact
                .start()
                .checked_add(1)
                .unwrap_or_else(|| range::panic_end_overflow()),
        );
        let after = IndexRange::unchecked(subset.start(), subset.end() - contact.len());
        Ok(DrainRange {
            contact,
            before,
            after,
        })
    }
}

#[derive(Debug)]
struct DrainOnlyRangeSubset<'a, T> {
    drain: Drain<'a, T>,
    range: &'a mut IndexRange,
    after: IndexRange,
}

impl<T> AsRef<[T]> for DrainOnlyRangeSubset<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.drain.as_ref()
    }
}

impl<T> DoubleEndedIterator for DrainOnlyRangeSubset<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<T> Drop for DrainOnlyRangeSubset<'_, T> {
    fn drop(&mut self) {
        *self.range = self.after;
    }
}

impl<T> ExactSizeIterator for DrainOnlyRangeSubset<'_, T> {
    fn len(&self) -> usize {
        self.drain.len()
    }
}

impl<T> FusedIterator for DrainOnlyRangeSubset<'_, T> {}

impl<T> Iterator for DrainOnlyRangeSubset<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }
}

#[derive(Debug)]
pub struct SwapDrainOnlyRangeSubset<'a, T> {
    drain: DrainOnlyRangeSubset<'a, T>,
    swapped: Option<T>,
}

impl<T> DoubleEndedIterator for SwapDrainOnlyRangeSubset<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let next = self.drain.next();
        next.or_else(|| self.swapped.take())
    }
}

impl<T> ExactSizeIterator for SwapDrainOnlyRangeSubset<'_, T> {
    fn len(&self) -> usize {
        self.drain
            .len()
            .checked_add(if self.swapped.is_some() { 1 } else { 0 })
            .expect("overflow in iterator length")
    }
}

impl<T> FusedIterator for SwapDrainOnlyRangeSubset<'_, T> {}

impl<T> Iterator for SwapDrainOnlyRangeSubset<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let swapped = self.swapped.take();
        swapped.or_else(|| self.drain.next())
    }
}

pub type OnlyRangeSubset<'a, T> = subset::OnlyRangeSubset<'a, Vec<T>, IndexRange>;

impl<T> OnlyRangeSubset<'_, T> {
    pub fn split_off(&mut self, at: usize) -> Vec<T> {
        let at = self
            .range
            .project_point(at)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        let range = IndexRange::unchecked(at, self.range.end());
        let items = self.items.drain(range).collect();
        self.range = range;
        items
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

    // This implementation assumes that no items before the start of the drain range are ever
    // forgotten in the target `Vec`. The `Vec` documentation does not specify this, but the
    // implementation behaves this way and it is very reasonable behavior that is very unlikely to
    // change. This API is unsound if this assumption does not hold.
    pub fn swap_drain<R>(&mut self, range: R) -> SwapDrainOnlyRangeSubset<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let DrainRange {
            mut contact,
            before,
            after,
        } = DrainRange::project_and_contact(self.range, range).expect("invalid drain range");
        if self.range.is_prefix() && contact.is_prefix() {
            // If both the subset and drain ranges are prefixes, then the target `Vec` may be left
            // empty if the drain iterator leaks (e.g., via `mem::forget`). Before the drain
            // operation begins and data beyond the start of the drain range may be left
            // uninitialized, the target `Vec` and drain must be configured such that the non-empty
            // invariant cannot be compromised.
            //
            // Because the `OnlyRangeSubset` is a prefix and strict subset of a `Vec1`, this code
            // can assume that there is at least one item beyond the end of the drain range. This
            // item is swapped with the first in the target `Vec` and the drain range is advanced by
            // one. This guarantees that the target `Vec` is not empty if the drain iterator leaks.
            // If the drain iterator is dropped, its range is removed from the target `Vec` and its
            // remainder is restored with the swapped item in the correct position.
            self.items
                .as_mut_slice()
                .swap(contact.start(), contact.end());
            contact.advance_by(1);
            self.range = before;
            let mut drain = DrainOnlyRangeSubset {
                drain: self.items.drain(contact),
                range: &mut self.range,
                after,
            };
            let swapped = drain.next_back();
            SwapDrainOnlyRangeSubset { drain, swapped }
        }
        else {
            self.range = before;
            SwapDrainOnlyRangeSubset {
                drain: DrainOnlyRangeSubset {
                    drain: self.items.drain(contact),
                    range: &mut self.range,
                    after,
                },
                swapped: None,
            }
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        let index = self
            .range
            .project_point(index)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        let item = self.items.remove(index);
        self.range.take_from_end(1);
        item
    }

    pub fn remove_back(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            let item = self.items.remove(self.range.end() - 1);
            self.range.take_from_end(1);
            Some(item)
        }
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        if self.range.is_empty() {
            range::panic_index_out_of_bounds()
        }
        else {
            let index = self
                .range
                .project_point(index)
                .unwrap_or_else(|_| range::panic_index_out_of_bounds());
            let swapped = self.range.end() - 1;
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

    pub fn iter(&self) -> Take<Skip<slice::Iter<'_, T>>> {
        self.items.iter().skip(self.range.start()).take(self.len())
    }

    pub fn iter_mut(&mut self) -> Take<Skip<slice::IterMut<'_, T>>> {
        let body = self.len();
        self.items.iter_mut().skip(self.range.start()).take(body)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items.as_slice()[self.range.start()..self.range.end()]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items.as_mut_slice()[self.range.start()..self.range.end()]
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

impl<T> OnlyRangeSubset<'_, T> {
    pub fn only<R>(&mut self, range: R) -> Result<OnlyRangeSubset<'_, T>, RangeError<usize>>
    where
        R: RangeBounds<usize>,
    {
        self.project_and_contact(range)
    }

    pub fn tail(&mut self) -> OnlyRangeSubset<'_, T> {
        self.project_tail_range()
    }

    pub fn rtail(&mut self) -> OnlyRangeSubset<'_, T> {
        let n = self.len();
        self.project_rtail_range(n)
    }
}

impl<T> AsMut<[T]> for OnlyRangeSubset<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for OnlyRangeSubset<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Borrow<[T]> for OnlyRangeSubset<'_, T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> BorrowMut<[T]> for OnlyRangeSubset<'_, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Deref for OnlyRangeSubset<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for OnlyRangeSubset<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Eq for OnlyRangeSubset<'_, T> where T: Eq {}

impl<T> Ord for OnlyRangeSubset<'_, T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, T, U> PartialEq<OnlyRangeSubset<'a, U>> for OnlyRangeSubset<'a, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &OnlyRangeSubset<'a, U>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T> PartialOrd<Self> for OnlyRangeSubset<'_, T>
where
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
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
    ($item:expr ; $N:expr) => {{
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
    use alloc::vec::Vec;
    use core::iter;
    use core::mem;
    use core::ops::RangeBounds;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    #[cfg(feature = "schemars")]
    use crate::schemars;
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};
    use crate::slice1::{Slice1, slice1};
    use crate::vec1::Vec1;
    use crate::vec1::harness::{self, xs1};
    use crate::{Many, One};

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

    // SAFETY: The `FnMut`s constructed in cases (the parameter `f`) must not stash or otherwise
    //         allow access to the parameter beyond the scope of their bodies. (This is difficult
    //         to achieve in this context.)
    #[rstest]
    #[case::ignore_and_retain(|_| true, (None, slice1![0, 1, 2, 3, 4]))]
    #[case::ignore_and_do_not_retain(|_| false, (Some(4), slice1![4]))]
    #[case::compare_and_retain_none(
        |x: *const _| unsafe {
            *x > 4
        },
        (Some(4), slice1![4]),
    )]
    #[case::compare_and_retain_some(
        |x: *const _| unsafe {
            *x < 3
        },
        (None, slice1![0, 1, 2]),
    )]
    fn retain_until_only_from_vec1_then_output_and_vec1_eq<F>(
        mut xs1: Vec1<u8>,
        #[case] mut f: F,
        #[case] expected: (Option<u8>, &Slice1<u8>),
    ) where
        F: FnMut(*const u8) -> bool,
    {
        // TODO: The type parameter `F` must be a `FnMut` over `*const u8` instead of `&u8` here,
        //       because `rstest` constructs the case in a way that the `&u8` has a lifetime that
        //       is too specific and too long (it would borrow the item beyond
        //       `retain_until_only`). Is there a way to prevent this without introducing `*const
        //       u8` and unsafe code in cases for `f`? If so, do that instead!
        let x = xs1.retain_until_only(|x| f(x as *const u8)).copied();
        assert_eq!((x, xs1.as_slice1()), expected);
    }

    #[rstest]
    fn pop_if_many_from_vec1_until_and_after_only_then_vec1_eq_first(mut xs1: Vec1<u8>) {
        let mut first = *xs1.first();
        let mut tail = xs1.as_slice()[1..].to_vec();
        while let Many(x) = xs1.pop_if_many() {
            assert_eq!(tail.pop().unwrap(), x);
        }
        for _ in 0..3 {
            assert_eq!(xs1.pop_if_many(), One(&mut first));
        }
        assert_eq!(xs1.as_slice(), &[first]);
    }

    // SAFETY: The `FnOnce`s constructed in cases (the parameter `f`) must not stash or otherwise
    //         allow access to the parameter beyond the scope of their bodies. (This is difficult
    //         to achieve in this context.)
    #[rstest]
    #[case::ignore_and_pop(|_| true, (Some(4), slice1![0, 1, 2, 3]))]
    #[case::ignore_and_do_not_pop(|_| false, (None, slice1![0, 1, 2, 3, 4]))]
    #[case::compare_and_pop(|x: *mut _| unsafe { *x > 1 }, (Some(4), slice1![0, 1, 2, 3]))]
    #[case::compare_and_do_not_pop(|x: *mut _| unsafe { *x < 1 }, (None, slice1![0, 1, 2, 3, 4]))]
    #[case::mutate_and_pop(
        |x: *mut _| unsafe {
            *x = 42;
            true
        },
        (Some(42), slice1![0, 1, 2, 3]),
    )]
    #[case::mutate_and_do_not_pop(
        |x: *mut _| unsafe {
            *x = 42;
            false
        },
        (None, slice1![0, 1, 2, 3, 42]),
    )]
    fn pop_if_many_and_from_vec1_then_popped_and_vec1_eq<F>(
        mut xs1: Vec1<u8>,
        #[case] f: F,
        #[case] expected: (Option<u8>, &Slice1<u8>),
    ) where
        F: FnOnce(*mut u8) -> bool,
    {
        // TODO: The type parameter `F` must be a `FnOnce` over `*mut u8` instead of `&mut u8`
        //       here, because `rstest` constructs the case in a way that the `&mut u8` has a
        //       lifetime that is too specific and too long (it would borrow the item beyond
        //       `pop_if_many_and`). Is there a way to prevent this without introducing `*mut u8`
        //       and unsafe code in cases for `f`? If so, do that instead!
        let x = xs1.pop_if_many_and(|x| f(x as *mut u8)).many().unwrap();
        assert_eq!((x, xs1.as_slice1()), expected);
    }

    #[rstest]
    #[case::empty_at_front(0..0, &[])]
    #[case::empty_at_back(4..4, &[])]
    #[case::one_at_front(0..1, &[0])]
    #[case::one_at_back(4.., &[4])]
    #[case::middle(1..4, &[1, 2, 3])]
    #[case::tail(1.., &[1, 2, 3, 4])]
    #[case::rtail(..4, &[0, 1, 2, 3])]
    fn collect_only_range_subset_iter_of_vec1_into_vec_then_eq<R>(
        mut xs1: Vec1<u8>,
        #[case] range: R,
        #[case] expected: &[u8],
    ) where
        R: RangeBounds<usize>,
    {
        let xss = xs1.only(range).unwrap();
        let xs: Vec<_> = xss.iter().copied().collect();
        assert_eq!(xs.as_slice(), expected);
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
    fn retain_none_from_vec1_only_range_subset_then_only_range_subset_is_empty<R>(
        #[case] mut xs1: Vec1<u8>,
        #[case] range: R,
    ) where
        R: RangeBounds<usize>,
    {
        let mut xss = xs1.only(range).unwrap();
        xss.retain(|_| false);
        assert!(xss.is_empty());
    }

    #[rstest]
    #[case::tail(harness::xs1(3), 1.., slice1![0])]
    #[case::rtail(harness::xs1(3), ..3, slice1![3])]
    #[case::middle(harness::xs1(9), 4..8, slice1![0, 1, 2, 3, 8, 9])]
    fn retain_none_from_vec1_only_range_subset_then_vec1_eq<R>(
        #[case] mut xs1: Vec1<u8>,
        #[case] range: R,
        #[case] expected: &Slice1<u8>,
    ) where
        R: RangeBounds<usize>,
    {
        xs1.only(range).unwrap().retain(|_| false);
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
    fn swap_drain_from_vec1_only_range_subset_then_vec1_eq<S, D>(
        #[case] mut xs1: Vec1<u8>,
        #[case] subset: S,
        #[case] drain: D,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
        D: RangeBounds<usize>,
    {
        xs1.only(subset).unwrap().swap_drain(drain);
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
    fn leak_swap_drain_of_vec1_only_range_subset_then_vec1_eq<S, D>(
        #[case] mut xs1: Vec1<u8>,
        #[case] subset: S,
        #[case] drain: D,
        #[case] expected: &Slice1<u8>,
    ) where
        S: RangeBounds<usize>,
        D: RangeBounds<usize>,
    {
        let mut xss = xs1.only(subset).unwrap();
        mem::forget(xss.swap_drain(drain));
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
    /// let xs = mitsein::vec1::vec1![];
    /// ```
    #[doc(hidden)]
    const fn _empty_rep_expr_then_vec1_macro_compilation_fails() {}

    /// ```compile_fail,E0277
    /// let xs = mitsein::vec1::vec1![0u8; 0];
    /// ```
    #[doc(hidden)]
    const fn _empty_expr_count_then_vec1_macro_compilation_fails() {}
}

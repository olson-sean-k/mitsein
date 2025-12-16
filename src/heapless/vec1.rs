//! A non-empty [`Vec`][`::heapless::vec`].

use ::heapless::CapacityError;
use ::heapless::vec::{self, Vec, VecInner, VecStorage, VecView};
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter::{Skip, Take};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, RangeBounds};
use core::slice;

use crate::array1::Array1;
use crate::heapless;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, IndexRange, Project, RangeError};
use crate::segment::{self, ByRange, ByTail, Segmentation};
use crate::slice1::Slice1;
use crate::take;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

type ItemFor<K> = <K as ClosedVec>::Item;
type StorageFor<K> = <K as ClosedVec>::Storage;

// The name `Vec` here is used very generally; this trait operates against `VecInner` and so covers
// the family of `Vec` types from `heapless`. This module makes `VecInner` types less explicit.
pub trait ClosedVec {
    type Item;
    type Storage: ?Sized + VecStorage<Self::Item>;

    fn as_vec_inner(&self) -> &VecInner<Self::Item, usize, Self::Storage>;
}

impl<T, S> ClosedVec for VecInner<T, usize, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Item = T;
    type Storage = S;

    fn as_vec_inner(&self) -> &VecInner<Self::Item, usize, Self::Storage> {
        self
    }
}

impl<T, S, R> ByRange<usize, R> for VecInner<T, usize, S>
where
    S: ?Sized + VecStorage<T>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self>, Self::Error> {
        let n = self.len();
        Segment::intersected(self, n, range)
    }
}

impl<T, S> ByTail for VecInner<T, usize, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self> {
        let n = self.len();
        Segment::from_tail_range(self, n)
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        let n = self.len();
        Segment::from_rtail_range(self, n)
    }
}

unsafe impl<T, S> MaybeEmpty for VecInner<T, usize, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_slice().cardinality()
    }
}

impl<T, S> Segmentation for VecInner<T, usize, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Kind = Self;
    type Target = Self;
}

type TakeIfMany<'a, T, S, N = ()> = take::TakeIfMany<'a, VecInner<T, usize, S>, T, N>;

pub type PopIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, StorageFor<K>, ()>;

pub type RemoveIfMany<'a, K> = TakeIfMany<'a, ItemFor<K>, StorageFor<K>, usize>;

impl<'a, T, S, N> TakeIfMany<'a, T, S, N>
where
    S: ?Sized + VecStorage<T>,
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

impl<'a, T, S> TakeIfMany<'a, T, S, usize>
where
    S: ?Sized + VecStorage<T>,
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

type VecInner1<T, S> = NonEmpty<VecInner<T, usize, S>>;

impl<T, S> VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
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

    pub fn push(&mut self, item: T) -> Result<(), T> {
        self.items.push(item)
    }

    /// # Safety
    pub unsafe fn push_unchecked(&mut self, item: T) {
        unsafe { self.items.push_unchecked(item) }
    }

    pub fn pop_if_many(&mut self) -> PopIfMany<'_, Self> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: T) -> Result<(), T> {
        self.items.insert(index, item)
    }

    pub fn remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self> {
        TakeIfMany::with(self, index, |items, index| items.items.swap_remove(index))
    }

    pub fn extend_from_slice(&mut self, items: &[T]) -> Result<(), CapacityError>
    where
        T: Clone,
    {
        self.items.extend_from_slice(items)
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
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

    pub fn is_full(&self) -> bool {
        self.items.is_full()
    }
}

impl<T, S> AsMut<[T]> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T, S> AsMut<Slice1<T>> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn as_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T, S> AsRef<[T]> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T, S> AsRef<Slice1<T>> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn as_ref(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T, S> Borrow<[T]> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn borrow(&self) -> &[T] {
        self.items.borrow()
    }
}

impl<T, S> Borrow<Slice1<T>> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn borrow(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<T, S> BorrowMut<[T]> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.items.borrow_mut()
    }
}

impl<T, S> BorrowMut<Slice1<T>> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn borrow_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T, S, R> ByRange<usize, R> for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self>, Self::Error> {
        let n = self.items.len();
        Segment::intersected_strict_subset(&mut self.items, n, range)
    }
}

impl<T, S> ByTail for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self> {
        self.items.tail().rekind()
    }

    fn rtail(&mut self) -> Segment<'_, Self> {
        self.items.rtail().rekind()
    }
}

impl<T, S> ClosedVec for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Item = T;
    type Storage = S;

    fn as_vec_inner(&self) -> &VecInner<Self::Item, usize, Self::Storage> {
        &self.items
    }
}

impl<T, S> Debug for VecInner1<T, S>
where
    T: Debug,
    S: ?Sized + VecStorage<T>,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T, S> Deref for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Target = Slice1<T>;

    fn deref(&self) -> &Self::Target {
        self.as_slice1()
    }
}

impl<T, S> DerefMut for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice1()
    }
}

impl<'a, T, S> IntoIterator for &'a VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.as_slice().iter()
    }
}

impl<'a, T, S> IntoIterator for &'a mut VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.as_mut_slice().iter_mut()
    }
}

impl<T, S> IntoIterator1 for &VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self) }
    }
}

impl<T, S> IntoIterator1 for &mut VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self) }
    }
}

heapless::impl_partial_eq_for_non_empty!([for U, const N: usize in [U; N]] <= [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U, const N: usize in &[U; N]] <= [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U in [U]] <= [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U in &[U]] <= [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U in &mut [U]] <= [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U in &Slice1<U>] == [for T, S as VecStorage in VecInner1<T, S>]);
heapless::impl_partial_eq_for_non_empty!([for U in &mut Slice1<U>] == [for T, S as VecStorage in VecInner1<T, S>]);

// TODO: In the `heapless` crate, the implementations of `PartialEq` for slices (where slices are
//       the left-hand side) do not bound the storage type parameter on `?Sized`, and so only sized
//       storage types are supported (i.e., `OwnedVecStorage`). It's not clear why this is the
//       case, as the reciprocal implementations do not have this requirement. This breaks the
//       macro invocations below, because they necessarily include a bound on `?Sized` for any
//       storage type parameters.
//
//       Determine if this can be changed upstream and, if so, implement these traits via the
//       `impl_partial_eq_for_non_empty` macro when possible. See
//       https://github.com/rust-embedded/heapless/issues/636
//heapless::impl_partial_eq_for_non_empty!([for U, S as VecStorage in VecInner1<U, S>] => [for T in [T]]);
//heapless::impl_partial_eq_for_non_empty!([for U, S as VecStorage in VecInner1<U, S>] => [for T in &[T]]);
//heapless::impl_partial_eq_for_non_empty!([for U, S as VecStorage in VecInner1<U, S>] => [for T in &mut [T]]);
//heapless::impl_partial_eq_for_non_empty!([for U, S as VecStorage in VecInner1<U, S>] == [for T in &Slice1<T>]);
//heapless::impl_partial_eq_for_non_empty!([for U, S as VecStorage in VecInner1<U, S>] == [for T in &mut Slice1<T>]);

impl<T, S> Segmentation for VecInner1<T, S>
where
    S: ?Sized + VecStorage<T>,
{
    type Kind = Self;
    type Target = VecInner<T, usize, S>;
}

pub type Vec1<T, const N: usize> = NonEmpty<Vec<T, N, usize>>;

impl<T, const N: usize> Vec1<T, N>
where
    [T; N]: Array1,
{
    /// # Safety
    pub unsafe fn from_vec_unchecked(items: Vec<T, N>) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_array<const M: usize>(items: [T; M]) -> Self
    where
        [T; M]: Array1,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(Vec::from_array(items)) }
    }

    pub fn into_array<const M: usize>(self) -> Result<[T; M], Self> {
        self.into_vec()
            .into_array::<M>()
            // SAFETY: `self` and therefore `items` must be non-empty.
            .map_err(|items| unsafe { Vec1::from_vec_unchecked(items) })
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<Vec<T, N>>>
    where
        F: FnMut(&T) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }
}

impl<T, const N: usize> Vec1<T, N> {
    pub fn into_vec(self) -> Vec<T, N> {
        self.items
    }

    pub const fn as_vec(&self) -> &Vec<T, N> {
        &self.items
    }

    /// # Safety
    pub const unsafe fn as_mut_vec(&mut self) -> &mut Vec<T, N> {
        &mut self.items
    }

    // Explicit view conversions can be implemented more generally for `NonEmpty<VecInner<_>>`, but
    // such an implementation would be non-`const` and require unsafe code. Since such explicit
    // conversions are only useful for `Vec1`, they are implemented more specifically instead.
    pub const fn as_view1(&self) -> &VecView1<T> {
        self
    }

    pub const fn as_mut_view1(&mut self) -> &mut VecView1<T> {
        self
    }
}

impl<T, const N: usize> From<Vec1<T, N>> for Vec<T, N> {
    fn from(items: Vec1<T, N>) -> Self {
        items.items
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for Vec1<T, N>
where
    [T; N]: Array1,
    [T; M]: Array1,
{
    fn from(items: [T; M]) -> Self {
        Vec1::from_array(items)
    }
}

impl<T, const N: usize> FromIterator1<T> for Vec1<T, N>
where
    [T; N]: Array1,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { Vec1::from_vec_unchecked(items.into_iter().collect()) }
    }
}

impl<T, const N: usize> IntoIterator for Vec1<T, N> {
    type Item = T;
    type IntoIter = vec::IntoIter<T, N, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T, const N: usize> IntoIterator1 for Vec1<T, N> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, const N: usize> TryFrom<Vec<T, N>> for Vec1<T, N>
where
    [T; N]: Array1,
{
    type Error = EmptyError<Vec<T, N>>;

    fn try_from(items: Vec<T, N>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

pub type VecView1<T> = NonEmpty<VecView<T, usize>>;

impl<T> VecView1<T> {
    pub const fn as_vec_view(&self) -> &VecView<T> {
        &self.items
    }

    /// # Safety
    pub const unsafe fn as_mut_vec_view(&mut self) -> &mut VecView<T> {
        &mut self.items
    }
}

pub type Segment<'a, K> =
    segment::Segment<'a, K, VecInner<ItemFor<K>, usize, StorageFor<K>>, IndexRange>;

impl<K, T, S> Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>> + ?Sized,
    S: ?Sized + VecStorage<T>,
{
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

    pub fn insert(&mut self, index: usize, item: T) -> Result<(), T> {
        let index = self
            .range
            .project(index)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        self.items.insert(index, item)?;
        self.range.put_from_end(1);
        Ok(())
    }

    pub fn insert_back(&mut self, item: T) -> Result<(), T> {
        self.items.insert(self.range.end(), item)?;
        self.range.put_from_end(1);
        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> T {
        let index = self
            .range
            .project(index)
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
                .project(index)
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

impl<K, T, S> AsMut<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, S> AsRef<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, S> Borrow<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, S> BorrowMut<[T]> for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, S, R> ByRange<usize, R> for Segment<'_, K>
where
    IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, K>, Self::Error> {
        self.project_and_intersect(range)
    }
}

impl<K, T, S> ByTail for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, K> {
        self.project_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, K> {
        let n = self.len();
        self.project_rtail_range(n)
    }
}

impl<K, T, S> Deref for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<K, T, S> DerefMut for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    S: ?Sized + VecStorage<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<K, T, S> Eq for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    T: Eq,
    S: ?Sized + VecStorage<T>,
{
}

impl<K, T, S> Ord for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    T: Ord,
    S: ?Sized + VecStorage<T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<'a, KT, KU, T, U, ST, SU> PartialEq<Segment<'a, KU>> for Segment<'a, KT>
where
    KT: ClosedVec<Item = T, Storage = ST> + Segmentation<Target = VecInner<T, usize, ST>>,
    KU: ClosedVec<Item = U, Storage = SU> + Segmentation<Target = VecInner<U, usize, SU>>,
    T: PartialEq<U>,
    ST: ?Sized + VecStorage<T>,
    SU: ?Sized + VecStorage<U>,
{
    fn eq(&self, other: &Segment<'a, KU>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<K, T, S> PartialOrd<Self> for Segment<'_, K>
where
    K: ClosedVec<Item = T, Storage = S> + Segmentation<Target = VecInner<T, usize, S>>,
    T: PartialOrd<T>,
    S: ?Sized + VecStorage<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::heapless::vec1::Vec1;
    use crate::iter1::{self, FromIterator1};

    pub const N: usize = 32;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> Vec1<u8, N> {
        Vec1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {
    use ::heapless::vec::Vec;
    use core::iter;
    use core::ops::RangeBounds;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::heapless::vec1::Vec1;
    use crate::heapless::vec1::harness::{self, N, xs1};
    use crate::iter1::IntoIterator1;
    use crate::segment::{ByRange, ByTail};
    #[cfg(feature = "serde")]
    use crate::serde::{self, harness::sequence};
    use crate::slice1::{Slice1, slice1};

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
        mut xs1: Vec1<u8, N>,
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
    fn pop_if_many_from_vec1_until_and_after_only_then_vec1_eq_first(mut xs1: Vec1<u8, N>) {
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
    #[case::empty_at_front(0..0, &[])]
    #[case::empty_at_back(4..4, &[])]
    #[case::one_at_front(0..1, &[0])]
    #[case::one_at_back(4.., &[4])]
    #[case::middle(1..4, &[1, 2, 3])]
    #[case::tail(1.., &[1, 2, 3, 4])]
    #[case::rtail(..4, &[0, 1, 2, 3])]
    fn collect_segment_iter_of_vec1_into_vec_then_eq<R>(
        mut xs1: Vec1<u8, N>,
        #[case] range: R,
        #[case] expected: &[u8],
    ) where
        R: RangeBounds<usize>,
    {
        let xss = xs1.segment(range).unwrap();
        let xs: Vec<_, N> = xss.iter().copied().collect();
        assert_eq!(xs.as_slice(), expected);
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
    fn insert_back_into_vec1_segment_then_vec1_eq<R, T>(
        mut xs1: Vec1<u8, N>,
        #[case] range: R,
        #[case] items: T,
        #[case] expected: &Slice1<u8>,
    ) where
        R: RangeBounds<usize>,
        T: IntoIterator1<Item = u8>,
    {
        let mut xss = xs1.segment(range).unwrap();
        for item in items {
            xss.insert_back(item).unwrap();
        }
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail(harness::xs1(1))]
    #[case::many_tail(harness::xs1(2))]
    fn remove_back_all_from_tail_of_vec1_then_vec1_eq_head(#[case] mut xs1: Vec1<u8, N>) {
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
    fn clear_tail_of_vec1_then_vec1_eq_head(#[case] mut xs1: Vec1<u8, N>) {
        xs1.tail().clear();
        assert_eq!(xs1.as_slice(), &[0]);
    }

    #[rstest]
    #[case::empty_rtail(harness::xs1(0))]
    #[case::one_rtail(harness::xs1(1))]
    #[case::many_rtail(harness::xs1(2))]
    fn clear_rtail_of_vec1_then_vec1_eq_tail(#[case] mut xs1: Vec1<u8, N>) {
        let tail = *xs1.last();
        xs1.rtail().clear();
        assert_eq!(xs1.as_slice(), &[tail]);
    }

    #[rstest]
    #[case::empty_tail(harness::xs1(0))]
    #[case::one_tail_empty_rtail(harness::xs1(1))]
    #[case::many_tail_one_rtail(harness::xs1(2))]
    #[case::many_tail_many_rtail(harness::xs1(3))]
    fn clear_tail_rtail_of_vec1_then_vec1_eq_head_and_tail(#[case] mut xs1: Vec1<u8, N>) {
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
    fn retain_none_from_vec1_segment_then_segment_is_empty<R>(
        #[case] mut xs1: Vec1<u8, N>,
        #[case] range: R,
    ) where
        R: RangeBounds<usize>,
    {
        let mut xss = xs1.segment(range).unwrap();
        xss.retain(|_| false);
        assert!(xss.is_empty());
    }

    #[rstest]
    #[case::tail(harness::xs1(3), 1.., slice1![0])]
    #[case::rtail(harness::xs1(3), ..3, slice1![3])]
    #[case::middle(harness::xs1(9), 4..8, slice1![0, 1, 2, 3, 8, 9])]
    fn retain_none_from_vec1_segment_then_vec1_eq<R>(
        #[case] mut xs1: Vec1<u8, N>,
        #[case] range: R,
        #[case] expected: &Slice1<u8>,
    ) where
        R: RangeBounds<usize>,
    {
        xs1.segment(range).unwrap().retain(|_| false);
        assert_eq!(xs1.as_slice1(), expected);
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn de_serialize_vec1_into_and_from_tokens_eq(
        xs1: Vec1<u8, N>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_, N>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_vec1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<Vec1<u8, N>, Vec<_, N>>(sequence)
    }
}

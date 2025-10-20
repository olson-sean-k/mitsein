//! A non-empty [`ArrayVec`].

#![cfg(feature = "arrayvec")]
#![cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]

#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use arrayvec::{ArrayVec, CapacityError};
use core::borrow::{Borrow, BorrowMut};
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Skip, Take};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, RangeBounds};
use core::slice;
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};
#[cfg(feature = "std")]
use {
    std::cmp,
    std::io::{self, Write},
};

use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
use crate::safety::{self, ArrayVecExt as _, OptionExt as _};
use crate::segment::range::{self, IndexRange, Project, RangeError};
use crate::segment::{self, Query, Segmentation, Tail};
use crate::slice1::Slice1;
use crate::take;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

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

impl<T, R, const N: usize> Query<usize, R> for ArrayVec<T, N>
where
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self, N>, Self::Error> {
        let n = self.len();
        Segment::intersected(self, n, range)
    }
}

impl<T, const N: usize> Segmentation for ArrayVec<T, N> {
    type Kind = Self;
    type Target = Self;
}

impl<T, const N: usize> Tail for ArrayVec<T, N> {
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self, N> {
        let n = self.len();
        Segment::from_tail_range(self, n)
    }

    fn rtail(&mut self) -> Segment<'_, Self, N> {
        let n = self.len();
        Segment::from_rtail_range(self, n)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CardinalityError<T> {
    Empty(EmptyError<T>),
    // Unlike `EmptyError`, the input type parameter and payload of `CapacityError` is meant to
    // represent an item rather than a collection, so this parameter is always the unit type here.
    Capacity(CapacityError<()>),
}

impl<T> CardinalityError<T> {
    pub fn empty(self) -> Option<EmptyError<T>> {
        match self {
            CardinalityError::Empty(error) => Some(error),
            _ => None,
        }
    }

    pub fn capacity(self) -> Option<CapacityError<()>> {
        match self {
            CardinalityError::Capacity(error) => Some(error),
            _ => None,
        }
    }
}

impl<T> Debug for CardinalityError<T> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            CardinalityError::Empty(error) => formatter.debug_tuple("Empty").field(error).finish(),
            CardinalityError::Capacity(error) => {
                formatter.debug_tuple("Capacity").field(error).finish()
            },
        }
    }
}

impl<T> Display for CardinalityError<T> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            CardinalityError::Empty(error) => write!(formatter, "{error}"),
            CardinalityError::Capacity(error) => write!(formatter, "{error}"),
        }
    }
}

impl<T> From<CapacityError<()>> for CardinalityError<T> {
    fn from(error: CapacityError<()>) -> Self {
        CardinalityError::Capacity(error)
    }
}

impl<T> From<EmptyError<T>> for CardinalityError<T> {
    fn from(error: EmptyError<T>) -> Self {
        CardinalityError::Empty(error)
    }
}

impl<T> Error for CardinalityError<T> {}

type TakeIfMany<'a, T, U, M, const N: usize> = take::TakeIfMany<'a, ArrayVec<T, N>, U, M>;

pub type PopIfMany<'a, K, const N: usize> = TakeIfMany<'a, ItemFor<K, N>, ItemFor<K, N>, (), N>;

pub type SwapPopIfMany<'a, K, const N: usize> =
    TakeIfMany<'a, ItemFor<K, N>, Option<ItemFor<K, N>>, usize, N>;

pub type RemoveIfMany<'a, K, const N: usize> =
    TakeIfMany<'a, ItemFor<K, N>, ItemFor<K, N>, usize, N>;

impl<'a, T, M, const N: usize> TakeIfMany<'a, T, T, M, N>
where
    [T; N]: Array1,
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

impl<'a, T, const N: usize> TakeIfMany<'a, T, T, usize, N>
where
    [T; N]: Array1,
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

impl<'a, T, const N: usize> TakeIfMany<'a, T, Option<T>, usize, N>
where
    [T; N]: Array1,
{
    pub fn or_get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, index| items.get(index))
    }

    pub fn or_replace(self, replacement: T) -> Option<Result<T, T>> {
        self.or_else_replace(move || replacement)
    }

    pub fn or_else_replace<F>(self, f: F) -> Option<Result<T, T>>
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
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
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

    pub fn try_from_ref(
        items: &ArrayVec<T, N>,
    ) -> Result<&'_ Self, EmptyError<&'_ ArrayVec<T, N>>> {
        items.try_into()
    }

    pub fn try_from_mut_ref(
        items: &mut ArrayVec<T, N>,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut ArrayVec<T, N>>> {
        items.try_into()
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
        unsafe { self.items.into_inner_unchecked() }
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<ArrayVec<T, N>>>
    where
        F: FnMut(&mut T) -> bool,
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
                self.pop_if_many().or_none();
            }
            None
        }
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
        unsafe { self.items.push_unchecked(item) }
    }

    pub fn pop_if_many(&mut self) -> PopIfMany<'_, Self, N> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn swap_pop_if_many(&mut self, index: usize) -> SwapPopIfMany<'_, Self, N> {
        TakeIfMany::with(self, index, |items, index| items.items.swap_pop(index))
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn try_insert(&mut self, index: usize, item: T) -> Result<(), CapacityError<T>> {
        self.items.try_insert(index, item)
    }

    pub fn remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self, N> {
        TakeIfMany::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn swap_remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_, Self, N> {
        TakeIfMany::with(self, index, |items, index| items.items.swap_remove(index))
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

    /// # Safety
    ///
    /// The [`ArrayVec`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::array_vec1::ArrayVec1;
    ///
    /// let mut xs = ArrayVec1::from([0i32, 1, 2, 3]);
    /// // This block is unsound. The `&mut ArrayVec` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_array_vec().clear();
    /// }
    /// let x = xs.first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_array_vec(&mut self) -> &mut ArrayVec<T, N> {
        &mut self.items
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

impl<'a, T, const N: usize> IntoIterator for &'a ArrayVec1<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut ArrayVec1<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T, const N: usize> IntoIterator1 for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, const N: usize> IntoIterator1 for &'_ ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<T, const N: usize> IntoIterator1 for &'_ mut ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

impl<T, const N: usize> PartialEq<[T]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: PartialEq<T>,
{
    fn eq(&self, other: &[T]) -> bool {
        PartialEq::eq(self.as_array_vec(), other)
    }
}

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl<T, const N: usize> JsonSchema for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        ArrayVec::<T, N>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<ArrayVec<T, N>>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        ArrayVec::<T, N>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        ArrayVec::<T, N>::schema_id()
    }
}

impl<T, R, const N: usize> Query<usize, R> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, Self, N>, Self::Error> {
        let n = self.items.len();
        Segment::intersected_strict_subset(&mut self.items, n, range)
    }
}

impl<T, const N: usize> Segmentation for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Kind = Self;
    type Target = ArrayVec<T, N>;
}

impl<T, const N: usize> Tail for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, Self, N> {
        self.items.tail().rekind()
    }

    fn rtail(&mut self) -> Segment<'_, Self, N> {
        self.items.rtail().rekind()
    }
}

impl<'a, T, const N: usize> TryFrom<&'a [T]> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Clone,
{
    type Error = CardinalityError<&'a [T]>;

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        Slice1::try_from_slice(items)
            .map_err(From::from)
            .and_then(|items| ArrayVec1::try_from(items).map_err(From::from))
    }
}

impl<'a, T, const N: usize> TryFrom<&'a Slice1<T>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
    T: Clone,
{
    type Error = CapacityError;

    fn try_from(items: &'a Slice1<T>) -> Result<Self, Self::Error> {
        ArrayVec::try_from(items.as_slice())
            // SAFETY: `items` is non-empty.
            .map(|items| unsafe { ArrayVec1::from_array_vec_unchecked(items) })
    }
}

impl<T, const N: usize> TryFrom<ArrayVec<T, N>> for ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = EmptyError<ArrayVec<T, N>>;

    fn try_from(items: ArrayVec<T, N>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T, const N: usize> TryFrom<&'a ArrayVec<T, N>> for &'a ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = EmptyError<&'a ArrayVec<T, N>>;

    fn try_from(items: &'a ArrayVec<T, N>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T, const N: usize> TryFrom<&'a mut ArrayVec<T, N>> for &'a mut ArrayVec1<T, N>
where
    [T; N]: Array1,
{
    type Error = EmptyError<&'a mut ArrayVec<T, N>>;

    fn try_from(items: &'a mut ArrayVec<T, N>) -> Result<Self, Self::Error> {
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

pub type Segment<'a, K, const N: usize> =
    segment::Segment<'a, K, ArrayVec<ItemFor<K, N>, N>, IndexRange>;

impl<K, T, const N: usize> Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    pub fn truncate(&mut self, len: usize) {
        if let Some(range) = self.range.truncate_from_end(len) {
            self.items.drain(range);
        }
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.items.retain(self.range.retain_mut_from_end(f))
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let index = self
            .range
            .project(index)
            .unwrap_or_else(|_| range::panic_index_out_of_bounds());
        self.items.insert(index, item);
        self.range.put_from_end(1);
    }

    pub fn insert_back(&mut self, item: T) {
        self.items.insert(self.range.end(), item);
        self.range.put_from_end(1);
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
            panic!("index out of bounds")
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

impl<K, T, const N: usize> AsMut<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> AsRef<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, const N: usize> Borrow<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<K, T, const N: usize> BorrowMut<[T]> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> Deref for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<K, T, const N: usize> DerefMut for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<K, T, const N: usize> Eq for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
    T: Eq,
{
}

impl<K, T, const N: usize> Extend<T> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let n = self.items.len();
        // Split off the remainder beyond the segment to avoid spurious inserts and copying. This
        // comes at the cost of a necessary array on the stack and bulk copy.
        let tail: ArrayVec<_, N> = self.items.drain(self.range.end()..).collect();
        self.items.extend(items);
        self.items.extend(tail);
        let n = self.items.len() - n;
        self.range.put_from_end(n);
    }
}

impl<K, T, const N: usize> Ord for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<K, T, const N: usize> PartialEq<Self> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<K, T, const N: usize> PartialOrd<Self> for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<K, T, R, const N: usize> Query<usize, R> for Segment<'_, K, N>
where
    IndexRange: Project<R, Output = IndexRange, Error = RangeError<usize>>,
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
    R: RangeBounds<usize>,
{
    type Range = IndexRange;
    type Error = RangeError<usize>;

    fn segment(&mut self, range: R) -> Result<Segment<'_, K, N>, Self::Error> {
        self.project_and_intersect(range)
    }
}

impl<K, T, const N: usize> Tail for Segment<'_, K, N>
where
    K: ClosedArrayVec<N, Item = T> + Segmentation<Target = ArrayVec<T, N>>,
{
    type Range = IndexRange;

    fn tail(&mut self) -> Segment<'_, K, N> {
        self.project_tail_range()
    }

    fn rtail(&mut self) -> Segment<'_, K, N> {
        let n = self.len();
        self.project_rtail_range(n)
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

    use crate::array_vec1::ArrayVec1;
    use crate::array_vec1::harness::{self, CAPACITY};
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::segment::Tail;
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

    #[cfg(feature = "schemars")]
    #[rstest]
    fn array_vec1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<ArrayVec1<u8, 5>>(
            schemars::NON_EMPTY_KEY_ARRAY,
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

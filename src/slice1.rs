//! A non-empty [slice][`prim@slice`].

use core::fmt::{self, Debug, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::{self, Chunks, ChunksMut, RChunks, RChunksMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
#[cfg(feature = "alloc")]
use {alloc::borrow::ToOwned, alloc::vec::Vec};

#[cfg(feature = "rayon")]
use crate::iter1::ParallelIterator1;
use crate::iter1::{IntoIterator1, Iterator1};
use crate::safety;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};
#[cfg(feature = "alloc")]
use {crate::boxed1::BoxedSlice1, crate::vec1::Vec1};

unsafe impl<T> MaybeEmpty for &'_ [T] {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

unsafe impl<T> MaybeEmpty for &'_ mut [T] {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        (&**self).cardinality()
    }
}

pub type Slice1<T> = NonEmpty<[T]>;

// TODO: At time of writing, `const` functions are not supported in traits, so
//       `FromMaybeEmpty::from_maybe_empty_unchecked` cannot be used to construct a `Slice1` yet.
//       Use that function instead of `mem::transmute` when possible.
impl<T> Slice1<T> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice literal `&[]`.
    pub const unsafe fn from_slice_unchecked(items: &[T]) -> &Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `[T]` and
        //         `Slice1<T>` are the same.
        mem::transmute::<&'_ [T], &'_ Slice1<T>>(items)
    }

    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice literal `&mut []`.
    pub const unsafe fn from_mut_slice_unchecked(items: &mut [T]) -> &mut Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `[T]` and
        //         `Slice1<T>` are the same.
        mem::transmute::<&'_ mut [T], &'_ mut Slice1<T>>(items)
    }

    pub fn try_from_slice(items: &[T]) -> Result<&Self, &[T]> {
        items.try_into()
    }

    pub fn try_from_mut_slice(items: &mut [T]) -> Result<&mut Self, &mut [T]> {
        items.try_into()
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn to_vec1(&self) -> Vec1<T>
    where
        T: Clone,
    {
        Vec1::from(self)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_vec1(self: BoxedSlice1<T>) -> Vec1<T> {
        Vec1::from(self)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn once_and_then_repeat(&self, n: usize) -> Vec1<T>
    where
        T: Copy,
    {
        // SAFETY: `self` must be non-empty.
        unsafe {
            Vec1::from_vec_unchecked(
                self.items
                    .repeat(n.checked_add(1).expect("overflow in slice repetition")),
            )
        }
    }

    pub const fn split_first(&self) -> (&T, &[T]) {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.split_first()) }
    }

    pub const fn split_first_mut(&mut self) -> (&mut T, &mut [T]) {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.split_first_mut()) }
    }

    pub const fn first(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.first()) }
    }

    pub const fn first_mut(&mut self) -> &mut T {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.first_mut()) }
    }

    pub const fn last(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.last()) }
    }

    pub const fn last_mut(&mut self) -> &mut T {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::unwrap_option_maybe_unchecked(self.items.last_mut()) }
    }

    pub fn chunks(&self, n: usize) -> Iterator1<Chunks<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunks(n)) }
    }

    pub fn chunks_mut(&mut self, n: usize) -> Iterator1<ChunksMut<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunks_mut(n)) }
    }

    pub fn rchunks(&self, n: usize) -> Iterator1<RChunks<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.rchunks(n)) }
    }

    pub fn rchunks_mut(&mut self, n: usize) -> Iterator1<RChunksMut<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.rchunks_mut(n)) }
    }

    pub fn iter1(&self) -> Iterator1<slice::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_slice().iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<slice::IterMut<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_mut_slice().iter_mut()) }
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.len()) }
    }

    pub const fn as_slice(&self) -> &'_ [T] {
        &self.items
    }

    pub const fn as_mut_slice(&mut self) -> &'_ mut [T] {
        &mut self.items
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> Slice1<T> {
    pub fn par_iter1(&self) -> ParallelIterator1<<&'_ [T] as IntoParallelIterator>::Iter>
    where
        T: Sync,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter()) }
    }

    pub fn par_iter1_mut(
        &mut self,
    ) -> ParallelIterator1<<&'_ mut [T] as IntoParallelIterator>::Iter>
    where
        T: Send,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_iter_mut()) }
    }
}

impl<T> AsMut<[T]> for Slice1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.items
    }
}

impl<T> AsRef<[T]> for Slice1<T> {
    fn as_ref(&self) -> &[T] {
        &self.items
    }
}

impl<T> Debug for Slice1<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries(self.as_slice().iter())
            .finish()
    }
}

impl<T> Deref for Slice1<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Slice1<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'a, T> From<&'a Slice1<T>> for Vec<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        Vec::from(items.as_slice())
    }
}

impl<T, I> Index<I> for Slice1<T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T, I> IndexMut<I> for Slice1<T>
where
    [T]: IndexMut<I>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<'a, T> IntoIterator for &'a Slice1<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Slice1<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T> IntoIterator1 for &'_ Slice1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<T> IntoIterator1 for &'_ mut Slice1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

crate::impl_partial_eq_for_non_empty!([for U in [U]] <= [for T in Slice1<T>]);
crate::impl_partial_eq_for_non_empty!([for U in Slice1<U>] => [for T in [T]]);

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<T> ToOwned for Slice1<T>
where
    T: Clone,
{
    type Owned = Vec1<T>;

    fn to_owned(&self) -> Self::Owned {
        Vec1::from(self)
    }
}

impl<'a, T> TryFrom<&'a [T]> for &'a Slice1<T> {
    type Error = &'a [T];

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T> TryFrom<&'a mut [T]> for &'a mut Slice1<T> {
    type Error = &'a mut [T];

    fn try_from(items: &'a mut [T]) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

pub const fn from_ref<T>(item: &T) -> &Slice1<T> {
    // SAFETY: The input slice is non-empty.
    unsafe { Slice1::from_slice_unchecked(slice::from_ref(item)) }
}

pub fn from_mut<T>(item: &mut T) -> &mut Slice1<T> {
    // SAFETY: The input slice is non-empty.
    unsafe { Slice1::from_mut_slice_unchecked(slice::from_mut(item)) }
}

#[macro_export]
macro_rules! slice1 {
    ($($item:expr $(,)?)+) => {{
        let slice: &[_] = &[$($item,)+];
        // SAFETY: There must be one or more `item` metavariables in the repetition.
        unsafe { $crate::slice1::Slice1::from_slice_unchecked(slice) }
    }};
}
pub use slice1;

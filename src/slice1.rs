//! A non-empty [slice][`prim@slice`].

#[cfg(feature = "serde")]
use ::serde::{Deserialize, Deserializer};
use core::fmt::{self, Debug, Formatter};
use core::iter::{DoubleEndedIterator, FusedIterator};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::{self, ChunkBy, ChunkByMut, Chunks, ChunksMut, RChunks, RChunksMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
#[cfg(feature = "schemars")]
use {
    alloc::borrow::Cow,
    schemars::{JsonSchema, Schema, SchemaGenerator},
};
#[cfg(feature = "alloc")]
use {alloc::borrow::ToOwned, alloc::vec::Vec};

use crate::iter1::{IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::iter1::{IntoParallelIterator1, ParallelIterator1};
use crate::safety;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};
#[cfg(feature = "alloc")]
use {crate::boxed1::BoxedSlice1, crate::vec1::Vec1};

unsafe impl<T> MaybeEmpty for [T] {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

pub trait SliceExt<T> {
    fn chunk_by1<F>(&self, f: F) -> ChunkBy1<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool;

    fn chunk_by1_mut<F>(&mut self, f: F) -> ChunkBy1Mut<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool;
}

impl<T> SliceExt<T> for [T] {
    fn chunk_by1<F>(&self, f: F) -> ChunkBy1<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        ChunkBy1 {
            chunks: self.chunk_by(f),
        }
    }

    fn chunk_by1_mut<F>(&mut self, f: F) -> ChunkBy1Mut<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        ChunkBy1Mut {
            chunks: self.chunk_by_mut(f),
        }
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
        unsafe { mem::transmute::<&'_ [T], &'_ Slice1<T>>(items) }
    }

    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is undefined behavior to call this function with
    /// an empty slice literal `&mut []`.
    pub const unsafe fn from_mut_slice_unchecked(items: &mut [T]) -> &mut Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `[T]` and
        //         `Slice1<T>` are the same.
        unsafe { mem::transmute::<&'_ mut [T], &'_ mut Slice1<T>>(items) }
    }

    pub fn try_from_slice(items: &[T]) -> Result<&Self, EmptyError<&[T]>> {
        items.try_into()
    }

    pub fn try_from_mut_slice(items: &mut [T]) -> Result<&mut Self, EmptyError<&mut [T]>> {
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

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn repeat_non_zero(&self, n: NonZeroUsize) -> Vec1<T>
    where
        T: Copy,
    {
        // SAFETY: `self` must be non-empty and `n` must be greater than zero.
        unsafe { Vec1::from_vec_unchecked(self.items.repeat(n.get())) }
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

    pub fn chunks1(&self, n: usize) -> Iterator1<Chunks<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunks(n)) }
    }

    pub fn chunks1_mut(&mut self, n: usize) -> Iterator1<ChunksMut<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunks_mut(n)) }
    }

    pub fn rchunks1(&self, n: usize) -> Iterator1<RChunks<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.rchunks(n)) }
    }

    pub fn rchunks1_mut(&mut self, n: usize) -> Iterator1<RChunksMut<'_, T>> {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.rchunks_mut(n)) }
    }

    // Unlike other `1`-postfixed functions, `chunk_by1` and `chunk_by1_mut` shadow
    // `SliceExt::chunk_by1` and `SliceExt::chunk_by1_mut`, respectively. Generally, the postfix is
    // used for clarity and to avoid shadowing maybe-empty counterparts, but that would require yet
    // another distinction here and would likely just cause a lot more confusion.
    pub fn chunk_by1<F>(&self, f: F) -> Iterator1<ChunkBy1<'_, T, F>>
    where
        F: FnMut(&T, &T) -> bool,
    {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunk_by1(f)) }
    }

    pub fn chunk_by1_mut<F>(&mut self, f: F) -> Iterator1<ChunkBy1Mut<'_, T, F>>
    where
        F: FnMut(&T, &T) -> bool,
    {
        // SAFETY: This iterator cannot have a cardinality of zero.
        unsafe { Iterator1::from_iter_unchecked(self.items.chunk_by1_mut(f)) }
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

impl<T> AsMut<[T]> for Slice1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.items
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

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'a, 'de> Deserialize<'de> for &'a Slice1<u8>
where
    'de: 'a,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use ::serde::de::Error;

        let items = <&[u8]>::deserialize(deserializer)?;
        <&Slice1<u8>>::try_from(items).map_err(D::Error::custom)
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

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a Slice1<T>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = <&'a [T] as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, T> IntoParallelIterator for &'a mut Slice1<T>
where
    T: Send,
{
    type Item = &'a mut T;
    type Iter = <&'a mut [T] as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        (&mut self.items).into_par_iter()
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<T> IntoParallelIterator1 for &'_ Slice1<T>
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
impl<T> IntoParallelIterator1 for &'_ mut Slice1<T>
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
impl<T> JsonSchema for Slice1<T>
where
    T: JsonSchema,
{
    fn schema_name() -> Cow<'static, str> {
        <[T]>::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<[T]>(
            schemars::NON_EMPTY_KEY_ARRAY,
            generator,
        )
    }

    fn inline_schema() -> bool {
        <[T]>::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        <[T]>::schema_id()
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
    type Error = EmptyError<&'a [T]>;

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a, T> TryFrom<&'a mut [T]> for &'a mut Slice1<T> {
    type Error = EmptyError<&'a mut [T]>;

    fn try_from(items: &'a mut [T]) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

pub struct ChunkBy1<'a, T, F> {
    chunks: ChunkBy<'a, T, F>,
}

impl<T, F> Clone for ChunkBy1<'_, T, F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        ChunkBy1 {
            chunks: self.chunks.clone(),
        }
    }
}

impl<T, F> Debug for ChunkBy1<'_, T, F>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChunkBy1")
            .field("chunks", &self.chunks)
            .finish()
    }
}

impl<'a, T, F> DoubleEndedIterator for ChunkBy1<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.chunks
            .next_back()
            // SAFETY: `slice::chunk_by` never outputs empty chunks, so each chunk can be safely
            //         converted into a `Slice1`.
            .map(|chunk| unsafe { Slice1::from_slice_unchecked(chunk) })
    }
}

impl<'a, T, F> FusedIterator for ChunkBy1<'a, T, F> where F: FnMut(&T, &T) -> bool {}

impl<'a, T, F> Iterator for ChunkBy1<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = &'a Slice1<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks
            .next()
            // SAFETY: `slice::chunk_by` never outputs empty chunks, so each chunk can be safely
            //         converted into a `Slice1`.
            .map(|chunk| unsafe { Slice1::from_slice_unchecked(chunk) })
    }
}

pub struct ChunkBy1Mut<'a, T, F> {
    chunks: ChunkByMut<'a, T, F>,
}

impl<T, F> Debug for ChunkBy1Mut<'_, T, F>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChunkBy1Mut")
            .field("chunks", &self.chunks)
            .finish()
    }
}

impl<'a, T, F> DoubleEndedIterator for ChunkBy1Mut<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.chunks
            .next_back()
            // SAFETY: `slice::chunk_by_mut` never outputs empty chunks, so each chunk can be
            //         safely converted into a `Slice1`.
            .map(|chunk| unsafe { Slice1::from_mut_slice_unchecked(chunk) })
    }
}

impl<'a, T, F> FusedIterator for ChunkBy1Mut<'a, T, F> where F: FnMut(&T, &T) -> bool {}

impl<'a, T, F> Iterator for ChunkBy1Mut<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = &'a mut Slice1<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks
            .next()
            // SAFETY: `slice::chunk_by_mut` never outputs empty chunks, so each chunk can be
            //         safely converted into a `Slice1`.
            .map(|chunk| unsafe { Slice1::from_mut_slice_unchecked(chunk) })
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

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::slice1::Slice1;

    #[fixture]
    pub fn xs1() -> &'static Slice1<u8> {
        slice1![0, 1, 2, 3, 4]
    }
}

#[cfg(all(
    test,
    any(feature = "schemars", all(feature = "alloc", feature = "serde"))
))]
mod tests {
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use {alloc::vec::Vec, serde_test::Token};

    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::slice1::Slice1;
    #[cfg(feature = "serde")]
    use {
        crate::serde::{
            self,
            harness::{borrowed_bytes, sequence},
        },
        crate::slice1::harness::xs1,
    };

    #[cfg(feature = "schemars")]
    #[rstest]
    fn slice1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<Slice1<u8>>(
            schemars::NON_EMPTY_KEY_ARRAY,
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_ref_slice1_u8_from_tokens_eq(
        xs1: &Slice1<u8>,
        #[with(xs1)] borrowed_bytes: impl Iterator<Item = Token>,
    ) {
        let borrowed_bytes: Vec<_> = borrowed_bytes.collect();
        let borrowed_bytes = borrowed_bytes.as_slice();
        serde::harness::assert_ref_from_tokens_eq(xs1, borrowed_bytes)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn serialize_slice1_into_tokens_eq(xs1: &Slice1<u8>, sequence: impl Iterator<Item = Token>) {
        serde::harness::assert_into_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }
}

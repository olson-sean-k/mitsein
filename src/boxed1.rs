//! Non-empty [boxed][`boxed`] collections.
//!
//! [`boxed`]: alloc::boxed

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

#[cfg(feature = "serde")]
use ::serde::{Deserialize, Deserializer};
use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::slice;

use crate::EmptyError;
use crate::MaybeEmpty;
use crate::array1::Array1;
use crate::iter1::{FromIterator1, IntoIterator1, Iterator1};
use crate::rc1::{RcSlice1, RcSlice1Ext as _};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::string1::String1;
#[cfg(target_has_atomic = "ptr")]
use crate::sync1::{ArcSlice1, ArcSlice1Ext as _};
use crate::vec1::Vec1;

pub type BoxedSlice1<T> = Box<Slice1<T>>;

pub trait BoxedSlice1Ext<T>: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Box::<T>::from([])`][`Box::from`].
    ///
    /// [`Box::from`]: alloc::boxed::Box::from
    unsafe fn from_boxed_slice_unchecked(items: Box<[T]>) -> Self;

    fn try_from_boxed_slice(items: Box<[T]>) -> Result<Self, EmptyError<Box<[T]>>>;

    fn try_from_slice(items: &[T]) -> Result<Self, EmptyError<&[T]>>
    where
        T: Clone;

    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T>;

    fn into_boxed_slice(self) -> Box<[T]>;

    fn into_rc_slice1(self) -> RcSlice1<T>;

    fn into_vec1(self) -> Vec1<T>;

    fn as_slice1(&self) -> &Slice1<T>;

    fn as_mut_slice1(&mut self) -> &mut Slice1<T>;
}

impl<T> BoxedSlice1Ext<T> for BoxedSlice1<T> {
    unsafe fn from_boxed_slice_unchecked(items: Box<[T]>) -> Self {
        let items = Box::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input slice is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `[T]` and
        //         `Slice1<T>` have the same representation (`Slice1<T>` is `repr(transparent)`).
        //         Moreover, the allocator only requires that the memory location and layout are
        //         the same when deallocating, so dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut Slice1<T>) }
    }

    fn try_from_boxed_slice(items: Box<[T]>) -> Result<Self, EmptyError<Box<[T]>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { BoxedSlice1::from_boxed_slice_unchecked(items) }),
        }
    }

    fn try_from_slice(items: &[T]) -> Result<Self, EmptyError<&[T]>>
    where
        T: Clone,
    {
        match items.len() {
            0 => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { BoxedSlice1::from_boxed_slice_unchecked(Box::from(items)) }),
        }
    }

    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T> {
        ArcSlice1::from_boxed_slice1(self)
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        let items = Box::into_raw(self);
        // SAFETY: This transmutation is safe, because `[T]` and `Slice1<T>` have the same
        //         representation (`Slice1<T>` is `repr(transparent)`). Moreover, the allocator
        //         only requires that the memory location and layout are the same when
        //         deallocating, so dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut [T]) }
    }

    fn into_rc_slice1(self) -> RcSlice1<T> {
        RcSlice1::from_boxed_slice1(self)
    }

    fn into_vec1(self) -> Vec1<T> {
        Vec1::from(self)
    }

    fn as_slice1(&self) -> &Slice1<T> {
        self.as_ref()
    }

    fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        self.as_mut()
    }
}

impl<T> AsMut<[T]> for BoxedSlice1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T> AsRef<[T]> for BoxedSlice1<T> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de, T> Deserialize<'de> for BoxedSlice1<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use ::serde::de::Error;

        let items = Box::<[T]>::deserialize(deserializer)?;
        BoxedSlice1::try_from(items).map_err(D::Error::custom)
    }
}

impl<T, const N: usize> From<[T; N]> for BoxedSlice1<T>
where
    [T; N]: Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { BoxedSlice1::from_boxed_slice_unchecked(Box::from(items)) }
    }
}

impl<'a, T> From<&'a Slice1<T>> for BoxedSlice1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { BoxedSlice1::from_boxed_slice_unchecked(Box::from(items.as_slice())) }
    }
}

impl<T> From<BoxedSlice1<T>> for Box<[T]> {
    fn from(items: BoxedSlice1<T>) -> Self {
        items.into_boxed_slice()
    }
}

impl<T> From<BoxedSlice1<T>> for Vec<T> {
    fn from(items: BoxedSlice1<T>) -> Self {
        Vec::from(items.into_boxed_slice())
    }
}

impl<T> From<Vec1<T>> for BoxedSlice1<T> {
    fn from(items: Vec1<T>) -> Self {
        items.into_boxed_slice1()
    }
}

impl<T> IntoIterator for BoxedSlice1<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        <Box<[T]> as IntoIterator>::into_iter(self.into_boxed_slice())
    }
}

impl<'a, T> IntoIterator for &'a BoxedSlice1<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut BoxedSlice1<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T> IntoIterator1 for BoxedSlice1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self) }
    }
}

impl<T> IntoIterator1 for &'_ BoxedSlice1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1()
    }
}

impl<T> IntoIterator1 for &'_ mut BoxedSlice1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self.iter1_mut()
    }
}

impl<'a, T> TryFrom<&'a [T]> for BoxedSlice1<T>
where
    T: Clone,
{
    type Error = EmptyError<&'a [T]>;

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        BoxedSlice1::try_from_slice(items)
    }
}

impl<T> TryFrom<Box<[T]>> for BoxedSlice1<T> {
    type Error = EmptyError<Box<[T]>>;

    fn try_from(items: Box<[T]>) -> Result<Self, Self::Error> {
        BoxedSlice1::try_from_boxed_slice(items)
    }
}

pub type BoxedStr1 = Box<Str1>;

pub trait BoxedStr1Ext: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Box::<str>::from("")`][`Box::from`].
    ///
    /// [`Box::from`]: alloc::boxed::Box::from
    unsafe fn from_boxed_str_unchecked(items: Box<str>) -> Self;

    fn try_from_boxed_str(items: Box<str>) -> Result<Self, EmptyError<Box<str>>>;

    fn into_boxed_str(self) -> Box<str>;
}

impl BoxedStr1Ext for BoxedStr1 {
    unsafe fn from_boxed_str_unchecked(items: Box<str>) -> Self {
        let items = Box::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input string is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `str` and
        //         `Str1` have the same representation (`Str1` is `repr(transparent)`). Moreover,
        //         the allocator only requires that the memory location and layout are the same
        //         when deallocating, so dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut Str1) }
    }

    fn try_from_boxed_str(items: Box<str>) -> Result<Self, EmptyError<Box<str>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { BoxedStr1::from_boxed_str_unchecked(items) }),
        }
    }

    fn into_boxed_str(self) -> Box<str> {
        let items = Box::into_raw(self);
        // SAFETY: This transmutation is safe, because `str` and `Str1` have the same
        //         representation (`Str1` is `repr(transparent)`). Moreover, the allocator only
        //         requires that the memory location and layout are the same when deallocating, so
        //         dropping the transmuted `Box` is sound.
        unsafe { Box::from_raw(items as *mut str) }
    }
}

impl<T> FromIterator1<T> for BoxedStr1
where
    String1: FromIterator1<T>,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        String1::from_iter1(items).into_boxed_str1()
    }
}

#[cfg(all(test, feature = "serde"))]
pub mod harness {
    use rstest::fixture;

    use crate::boxed1::BoxedSlice1;
    use crate::slice1::slice1;

    #[fixture]
    pub fn xs1() -> BoxedSlice1<u8> {
        BoxedSlice1::from(slice1![0, 1, 2, 3, 4])
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use alloc::vec::Vec;
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::Token;

    use crate::boxed1::BoxedSlice1;
    use crate::boxed1::harness::xs1;
    use crate::serde::{self, harness::sequence};

    #[rstest]
    fn de_serialize_boxed_slice1_into_and_from_tokens_eq(
        xs1: BoxedSlice1<u8>,
        sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_into_and_from_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }

    #[cfg(feature = "serde")]
    #[rstest]
    fn deserialize_boxed_slice1_from_empty_tokens_then_empty_error(
        #[with(0)] sequence: impl Iterator<Item = Token>,
    ) {
        serde::harness::assert_deserialize_error_eq_empty_error::<BoxedSlice1<u8>, Vec<_>>(sequence)
    }
}

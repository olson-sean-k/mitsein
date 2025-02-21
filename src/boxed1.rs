//! Non-empty [boxed][`boxed`] collections.
//!
//! [`boxed`]: alloc::boxed

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::slice;

use crate::array1::Array1;
use crate::iter1::{IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::str1::Str1;
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

    fn try_from_boxed_slice(items: Box<[T]>) -> Result<Self, Box<[T]>>;

    fn try_from_slice(items: &[T]) -> Result<Self, &[T]>
    where
        T: Clone;

    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T>;

    fn into_boxed_slice(self) -> Box<[T]>;

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
        Box::from_raw(items as *mut Slice1<T>)
    }

    fn try_from_boxed_slice(items: Box<[T]>) -> Result<Self, Box<[T]>> {
        match items.len() {
            0 => Err(items),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { BoxedSlice1::from_boxed_slice_unchecked(items) }),
        }
    }

    fn try_from_slice(items: &[T]) -> Result<Self, &[T]>
    where
        T: Clone,
    {
        match items.len() {
            0 => Err(items),
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
    type Error = &'a [T];

    fn try_from(items: &'a [T]) -> Result<Self, Self::Error> {
        BoxedSlice1::try_from_slice(items)
    }
}

impl<T> TryFrom<Box<[T]>> for BoxedSlice1<T> {
    type Error = Box<[T]>;

    fn try_from(items: Box<[T]>) -> Result<Self, Self::Error> {
        BoxedSlice1::try_from_boxed_slice(items)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<Box<[T]>>> for BoxedSlice1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<Box<[T]>>) -> Result<Self, Self::Error> {
        BoxedSlice1::try_from_boxed_slice(serde.items).map_err(|_| EmptyError)
    }
}

pub type BoxedStr1 = Box<Str1>;

pub trait BoxedStr1Ext {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Box::<str>::from("")`][`Box::from`].
    ///
    /// [`Box::from`]: alloc::boxed::Box::from
    unsafe fn from_boxed_str_unchecked(items: Box<str>) -> Self;

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
        Box::from_raw(items as *mut Str1)
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

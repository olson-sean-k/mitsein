#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};

#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::vec1::Vec1;
use crate::NonEmpty;

pub type BoxedSlice1<T> = NonEmpty<Box<[T]>>;

impl<T> BoxedSlice1<T> {
    pub(crate) fn from_boxed_slice_unchecked(items: Box<[T]>) -> Self {
        BoxedSlice1 { items }
    }

    pub fn try_from_boxed_slice(items: Box<[T]>) -> Result<Self, Box<[T]>> {
        match items.len() {
            0 => Err(items),
            _ => Ok(BoxedSlice1::from_boxed_slice_unchecked(items)),
        }
    }

    pub fn try_from_slice(items: &[T]) -> Result<Self, &[T]>
    where
        T: Clone,
    {
        match items.len() {
            0 => Err(items),
            _ => Ok(BoxedSlice1::from_boxed_slice_unchecked(Box::from(items))),
        }
    }

    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.items
    }

    pub fn into_vec1(self) -> Vec1<T> {
        Vec1::from(self)
    }

    pub fn leak<'a>(items: Self) -> &'a mut Slice1<T> {
        Slice1::from_mut_slice_unchecked(Box::leak(items.into_boxed_slice()))
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        Slice1::from_slice_unchecked(self.items.as_ref())
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        Slice1::from_mut_slice_unchecked(self.items.as_mut())
    }
}

impl<T> AsMut<[T]> for BoxedSlice1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T> AsMut<Slice1<T>> for BoxedSlice1<T> {
    fn as_mut(&mut self) -> &mut Slice1<T> {
        self.as_mut_slice1()
    }
}

impl<T> AsRef<[T]> for BoxedSlice1<T> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T> AsRef<Slice1<T>> for BoxedSlice1<T> {
    fn as_ref(&self) -> &Slice1<T> {
        self.as_slice1()
    }
}

impl<'a, T> From<&'a Slice1<T>> for BoxedSlice1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        BoxedSlice1::from_boxed_slice_unchecked(Box::from(items.as_slice()))
    }
}

impl<T> From<Vec1<T>> for BoxedSlice1<T> {
    fn from(items: Vec1<T>) -> Self {
        items.into_boxed_slice1()
    }
}

impl<T> From<BoxedSlice1<T>> for Vec<T> {
    fn from(items: BoxedSlice1<T>) -> Self {
        Vec::from(items.into_boxed_slice())
    }
}

impl<T> Deref for BoxedSlice1<T> {
    type Target = Slice1<T>;

    fn deref(&self) -> &Self::Target {
        Slice1::from_slice_unchecked(self.items.deref())
    }
}

impl<T> DerefMut for BoxedSlice1<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Slice1::from_mut_slice_unchecked(self.items.deref_mut())
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

macro_rules! impl_from_array_for_boxed_slice1 {
    ($N:literal) => {
        impl<T> From<[T; $N]> for $crate::boxed1::BoxedSlice1<T> {
            fn from(items: [T; $N]) -> Self {
                $crate::boxed1::BoxedSlice1::from_boxed_slice_unchecked(alloc::boxed::Box::from(
                    items,
                ))
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_from_array_for_boxed_slice1);

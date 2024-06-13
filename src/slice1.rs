use std::borrow::ToOwned;
use std::fmt::{self, Debug, Formatter};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;

use crate::iter1::Iterator1;
use crate::vec1::Vec1;

#[derive(Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Slice1<T>
where
    T: ?Sized,
{
    slice: T,
}

// TODO: As with `Vec1`, provide functions with stronger guarantees for slices of one or more
//       items. For example, functions that isolate terminal items like `split_first` need not be
//       fallible.
impl<T> Slice1<[T]> {
    pub(crate) fn from_slice_unchecked(slice: &[T]) -> &Self {
        // SAFETY:
        unsafe { mem::transmute::<&'_ [T], &'_ Slice1<[T]>>(slice) }
    }

    pub(crate) fn from_mut_slice_unchecked(slice: &mut [T]) -> &mut Self {
        // SAFETY:
        unsafe { mem::transmute::<&'_ mut [T], &'_ mut Slice1<[T]>>(slice) }
    }

    pub fn try_from_slice(slice: &[T]) -> Result<&Self, &[T]> {
        match slice.len() {
            0 => Err(slice),
            _ => Ok(Slice1::from_slice_unchecked(slice)),
        }
    }

    pub fn try_from_mut_slice(slice: &mut [T]) -> Result<&mut Self, &mut [T]> {
        match slice.len() {
            0 => Err(slice),
            _ => Ok(Slice1::from_mut_slice_unchecked(slice)),
        }
    }

    // TODO: It isn't clear that this transmutation through `Box` is safe, even though `Slice1` has
    //       a `transparent` representation.
    //
    // pub fn into_vec1(self: Box<Slice1<[T]>>) -> Vec1<T> {
    //     // SAFETY:
    //     Vec1::from_vec_unchecked(unsafe {
    //         mem::transmute::<Box<Slice1<[T]>>, Box<[T]>>(self).into_vec()
    //     })
    // }

    pub fn to_vec1(&self) -> Vec1<T>
    where
        T: Clone,
    {
        Vec1::from(self)
    }

    pub fn iter1(&self) -> Iterator1<slice::Iter<'_, T>> {
        Iterator1::from_iter_unchecked(self.as_slice().iter())
    }

    pub fn iter1_mut(&mut self) -> Iterator1<slice::IterMut<'_, T>> {
        Iterator1::from_iter_unchecked(self.as_mut_slice().iter_mut())
    }

    pub fn as_slice(&self) -> &'_ [T] {
        // SAFETY:
        unsafe { mem::transmute::<&'_ Slice1<[T]>, &'_ [T]>(self) }
    }

    pub fn as_mut_slice(&mut self) -> &'_ mut [T] {
        // SAFETY:
        unsafe { mem::transmute::<&'_ mut Slice1<[T]>, &'_ mut [T]>(self) }
    }
}

impl<T> AsMut<[T]> for &'_ mut Slice1<[T]> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.slice
    }
}

impl<T> AsRef<[T]> for &'_ Slice1<[T]> {
    fn as_ref(&self) -> &[T] {
        &self.slice
    }
}

impl<T> Debug for Slice1<[T]>
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

impl<T> Deref for Slice1<[T]> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Slice1<[T]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> ToOwned for Slice1<[T]>
where
    T: Clone,
{
    type Owned = Vec1<T>;

    fn to_owned(&self) -> Self::Owned {
        Vec1::from(self)
    }
}

#[cfg(test)]
mod tests {}

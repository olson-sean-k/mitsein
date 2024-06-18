use core::convert::Infallible;
use core::option;

use crate::iter1::{AtMostOne, ExactlyOne, IntoIterator1, Iterator1};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{FnInto, NonEmpty};

pub trait OptionExt<T> {
    fn call(self) -> Option<T::Into>
    where
        T: FnInto;
}

impl<T> OptionExt<T> for Option<T> {
    fn call(self) -> Option<T::Into>
    where
        T: FnInto,
    {
        self.map(FnInto::call)
    }
}

pub type Option1<T> = NonEmpty<Option<T>>;

impl<T> Option1<T> {
    pub fn from_item(item: T) -> Self {
        NonEmpty { items: Some(item) }
    }

    pub fn some(self) -> Option<T> {
        self.items
    }

    pub fn ok(self) -> Result<T, Infallible> {
        Ok(self.take())
    }

    pub fn take(mut self) -> T {
        // SAFETY:
        unsafe { self.items.take().unwrap_unchecked() }
    }

    pub fn call(self) -> T::Into
    where
        T: FnInto,
    {
        self.take().call()
    }

    pub fn map<U, F>(self, f: F) -> Option1<U>
    where
        F: FnOnce(T) -> U,
    {
        Option1::from_item(f(self.take()))
    }

    pub fn get(&self) -> &T {
        // SAFETY:
        unsafe { self.items.as_ref().unwrap_unchecked() }
    }

    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.as_mut().unwrap_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<option::Iter<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter())
    }

    pub fn iter1_mut(&mut self) -> Iterator1<option::IterMut<'_, T>> {
        Iterator1::from_iter_unchecked(self.items.iter_mut())
    }

    pub fn as_option(&self) -> &Option<T> {
        &self.items
    }

    pub fn as_slice1(&self) -> &Slice1<T> {
        Slice1::from_slice_unchecked(self.items.as_slice())
    }

    pub fn as_mut_slice1(&mut self) -> &mut Slice1<T> {
        Slice1::from_mut_slice_unchecked(self.items.as_mut_slice())
    }

    pub fn as_ref(&self) -> Option1<&T> {
        Option1::from_item(self.get())
    }

    pub fn as_mut(&mut self) -> Option1<&mut T> {
        Option1::from_item(self.get_mut())
    }
}

impl<'a, T> Option1<&'a T> {
    pub fn cloned(self) -> Option1<T>
    where
        T: Clone,
    {
        Option1::from_item(self.take().clone())
    }

    pub fn copied(self) -> Option1<T>
    where
        T: Copy,
    {
        Option1::from_item(*self.take())
    }
}

impl<'a, T> Option1<&'a mut T> {
    pub fn cloned(self) -> Option1<T>
    where
        T: Clone,
    {
        Option1::from_item(self.take().clone())
    }

    pub fn copied(self) -> Option1<T>
    where
        T: Copy,
    {
        Option1::from_item(*self.take())
    }
}

impl<T> Option1<Option1<T>> {
    pub fn flatten(self) -> Option1<T> {
        self.take()
    }
}

impl<T> Option1<Option<T>> {
    pub fn transpose(self) -> Option<Option1<T>> {
        self.take().map(Option1::from_item)
    }
}

impl<T, E> Option1<Result<T, E>> {
    pub fn transpose(self) -> Result<Option1<T>, E> {
        self.take().map(Option1::from_item)
    }
}

impl<T> From<T> for Option1<T> {
    fn from(item: T) -> Self {
        Option1::from_item(item)
    }
}

impl<T> From<Option1<T>> for Option<T> {
    fn from(item: Option1<T>) -> Self {
        item.items
    }
}

impl<T> IntoIterator for Option1<T> {
    type Item = T;
    type IntoIter = AtMostOne<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for Option1<T> {
    fn into_iter1(self) -> ExactlyOne<T> {
        Iterator1::from_iter_unchecked(self.items)
    }
}

impl<T> TryFrom<Option<T>> for Option1<T> {
    type Error = Option<Infallible>;

    fn try_from(option: Option<T>) -> Result<Self, Self::Error> {
        match option {
            Some(item) => Ok(Option1::from_item(item)),
            _ => Err(None),
        }
    }
}

impl<T, E> TryFrom<Result<T, E>> for Option1<T> {
    type Error = Result<Infallible, E>;

    fn try_from(result: Result<T, E>) -> Result<Self, Self::Error> {
        match result {
            Ok(item) => Ok(Option1::from_item(item)),
            Err(error) => Err(Err(error)),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<Option<T>>> for Option1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<Option<T>>) -> Result<Self, Self::Error> {
        Option1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

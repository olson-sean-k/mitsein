//! Non-empty copy-on-write types.

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::Cow;

use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::string1::String1;
#[cfg(target_has_atomic = "ptr")]
use crate::sync1::{ArcSlice1, ArcSlice1Ext as _, ArcStr1, ArcStr1Ext as _};
use crate::vec1::Vec1;

pub type CowSlice1<'a, T> = Cow<'a, Slice1<T>>;

pub trait CowSlice1Ext<'a, T>
where
    T: Clone,
{
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T>;

    fn into_cow_slice(self) -> Cow<'a, [T]>;
}

impl<'a, T> CowSlice1Ext<'a, T> for CowSlice1<'a, T>
where
    T: Clone,
{
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T> {
        ArcSlice1::from_cow_slice1(self)
    }

    fn into_cow_slice(self) -> Cow<'a, [T]> {
        match self {
            Cow::Borrowed(borrowed) => Cow::Borrowed(borrowed),
            Cow::Owned(owned) => Cow::Owned(owned.into_vec()),
        }
    }
}

impl<'a, T> From<&'a Slice1<T>> for CowSlice1<'a, T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        Cow::Borrowed(items)
    }
}

impl<T> From<Vec1<T>> for CowSlice1<'_, T>
where
    T: Clone,
{
    fn from(items: Vec1<T>) -> Self {
        Cow::Owned(items)
    }
}

impl<'a, T> From<&'a Vec1<T>> for CowSlice1<'a, T>
where
    T: Clone,
{
    fn from(items: &'a Vec1<T>) -> Self {
        Cow::Borrowed(items.as_slice1())
    }
}

pub type CowStr1<'a> = Cow<'a, Str1>;

pub trait CowStr1Ext<'a> {
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_str1(self) -> ArcStr1;

    fn into_cow_str(self) -> Cow<'a, str>;
}

impl<'a> CowStr1Ext<'a> for CowStr1<'a> {
    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_str1(self) -> ArcStr1 {
        ArcStr1::from_cow_str1(self)
    }

    fn into_cow_str(self) -> Cow<'a, str> {
        match self {
            Cow::Borrowed(borrowed) => Cow::Borrowed(borrowed),
            Cow::Owned(owned) => Cow::Owned(owned.into_string()),
        }
    }
}

impl<'a> From<&'a Str1> for CowStr1<'a> {
    fn from(items: &'a Str1) -> Self {
        Cow::Borrowed(items)
    }
}

impl From<String1> for CowStr1<'_> {
    fn from(items: String1) -> Self {
        Cow::Owned(items)
    }
}

impl<'a> From<&'a String1> for CowStr1<'a> {
    fn from(items: &'a String1) -> Self {
        Cow::Borrowed(items.as_str1())
    }
}

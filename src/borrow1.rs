//! Non-empty copy-on-write types.

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Cow, ToOwned};

use crate::EmptyError;
use crate::rc1::{RcSlice1, RcSlice1Ext as _, RcStr1, RcStr1Ext as _};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::string1::String1;
#[cfg(target_has_atomic = "ptr")]
use crate::sync1::{ArcSlice1, ArcSlice1Ext as _, ArcStr1, ArcStr1Ext as _};
use crate::vec1::Vec1;

pub type CowSlice1<'a, T> = Cow<'a, Slice1<T>>;

pub trait CowSlice1Ext<'a, T>: Sized
where
    T: Clone,
{
    fn try_from_cow_slice(items: Cow<'a, [T]>) -> Result<Self, EmptyError<Cow<'a, [T]>>>;

    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_slice1(self) -> ArcSlice1<T>;

    fn into_cow_slice(self) -> Cow<'a, [T]>;

    fn into_rc_slice1(self) -> RcSlice1<T>;
}

impl<'a, T> CowSlice1Ext<'a, T> for CowSlice1<'a, T>
where
    T: Clone,
{
    fn try_from_cow_slice(items: Cow<'a, [T]>) -> Result<Self, EmptyError<Cow<'a, [T]>>> {
        match items {
            Cow::Borrowed(borrowed) => Slice1::try_from_slice(borrowed)
                .map(From::from)
                .map_err(|error| error.map(From::from)),
            Cow::Owned(owned) => Vec1::try_from(owned)
                .map(From::from)
                .map_err(|error| error.map(From::from)),
        }
    }

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

    fn into_rc_slice1(self) -> RcSlice1<T> {
        RcSlice1::from_cow_slice1(self)
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

impl<T, U> PartialEq<Vec1<U>> for CowSlice1<'_, T>
where
    Slice1<T>: ToOwned,
    T: PartialEq<U>,
{
    fn eq(&self, other: &Vec1<U>) -> bool {
        PartialEq::eq(self.as_ref(), other)
    }
}

pub type CowStr1<'a> = Cow<'a, Str1>;

pub trait CowStr1Ext<'a>: Sized {
    fn try_from_cow_str(items: Cow<'a, str>) -> Result<Self, EmptyError<Cow<'a, str>>>;

    #[cfg(target_has_atomic = "ptr")]
    #[cfg_attr(docsrs, doc(cfg(target_has_atomic = "ptr")))]
    fn into_arc_str1(self) -> ArcStr1;

    fn into_cow_str(self) -> Cow<'a, str>;

    fn into_rc_str1(self) -> RcStr1;
}

impl<'a> CowStr1Ext<'a> for CowStr1<'a> {
    fn try_from_cow_str(items: Cow<'a, str>) -> Result<Self, EmptyError<Cow<'a, str>>> {
        match items {
            Cow::Borrowed(borrowed) => Str1::try_from_str(borrowed)
                .map(From::from)
                .map_err(|error| error.map(From::from)),
            Cow::Owned(owned) => String1::try_from(owned)
                .map(From::from)
                .map_err(|error| error.map(From::from)),
        }
    }

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

    fn into_rc_str1(self) -> RcStr1 {
        RcStr1::from_cow_str1(self)
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

impl PartialEq<String1> for CowStr1<'_> {
    fn eq(&self, other: &String1) -> bool {
        PartialEq::eq(self.as_ref().as_str(), other.as_string())
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use alloc::vec::Vec;
    use rstest::rstest;
    use serde_test::Token;

    use crate::borrow1::CowSlice1;
    use crate::serde::{self, harness::sequence};
    use crate::slice1::Slice1;
    use crate::slice1::harness::xs1;

    #[rstest]
    fn serialize_cow_slice1_from_tokens_eq(
        xs1: &Slice1<u8>,
        sequence: impl Iterator<Item = Token>,
    ) {
        let xs1 = CowSlice1::from(xs1);
        serde::harness::assert_into_tokens_eq::<_, Vec<_>>(xs1, sequence)
    }
}

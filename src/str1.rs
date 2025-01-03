//! A non-empty [str][`prim@str`].

use core::fmt::{self, Debug, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::SliceIndex;
use core::str::{self, Bytes, Lines, Utf8Error};
#[cfg(feature = "alloc")]
use {alloc::borrow::ToOwned, alloc::string::String};

use crate::iter1::Iterator1;
use crate::safety;
use crate::slice1::Slice1;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};
#[cfg(feature = "alloc")]
use {crate::boxed1::BoxedStr1, crate::string1::String1};

unsafe impl MaybeEmpty for &'_ str {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        // Unlike other containers, the items (bytes) in UTF-8 encoded strings are incongruent with
        // the items manipulated by insertions and removals: code points (`char`s). Cardinality is
        // based on code points formed from the UTF-8 rather than the count of bytes. This can
        // present some edge cases, such as a non-zero number of invalid UTF-8 bytes that cannot be
        // interpreted as code points.
        match self.chars().take(2).count() {
            0 => None,
            1 => Some(Cardinality::One(())),
            _ => Some(Cardinality::Many(())),
        }
    }
}

unsafe impl MaybeEmpty for &'_ mut str {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        (&**self).cardinality()
    }
}

pub type Str1 = NonEmpty<str>;

// TODO: At time of writing, `const` functions are not supported in traits, so
//       `FromMaybeEmpty::from_maybe_empty_unchecked` cannot be used to construct a `Str1` yet. Use
//       that function instead of `mem::transmute` when possible.
impl Str1 {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with an empty
    /// string literal `""`.
    pub const unsafe fn from_str_unchecked(items: &str) -> &Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `str` and `Str1`
        //         are the same.
        mem::transmute::<&'_ str, &'_ Str1>(items)
    }

    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with an empty
    /// string literal `""`.
    pub const unsafe fn from_mut_str_unchecked(items: &mut str) -> &mut Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `str` and `Str1`
        //         are the same.
        mem::transmute::<&'_ mut str, &'_ mut Str1>(items)
    }

    pub fn try_from_str(items: &str) -> Result<&Self, &str> {
        items.try_into()
    }

    pub fn try_from_mut_str(items: &mut str) -> Result<&mut Self, &mut str> {
        items.try_into()
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn to_string1(&self) -> String1 {
        String1::from(self)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_string1(self: BoxedStr1) -> String1 {
        String1::from(self)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn once_and_then_repeat(&self, n: usize) -> String1 {
        // SAFETY: `self` must be non-empty.
        unsafe {
            String1::from_string_unchecked(
                self.items
                    .repeat(n.checked_add(1).expect("overflow in slice repetition")),
            )
        }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn first(&self) -> char {
        use crate::safety::OptionExt as _;

        // SAFETY: `self` must be non-empty.
        unsafe { self.items.chars().next().unwrap_maybe_unchecked() }
    }

    pub fn bytes1(&self) -> Iterator1<Bytes<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_str().bytes()) }
    }

    pub fn lines1(&self) -> Iterator1<Lines<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_str().lines()) }
    }

    pub const fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { safety::non_zero_from_usize_maybe_unchecked(self.items.len()) }
    }

    pub const fn as_bytes1(&self) -> &Slice1<u8> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_slice_unchecked(self.as_str().as_bytes()) }
    }

    pub const fn as_bytes1_mut(&mut self) -> &mut Slice1<u8> {
        // SAFETY: `self` must be non-empty.
        unsafe { Slice1::from_mut_slice_unchecked(self.as_mut_str().as_bytes_mut()) }
    }

    pub const fn as_str(&self) -> &'_ str {
        &self.items
    }

    pub const fn as_mut_str(&mut self) -> &'_ mut str {
        &mut self.items
    }
}

impl AsMut<str> for &'_ mut Str1 {
    fn as_mut(&mut self) -> &mut str {
        &mut self.items
    }
}

impl AsRef<str> for &'_ Str1 {
    fn as_ref(&self) -> &str {
        &self.items
    }
}

impl Debug for Str1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.as_str())
    }
}

impl Deref for Str1 {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl DerefMut for Str1 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<'a> From<&'a Str1> for String {
    fn from(items: &'a Str1) -> Self {
        String::from(items.as_str())
    }
}

impl<I> Index<I> for Str1
where
    I: SliceIndex<str>,
{
    type Output = <I as SliceIndex<str>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<I> IndexMut<I> for Str1
where
    I: SliceIndex<str>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl ToOwned for Str1 {
    type Owned = String1;

    fn to_owned(&self) -> Self::Owned {
        String1::from(self)
    }
}

impl<'a> TryFrom<&'a str> for &'a Str1 {
    type Error = &'a str;

    fn try_from(items: &'a str) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a mut str> for &'a mut Str1 {
    type Error = &'a mut str;

    fn try_from(items: &'a mut str) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

// There is no counterpart to `str::from_utf8_unchecked` here, because it is not clear that a `str`
// consisting of one or more bytes of invalid UTF-8 will yield at least one `char` (code point) in
// its API. This forms a deceptively complex API, because both the data format and non-empty
// guarantees must be maintained by callers (despite accepting a `Slice1`). Contrast this with
// `Str1::from_str_unchecked`, which is only sensitive to the non-empty guarantee.
pub const fn from_utf8(items: &Slice1<u8>) -> Result<&Str1, Utf8Error> {
    match str::from_utf8(items.as_slice()) {
        // SAFETY: `items` is non-empty and `str::from_utf8` has parsed it as valid UTF-8, so there
        //         must be one or more code points.
        Ok(items) => Ok(unsafe { Str1::from_str_unchecked(items) }),
        Err(error) => Err(error),
    }
}

pub fn from_utf8_mut(items: &mut Slice1<u8>) -> Result<&mut Str1, Utf8Error> {
    match str::from_utf8_mut(items.as_mut_slice()) {
        // SAFETY: `items` is non-empty and `str::from_utf8_mut` has parsed it as valid UTF-8, so
        //         there must be one or more code points.
        Ok(items) => Ok(unsafe { Str1::from_mut_str_unchecked(items) }),
        Err(error) => Err(error),
    }
}

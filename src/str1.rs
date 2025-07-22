//! A non-empty [str][`prim@str`].

#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::fmt::{self, Debug, Display, Formatter};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::SliceIndex;
use core::str::{
    self, Bytes, CharIndices, Chars, EncodeUtf16, Lines, Split, SplitInclusive, SplitTerminator,
    Utf8Error,
};
#[cfg(feature = "rayon")]
use rayon::str::ParallelString;
#[cfg(feature = "alloc")]
use {alloc::borrow::ToOwned, alloc::string::String};

use crate::iter1::Iterator1;
#[cfg(feature = "rayon")]
use crate::iter1::ParallelIterator1;
use crate::safety;
use crate::slice1::Slice1;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};
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

    pub fn try_from_str(items: &str) -> Result<&Self, EmptyError<&str>> {
        items.try_into()
    }

    pub fn try_from_mut_str(items: &mut str) -> Result<&mut Self, EmptyError<&mut str>> {
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

    pub fn encode1_utf16(&self) -> Iterator1<EncodeUtf16<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.encode_utf16()) }
    }

    pub fn split1<'a, P>(&'a self, separator: &'a P) -> Iterator1<Split<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.split(separator.as_ref())) }
    }

    pub fn split1_inclusive<'a, P>(
        &'a self,
        separator: &'a P,
    ) -> Iterator1<SplitInclusive<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.split_inclusive(separator.as_ref())) }
    }

    pub fn split1_terminator<'a, P>(
        &'a self,
        separator: &'a P,
    ) -> Iterator1<SplitTerminator<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.split_terminator(separator.as_ref())) }
    }

    #[cfg(feature = "alloc")]
    pub(crate) fn first(&self) -> char {
        use crate::safety::OptionExt as _;

        // SAFETY: `self` must be non-empty.
        unsafe { self.items.chars().next().unwrap_maybe_unchecked() }
    }

    pub fn chars1(&self) -> Iterator1<Chars<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_str().chars()) }
    }

    pub fn char_indices1(&self) -> Iterator1<CharIndices<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.as_str().char_indices()) }
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

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl Str1 {
    pub fn par_encode1_utf16(&self) -> ParallelIterator1<rayon::str::EncodeUtf16<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_encode_utf16()) }
    }

    pub fn par_split1<'a, P>(
        &'a self,
        separator: &'a P,
    ) -> ParallelIterator1<rayon::str::Split<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_split(separator.as_ref())) }
    }

    pub fn par_split1_inclusive<'a, P>(
        &'a self,
        separator: &'a P,
    ) -> ParallelIterator1<rayon::str::SplitInclusive<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe {
            ParallelIterator1::from_par_iter_unchecked(self.par_split_inclusive(separator.as_ref()))
        }
    }

    pub fn par_split1_terminator<'a, P>(
        &'a self,
        separator: &'a P,
    ) -> ParallelIterator1<rayon::str::SplitTerminator<'a, &'a [char]>>
    where
        P: 'a + AsRef<[char]>,
    {
        // SAFETY: `self` must be non-empty.
        unsafe {
            ParallelIterator1::from_par_iter_unchecked(
                self.par_split_terminator(separator.as_ref()),
            )
        }
    }

    pub fn par_chars1(&self) -> ParallelIterator1<rayon::str::Chars<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_chars()) }
    }

    pub fn par_char_indices1(&self) -> ParallelIterator1<rayon::str::CharIndices<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_char_indices()) }
    }

    pub fn par_bytes1(&self) -> ParallelIterator1<rayon::str::Bytes<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_bytes()) }
    }

    pub fn par_lines1(&self) -> ParallelIterator1<rayon::str::Lines<'_>> {
        // SAFETY: `self` must be non-empty.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.par_lines()) }
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> Arbitrary<'a> for &'a Str1 {
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        loop {
            if let Ok(items) = Str1::try_from_str(<&'a str>::arbitrary(unstructured)?) {
                break Ok(items);
            }
        }
    }

    fn size_hint(_: usize) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl AsMut<str> for Str1 {
    fn as_mut(&mut self) -> &mut str {
        &mut self.items
    }
}

impl Debug for Str1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", &self.items)
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

impl Display for Str1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", &self.items)
    }
}

impl<'a> From<&'a Str1> for &'a str {
    fn from(items: &'a Str1) -> Self {
        &items.items
    }
}

impl<'a> From<&'a mut Str1> for &'a mut str {
    fn from(items: &'a mut Str1) -> Self {
        &mut items.items
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

crate::impl_partial_eq_for_non_empty!([in str] <= [in Str1]);
crate::impl_partial_eq_for_non_empty!([in Str1] => [in str]);

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl ToOwned for Str1 {
    type Owned = String1;

    fn to_owned(&self) -> Self::Owned {
        String1::from(self)
    }
}

impl<'a> TryFrom<&'a str> for &'a Str1 {
    type Error = EmptyError<&'a str>;

    fn try_from(items: &'a str) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a mut str> for &'a mut Str1 {
    type Error = EmptyError<&'a mut str>;

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

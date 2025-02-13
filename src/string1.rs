//! A non-empty [`String`][`string`].
//!
//! [`string`]: alloc::string

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, Cow};
use alloc::string::{FromUtf16Error, FromUtf8Error, String};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::fmt::{self, Debug, Formatter, Write};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice::SliceIndex;

use crate::borrow1::CowStr1;
use crate::boxed1::{BoxedStr1, BoxedStr1Ext as _};
use crate::iter1::{Extend1, FromIterator1, IntoIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::take;
use crate::vec1::Vec1;
use crate::{Cardinality, FromMaybeEmpty, MaybeEmpty, NonEmpty};

impl Extend1<char> for String {
    fn extend_non_empty<I>(mut self, items: I) -> String1
    where
        I: IntoIterator1<Item = char>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { String1::from_maybe_empty_unchecked(self) }
    }
}

unsafe impl MaybeEmpty for String {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_str().cardinality()
    }
}

type TakeOr<'a, N = ()> = take::TakeOr<'a, String, char, N>;

pub type PopOr<'a> = TakeOr<'a, ()>;

pub type RemoveOr<'a> = TakeOr<'a, usize>;

impl<N> TakeOr<'_, N> {
    pub fn only(self) -> Result<char, char> {
        self.take_or_else(|items, _| items.first())
    }

    pub fn replace_only(self, replacement: char) -> Result<char, char> {
        self.else_replace_only(move || replacement)
    }

    pub fn else_replace_only<F>(self, f: F) -> Result<char, char>
    where
        F: FnOnce() -> char,
    {
        self.take_or_else(move |items, _| {
            let target = items.first();
            items.items.clear();
            items.items.push(f());
            target
        })
    }
}

impl TakeOr<'_, usize> {
    pub fn get(self) -> Result<char, char> {
        self.take_or_else(|items, index| {
            if items.is_char_boundary(index) {
                items.first()
            }
            else {
                self::panic_index_is_not_char_boundary()
            }
        })
    }

    pub fn replace(self, replacement: char) -> Result<char, char> {
        self.else_replace(move || replacement)
    }

    pub fn else_replace<F>(self, f: F) -> Result<char, char>
    where
        F: FnOnce() -> char,
    {
        self.take_or_else(move |items, index| {
            if items.is_char_boundary(index) {
                let target = items.items.remove(index);
                items.items.push(f());
                target
            }
            else {
                self::panic_index_is_not_char_boundary()
            }
        })
    }
}

pub type String1 = NonEmpty<String>;

impl String1 {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`Vec::new()`][`Vec::new`].
    ///
    /// [`Vec::new`]: alloc::vec::Vec::new
    pub unsafe fn from_string_unchecked(items: String) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn from_one_with_capacity<U>(item: char, capacity: usize) -> Self {
        String1::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        String: Extend1<U::Item>,
        U: IntoIterator1,
    {
        String::with_capacity(capacity).extend_non_empty(items)
    }

    pub fn from_utf8(items: Vec1<u8>) -> Result<Self, FromUtf8Error> {
        // SAFETY: `items` is non-empty and `String::from_utf8` checks for valid UTF-8, so there
        //         must be one or more code points.
        String::from_utf8(items.into_vec())
            .map(|items| unsafe { String1::from_string_unchecked(items) })
    }

    pub fn from_utf8_lossy(items: &Slice1<u8>) -> CowStr1 {
        // SAFETY: `items` is non-empty and `String::from_utf8_lossy` checks for valid UTF-8 or
        //         introduces replacement characters, so there must be one or more code points.
        unsafe {
            match String::from_utf8_lossy(items.as_slice()) {
                Cow::Borrowed(items) => Cow::Borrowed(Str1::from_str_unchecked(items)),
                Cow::Owned(items) => Cow::Owned(String1::from_string_unchecked(items)),
            }
        }
    }

    pub fn from_utf16(items: &Slice1<u16>) -> Result<Self, FromUtf16Error> {
        // SAFETY: `items` is non-empty and `String::from_utf16` checks for valid UTF-16, so there
        //         must be one or more code points.
        String::from_utf16(items.as_slice())
            .map(|items| unsafe { String1::from_string_unchecked(items) })
    }

    pub fn from_utf16_lossy(items: &Slice1<u16>) -> String1 {
        // SAFETY: `items` is non-empty and `String::from_utf16` checks for valid UTF-16, so there
        //         must be one or more code points.
        unsafe { String1::from_string_unchecked(String::from_utf16_lossy(items.as_slice())) }
    }

    pub fn into_string(self) -> String {
        self.items
    }

    pub fn into_boxed_str1(self) -> BoxedStr1 {
        // SAFETY: `self` must be non-empty.
        unsafe { BoxedStr1::from_boxed_str_unchecked(self.items.into_boxed_str()) }
    }

    pub fn leak<'a>(self) -> &'a mut Str1 {
        // SAFETY: `self` must be non-empty.
        unsafe { Str1::from_mut_str_unchecked(self.items.leak()) }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.items.reserve_exact(additional)
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn split_off_tail(&mut self) -> String {
        let index = unsafe {
            // SAFETY: `self` must be non-empty.
            self.items
                .char_indices()
                .take(2)
                .last()
                .map(|(index, _)| index)
                .unwrap_maybe_unchecked()
        };
        self.items.split_off(index)
    }

    pub fn push(&mut self, item: char) {
        self.items.push(item)
    }

    pub fn push_str(&mut self, items: &str) {
        self.items.push_str(items)
    }

    pub fn pop_or(&mut self) -> PopOr<'_> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: char) {
        self.items.insert(index, item)
    }

    pub fn remove_or(&mut self, index: usize) -> RemoveOr<'_> {
        TakeOr::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub const fn as_string(&self) -> &String {
        &self.items
    }

    /// # Safety
    ///
    /// The returned [`Vec1`] must contain valid UTF-8 when the reference is dropped. Note that the
    /// non-empty guarantee of `String1` may also be violated by invalid UTF-8, because invalid
    /// UTF-8 bytes may yield no code points.
    pub unsafe fn as_mut_vec1(&mut self) -> &mut Vec1<u8> {
        unsafe { mem::transmute(self.items.as_mut_vec()) }
    }

    pub fn as_str1(&self) -> &Str1 {
        // SAFETY: `self` must be non-empty.
        unsafe { Str1::from_str_unchecked(self.items.as_str()) }
    }

    pub fn as_mut_str1(&mut self) -> &mut Str1 {
        // SAFETY: `self` must be non-empty.
        unsafe { Str1::from_mut_str_unchecked(self.items.as_mut_str()) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.items.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.items.as_mut_ptr()
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> Arbitrary<'a> for String1 {
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        <&'a Str1>::arbitrary(unstructured).map(String1::from)
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        (<&'a Str1>::size_hint(depth).0, None)
    }
}

impl AsMut<str> for String1 {
    fn as_mut(&mut self) -> &mut str {
        self.items.as_mut()
    }
}

impl AsMut<Str1> for String1 {
    fn as_mut(&mut self) -> &mut Str1 {
        self.as_mut_str1()
    }
}

impl AsRef<str> for String1 {
    fn as_ref(&self) -> &str {
        self.items.as_ref()
    }
}

impl AsRef<Str1> for String1 {
    fn as_ref(&self) -> &Str1 {
        self.as_str1()
    }
}

impl Borrow<str> for String1 {
    fn borrow(&self) -> &str {
        self.items.borrow()
    }
}

impl Borrow<Str1> for String1 {
    fn borrow(&self) -> &Str1 {
        self.as_str1()
    }
}

impl BorrowMut<str> for String1 {
    fn borrow_mut(&mut self) -> &mut str {
        self.items.borrow_mut()
    }
}

impl BorrowMut<Str1> for String1 {
    fn borrow_mut(&mut self) -> &mut Str1 {
        self.as_mut_str1()
    }
}

impl Debug for String1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", &self.items)
    }
}

impl Deref for String1 {
    type Target = Str1;

    fn deref(&self) -> &Self::Target {
        self.as_str1()
    }
}

impl DerefMut for String1 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str1()
    }
}

impl<T> Extend<T> for String1
where
    String: Extend<T>,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl From<BoxedStr1> for String1 {
    fn from(items: BoxedStr1) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { String1::from_string_unchecked(String::from(items.into_boxed_str())) }
    }
}

impl<'a> From<&'a Str1> for String1 {
    fn from(items: &'a Str1) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { String1::from_string_unchecked(String::from(items.as_str())) }
    }
}

impl<'a> From<&'a mut Str1> for String1 {
    fn from(items: &'a mut Str1) -> Self {
        // SAFETY: `items` must be non-empty.
        unsafe { String1::from_string_unchecked(String::from(items.as_str())) }
    }
}

impl From<String1> for String {
    fn from(items: String1) -> Self {
        items.items
    }
}

impl FromIterator1<char> for String1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = char>,
    {
        // SAFETY: `items` must be non-empty.
        unsafe { String1::from_string_unchecked(items.into_iter().collect()) }
    }
}

impl<I> Index<I> for String1
where
    I: SliceIndex<str>,
{
    type Output = I::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<I> IndexMut<I> for String1
where
    I: SliceIndex<str>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<'a> TryFrom<&'a str> for String1 {
    type Error = &'a str;

    fn try_from(items: &'a str) -> Result<Self, Self::Error> {
        Str1::try_from_str(items).map(String1::from)
    }
}

impl<'a> TryFrom<&'a mut str> for String1 {
    type Error = &'a mut str;

    fn try_from(items: &'a mut str) -> Result<Self, Self::Error> {
        Str1::try_from_mut_str(items).map(String1::from)
    }
}

impl TryFrom<String> for String1 {
    type Error = String;

    fn try_from(items: String) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl Write for String1 {
    fn write_str(&mut self, items: &str) -> fmt::Result {
        self.items.write_str(items)
    }

    fn write_char(&mut self, item: char) -> fmt::Result {
        self.items.write_char(item)
    }
}

const fn panic_index_is_not_char_boundary() -> ! {
    panic!("index is not at a UTF-8 code point boundary")
}

#[cfg(test)]
pub mod harness {
    use rstest::fixture;

    use crate::iter1;
    use crate::string1::String1;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> String1 {
        iter1::one('x')
            .first_and_then_take(usize::from(end))
            .collect1()
    }
}

#[cfg(test)]
mod tests {}

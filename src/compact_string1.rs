//! A non-empty [`CompactString`].

#![cfg(feature = "compact-str")]
#![cfg_attr(docsrs, doc(cfg(feature = "compact-str")))]

use alloc::borrow::Cow;
use alloc::string::String;
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use compact_str::{CompactString, CompactStringExt as _, ReserveError, Utf16Error};
use core::borrow::{Borrow, BorrowMut};
use core::fmt::{self, Debug, Display, Formatter, Write};
use core::num::NonZeroUsize;
use core::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::slice::SliceIndex;
use core::str::{FromStr, Utf8Error};
#[cfg(feature = "schemars")]
use schemars::{JsonSchema, Schema, SchemaGenerator};
#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::error::Error as StdError;

use crate::borrow1::{CowStr1, CowStr1Ext as _};
use crate::boxed1::{BoxedStr1, BoxedStr1Ext as _};
use crate::iter1::{Extend1, FromIterator1, IntoIterator1};
use crate::rc1::{RcStr1, RcStr1Ext as _};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::sealed::MaybeEmpty;
use crate::slice1::Slice1;
#[cfg(feature = "smallvec")]
use crate::small_vec1::SmallVec1;
use crate::str1::Str1;
use crate::string1::String1;
use crate::sync1::{ArcStr1, ArcStr1Ext as _};
use crate::vec1::Vec1;
use crate::{Cardinality, FromMaybeEmpty, NonEmpty};
use crate::{EmptyError, take};

impl Add<&Str1> for CompactString {
    type Output = Self;

    fn add(mut self, rhs: &Str1) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl AddAssign<&Str1> for CompactString {
    fn add_assign(&mut self, rhs: &Str1) {
        self.push_str(rhs);
    }
}

impl<'a> Extend<&'a Str1> for CompactString {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = &'a Str1>,
    {
        self.extend(items.into_iter().map(Str1::as_str))
    }
}

impl Extend<BoxedStr1> for CompactString {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = BoxedStr1>,
    {
        self.extend(
            items
                .into_iter()
                .map(|items| CompactString::from(items.as_str())),
        )
    }
}

impl Extend<CompactString1> for CompactString {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = CompactString1>,
    {
        self.extend(items.into_iter().map(CompactString1::into_compact_string))
    }
}

impl Extend<CompactString1> for Cow<'_, str> {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = CompactString1>,
    {
        self.extend(items.into_iter().map(CompactString1::into_compact_string))
    }
}

impl Extend<CompactString1> for String {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = CompactString1>,
    {
        self.extend(items.into_iter().map(CompactString1::into_compact_string))
    }
}

impl Extend1<char> for CompactString {
    fn extend_non_empty<I>(mut self, items: I) -> CompactString1
    where
        I: IntoIterator1<Item = char>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { CompactString1::from_maybe_empty_unchecked(self) }
    }
}

unsafe impl MaybeEmpty for CompactString {
    fn cardinality(&self) -> Option<Cardinality<(), ()>> {
        self.as_str().cardinality()
    }
}

type TakeIfMany<'a, N = ()> = take::TakeIfMany<'a, CompactString, char, N>;

pub type PopIfMany<'a> = TakeIfMany<'a, ()>;

pub type RemoveIfMany<'a> = TakeIfMany<'a, usize>;

impl<N> TakeIfMany<'_, N> {
    pub fn or_get_only(self) -> Result<char, char> {
        self.take_or_else(|items, _| items.first())
    }

    pub fn or_replace_only(self, replacement: char) -> Result<char, char> {
        self.or_else_replace_only(move || replacement)
    }

    pub fn or_else_replace_only<F>(self, f: F) -> Result<char, char>
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

impl TakeIfMany<'_, usize> {
    pub fn or_get(self) -> Result<char, char> {
        self.take_or_else(|items, index| {
            if items.is_char_boundary(index) {
                items.first()
            }
            else {
                self::panic_index_is_not_char_boundary()
            }
        })
    }

    pub fn or_replace(self, replacement: char) -> Result<char, char> {
        self.or_else_replace(move || replacement)
    }

    pub fn or_else_replace<F>(self, f: F) -> Result<char, char>
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

pub type CompactString1 = NonEmpty<CompactString>;

impl CompactString1 {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`CompactString::default()`][`CompactString::default`].
    ///
    /// [`CompactString::default`]: compact_str::CompactString::default
    pub unsafe fn from_compact_string_unchecked(items: CompactString) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
    }

    #[cfg(feature = "smallvec")]
    pub fn into_bytes1(self) -> SmallVec1<[u8; 24]> {
        // SAFETY: `self` is non-empty.
        unsafe { SmallVec1::from_small_vec_unchecked(self.items.into_bytes()) }
    }

    pub fn new_non_empty<T: AsRef<Str1>>(items: T) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { Self::from_compact_string_unchecked(CompactString::new(items.as_ref())) }
    }

    pub fn try_new_non_empty<T: AsRef<Str1>>(items: T) -> Result<Self, ReserveError> {
        // SAFETY: `items` is non-empty.
        CompactString::try_new(items.as_ref())
            .map(|items| unsafe { Self::from_compact_string_unchecked(items) })
    }

    pub const fn const_new_non_empty(items: &'static Str1) -> Self {
        // SAFETY: `items` is non-empty.
        #[allow(unused_unsafe)]
        unsafe {
            // `FromMaybeEmpty::from_maybe_empty_unchecked` would be cleaner,
            // but traits don't support const functinos.
            NonEmpty {
                items: CompactString::const_new(items.as_str()),
            }
        }
    }

    pub const fn as_static_str1(&self) -> Option<&'static Str1> {
        match self.items.as_static_str() {
            // SAFETY: `items` is non-empty.
            Some(items) => Some(unsafe { Str1::from_str_unchecked(items) }),
            None => None,
        }
    }

    pub fn from_one_with_capacity<U>(item: char, capacity: usize) -> Self {
        Self::from_iter1_with_capacity([item], capacity)
    }

    pub fn from_iter1_with_capacity<U>(items: U, capacity: usize) -> Self
    where
        CompactString: Extend1<U::Item>,
        U: IntoIterator1,
    {
        CompactString::with_capacity(capacity).extend_non_empty(items)
    }

    pub fn try_from_one_with_capacity<U>(
        item: char,
        capacity: usize,
    ) -> Result<Self, ReserveError> {
        Self::try_from_iter1_with_capacity([item], capacity)
    }

    pub fn try_from_iter1_with_capacity<U>(items: U, capacity: usize) -> Result<Self, ReserveError>
    where
        CompactString: Extend1<U::Item>,
        U: IntoIterator1,
    {
        // Is this robust enough?
        // Can panic if `items` exceed the available memory.
        CompactString::try_with_capacity(capacity).map(|s| s.extend_non_empty(items))
    }

    pub fn from_utf8<B: AsRef<Slice1<u8>>>(items: B) -> Result<Self, Utf8Error> {
        // SAFETY: `items` is non-empty and `CompactString::from_utf8` checks for valid UTF-8,
        //         so there must be one or more code points.
        CompactString::from_utf8(items.as_ref())
            .map(|items| unsafe { Self::from_compact_string_unchecked(items) })
    }

    pub fn from_utf8_lossy<B: AsRef<Slice1<u8>>>(items: B) -> Self {
        // SAFETY: `items` is non-empty and `CompactString::from_utf8_lossy` checks for valid UTF-8
        //         or introduces replacement characters, so there must be one or more code points.
        unsafe {
            Self::from_compact_string_unchecked(CompactString::from_utf8_lossy(items.as_ref()))
        }
    }

    pub fn from_utf16<B: AsRef<Slice1<u16>>>(items: B) -> Result<Self, Utf16Error> {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16` checks for valid UTF-16,
        //         so there must be one or more code points.
        CompactString::from_utf16(items.as_ref())
            .map(|items| unsafe { Self::from_compact_string_unchecked(items) })
    }

    pub fn from_utf16_lossy<B: AsRef<Slice1<u16>>>(items: B) -> Self {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16_lossy` checks for valid
        //         UTF-16 or introduces replacement characters, so there must be one or more code
        //         points.
        unsafe {
            Self::from_compact_string_unchecked(CompactString::from_utf16_lossy(items.as_ref()))
        }
    }

    pub fn from_utf16le<B: AsRef<Slice1<u8>>>(items: B) -> Result<Self, Utf16Error> {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16le` checks for valid UTF-16,
        //         so there must be one or more code points.
        CompactString::from_utf16le(items.as_ref())
            .map(|items| unsafe { CompactString1::from_compact_string_unchecked(items) })
    }

    pub fn from_utf16le_lossy<B: AsRef<Slice1<u8>>>(items: B) -> Self {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16le_lossy` checks for valid
        //         UTF-16 or introduces replacement characters, so there must be one or more code
        //         points.
        unsafe {
            Self::from_compact_string_unchecked(CompactString::from_utf16le_lossy(items.as_ref()))
        }
    }

    pub fn from_utf16be<B: AsRef<Slice1<u8>>>(items: B) -> Result<Self, Utf16Error> {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16be` checks for valid UTF-16,
        //         so there must be one or more code points.
        CompactString::from_utf16be(items.as_ref())
            .map(|items| unsafe { CompactString1::from_compact_string_unchecked(items) })
    }

    pub fn from_utf16be_lossy<B: AsRef<Slice1<u8>>>(items: B) -> Self {
        // SAFETY: `items` is non-empty and `CompactString::from_utf16be_lossy` checks for valid
        //         UTF-16 or introduces replacement characters, so there must be one or more code
        //         points.
        unsafe {
            Self::from_compact_string_unchecked(CompactString::from_utf16be_lossy(items.as_ref()))
        }
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` is non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY: `self` is non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional)
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), ReserveError> {
        self.items.try_reserve(additional)
    }

    pub fn as_str1(&self) -> &Str1 {
        // SAFETY: `self` is non-empty.
        unsafe { Str1::from_str_unchecked(self.items.as_str()) }
    }

    pub fn as_mut_str1(&mut self) -> &mut Str1 {
        // SAFETY: `self` is non-empty.
        unsafe { Str1::from_mut_str_unchecked(self.items.as_mut_str()) }
    }

    pub fn push(&mut self, item: char) {
        self.items.push(item)
    }

    pub fn pop_if_many(&mut self) -> PopIfMany<'_> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn push_str(&mut self, items: &str) {
        self.items.push_str(items)
    }

    pub fn remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_> {
        TakeIfMany::with(self, index, |items, index| items.items.remove(index))
    }

    pub fn is_heap_allocated(&self) -> bool {
        self.items.is_heap_allocated()
    }

    pub fn replace_range(&mut self, range: impl RangeBounds<usize>, items: &Str1) {
        // SAFETY: `items` is non-empty.
        #[allow(unused_unsafe)]
        unsafe {
            self.items.replace_range(range, items)
        }
    }

    pub fn repeat(&self, n: usize) -> Self {
        // SAFETY: `self` is non-empty.
        unsafe { Self::from_compact_string_unchecked(self.items.repeat(n)) }
    }

    pub fn truncate(&mut self, new_len: NonZeroUsize) {
        self.items.truncate(new_len.get());
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.items.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.items.as_mut_ptr()
    }

    pub fn insert_str(&mut self, idx: usize, items: &str) {
        self.items.insert_str(idx, items)
    }

    pub fn insert(&mut self, index: usize, item: char) {
        self.items.insert(index, item)
    }

    pub fn split_off_tail(&mut self) -> CompactString {
        // Like `String1::split_off_tail`, `CompactString1` must consider UTF-8 byte boundaries
        // here. The byte index of the tail may not be one!
        match self.items.char_indices().nth(1) {
            Some((index, _)) => self.items.split_off(index),
            _ => CompactString::default(),
        }
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<CompactString>>
    where
        F: FnMut(char) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
    }

    pub fn into_string1(self) -> String1 {
        // SAFETY: `self` is non-empty.
        unsafe { String1::from_string_unchecked(self.items.into_string()) }
    }

    pub fn from_string1_buffer(items: String1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe {
            Self::from_compact_string_unchecked(CompactString::from_string_buffer(items.items))
        }
    }

    pub fn to_ascii_lowercase(&self) -> Self {
        // SAFETY: `self` is non-empty.
        unsafe { Self::from_compact_string_unchecked(self.items.to_ascii_lowercase()) }
    }

    pub fn to_ascii_uppercase(&self) -> Self {
        // SAFETY: `self` is non-empty.
        unsafe { Self::from_compact_string_unchecked(self.items.to_ascii_uppercase()) }
    }

    pub fn to_lowercase(&self) -> Self {
        // SAFETY: `self` is non-empty.
        unsafe { Self::from_compact_string_unchecked(self.items.to_lowercase()) }
    }

    pub fn from_str1_to_lowercase(items: &Str1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { Self::from_compact_string_unchecked(CompactString::from_str_to_lowercase(items)) }
    }

    pub fn to_uppercase(&self) -> Self {
        // SAFETY: `self` is non-empty.
        unsafe { Self::from_compact_string_unchecked(self.items.to_uppercase()) }
    }

    pub fn from_str1_to_uppercase(items: &Str1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { Self::from_compact_string_unchecked(CompactString::from_str_to_uppercase(items)) }
    }

    pub fn into_compact_string(self) -> CompactString {
        self.items
    }

    pub fn from_one<U>(item: char) -> Self {
        Self::from_iter1([item])
    }

    pub fn try_from_ref(items: &CompactString) -> Result<&'_ Self, EmptyError<&'_ CompactString>> {
        items.try_into()
    }

    pub fn try_from_mut(
        items: &mut CompactString,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut CompactString>> {
        items.try_into()
    }

    pub const fn as_compact_string(&self) -> &CompactString {
        &self.items
    }

    /// # Safety
    ///
    /// The [`CompactString`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::compact_string1::CompactString1;
    ///
    /// let mut xs = CompactString1::try_from("abc").unwrap();
    /// // This block is unsound. The `&mut CompactString` is dropped in the block and so `xs` can
    /// // be freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_compact_string().clear();
    /// }
    /// let x = xs.as_bytes1().first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_compact_string(&mut self) -> &mut CompactString {
        &mut self.items
    }
}

pub trait CompactString1Ext {
    fn concat_compact1(self) -> CompactString1;

    fn join_compact1<S: AsRef<str>>(self, separator: S) -> CompactString1;
}

impl<I, C> CompactString1Ext for C
where
    I: AsRef<Str1>,
    C: IntoIterator1<Item = I>,
{
    fn concat_compact1(self) -> CompactString1 {
        let (head, tail) = self.into_iter1().into_head_and_tail();
        let mut head = CompactString1::new_non_empty(head);
        for item in tail {
            head += item.as_ref();
        }
        head
    }

    fn join_compact1<S: AsRef<str>>(self, separator: S) -> CompactString1 {
        let (head, tail) = self.into_iter1().into_head_and_tail();
        let mut head = CompactString1::new_non_empty(head);
        for item in tail {
            head += separator.as_ref();
            head += item.as_ref();
        }
        head
    }
}

impl Add<&str> for CompactString1 {
    type Output = Self;

    fn add(mut self, rhs: &str) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl Add<&Str1> for CompactString1 {
    type Output = Self;

    fn add(mut self, rhs: &Str1) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl AddAssign<&str> for CompactString1 {
    fn add_assign(&mut self, rhs: &str) {
        self.push_str(rhs);
    }
}

impl AddAssign<&Str1> for CompactString1 {
    fn add_assign(&mut self, rhs: &Str1) {
        self.push_str(rhs);
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> Arbitrary<'a> for CompactString1 {
    fn arbitrary(unstructured: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // SAFETY: `items` is non-empty.
        CompactString::arbitrary(unstructured)
            .map(|items| unsafe { CompactString1::from_compact_string_unchecked(items) })
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&'a Str1>::size_hint(depth)
    }
}

impl AsMut<str> for CompactString1 {
    fn as_mut(&mut self) -> &mut str {
        self.items.as_mut()
    }
}

impl AsMut<Str1> for CompactString1 {
    fn as_mut(&mut self) -> &mut Str1 {
        self.as_mut_str1()
    }
}

impl AsRef<str> for CompactString1 {
    fn as_ref(&self) -> &str {
        self.items.as_ref()
    }
}

impl AsRef<Str1> for CompactString1 {
    fn as_ref(&self) -> &Str1 {
        self.as_str1()
    }
}

impl Borrow<str> for CompactString1 {
    fn borrow(&self) -> &str {
        self.items.borrow()
    }
}

impl Borrow<Str1> for CompactString1 {
    fn borrow(&self) -> &Str1 {
        self.as_str1()
    }
}

impl BorrowMut<str> for CompactString1 {
    fn borrow_mut(&mut self) -> &mut str {
        self.items.borrow_mut()
    }
}

impl BorrowMut<Str1> for CompactString1 {
    fn borrow_mut(&mut self) -> &mut Str1 {
        self.as_mut_str1()
    }
}

impl Debug for CompactString1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.items)
    }
}

impl Deref for CompactString1 {
    type Target = Str1;

    fn deref(&self) -> &Self::Target {
        self.as_str1()
    }
}

impl DerefMut for CompactString1 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str1()
    }
}

impl Display for CompactString1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.items)
    }
}

// This unfortunately cannot support extending from `CowStr1`s, because `Extend<CowStr1<'_>>`
// cannot be implemented for `CompactString` in this crate. It cannot be implemented directly for
// `CompactString1` either, because it conflicts with this implementation.
impl<T> Extend<T> for CompactString1
where
    CompactString: Extend<T>,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl From<CompactString1> for CompactString {
    fn from(items: CompactString1) -> Self {
        items.items
    }
}

impl<'a> From<&'a CompactString1> for CowStr1<'a> {
    fn from(items: &'a CompactString1) -> Self {
        Self::Borrowed(items)
    }
}

impl<'a> From<&'a String1> for CompactString1 {
    fn from(items: &'a String1) -> Self {
        CompactString1::new_non_empty(items)
    }
}

impl From<&Str1> for CompactString1 {
    fn from(items: &Str1) -> Self {
        CompactString1::new_non_empty(items)
    }
}

impl From<CompactString1> for ArcStr1 {
    fn from(items: CompactString1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcStr1::from_arc_str_unchecked(items.into_compact_string().into()) }
    }
}

#[cfg(feature = "std")]
impl From<CompactString1> for Box<dyn StdError> {
    fn from(items: CompactString1) -> Self {
        items.into_compact_string().into()
    }
}

#[cfg(feature = "std")]
impl From<CompactString1> for Box<dyn StdError + Send + Sync> {
    fn from(items: CompactString1) -> Self {
        items.into_compact_string().into()
    }
}

impl From<CompactString1> for BoxedStr1 {
    fn from(items: CompactString1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { BoxedStr1::from_boxed_str_unchecked(items.into_compact_string().into()) }
    }
}

impl From<CompactString1> for CowStr1<'_> {
    fn from(items: CompactString1) -> Self {
        match items.as_static_str1() {
            Some(items) => Self::Borrowed(items),
            None => Self::Owned(items.into_string1()),
        }
    }
}

impl From<CompactString1> for RcStr1 {
    fn from(items: CompactString1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcStr1::from_rc_str_unchecked(items.into_compact_string().into()) }
    }
}

impl From<CompactString1> for String1 {
    fn from(items: CompactString1) -> Self {
        items.into_string1()
    }
}

impl From<CompactString1> for Vec1<u8> {
    fn from(items: CompactString1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { Vec1::from_vec_unchecked(items.into_compact_string().into()) }
    }
}

impl<'a> From<CowStr1<'a>> for CompactString1 {
    fn from(items: CowStr1<'a>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { CompactString1::from_compact_string_unchecked(items.into_cow_str().into()) }
    }
}

impl From<String1> for CompactString1 {
    fn from(items: String1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { Self::from_compact_string_unchecked(CompactString::from(items.into_string())) }
    }
}

impl<'a> FromIterator1<&'a char> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = &'a char>,
    {
        CompactString1::from_iter1(items.into_iter1().cloned())
    }
}

impl<'a> FromIterator1<&'a Str1> for CompactString {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = &'a Str1>,
    {
        items.concat_compact()
    }
}

impl<'a> FromIterator1<&'a Str1> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = &'a Str1>,
    {
        items.concat_compact1()
    }
}

impl FromIterator1<BoxedStr1> for CompactString {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = BoxedStr1>,
    {
        items.concat_compact1().into_compact_string()
    }
}

impl FromIterator1<BoxedStr1> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = BoxedStr1>,
    {
        items.concat_compact1()
    }
}

impl FromIterator1<CompactString1> for CompactString {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = CompactString1>,
    {
        items.concat_compact()
    }
}

impl FromIterator1<Self> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = Self>,
    {
        items.concat_compact1()
    }
}

impl<'a> FromIterator1<CowStr1<'a>> for CompactString {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = CowStr1<'a>>,
    {
        items.concat_compact1().into_compact_string()
    }
}

impl<'a> FromIterator1<CowStr1<'a>> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = CowStr1<'a>>,
    {
        items.concat_compact1()
    }
}

impl FromIterator1<String1> for CompactString {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = String1>,
    {
        items.concat_compact()
    }
}

impl FromIterator1<String1> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = String1>,
    {
        items.concat_compact1()
    }
}

impl FromIterator1<char> for CompactString1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = char>,
    {
        // SAFETY: `items` is non-empty and each item (`char`) is intrinsically non-empty. A
        //         `CompactString` constructed from one or more `char`s is never empty.
        unsafe { CompactString1::from_compact_string_unchecked(items.into_iter().collect()) }
    }
}

impl FromStr for CompactString1 {
    // Has to be owned.
    type Err = EmptyError<CompactString>;

    fn from_str(items: &str) -> Result<Self, Self::Err> {
        Str1::try_from_str(items)
            .map(Self::from)
            .map_err(|err| err.map(CompactString::new))
    }
}

impl<I> Index<I> for CompactString1
where
    I: SliceIndex<str>,
{
    type Output = I::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<I> IndexMut<I> for CompactString1
where
    I: SliceIndex<str>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

// `JsonSchema` is not implemented for `CompactString` natively, yet.
// However, this implementation can still be useful to some.
#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl JsonSchema for CompactString1 {
    fn schema_name() -> Cow<'static, str> {
        str::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<str>(
            schemars::NON_EMPTY_KEY_STRING,
            generator,
        )
    }

    fn inline_schema() -> bool {
        str::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        str::schema_id()
    }
}

impl<'a> TryFrom<&'a str> for CompactString1 {
    type Error = EmptyError<&'a str>;

    fn try_from(items: &'a str) -> Result<Self, Self::Error> {
        Str1::try_from_str(items).map(Self::from)
    }
}

impl TryFrom<CompactString> for CompactString1 {
    type Error = EmptyError<CompactString>;

    fn try_from(items: CompactString) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a CompactString> for &'a CompactString1 {
    type Error = EmptyError<&'a CompactString>;

    fn try_from(items: &'a CompactString) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a mut CompactString> for &'a mut CompactString1 {
    type Error = EmptyError<&'a mut CompactString>;

    fn try_from(items: &'a mut CompactString) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

crate::impl_partial_eq_for_non_empty!([in &CompactString] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString] <= [in Str1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString] <= [in String1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in &&Str1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in &Str1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in Str1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in &String1]);
crate::impl_partial_eq_for_non_empty!([in CompactString] <= [in String1]);
crate::impl_partial_eq_for_non_empty!([in &Cow<'_, str>] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in Cow<'_, str>] <= [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in Cow<'_, str>] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &String] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in String] <= [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in String] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &&str] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &str] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in str] <= [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in str] <= [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] == [in CowStr1<'_>]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] == [in Str1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] == [in String1]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in &CowStr1<'_>]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in CowStr1<'_>]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in &&Str1]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in &Str1]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] == [in &String1]);
crate::impl_partial_eq_for_non_empty!([in &CowStr1<'_>] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in CowStr1<'_>] == [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in CowStr1<'_>] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &&Str1] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &Str1] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in Str1] == [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &String1] == [in CompactString1]);
crate::impl_partial_eq_for_non_empty!([in String1] == [in &CompactString1]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] => [in Cow<'_, str>]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] => [in String]);
crate::impl_partial_eq_for_non_empty!([in &CompactString1] => [in str]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in &CompactString]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in &Cow<'_, str>]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in Cow<'_, str>]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in &&str]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in &str]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in str]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in &String]);
crate::impl_partial_eq_for_non_empty!([in CompactString1] => [in String]);
crate::impl_partial_eq_for_non_empty!([in String1] => [in &CompactString]);

impl Write for CompactString1 {
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
mod tests {
    #[cfg(feature = "serde")]
    use alloc::borrow::ToOwned as _;
    #[cfg(feature = "serde")]
    use alloc::string::String;
    use compact_str::CompactString;
    use rstest::rstest;

    use crate::compact_string1::{CompactString1, CompactString1Ext as _};
    use crate::iter1::IteratorExt as _;
    #[cfg(feature = "schemars")]
    use crate::schemars;
    use crate::str1::Str1;

    #[rstest]
    #[case("")]
    #[case(CompactString::default())]
    fn compact_string1_cannot_be_empty(#[case] input: impl TryInto<CompactString1>) {
        assert!(input.try_into().is_err());
    }

    #[rstest]
    #[case(["lorem", "ipsum", "dolor"].as_slice(), "loremipsumdolor")]
    #[case(
        ["  a.more ", "  dif_icu1t ", "3x-mple "].as_slice(),
        "  a.more   dif_icu1t 3x-mple ",
    )]
    fn compact_string1_ext_concat_works(#[case] items: &[&str], #[case] expected_output: &str) {
        let actual_output: CompactString1 = {
            items
                .iter()
                .copied()
                .try_into_iter1()
                .unwrap()
                .map(|item| Str1::try_from_str(item).unwrap())
                .concat_compact1()
        };
        assert_eq!(actual_output, expected_output);
    }

    #[rstest]
    #[case(["lorem", "ipsum", "dolor"].as_slice(), "", "loremipsumdolor")]
    #[case(
        ["  a.more ", "  dif_icu1t ", "3x-mple "].as_slice(),
        "",
        "  a.more   dif_icu1t 3x-mple ",
    )]
    #[case(["lorem", "ipsum", "dolor"].as_slice(), " - ", "lorem - ipsum - dolor")]
    #[case(
        ["  a.more ", "  dif_icu1t ", "3x-mple "].as_slice(),
        "!@# 123 ",
        "  a.more !@# 123   dif_icu1t !@# 123 3x-mple ",
    )]
    fn compact_string1_ext_join_works(
        #[case] items: &[&str],
        #[case] separator: &str,
        #[case] expected_output: &str,
    ) {
        let actual_output: CompactString1 = {
            items
                .iter()
                .copied()
                .try_into_iter1()
                .unwrap()
                .map(|item| Str1::try_from_str(item).unwrap())
                .join_compact1(separator)
        };
        assert_eq!(actual_output, expected_output);
    }

    #[cfg(feature = "schemars")]
    #[test]
    fn compact_string1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<CompactString1>(
            schemars::NON_EMPTY_KEY_STRING,
        );
    }

    #[cfg(feature = "serde")]
    #[rstest]
    #[case("x")]
    #[case("ShortStr")]
    #[case(" space  ~around ")]
    #[case("español")]
    #[case("One very long string that cannot possibly fit in twenty four characters.")]
    fn compact_string1_deserializes_correctly(#[case] input: &str) {
        let input_json = serde_json::Value::String(input.to_owned());
        let expected_output = CompactString::new(input);
        let actual_output: CompactString1 = serde_json::from_value(input_json).unwrap();
        assert_eq!(actual_output, expected_output);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn compact_string1_cannot_deserialize_from_empty_string() {
        let empty_string_json = serde_json::Value::String(String::default());
        let actual_output = serde_json::from_value::<CompactString1>(empty_string_json);
        assert!(actual_output.is_err());
    }
}

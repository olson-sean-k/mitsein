//! A non-empty [`String`][`string`].
//!
//! [`string`]: alloc::string

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::borrow::{Borrow, BorrowMut, Cow};
use alloc::string::{FromUtf8Error, FromUtf16Error, String};
#[cfg(feature = "arbitrary")]
use arbitrary::{Arbitrary, Unstructured};
use core::fmt::{self, Debug, Display, Formatter, Write};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut};
use core::slice::SliceIndex;
#[cfg(feature = "schemars")]
use schemars::{JsonSchema, Schema, SchemaGenerator};

use crate::borrow1::{CowStr1, CowStr1Ext as _};
use crate::boxed1::{BoxedStr1, BoxedStr1Ext as _};
use crate::iter1::{Extend1, FromIterator1, IntoIterator1};
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::take;
use crate::vec1::Vec1;
use crate::{Cardinality, EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty};

impl Add<&Str1> for String {
    type Output = String;

    fn add(mut self, rhs: &Str1) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl AddAssign<&Str1> for String {
    fn add_assign(&mut self, rhs: &Str1) {
        self.push_str(rhs);
    }
}

impl<'a> Extend<&'a Str1> for String {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = &'a Str1>,
    {
        self.extend(items.into_iter().map(Str1::as_str))
    }
}

impl Extend<String1> for String {
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = String1>,
    {
        self.extend(items.into_iter().map(String1::into_string))
    }
}

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

type TakeIfMany<'a, N = ()> = take::TakeIfMany<'a, String, char, N>;

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

pub type String1 = NonEmpty<String>;

impl String1 {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`Vec::new()`][`Vec::new`].
    ///
    /// [`Vec::new`]: alloc::vec::Vec::new
    pub unsafe fn from_string_unchecked(items: String) -> Self {
        unsafe { FromMaybeEmpty::from_maybe_empty_unchecked(items) }
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

    pub fn try_from_ref(items: &String) -> Result<&'_ Self, EmptyError<&'_ String>> {
        items.try_into()
    }

    pub fn try_from_mut_ref(
        items: &mut String,
    ) -> Result<&'_ mut Self, EmptyError<&'_ mut String>> {
        items.try_into()
    }

    pub fn from_utf8(items: Vec1<u8>) -> Result<Self, FromUtf8Error> {
        // SAFETY: `items` is non-empty and `String::from_utf8` checks for valid UTF-8, so there
        //         must be one or more code points.
        String::from_utf8(items.into_vec())
            .map(|items| unsafe { String1::from_string_unchecked(items) })
    }

    pub fn from_utf8_lossy(items: &Slice1<u8>) -> CowStr1<'_> {
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
        // SAFETY: `items` is non-empty and `String::from_utf16_lossy` checks for valid UTF-16 or
        //         introduces replacement characters, so there must be one or more code points.
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

    pub fn try_retain<F>(self, f: F) -> Result<Self, EmptyError<String>>
    where
        F: FnMut(char) -> bool,
    {
        self.and_then_try(|items| items.retain(f))
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

    pub fn pop_if_many(&mut self) -> PopIfMany<'_> {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeIfMany::with(self, (), |items, ()| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn insert(&mut self, index: usize, item: char) {
        self.items.insert(index, item)
    }

    pub fn remove_if_many(&mut self, index: usize) -> RemoveIfMany<'_> {
        TakeIfMany::with(self, index, |items, index| items.items.remove(index))
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
    /// The [`String`] behind the returned mutable reference **must not** be empty when the
    /// reference is dropped. Consider the following example:
    ///
    /// ```rust,no_run
    /// use mitsein::string1::String1;
    ///
    /// let mut xs = String1::try_from("abc").unwrap();
    /// // This block is unsound. The `&mut String` is dropped in the block and so `xs` can be
    /// // freely manipulated after the block despite violation of the non-empty guarantee.
    /// unsafe {
    ///     xs.as_mut_string().clear();
    /// }
    /// let x = xs.as_bytes1().first(); // Undefined behavior!
    /// ```
    pub const unsafe fn as_mut_string(&mut self) -> &mut String {
        &mut self.items
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

impl Add<&str> for String1 {
    type Output = String1;

    fn add(mut self, rhs: &str) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl Add<&Str1> for String1 {
    type Output = String1;

    fn add(mut self, rhs: &Str1) -> Self::Output {
        self.push_str(rhs);
        self
    }
}

impl AddAssign<&str> for String1 {
    fn add_assign(&mut self, rhs: &str) {
        self.push_str(rhs);
    }
}

impl AddAssign<&Str1> for String1 {
    fn add_assign(&mut self, rhs: &Str1) {
        self.push_str(rhs);
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

impl Display for String1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", &self.items)
    }
}

// This unfortunately cannot support extending from `CowStr1`s, because `Extend<CowStr1<'_>>`
// cannot be implemented for `String` in this crate. It cannot be implemented directly for
// `String1` either, because it conflicts with this implementation.
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

impl From<char> for String1 {
    fn from(point: char) -> Self {
        // SAFETY: The `From<char>` implementation for `String` never constructs an empty `String`.
        unsafe { String1::from_string_unchecked(String::from(point)) }
    }
}

impl<'a> From<CowStr1<'a>> for String1 {
    fn from(items: CowStr1<'a>) -> Self {
        items.into_owned()
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
        // SAFETY: `items` is non-empty and each item (`char`) is intrinsically non-empty. A
        //         `String` constructed from one or more `char`s is never empty.
        unsafe { String1::from_string_unchecked(items.into_iter().collect()) }
    }
}

impl<'a> FromIterator1<&'a char> for String1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = &'a char>,
    {
        String1::from_iter1(items.into_iter1().cloned())
    }
}

impl<'a> FromIterator1<CowStr1<'a>> for String1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = CowStr1<'a>>,
    {
        let (head, tail) = items.into_iter1().into_head_and_tail();
        let mut head = head.into_owned();
        head.items.extend(tail.map(CowStr1::into_cow_str));
        head
    }
}

impl<'a> FromIterator1<&'a Str1> for String1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = &'a Str1>,
    {
        let (head, tail) = items.into_iter1().into_head_and_tail();
        let mut head = String1::from(head);
        head.extend(tail);
        head
    }
}

impl FromIterator1<String1> for String1 {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = String1>,
    {
        let (mut head, tail) = items.into_iter1().into_head_and_tail();
        head.extend(tail);
        head
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

#[cfg(feature = "schemars")]
#[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
impl JsonSchema for String1 {
    fn schema_name() -> Cow<'static, str> {
        String::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        use crate::schemars;

        schemars::json_subschema_with_non_empty_property_for::<String>(
            schemars::NON_EMPTY_KEY_STRING,
            generator,
        )
    }

    fn inline_schema() -> bool {
        String::inline_schema()
    }

    fn schema_id() -> Cow<'static, str> {
        String::schema_id()
    }
}

crate::impl_partial_eq_for_non_empty!([in str] <= [in String1]);
crate::impl_partial_eq_for_non_empty!([in &str] <= [in String1]);
crate::impl_partial_eq_for_non_empty!([in &Str1] == [in String1]);
crate::impl_partial_eq_for_non_empty!([in CowStr1<'_>] == [in String1]);
crate::impl_partial_eq_for_non_empty!([in String1] => [in str]);
crate::impl_partial_eq_for_non_empty!([in String1] => [in &str]);
crate::impl_partial_eq_for_non_empty!([in String1] == [in &Str1]);

impl<'a> TryFrom<&'a str> for String1 {
    type Error = EmptyError<&'a str>;

    fn try_from(items: &'a str) -> Result<Self, Self::Error> {
        Str1::try_from_str(items).map(String1::from)
    }
}

impl<'a> TryFrom<&'a mut str> for String1 {
    type Error = EmptyError<&'a mut str>;

    fn try_from(items: &'a mut str) -> Result<Self, Self::Error> {
        Str1::try_from_mut_str(items).map(String1::from)
    }
}

impl TryFrom<String> for String1 {
    type Error = EmptyError<String>;

    fn try_from(items: String) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a String> for &'a String1 {
    type Error = EmptyError<&'a String>;

    fn try_from(items: &'a String) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl<'a> TryFrom<&'a mut String> for &'a mut String1 {
    type Error = EmptyError<&'a mut String>;

    fn try_from(items: &'a mut String) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

impl TryFrom<Vec1<u8>> for String1 {
    type Error = FromUtf8Error;

    fn try_from(items: Vec1<u8>) -> Result<Self, Self::Error> {
        String1::from_utf8(items)
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

#[cfg(all(test, feature = "schemars"))]
mod tests {
    use rstest::rstest;

    use crate::schemars;
    use crate::string1::String1;

    #[rstest]
    fn string1_json_schema_has_non_empty_property() {
        schemars::harness::assert_json_schema_has_non_empty_property::<String1>(
            schemars::NON_EMPTY_KEY_STRING,
        );
    }
}

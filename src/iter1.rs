use std::cmp;
use std::iter::{
    Chain, Cloned, Copied, Cycle, Enumerate, Inspect, Map, Peekable, Rev, StepBy, Take,
};
use std::num::NonZeroUsize;
use std::option;

use crate::option1::{Option1, OptionExt as _};
use crate::FnInto;

pub trait IteratorExt: Iterator {
    fn try_into_iter1(self) -> Remainder<Self>
    where
        Self: Sized;

    fn chain1<T>(self, items: T) -> Iterator1<Chain<Self, T::IntoIter>>
    where
        Self: Sized,
        T: IntoIterator1<Item = Self::Item>;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn try_into_iter1(self) -> Remainder<Self> {
        Iterator1::try_from_iter(self)
    }

    fn chain1<T>(self, items: T) -> Iterator1<Chain<Self, T::IntoIter>>
    where
        T: IntoIterator1<Item = Self::Item>,
    {
        Iterator1::from_iter_unchecked(self.chain(items.into_iter1()))
    }
}

pub trait FromIterator1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>;
}

// TODO: This should work, but `rustc` incorrectly believes that this is a coherence error (E0119).
//       The error claims that downstream crates may implement `FromIterator` for collection types
//       in this crate, but they cannot implement foreign traits for foreign types!
//
// impl<T, U> FromIterator1<U> for T
// where
//     T: FromIterator<U>,
// {
//     fn from_iter1<I>(items: I) -> Self
//     where
//         I: IntoIterator1<Item = U>,
//     {
//         items.into_iter1().into_iter().collect()
//     }
// }

pub trait IntoIterator1: IntoIterator {
    fn into_iter1(self) -> Iterator1<Self::IntoIter>;
}

impl<I> IntoIterator1 for Iterator1<I>
where
    I: Iterator,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self
    }
}

pub type AtMostOne<T> = option::IntoIter<T>;

pub type ExactlyOne<T> = Iterator1<AtMostOne<T>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtMostOneFn<F>
where
    F: FnInto,
{
    item: Option<F>,
}

impl<F> AtMostOneFn<F>
where
    F: FnInto,
{
    fn some(f: F) -> Self {
        AtMostOneFn { item: Some(f) }
    }

    fn none() -> Self {
        AtMostOneFn { item: None }
    }
}

impl<F> DoubleEndedIterator for AtMostOneFn<F>
where
    F: FnInto,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl<F> ExactSizeIterator for AtMostOneFn<F>
where
    F: FnInto,
{
    fn len(&self) -> usize {
        if self.item.is_some() {
            1
        }
        else {
            0
        }
    }
}

impl<F> Iterator for AtMostOneFn<F>
where
    F: FnInto,
{
    type Item = F::Into;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.item.take().call()
    }
}

pub type ExactlyOneFn<F> = Iterator1<AtMostOneFn<F>>;

pub type OrItem<I> = Iterator1<Chain<Peekable<I>, AtMostOne<<I as Iterator>::Item>>>;

pub type OrElseItem<I, F> = Iterator1<Chain<Peekable<I>, AtMostOneFn<F>>>;

pub type HeadAndTail<T> =
    Iterator1<Chain<AtMostOne<<T as IntoIterator>::Item>, <T as IntoIterator>::IntoIter>>;

pub type Remainder<I> = Result<Iterator1<Peekable<I>>, Peekable<I>>;

pub trait RemainderExt<I>
where
    I: Iterator,
{
    fn or_item(self, item: I::Item) -> OrItem<I>;

    fn or_else_item<F>(self, f: F) -> OrElseItem<I, F>
    where
        F: FnInto<Into = I::Item>;

    fn into_iter(self) -> Peekable<I>;
}

impl<I> RemainderExt<I> for Remainder<I>
where
    I: Iterator,
{
    fn or_item(self, item: I::Item) -> OrItem<I> {
        Iterator1::from_iter_unchecked(match self {
            Ok(items) => items.into_iter().chain(None),
            Err(empty) => empty.chain(Some(item)),
        })
    }

    fn or_else_item<F>(self, f: F) -> OrElseItem<I, F>
    where
        F: FnInto<Into = I::Item>,
    {
        Iterator1::from_iter_unchecked(match self {
            Ok(items) => items.into_iter().chain(AtMostOneFn::none()),
            Err(empty) => empty.chain(AtMostOneFn::some(f)),
        })
    }

    fn into_iter(self) -> Peekable<I> {
        match self {
            Ok(items) => items.into_iter(),
            Err(empty) => empty,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Iterator1<I> {
    items: I,
}

impl<I> Iterator1<I> {
    pub(crate) fn from_iter_unchecked<T>(items: T) -> Self
    where
        T: IntoIterator<IntoIter = I>,
    {
        Iterator1 {
            items: items.into_iter(),
        }
    }
}

impl<I> Iterator1<I>
where
    I: Iterator,
{
    pub fn try_from_iter<T>(items: T) -> Remainder<I>
    where
        T: IntoIterator<IntoIter = I>,
    {
        let mut items = items.into_iter().peekable();
        match items.peek() {
            Some(_) => Ok(Iterator1::from_iter_unchecked(items)),
            _ => Err(items),
        }
    }

    #[inline(always)]
    fn maybe_empty<T, F>(mut self, f: F) -> (T, Remainder<I>)
    where
        F: FnOnce(&mut I) -> T,
    {
        let output = f(&mut self.items);
        (output, Iterator1::try_from_iter(self.items))
    }

    #[inline(always)]
    fn non_empty<J, F>(self, f: F) -> Iterator1<J>
    where
        J: Iterator,
        F: FnOnce(I) -> J,
    {
        Iterator1::from_iter_unchecked(f(self.items))
    }

    pub fn size_hint(&self) -> (NonZeroUsize, Option<NonZeroUsize>) {
        let (lower, upper) = self.items.size_hint();
        // SAFETY:
        unsafe {
            (
                NonZeroUsize::new_unchecked(cmp::max(1, lower)),
                upper.and_then(|upper| match upper {
                    0 => None,
                    upper => Some(NonZeroUsize::new_unchecked(upper)),
                }),
            )
        }
    }

    pub fn len(&self) -> NonZeroUsize
    where
        I: ExactSizeIterator,
    {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(cmp::max(1, self.items.len())) }
    }

    pub fn as_iter(&self) -> &I {
        &self.items
    }

    pub fn count(self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.count()) }
    }

    pub fn last(self) -> I::Item {
        // SAFETY:
        unsafe { self.items.last().unwrap_unchecked() }
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.min().unwrap_unchecked() }
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.max().unwrap_unchecked() }
    }

    pub fn reduce<F>(self, f: F) -> I::Item
    where
        F: FnMut(I::Item, I::Item) -> I::Item,
    {
        // SAFETY:
        unsafe { self.items.reduce(f).unwrap_unchecked() }
    }

    pub fn any<F>(self, f: F) -> (bool, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.any(f))
    }

    pub fn all<F>(self, f: F) -> (bool, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.all(f))
    }

    pub fn nth(self, n: usize) -> (Option<I::Item>, Remainder<I>) {
        self.maybe_empty(move |items| items.nth(n))
    }

    pub fn find<F>(self, f: F) -> (Option<I::Item>, Remainder<I>)
    where
        F: FnMut(&I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.find(f))
    }

    pub fn find_map<T, F>(self, f: F) -> (Option<T>, Remainder<I>)
    where
        F: FnMut(I::Item) -> Option<T>,
    {
        self.maybe_empty(move |items| items.find_map(f))
    }

    pub fn position<F>(self, f: F) -> (Option<usize>, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.position(f))
    }

    pub fn rposition<F>(self, f: F) -> (Option<usize>, Remainder<I>)
    where
        I: DoubleEndedIterator + ExactSizeIterator,
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.rposition(f))
    }

    pub fn copied<'a, T>(self) -> Iterator1<Copied<I>>
    where
        T: 'a + Copy,
        I: Iterator<Item = &'a T>,
    {
        self.non_empty(I::copied)
    }

    pub fn cloned<'a, T>(self) -> Iterator1<Cloned<I>>
    where
        T: 'a + Clone,
        I: Iterator<Item = &'a T>,
    {
        self.non_empty(I::cloned)
    }

    pub fn enumerate(self) -> Iterator1<Enumerate<I>> {
        self.non_empty(I::enumerate)
    }

    pub fn map<T, F>(self, f: F) -> Iterator1<Map<I, F>>
    where
        F: FnMut(I::Item) -> T,
    {
        self.non_empty(move |items| items.map(f))
    }

    pub fn take(self, n: NonZeroUsize) -> Iterator1<Take<I>> {
        self.non_empty(move |items| items.take(n.into()))
    }

    pub fn cycle(self) -> Iterator1<Cycle<I>>
    where
        I: Clone,
    {
        self.non_empty(I::cycle)
    }

    pub fn chain<T>(self, chained: T) -> Iterator1<Chain<I, T::IntoIter>>
    where
        T: IntoIterator<Item = I::Item>,
    {
        self.non_empty(move |items| items.chain(chained))
    }

    pub fn step_by(self, step: usize) -> Iterator1<StepBy<I>> {
        self.non_empty(move |items| items.step_by(step))
    }

    pub fn inspect<F>(self, f: F) -> Iterator1<Inspect<I, F>>
    where
        F: FnMut(&I::Item),
    {
        self.non_empty(move |items| items.inspect(f))
    }

    pub fn peekable(self) -> Iterator1<Peekable<I>> {
        self.non_empty(I::peekable)
    }

    pub fn rev(self) -> Iterator1<Rev<I>>
    where
        I: DoubleEndedIterator,
    {
        self.non_empty(I::rev)
    }

    pub fn collect<T>(self) -> T
    where
        T: FromIterator1<I::Item>,
    {
        T::from_iter1(self)
    }
}

impl<I> Iterator1<Peekable<I>>
where
    I: Iterator,
{
    pub fn peek(&mut self) -> &I::Item {
        // SAFETY:
        unsafe { self.items.peek().unwrap_unchecked() }
    }
}

impl<I> IntoIterator for Iterator1<I>
where
    I: Iterator,
{
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> Self::IntoIter {
        self.items
    }
}

pub fn item<T>(item: T) -> ExactlyOne<T> {
    Option1::from_item(item).into_iter1()
}

pub fn from_fn<F>(f: F) -> ExactlyOneFn<F>
where
    F: FnInto,
{
    Iterator1::from_iter_unchecked(AtMostOneFn::some(f))
}

pub fn head_and_tail<T, I>(head: T, tail: I) -> HeadAndTail<I>
where
    I: IntoIterator<Item = T>,
{
    Option1::from_item(head).into_iter1().chain(tail)
}

#[cfg(test)]
mod tests {}

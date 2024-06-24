use core::cmp;
use core::iter::{
    self, Chain, Cloned, Copied, Cycle, Enumerate, Flatten, Inspect, Map, Peekable, Repeat, Rev,
    StepBy, Take,
};
use core::num::NonZeroUsize;
use core::option;
#[cfg(all(feature = "alloc", feature = "itertools"))]
use {alloc::vec, itertools::MultiPeek};
#[cfg(feature = "itertools")]
use {
    core::borrow::Borrow,
    itertools::{Itertools, MapInto, MapOk, WithPosition},
};

use crate::option1::Option1;
use crate::FnInto;

pub trait Then1<I>
where
    I: Iterator,
{
    type MaybeEmpty: Iterator<Item = I::Item>;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::MaybeEmpty, T::IntoIter>>
    where
        Self: Sized,
        T: IntoIterator1<Item = I::Item>;

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<I, T>
    where
        Self: Sized,
        T: IntoIterator1<Item = I::Item>;

    fn or_else_non_empty<F>(self, f: F) -> OrElseNonEmpty<I, F>
    where
        Self: Sized,
        F: FnInto,
        F::Into: IntoIterator1<Item = I::Item>;

    fn or_one<T>(self, item: I::Item) -> OrNonEmpty<I, [I::Item; 1]>
    where
        Self: Sized,
    {
        self.or_non_empty([item])
    }
}

impl<I> Then1<I> for I
where
    I: Iterator,
{
    type MaybeEmpty = I;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::MaybeEmpty, T::IntoIter>>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        Iterator1::from_iter_unchecked(self.chain(items.into_iter1()))
    }

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<I, T>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        Iterator1::try_from_iter(self).or_non_empty(items)
    }

    fn or_else_non_empty<F>(self, f: F) -> OrElseNonEmpty<I, F>
    where
        F: FnInto,
        F::Into: IntoIterator1<Item = I::Item>,
    {
        Iterator1::try_from_iter(self).or_else_non_empty(f)
    }
}

pub trait IteratorExt: Iterator + Sized + Then1<Self> {
    fn try_into_iter1(self) -> Remainder<Self>;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn try_into_iter1(self) -> Remainder<Self> {
        Iterator1::try_from_iter(self)
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

pub type AtMostOneWith<F> = iter::OnceWith<F>;

pub type ExactlyOneWith<F> = Iterator1<AtMostOneWith<F>>;

pub type HeadAndTail<T> =
    Iterator1<Chain<AtMostOne<<T as IntoIterator>::Item>, <T as IntoIterator>::IntoIter>>;

pub type TailAndHead<T> =
    Iterator1<Chain<<T as IntoIterator>::IntoIter, AtMostOne<<T as IntoIterator>::Item>>>;

pub type EmptyOrInto<T> = Flatten<AtMostOne<<T as IntoIterator>::IntoIter>>;

pub type OrNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

pub type OrElseNonEmpty<I, F> = Iterator1<Chain<Peekable<I>, EmptyOrInto<<F as FnInto>::Into>>>;

pub type Remainder<I> = Result<Iterator1<Peekable<I>>, Peekable<I>>;

pub trait RemainderExt<I>: Then1<I>
where
    I: Iterator,
{
    fn into_iter(self) -> Peekable<I>;
}

impl<I> RemainderExt<I> for Remainder<I>
where
    I: Iterator,
{
    fn into_iter(self) -> Peekable<I> {
        match self {
            Ok(items) => items.into_iter(),
            Err(empty) => empty,
        }
    }
}

impl<I> Then1<I> for Remainder<I>
where
    I: Iterator,
{
    type MaybeEmpty = Peekable<I>;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::MaybeEmpty, T::IntoIter>>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        Iterator1::from_iter_unchecked(RemainderExt::into_iter(self).chain(items.into_iter1()))
    }

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<I, T>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        Iterator1::from_iter_unchecked(match self {
            Ok(items) => items.into_iter().chain(empty_or_into::<T>(None)),
            Err(empty) => empty.chain(empty_or_into(Some(items))),
        })
    }

    fn or_else_non_empty<F>(self, f: F) -> OrElseNonEmpty<I, F>
    where
        F: FnInto,
        F::Into: IntoIterator1<Item = I::Item>,
    {
        Iterator1::from_iter_unchecked(match self {
            Ok(items) => items.into_iter().chain(empty_or_into::<F::Into>(None)),
            Err(empty) => empty.chain(empty_or_into(Some(f.call()))),
        })
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
                upper.map(|upper| NonZeroUsize::new_unchecked(cmp::max(1, upper))),
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

    pub fn first(mut self) -> I::Item {
        // SAFETY:
        unsafe { self.items.next().unwrap_unchecked() }
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

#[cfg(feature = "itertools")]
#[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
impl<I> Iterator1<I>
where
    I: Iterator,
{
    pub fn map_into<T>(self) -> Iterator1<MapInto<I, T>>
    where
        I::Item: Into<T>,
    {
        self.non_empty(I::map_into)
    }

    pub fn map_ok<F, T, U, E>(self, f: F) -> Iterator1<MapOk<I, F>>
    where
        I: Iterator<Item = Result<T, E>>,
        F: FnMut(T) -> U,
    {
        self.non_empty(move |items| items.map_ok(f))
    }

    pub fn with_position(self) -> Iterator1<WithPosition<I>> {
        self.non_empty(I::with_position)
    }

    pub fn contains<Q>(self, query: &Q) -> (bool, Remainder<I>)
    where
        I::Item: Borrow<Q>,
        Q: PartialEq,
    {
        self.maybe_empty(move |items| items.contains(query))
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn multipeek(self) -> Iterator1<MultiPeek<I>> {
        self.non_empty(I::multipeek)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted(self) -> Iterator1<vec::IntoIter<I::Item>>
    where
        I::Item: Ord,
    {
        self.non_empty(I::sorted)
    }
}

#[cfg(all(feature = "alloc", feature = "itertools"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "itertools"))))]
impl<I> Iterator1<MultiPeek<I>>
where
    I: Iterator,
{
    pub fn peek(&mut self) -> Option<&I::Item> {
        self.items.peek()
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

pub fn from_one<T>(item: T) -> ExactlyOne<T> {
    Option1::from_one(item).into_iter1()
}

pub fn from_one_with<F>(f: F) -> ExactlyOneWith<F>
where
    F: FnInto,
{
    Iterator1::from_iter_unchecked(iter::once_with(f))
}

pub fn from_head_and_tail<T, I>(head: T, tail: I) -> HeadAndTail<I>
where
    I: IntoIterator<Item = T>,
{
    Option1::from_one(head).into_iter1().chain(tail)
}

pub fn from_tail_and_head<I, T>(tail: I, head: T) -> TailAndHead<I>
where
    I: IntoIterator<Item = T>,
{
    tail.into_iter().chain_non_empty(Option1::from_one(head))
}

pub fn repeat<T>(item: T) -> Iterator1<Repeat<T>>
where
    T: Clone,
{
    Iterator1::from_iter_unchecked(iter::repeat(item))
}

fn empty_or_into<T>(items: Option<T>) -> EmptyOrInto<T>
where
    T: IntoIterator,
{
    items.map(T::into_iter).into_iter().flatten()
}

#[cfg(test)]
mod tests {}

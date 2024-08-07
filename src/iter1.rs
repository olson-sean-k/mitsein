use core::fmt::Debug;
use core::iter::{
    self, Chain, Cloned, Copied, Cycle, Enumerate, Flatten, Inspect, Map, Peekable, Repeat, Rev,
    StepBy, Take,
};
use core::num::NonZeroUsize;
use core::option;
use core::result;
#[cfg(all(feature = "alloc", feature = "itertools"))]
use {alloc::vec, itertools::MultiPeek};
#[cfg(feature = "itertools")]
use {
    core::borrow::Borrow,
    itertools::{Itertools, MapInto, MapOk, WithPosition},
};

use crate::{NonZeroExt as _, OptionExt as _, Saturated};

pub trait ThenIterator1<I>
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

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<I, T>
    where
        Self: Sized,
        T: IntoIterator1<Item = I::Item>,
        F: FnOnce() -> T;

    fn or_one<T>(self, item: I::Item) -> OrNonEmpty<I, [I::Item; 1]>
    where
        Self: Sized,
    {
        self.or_non_empty([item])
    }
}

impl<I> ThenIterator1<I> for I
where
    I: Iterator,
{
    type MaybeEmpty = I;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::MaybeEmpty, T::IntoIter>>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.chain(items.into_iter1())) }
    }

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<I, T>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        Iterator1::try_from_iter(self).or_non_empty(items)
    }

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<I, T>
    where
        Self: Sized,
        T: IntoIterator1<Item = I::Item>,
        F: FnOnce() -> T,
    {
        Iterator1::try_from_iter(self).or_else_non_empty(f)
    }
}

pub trait IteratorExt: Iterator + Sized + ThenIterator1<Self> {
    fn try_into_iter1(self) -> Result<Self>;

    fn try_collect1<T>(self) -> result::Result<T, Peekable<Self>>
    where
        T: FromIterator1<Self::Item>,
    {
        T::try_from_iter(self)
    }

    fn saturate<T>(self) -> AndRemainder<T, T::Remainder>
    where
        T: Saturated<Self>,
    {
        T::saturated(self)
    }
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn try_into_iter1(self) -> Result<Self> {
        Iterator1::try_from_iter(self)
    }
}

pub trait FromIterator1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>;

    fn try_from_iter<I>(items: I) -> result::Result<Self, Peekable<I::IntoIter>>
    where
        Self: Sized,
        I: IntoIterator<Item = T>,
    {
        Iterator1::try_from_iter(items).map(Self::from_iter1)
    }
}

// TODO: This blanket implementation is a coherence error (E0119) at time of writing. However, the
//       error claims that downstream crates can implement `FromIterator` for types in this crate,
//       which they cannot. If this is fixed and this implementation becomes possible, provide it
//       and remove the distinction between `collect` and `collect1` in `Iterator1`.
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

pub type FirstAndThen<T, I> = Iterator1<Chain<AtMostOne<T>, I>>;

pub type EmptyOrInto<T> = Flatten<AtMostOne<<T as IntoIterator>::IntoIter>>;

pub type OrNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

pub type OrElseNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

pub type Result<I> = result::Result<Iterator1<Peekable<I>>, Peekable<I>>;

impl<I> ThenIterator1<I> for Result<I>
where
    I: Iterator,
{
    type MaybeEmpty = Peekable<I>;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::MaybeEmpty, T::IntoIter>>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        // SAFETY:
        unsafe {
            Iterator1::from_iter_unchecked(
                match self {
                    Ok(items) => items.into_iter(),
                    Err(empty) => empty,
                }
                .chain(items.into_iter1()),
            )
        }
    }

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<I, T>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        // SAFETY:
        unsafe {
            Iterator1::from_iter_unchecked(match self {
                Ok(items) => items.into_iter().chain(empty_or_into::<T>(None)),
                Err(empty) => empty.chain(empty_or_into(Some(items))),
            })
        }
    }

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<I, T>
    where
        Self: Sized,
        T: IntoIterator1<Item = I::Item>,
        F: FnOnce() -> T,
    {
        // SAFETY:
        unsafe {
            Iterator1::from_iter_unchecked(match self {
                Ok(items) => items.into_iter().chain(empty_or_into::<T>(None)),
                Err(empty) => empty.chain(empty_or_into(Some(f()))),
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct AndRemainder<T, I> {
    pub output: T,
    pub remainder: I,
}

impl<T, I> AndRemainder<T, I> {
    pub fn with_output_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(T),
    {
        let AndRemainder { output, remainder } = self;
        f(output);
        remainder
    }

    pub fn with_remainder_and_then_output<F>(self, f: F) -> T
    where
        F: FnOnce(I),
    {
        let AndRemainder { output, remainder } = self;
        f(remainder);
        output
    }
}

impl<T, I> From<AndRemainder<T, I>> for (T, I) {
    fn from(query: AndRemainder<T, I>) -> Self {
        let AndRemainder { output, remainder } = query;
        (output, remainder)
    }
}

pub type Matched<T, I> = AndRemainder<Option<T>, I>;

impl<T, I> Matched<T, I> {
    pub fn matched(self) -> Option<T> {
        Into::into(self)
    }

    pub fn some_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(T),
    {
        let Matched { output, remainder } = self;
        if let Some(output) = output {
            f(output);
        }
        remainder
    }

    pub fn none_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let Matched { output, remainder } = self;
        if output.is_none() {
            f();
        }
        remainder
    }
}

impl<T, I> From<Matched<T, I>> for Option<T> {
    fn from(query: Matched<T, I>) -> Self {
        query.output
    }
}

pub type IsMatch<I> = AndRemainder<bool, I>;

impl<I> IsMatch<I> {
    pub fn is_match(self) -> bool {
        Into::into(self)
    }

    pub fn if_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let IsMatch { output, remainder } = self;
        if output {
            f();
        }
        remainder
    }

    pub fn if_not_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let IsMatch { output, remainder } = self;
        if !output {
            f();
        }
        remainder
    }
}

impl<I> From<IsMatch<I>> for bool {
    fn from(query: IsMatch<I>) -> Self {
        query.output
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Iterator1<I> {
    items: I,
}

impl<I> Iterator1<I> {
    /// # Safety
    pub unsafe fn from_iter_unchecked<T>(items: T) -> Self
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
    pub fn try_from_iter<T>(items: T) -> Result<I>
    where
        T: IntoIterator<IntoIter = I>,
    {
        let mut items = items.into_iter().peekable();
        match items.peek() {
            // SAFETY:
            Some(_) => Ok(unsafe { Iterator1::from_iter_unchecked(items) }),
            _ => Err(items),
        }
    }

    #[inline(always)]
    fn maybe_empty<T, F>(mut self, f: F) -> AndRemainder<T, I>
    where
        F: FnOnce(&mut I) -> T,
    {
        let output = f(&mut self.items);
        AndRemainder {
            output,
            remainder: self.items,
        }
    }

    #[inline(always)]
    unsafe fn non_empty<J, F>(self, f: F) -> Iterator1<J>
    where
        J: Iterator,
        F: FnOnce(I) -> J,
    {
        Iterator1::from_iter_unchecked(f(self.items))
    }

    pub fn size_hint(&self) -> (NonZeroUsize, Option<NonZeroUsize>) {
        let (lower, upper) = self.items.size_hint();
        (
            NonZeroUsize::clamped(lower),
            upper.map(NonZeroUsize::clamped),
        )
    }

    pub fn len(&self) -> NonZeroUsize
    where
        I: ExactSizeIterator,
    {
        NonZeroUsize::clamped(self.items.len())
    }

    pub fn as_iter(&self) -> &I {
        &self.items
    }

    pub fn count(self) -> NonZeroUsize {
        // Though the count must be non-zero here, it may overflow to zero.
        NonZeroUsize::new(self.items.count())
            .expect("non-empty iterator has zero items or overflow in count")
    }

    pub fn first(mut self) -> I::Item {
        // SAFETY:
        unsafe { self.items.next().unwrap_maybe_unchecked() }
    }

    pub fn last(self) -> I::Item {
        // SAFETY:
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn into_head_and_tail(self) -> (I::Item, I) {
        let AndRemainder { output, remainder } = self.maybe_empty(|items| {
            // SAFETY:
            unsafe { items.next().unwrap_maybe_unchecked() }
        });
        (output, remainder)
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.min().unwrap_maybe_unchecked() }
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.max().unwrap_maybe_unchecked() }
    }

    pub fn reduce<F>(self, f: F) -> I::Item
    where
        F: FnMut(I::Item, I::Item) -> I::Item,
    {
        // SAFETY:
        unsafe { self.items.reduce(f).unwrap_maybe_unchecked() }
    }

    pub fn any<F>(self, f: F) -> IsMatch<I>
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.any(f))
    }

    pub fn all<F>(self, f: F) -> IsMatch<I>
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.all(f))
    }

    pub fn nth(self, n: usize) -> Matched<I::Item, I> {
        self.maybe_empty(move |items| items.nth(n))
    }

    pub fn find<F>(self, f: F) -> Matched<I::Item, I>
    where
        F: FnMut(&I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.find(f))
    }

    pub fn find_map<T, F>(self, f: F) -> Matched<T, I>
    where
        F: FnMut(I::Item) -> Option<T>,
    {
        self.maybe_empty(move |items| items.find_map(f))
    }

    pub fn position<F>(self, f: F) -> Matched<usize, I>
    where
        F: FnMut(I::Item) -> bool,
    {
        self.maybe_empty(move |items| items.position(f))
    }

    pub fn rposition<F>(self, f: F) -> Matched<usize, I>
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
        // SAFETY:
        unsafe { self.non_empty(I::copied) }
    }

    pub fn cloned<'a, T>(self) -> Iterator1<Cloned<I>>
    where
        T: 'a + Clone,
        I: Iterator<Item = &'a T>,
    {
        // SAFETY:
        unsafe { self.non_empty(I::cloned) }
    }

    pub fn enumerate(self) -> Iterator1<Enumerate<I>> {
        // SAFETY:
        unsafe { self.non_empty(I::enumerate) }
    }

    pub fn map<T, F>(self, f: F) -> Iterator1<Map<I, F>>
    where
        F: FnMut(I::Item) -> T,
    {
        // SAFETY:
        unsafe { self.non_empty(move |items| items.map(f)) }
    }

    pub fn map_first_and_then<U, J, H, T>(mut self, head: H, tail: T) -> FirstAndThen<U, J>
    where
        J: Iterator<Item = U>,
        H: FnOnce(I::Item) -> U,
        T: FnOnce(I) -> J,
    {
        let first = self.items.next().map(head);
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(first.into_iter().chain(tail(self.items))) }
    }

    pub fn first_and_then<J, F>(self, f: F) -> FirstAndThen<I::Item, J>
    where
        J: Iterator<Item = I::Item>,
        F: FnOnce(I) -> J,
    {
        self.map_first_and_then(|first| first, f)
    }

    pub fn first_and_then_take(self, n: usize) -> FirstAndThen<I::Item, Take<I>> {
        self.first_and_then(|items| items.take(n))
    }

    pub fn cycle(self) -> Iterator1<Cycle<I>>
    where
        I: Clone,
    {
        // SAFETY:
        unsafe { self.non_empty(I::cycle) }
    }

    pub fn chain<T>(self, chained: T) -> Iterator1<Chain<I, T::IntoIter>>
    where
        T: IntoIterator<Item = I::Item>,
    {
        // SAFETY:
        unsafe { self.non_empty(move |items| items.chain(chained)) }
    }

    pub fn step_by(self, step: usize) -> Iterator1<StepBy<I>> {
        // SAFETY:
        unsafe { self.non_empty(move |items| items.step_by(step)) }
    }

    pub fn inspect<F>(self, f: F) -> Iterator1<Inspect<I, F>>
    where
        F: FnMut(&I::Item),
    {
        // SAFETY:
        unsafe { self.non_empty(move |items| items.inspect(f)) }
    }

    pub fn peekable(self) -> Iterator1<Peekable<I>> {
        // SAFETY:
        unsafe { self.non_empty(I::peekable) }
    }

    pub fn rev(self) -> Iterator1<Rev<I>>
    where
        I: DoubleEndedIterator,
    {
        // SAFETY:
        unsafe { self.non_empty(I::rev) }
    }

    pub fn collect<T>(self) -> T
    where
        T: FromIterator<I::Item>,
    {
        T::from_iter(self)
    }

    pub fn collect1<T>(self) -> T
    where
        T: FromIterator1<I::Item>,
    {
        T::from_iter1(self)
    }

    pub fn saturate<T>(self) -> AndRemainder<T, T::Remainder>
    where
        T: Saturated<Self>,
    {
        T::saturated(self)
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
        // SAFETY:
        unsafe { self.non_empty(I::map_into) }
    }

    pub fn map_ok<F, T, U, E>(self, f: F) -> Iterator1<MapOk<I, F>>
    where
        I: Iterator<Item = result::Result<T, E>>,
        F: FnMut(T) -> U,
    {
        // SAFETY:
        unsafe { self.non_empty(move |items| items.map_ok(f)) }
    }

    pub fn with_position(self) -> Iterator1<WithPosition<I>> {
        // SAFETY:
        unsafe { self.non_empty(I::with_position) }
    }

    pub fn contains<Q>(self, query: &Q) -> IsMatch<I>
    where
        I::Item: Borrow<Q>,
        Q: PartialEq,
    {
        self.maybe_empty(move |items| items.contains(query))
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn multipeek(self) -> Iterator1<MultiPeek<I>> {
        // SAFETY:
        unsafe { self.non_empty(I::multipeek) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted(self) -> Iterator1<vec::IntoIter<I::Item>>
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.non_empty(I::sorted) }
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
        unsafe { self.items.peek().unwrap_maybe_unchecked() }
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

pub fn one<T>(item: T) -> ExactlyOne<T> {
    // SAFETY:
    unsafe { Iterator1::from_iter_unchecked(Some(item)) }
}

pub fn one_with<T, F>(f: F) -> ExactlyOneWith<F>
where
    F: FnOnce() -> T,
{
    // SAFETY:
    unsafe { Iterator1::from_iter_unchecked(iter::once_with(f)) }
}

pub fn head_and_tail<T, I>(head: T, tail: I) -> HeadAndTail<I>
where
    I: IntoIterator<Item = T>,
{
    self::one(head).chain(tail)
}

pub fn tail_and_head<I, T>(tail: I, head: T) -> TailAndHead<I>
where
    I: IntoIterator<Item = T>,
{
    tail.into_iter().chain_non_empty(self::one(head))
}

pub fn repeat<T>(item: T) -> Iterator1<Repeat<T>>
where
    T: Clone,
{
    // SAFETY:
    unsafe { Iterator1::from_iter_unchecked(iter::repeat(item)) }
}

fn empty_or_into<T>(items: Option<T>) -> EmptyOrInto<T>
where
    T: IntoIterator,
{
    items.map(T::into_iter).into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use crate::iter1;

    #[test]
    fn maybe_empty_query() {
        let (has_zero, remainder) = iter1::head_and_tail(0i32, [1]).any(|x| x == 0).into();
        assert!(has_zero);
        assert_eq!(remainder.into_iter().next(), Some(1));

        let x = iter1::head_and_tail(0i32, [1, 2, 3])
            .map(|x| x + 1)
            .find(|&x| x == 3)
            .matched();
        assert_eq!(x, Some(3));

        let x = iter1::head_and_tail(0i32, [1, 2, 3])
            .map(|x| x + 1)
            .find(|&x| x == 3)
            .with_remainder_and_then_output(|remainder| {
                assert_eq!(remainder.count(), 1);
            });
        assert_eq!(x, Some(3));
    }
}

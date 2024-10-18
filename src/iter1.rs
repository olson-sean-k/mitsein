//! Non-empty [iterators][`iter`].
//!
//! [`iter`]: core::iter

// SAFETY: `Iterator1` relies on the behavior of certain iterator types in `core` to maintain its
//         non-empty invariant, especially in unchecked constructions. Checked construction relies
//         on `Peekable`. Similarly, this implementation relies on the behavior of iterator types
//         in `itertools` when integration is enabled.

use core::fmt::Debug;
use core::iter::{
    self, Chain, Cloned, Copied, Cycle, Enumerate, Flatten, Inspect, Map, Peekable, Repeat, Rev,
    Skip, StepBy, Take,
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

use crate::safety::OptionExt as _;
use crate::NonZeroExt as _;
#[cfg(any(feature = "alloc", feature = "arrayvec"))]
use crate::Vacancy;

// The input type parameter `K` is unused in this trait, but is required to prevent a coherence
// error. This trait is implemented for any `Iterator` type `I` and for `iter1::Result<I>`.
// However, `core` may implement `Iterator` for `core::Result` and that presents a conflict. This
// parameter avoids the conflict by implementing a distinct `ThenIterator1` trait for any
// `Iterator` type `I` versus `Result<I>`.
//
// This is unfortunate and, though this trait is not meant to be used in this way, makes bounds on
// the trait and type definitions for its outputs unusual:
//
//   fn max<I, K>(items: I) -> i64
//   where
//       I: ThenIterator1<K, Item = i64>,
//   {
//       items.chain_non_empty([0]).max()
//   }
//
// Note the input type parameter `K`. `ThenIterator1` is an extension trait with a broad
// implementaion over `Iterator` types, so this is very unlikely to be a problem. Moreover,
// associated types are the more "correct" implementation for such an iterator-like trait. In fact,
// using the input type parameter `K` makes this sort of bound even more troublesome, as the
// relationship between the implementor and parameter `K` differs between the two implementations
// (requiring distinct bounds for each)!
pub trait ThenIterator1<K>: Sized {
    type Item;
    type MaybeEmpty: Iterator<Item = Self::Item>;
    type Chained: Iterator<Item = Self::Item>;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::Chained, T::IntoIter>>
    where
        T: IntoIterator1<Item = Self::Item>;

    fn or_non_empty<T>(self, items: T) -> OrNonEmpty<Self::MaybeEmpty, T>
    where
        T: IntoIterator1<Item = Self::Item>,
    {
        self.or_else_non_empty(move || items)
    }

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<Self::MaybeEmpty, T>
    where
        T: IntoIterator1<Item = Self::Item>,
        F: FnOnce() -> T;

    fn or_one<T>(self, item: Self::Item) -> OrNonEmpty<Self::MaybeEmpty, [Self::Item; 1]> {
        self.or_non_empty([item])
    }
}

impl<I> ThenIterator1<I> for I
where
    I: Iterator + Sized,
{
    type Item = I::Item;
    type MaybeEmpty = I;
    type Chained = I;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::Chained, T::IntoIter>>
    where
        T: IntoIterator1<Item = Self::Item>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.chain(items.into_iter1())) }
    }

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<Self::MaybeEmpty, T>
    where
        T: IntoIterator1<Item = Self::Item>,
        F: FnOnce() -> T,
    {
        Iterator1::try_from_iter(self).or_else_non_empty(f)
    }
}

pub trait QueryAnd {
    type Item;
    type Remainder: Iterator<Item = Self::Item>;

    fn nth_and(self, n: usize) -> Matched<Self::Item, Self::Remainder>;

    fn find_and<F>(self, f: F) -> Matched<Self::Item, Self::Remainder>
    where
        F: FnMut(&Self::Item) -> bool;

    fn find_map_and<T, F>(self, f: F) -> Matched<T, Self::Remainder>
    where
        F: FnMut(Self::Item) -> Option<T>;

    fn position_and<F>(self, f: F) -> Matched<usize, Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool;

    fn rposition_and<F>(self, f: F) -> Matched<usize, Self::Remainder>
    where
        Self::Remainder: DoubleEndedIterator + ExactSizeIterator,
        F: FnMut(Self::Item) -> bool;

    fn any_and<F>(self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool;

    fn all_and<F>(self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool;

    #[cfg(feature = "itertools")]
    #[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
    fn contains_and<Q>(self, query: &Q) -> IsMatch<Self::Remainder>
    where
        Self::Item: Borrow<Q>,
        Q: PartialEq;
}

impl<I> QueryAnd for I
where
    I: Iterator,
{
    type Item = I::Item;
    type Remainder = Self;

    fn nth_and(mut self, n: usize) -> Matched<Self::Item, Self::Remainder> {
        let output = self.nth(n);
        Feed(output, self)
    }

    fn find_and<F>(mut self, f: F) -> Matched<Self::Item, Self::Remainder>
    where
        F: FnMut(&Self::Item) -> bool,
    {
        let output = self.find(f);
        Feed(output, self)
    }

    fn find_map_and<T, F>(mut self, f: F) -> Matched<T, Self::Remainder>
    where
        F: FnMut(Self::Item) -> Option<T>,
    {
        let output = self.find_map(f);
        Feed(output, self)
    }

    fn position_and<F>(mut self, f: F) -> Matched<usize, Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        let output = self.position(f);
        Feed(output, self)
    }

    fn rposition_and<F>(mut self, f: F) -> Matched<usize, Self::Remainder>
    where
        Self::Remainder: DoubleEndedIterator + ExactSizeIterator,
        F: FnMut(Self::Item) -> bool,
    {
        let output = self.rposition(f);
        Feed(output, self)
    }

    fn any_and<F>(mut self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        let output = self.any(f);
        Feed(output, self)
    }

    fn all_and<F>(mut self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        let output = self.all(f);
        Feed(output, self)
    }

    #[cfg(feature = "itertools")]
    #[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
    fn contains_and<Q>(mut self, query: &Q) -> IsMatch<Self::Remainder>
    where
        Self::Item: Borrow<Q>,
        Q: PartialEq,
    {
        let output = self.contains(query);
        Feed(output, self)
    }
}

pub trait IteratorExt: Iterator + Sized {
    fn try_into_iter1(self) -> Result<Self> {
        Iterator1::try_from_iter(self)
    }

    fn try_collect1<T>(self) -> result::Result<T, Peekable<Self>>
    where
        T: FromIterator1<<Self as Iterator>::Item>,
    {
        T::try_from_iter(self)
    }

    fn saturate<T>(self) -> Feed<T, Self>
    where
        T: FromIteratorUntil<Self>,
    {
        T::saturated(self)
    }
}

impl<I> IteratorExt for I where I: Iterator {}

pub trait ExtendUntil<I>
where
    I: IntoIterator,
{
    fn saturate(&mut self, items: I) -> I::IntoIter;
}

pub trait FromIteratorUntil<I>: Sized
where
    I: IntoIterator,
{
    fn saturated(items: I) -> Feed<Self, I::IntoIter>;
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

impl<T, U> FromIterator1<Option<T>> for Option<U>
where
    U: FromIterator1<T>,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = Option<T>>,
    {
        let mut items = items.into_iter1().into_iter();
        let mut unit = None;
        let residual = &mut unit;

        let items = iter::from_fn(move || {
            items.next().and_then(|option| {
                if option.is_none() {
                    *residual = Some(());
                }
                option
            })
        })
        .fuse();
        // Because `U::try_from_iter` can be implemented arbitrarily, the `Iterator1` is
        // constructed more directly here since its constructor has a known implementation and
        // behavior.
        match Iterator1::try_from_iter(items).map(U::from_iter1) {
            Ok(value) => match unit {
                Some(_) => None,
                _ => Some(value),
            },
            // `Iterator1::try_from_iter` only returns an error if the `items` iterator encounters
            // `None`.
            _ => None,
        }
    }
}

impl<T, U, E> FromIterator1<result::Result<T, E>> for result::Result<U, E>
where
    U: FromIterator1<T>,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = result::Result<T, E>>,
    {
        let mut items = items.into_iter1().into_iter();
        let mut error = None;
        let residual = &mut error;

        let items = iter::from_fn(move || {
            items.next().and_then(|result| match result {
                Ok(value) => Some(value),
                Err(error) => {
                    *residual = Some(error);
                    None
                },
            })
        })
        .fuse();
        // Because `U::try_from_iter` can be implemented arbitrarily, the `Iterator1` is
        // constructed more directly here since its constructor has a known implementation and
        // behavior. See below.
        match Iterator1::try_from_iter(items).map(U::from_iter1) {
            Ok(value) => match error {
                Some(error) => Err(error),
                _ => Ok(value),
            },
            // SAFETY: The `Iterator1::try_from_iter` function never returns an error unless the
            //         `items` iterator encounters an error (at the first item) and so `error` must
            //         be `Some`. See the use of `iter::from_fn` above.
            Err(_) => Err(unsafe { error.unwrap_maybe_unchecked() }),
        }
    }
}

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

pub type OrElseNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

pub type Result<I> = result::Result<Iterator1<Peekable<I>>, Peekable<I>>;

impl<I> ThenIterator1<I> for Result<I>
where
    I: Iterator + Sized,
{
    type Item = I::Item;
    type MaybeEmpty = I;
    type Chained = Peekable<I>;

    fn chain_non_empty<T>(self, items: T) -> Iterator1<Chain<Self::Chained, T::IntoIter>>
    where
        T: IntoIterator1<Item = Self::Item>,
    {
        // SAFETY: `items` is non-empty and is chained with the input iterator.
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

    fn or_else_non_empty<T, F>(self, f: F) -> OrElseNonEmpty<Self::MaybeEmpty, T>
    where
        T: IntoIterator1<Item = Self::Item>,
        F: FnOnce() -> T,
    {
        // SAFETY: Both `T` and `self` (when `Ok`) are non-empty.
        unsafe {
            Iterator1::from_iter_unchecked(match self {
                Ok(items) => items.into_iter().chain(empty_or_into::<T>(None)),
                Err(empty) => empty.chain(empty_or_into(Some(f()))),
            })
        }
    }
}

#[derive(Clone, Debug)]
pub struct Feed<T, I>(pub T, pub I);

impl<T, I> Feed<T, I> {
    pub fn output(self) -> T {
        self.0
    }

    pub fn remainder(self) -> I {
        self.1
    }

    pub fn with_output_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(T),
    {
        let Feed(output, remainder) = self;
        f(output);
        remainder
    }

    pub fn with_remainder_and_then_output<F>(self, f: F) -> T
    where
        F: FnOnce(I),
    {
        let Feed(output, remainder) = self;
        f(remainder);
        output
    }
}

impl<T, I> From<Feed<T, I>> for (T, I) {
    fn from(feed: Feed<T, I>) -> Self {
        let Feed(output, remainder) = feed;
        (output, remainder)
    }
}

pub type Matched<T, I> = Feed<Option<T>, I>;

impl<T, I> Matched<T, I> {
    pub fn matched(self) -> Option<T> {
        Into::into(self)
    }

    pub fn some_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(T),
    {
        let Feed(output, remainder) = self;
        if let Some(output) = output {
            f(output);
        }
        remainder
    }

    pub fn none_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let Feed(output, remainder) = self;
        if output.is_none() {
            f();
        }
        remainder
    }
}

impl<T, I> From<Matched<T, I>> for Option<T> {
    fn from(feed: Matched<T, I>) -> Self {
        feed.0
    }
}

pub type IsMatch<I> = Feed<bool, I>;

impl<I> IsMatch<I> {
    pub fn is_match(&self) -> bool {
        self.0
    }

    pub fn if_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let Feed(output, remainder) = self;
        if output {
            f();
        }
        remainder
    }

    pub fn if_not_and_then_remainder<F>(self, f: F) -> I
    where
        F: FnOnce(),
    {
        let Feed(output, remainder) = self;
        if !output {
            f();
        }
        remainder
    }
}

impl<I> From<IsMatch<I>> for bool {
    fn from(feed: IsMatch<I>) -> Self {
        feed.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Iterator1<I> {
    items: I,
}

impl<I> Iterator1<I> {
    /// # Safety
    ///
    /// `items` must yield one or more items from its [`IntoIterator`] implementation. For example,
    /// it is unsound to call this function with [`None::<I>`][`Option::None`] or any other empty
    /// [`IntoIterator`].
    ///
    /// [`IntoIterator`]: core::iter::IntoIterator
    /// [`Option::None`]: core::option::Option::None
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
            // SAFETY: `items` is non-empty.
            Some(_) => Ok(unsafe { Iterator1::from_iter_unchecked(items) }),
            _ => Err(items),
        }
    }

    /// # Safety
    ///
    /// The combinator function `f` must not reduce the cardinality of the iterator to zero. For
    /// example, it is unsound to call this function with [`Iterator::skip`] for an arbitrary or
    /// unchecked number of skipped items, because this could skip all items and produce an empty
    /// iterator.
    ///
    /// [`Iterator::skip`]: core::iter::Iterator::skip
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
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.next().unwrap_maybe_unchecked() }
    }

    pub fn last(self) -> I::Item {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn into_head_and_tail(mut self) -> (I::Item, I) {
        // SAFETY: `self` must be non-empty.
        let head = unsafe { self.items.next().unwrap_maybe_unchecked() };
        (head, self.items)
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.min().unwrap_maybe_unchecked() }
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.max().unwrap_maybe_unchecked() }
    }

    pub fn reduce<F>(self, f: F) -> I::Item
    where
        F: FnMut(I::Item, I::Item) -> I::Item,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.reduce(f).unwrap_maybe_unchecked() }
    }

    pub fn eq<J>(self, other: J) -> bool
    where
        J: IntoIterator,
        I::Item: PartialEq<J::Item>,
    {
        self.into_iter().eq(other)
    }

    pub fn copied<'a, T>(self) -> Iterator1<Copied<I>>
    where
        T: 'a + Copy,
        I: Iterator<Item = &'a T>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::copied) }
    }

    pub fn cloned<'a, T>(self) -> Iterator1<Cloned<I>>
    where
        T: 'a + Clone,
        I: Iterator<Item = &'a T>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::cloned) }
    }

    pub fn enumerate(self) -> Iterator1<Enumerate<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::enumerate) }
    }

    pub fn map<T, F>(self, f: F) -> Iterator1<Map<I, F>>
    where
        F: FnMut(I::Item) -> T,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(move |items| items.map(f)) }
    }

    pub fn map_first_and_then<J, H, T>(mut self, head: H, tail: T) -> HeadAndTail<J>
    where
        J: Iterator,
        H: FnOnce(I::Item) -> J::Item,
        T: FnOnce(I) -> J,
    {
        // SAFETY: `self` must be non-empty so that `next` yields `Some` here.
        let first = unsafe { self.items.next().map(head).unwrap_maybe_unchecked() };
        self::head_and_tail(first, tail(self.items))
    }

    pub fn first_and_then<J, F>(self, f: F) -> HeadAndTail<J>
    where
        J: Iterator<Item = I::Item>,
        F: FnOnce(I) -> J,
    {
        self.map_first_and_then(|first| first, f)
    }

    pub fn first_and_then_take(self, n: usize) -> HeadAndTail<Take<I>> {
        self.first_and_then(|items| items.take(n))
    }

    pub fn first_and_then_skip(self, n: usize) -> HeadAndTail<Skip<I>> {
        self.first_and_then(|items| items.skip(n))
    }

    pub fn cycle(self) -> Iterator1<Cycle<I>>
    where
        I: Clone,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::cycle) }
    }

    pub fn chain<T>(self, chained: T) -> Iterator1<Chain<I, T::IntoIter>>
    where
        T: IntoIterator<Item = I::Item>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(move |items| items.chain(chained)) }
    }

    pub fn step_by(self, step: usize) -> Iterator1<StepBy<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(move |items| items.step_by(step)) }
    }

    pub fn inspect<F>(self, f: F) -> Iterator1<Inspect<I, F>>
    where
        F: FnMut(&I::Item),
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(move |items| items.inspect(f)) }
    }

    pub fn peekable(self) -> Iterator1<Peekable<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::peekable) }
    }

    pub fn rev(self) -> Iterator1<Rev<I>>
    where
        I: DoubleEndedIterator,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
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

    pub fn saturate<T>(self) -> Feed<T, I>
    where
        T: FromIteratorUntil<Self>,
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
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::map_into) }
    }

    pub fn map_ok<F, T, U, E>(self, f: F) -> Iterator1<MapOk<I, F>>
    where
        I: Iterator<Item = result::Result<T, E>>,
        F: FnMut(T) -> U,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(move |items| items.map_ok(f)) }
    }

    pub fn with_position(self) -> Iterator1<WithPosition<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::with_position) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn multipeek(self) -> Iterator1<MultiPeek<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.non_empty(I::multipeek) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted(self) -> Iterator1<vec::IntoIter<I::Item>>
    where
        I::Item: Ord,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
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
        // SAFETY: `self` must be non-empty.
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

impl<I> QueryAnd for Iterator1<I>
where
    I: Iterator,
{
    type Item = I::Item;
    type Remainder = I;

    fn nth_and(self, n: usize) -> Matched<Self::Item, Self::Remainder> {
        self.items.nth_and(n)
    }

    fn find_and<F>(self, f: F) -> Matched<Self::Item, Self::Remainder>
    where
        F: FnMut(&Self::Item) -> bool,
    {
        self.items.find_and(f)
    }

    fn find_map_and<T, F>(self, f: F) -> Matched<T, Self::Remainder>
    where
        F: FnMut(Self::Item) -> Option<T>,
    {
        self.items.find_map_and(f)
    }

    fn position_and<F>(self, f: F) -> Matched<usize, Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.items.position_and(f)
    }

    fn rposition_and<F>(self, f: F) -> Matched<usize, Self::Remainder>
    where
        Self::Remainder: DoubleEndedIterator + ExactSizeIterator,
        F: FnMut(Self::Item) -> bool,
    {
        self.items.rposition_and(f)
    }

    fn any_and<F>(self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.items.any_and(f)
    }

    fn all_and<F>(self, f: F) -> IsMatch<Self::Remainder>
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.items.all_and(f)
    }

    #[cfg(feature = "itertools")]
    #[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
    fn contains_and<Q>(self, query: &Q) -> IsMatch<Self::Remainder>
    where
        Self::Item: Borrow<Q>,
        Q: PartialEq,
    {
        self.items.contains_and(query)
    }
}

pub fn one<T>(item: T) -> ExactlyOne<T> {
    // SAFETY: `Some(item)` is non-empty.
    unsafe { Iterator1::from_iter_unchecked(Some(item)) }
}

pub fn one_with<T, F>(f: F) -> ExactlyOneWith<F>
where
    F: FnOnce() -> T,
{
    // SAFETY: The output `OnceWith` is non-empty.
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
    // SAFETY: The output `Repeat` is non-empty.
    unsafe { Iterator1::from_iter_unchecked(iter::repeat(item)) }
}

#[cfg(any(feature = "alloc", feature = "arrayvec"))]
pub(crate) fn saturate_positional_vacancy<T, I>(destination: &mut T, source: I) -> I::IntoIter
where
    T: Extend<I::Item> + Vacancy,
    I: IntoIterator,
{
    let n = destination.vacancy();
    let mut source = source.into_iter();
    destination.extend(source.by_ref().take(n));
    source
}

fn empty_or_into<T>(items: Option<T>) -> EmptyOrInto<T>
where
    T: IntoIterator,
{
    items.map(T::into_iter).into_iter().flatten()
}

#[cfg(test)]
pub mod harness {
    use core::fmt::Debug;
    use core::iter::Peekable;
    use core::ops::RangeInclusive;
    use rstest::fixture;

    use crate::iter1::{Feed, Iterator1, Matched};

    pub type Xs1 = Iterator1<Peekable<RangeInclusive<u8>>>;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> Xs1 {
        Iterator1::try_from_iter(0..=end).unwrap_or_else(|_| panic!("range `0..={}` is empty", end))
    }

    #[fixture]
    pub fn matched_some_non_empty() -> Matched<u8, impl Iterator<Item = u8>> {
        Feed(Some(0), [1, 2, 3].into_iter())
    }

    #[fixture]
    pub fn matched_none_non_empty() -> Matched<u8, impl Iterator<Item = u8>> {
        Feed(None, [2, 3].into_iter())
    }

    pub fn assert_feed_eq<T, U>(
        lhs: Feed<T, impl IntoIterator<Item = U>>,
        rhs: Feed<T, impl IntoIterator<Item = U>>,
    ) where
        T: Debug + PartialEq<T>,
        U: PartialEq<U>,
    {
        assert_eq!(lhs.0, rhs.0);
        assert!(lhs.1.into_iter().eq(rhs.1.into_iter()));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::iter1::harness::{self, matched_none_non_empty, matched_some_non_empty, xs1, Xs1};
    use crate::iter1::{Feed, IntoIterator1, IsMatch, Matched, QueryAnd, ThenIterator1};
    use crate::slice1::{slice1, Slice1};
    #[cfg(feature = "alloc")]
    use crate::vec1::Vec1;

    #[cfg(feature = "alloc")]
    #[rstest]
    #[case::some([Some(0), Some(1), Some(2)], Some(Vec1::from([0, 1, 2])))]
    #[case::none_at_first([None, Some(1), Some(2)], None)]
    #[case::none_at_last([Some(0), Some(1), None], None)]
    fn collect1_options_into_option_of_vec1_then_eq(
        #[case] xs: impl IntoIterator1<Item = Option<u8>>,
        #[case] expected: Option<Vec1<u8>>,
    ) {
        let xs: Option<Vec1<_>> = xs.into_iter1().collect1();
        assert_eq!(xs, expected);
    }

    #[cfg(feature = "alloc")]
    #[rstest]
    #[case::ok([Ok(0), Ok(1), Ok(2)], Ok(Vec1::from([0, 1, 2])))]
    #[case::error_at_first([Err(()), Ok(1), Ok(2)], Err(()))]
    #[case::error_at_last([Ok(0), Ok(1), Err(())], Err(()))]
    fn collect1_results_into_result_of_vec1_then_eq(
        #[case] xs: impl IntoIterator1<Item = Result<u8, ()>>,
        #[case] expected: Result<Vec1<u8>, ()>,
    ) {
        let xs: Result<Vec1<_>, _> = xs.into_iter1().collect1();
        assert_eq!(xs, expected);
    }

    #[rstest]
    #[case::non_empty_iter([0], [255], slice1![0])]
    #[case::empty_iter([], [255], slice1![255])]
    fn iter_or_non_empty_then_eq(
        #[case] xs: impl IntoIterator<Item = u8>,
        #[case] xs1: impl IntoIterator1<Item = u8>,
        #[case] expected: &Slice1<u8>,
    ) {
        assert!(xs
            .into_iter()
            .or_non_empty(xs1)
            .eq(expected.iter1().copied()));
    }

    #[rstest]
    #[should_panic]
    fn some_and_then_remainder_with_some_matched_then_call(
        matched_some_non_empty: Matched<u8, impl Iterator<Item = u8>>,
    ) {
        let _remainder = matched_some_non_empty.some_and_then_remainder(|_| panic!());
    }

    #[rstest]
    fn some_and_then_remainder_with_none_matched_then_no_call(
        matched_none_non_empty: Matched<u8, impl Iterator<Item = u8>>,
    ) {
        let _remainder = matched_none_non_empty.some_and_then_remainder(|_| panic!());
    }

    #[rstest]
    #[case::first(0, Feed(true, [1, 2, 3, 4]))]
    #[case::last(4, Feed(true, []))]
    #[case::none(5, Feed(false, []))]
    fn any_and_with_iter1_then_is_match_eq(
        xs1: Xs1,
        #[case] any: u8,
        #[case] expected: IsMatch<impl IntoIterator<Item = u8>>,
    ) {
        harness::assert_feed_eq(xs1.any_and(|x| x == any), expected);
    }

    #[rstest]
    #[case::first(0, Feed(false, [2, 3, 4]))]
    #[case::last(3, Feed(false, [1, 2, 3, 4]))]
    #[case::none(4, Feed(false, [1, 2, 3, 4]))]
    fn all_and_with_iter1_then_is_match_eq(
        xs1: Xs1,
        #[case] all: u8,
        #[case] expected: IsMatch<impl IntoIterator<Item = u8>>,
    ) {
        harness::assert_feed_eq(xs1.all_and(|x| x == all), expected);
    }

    #[rstest]
    #[case::first(0, Feed(Some(0), [1, 2, 3, 4]))]
    #[case::last(4, Feed(Some(4), []))]
    #[case::none(5, Feed(None, []))]
    fn find_and_with_iter1_then_matched_eq(
        xs1: Xs1,
        #[case] find: u8,
        #[case] expected: Matched<u8, impl IntoIterator<Item = u8>>,
    ) {
        harness::assert_feed_eq(xs1.find_and(|&x| x == find), expected);
    }
}

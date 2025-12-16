//! Non-empty [iterators][`iter`].

// SAFETY: `Iterator1` relies on the behavior of certain iterator types in `core` to maintain its
//         non-empty invariant, especially in unchecked constructions. Checked construction relies
//         on `Peekable`. Similarly, this implementation relies on the behavior of iterator types
//         in `itertools` when integration is enabled.

use core::cmp::Ordering;
use core::fmt::Debug;
use core::iter::{
    self, Chain, Cloned, Copied, Cycle, Enumerate, FlatMap, Flatten, Inspect, Map, Peekable,
    Repeat, Rev, Skip, StepBy, Take, Zip,
};
use core::num::NonZeroUsize;
use core::option;
use core::result;
#[cfg(feature = "either")]
use either::Either;
#[cfg(feature = "itertools")]
use itertools::{
    Dedup, DedupBy, DedupByWithCount, DedupWithCount, Itertools, MapInto, MapOk, Merge, MergeBy,
    MinMaxResult, PadUsing, Update, WithPosition, ZipLongest,
};
#[cfg(all(feature = "std", feature = "itertools"))]
use itertools::{Unique, UniqueBy};
#[cfg(feature = "rayon")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator,
};
#[cfg(all(feature = "alloc", feature = "itertools"))]
use {
    alloc::vec,
    core::hash::Hash,
    itertools::{MultiPeek, Powerset, Tee},
};

use crate::safety::OptionExt as _;
#[cfg(feature = "rayon")]
use crate::vec1::Vec1;
#[cfg(feature = "itertools")]
use crate::{Cardinality, safety};
use crate::{EmptyError, FromMaybeEmpty, MaybeEmpty, NonEmpty, NonZeroExt as _};

// Ideally, `Either` would implement `IntoIterator1`, but cannot because of its direct `Iterator`
// implementation. In particular, the `IntoIterator::IntoIter` type for `Either` is itself, and so
// it is impossible to output an appropriate `Iterator1` type in an `IntoIterator1` implementation.
#[cfg(feature = "either")]
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
pub trait EitherExt<L, R> {
    fn into_iter1(self) -> LeftOrRight<L, R>
    where
        L: Iterator,
        R: Iterator<Item = L::Item>;
}

#[cfg(feature = "either")]
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
impl<L, R> EitherExt<L, R> for Either<Iterator1<L>, Iterator1<R>>
where
    L: Iterator,
    R: Iterator<Item = L::Item>,
{
    fn into_iter1(self) -> LeftOrRight<L, R>
    where
        L: Iterator,
        R: Iterator<Item = L::Item>,
    {
        let items = match self {
            Either::Left(items) => self::empty_or_into::<L>(Some(items.into_iter()))
                .chain(self::empty_or_into::<R>(None)),
            Either::Right(items) => self::empty_or_into::<L>(None)
                .chain(self::empty_or_into::<R>(Some(items.into_iter()))),
        };
        // SAFETY: Both the left and right values are non-empty iterators, so one of the iterators
        //         in the chain in `items` is non-empty and therefore `items` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(items) }
    }
}

pub trait Extend1<T> {
    #[must_use]
    fn extend_non_empty<I>(self, items: I) -> NonEmpty<Self>
    where
        I: IntoIterator1<Item = T>;
}

pub trait FromIterator1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>;

    fn try_from_iter<I>(items: I) -> result::Result<Self, EmptyError<Peekable<I::IntoIter>>>
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
//       See https://github.com/rust-lang/rust/issues/48869
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

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub trait FromParallelIterator1<T> {
    fn from_par_iter1<I>(items: I) -> Self
    where
        I: IntoParallelIterator1<Item = T>;
}

pub trait IntoIterator1: IntoIterator {
    fn into_iter1(self) -> Iterator1<Self::IntoIter>;
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub trait IntoParallelIterator1: IntoParallelIterator {
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter>;
}

pub trait IteratorExt: Iterator + Sized {
    fn try_into_iter1(self) -> Result<Self> {
        Iterator1::try_from_iter(self)
    }

    fn try_collect1<T>(self) -> result::Result<T, EmptyError<Peekable<Self>>>
    where
        T: FromIterator1<<Self as Iterator>::Item>,
    {
        T::try_from_iter(self)
    }
}

impl<I> IteratorExt for I where I: Iterator {}

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

pub type AtMostOne<T> = option::IntoIter<T>;

pub type ExactlyOne<T> = Iterator1<AtMostOne<T>>;

pub type AtMostOneWith<F> = iter::OnceWith<F>;

pub type ExactlyOneWith<F> = Iterator1<AtMostOneWith<F>>;

pub type HeadAndTail<T> =
    Iterator1<Chain<AtMostOne<<T as IntoIterator>::Item>, <T as IntoIterator>::IntoIter>>;

pub type RTailAndHead<T> =
    Iterator1<Chain<<T as IntoIterator>::IntoIter, AtMostOne<<T as IntoIterator>::Item>>>;

pub type EmptyOrInto<T> = Flatten<AtMostOne<<T as IntoIterator>::IntoIter>>;

pub type OrNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

pub type OrElseNonEmpty<I, T> = Iterator1<Chain<Peekable<I>, EmptyOrInto<T>>>;

#[cfg(feature = "either")]
#[cfg_attr(docsrs, doc(cfg(feature = "either")))]
pub type LeftOrRight<L, R> = Iterator1<Chain<EmptyOrInto<L>, EmptyOrInto<R>>>;

pub type Result<I> = result::Result<Iterator1<Peekable<I>>, EmptyError<Peekable<I>>>;

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
                    Err(error) => error.into_empty(),
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
                Ok(items) => items.into_iter().chain(self::empty_or_into::<T>(None)),
                Err(error) => error.into_empty().chain(self::empty_or_into(Some(f()))),
            })
        }
    }
}

#[cfg(feature = "itertools")]
#[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
pub type MinMax<T> = Cardinality<T, (T, T)>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Iterator1<I> {
    items: I,
}

impl<I> Iterator1<I> {
    /// # Safety
    ///
    /// `items` must yield one or more items from its [`IntoIterator`] implementation. For example,
    /// it is undefined behavior to call this function with [`None::<I>`][`Option::None`] or any
    /// other empty [`IntoIterator`].
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

    pub fn and_then_try<J, F>(self, f: F) -> Result<J>
    where
        J: Iterator,
        F: FnOnce(I) -> J,
    {
        Iterator1::try_from_iter(f(self.items))
    }

    /// Maps the inner [`Iterator`] with the given function without checking that the output is
    /// non-empty.
    ///
    /// # Safety
    ///
    /// The iterator returned by the function `f` must yield one or more items (must never be
    /// empty). For example, calling this function with a closure
    /// [`|_| iter::empty::<I>()`][iter::empty] is undefined behavior.
    ///
    /// [`iter::empty`]: core::iter::empty
    pub unsafe fn and_then_unchecked<J, F>(self, f: F) -> Iterator1<J>
    where
        J: Iterator,
        F: FnOnce(I) -> J,
    {
        unsafe { Iterator1::from_iter_unchecked(f(self.items)) }
    }

    pub const fn as_iter(&self) -> &I {
        &self.items
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
            _ => Err(EmptyError::from_empty(items)),
        }
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

    pub fn count(self) -> NonZeroUsize {
        // Though the count must be non-zero here, it may overflow to zero.
        NonZeroUsize::new(self.items.count())
            .expect("non-empty iterator has zero items or overflow in count")
    }

    pub fn eq<J>(self, other: J) -> bool
    where
        J: IntoIterator,
        I::Item: PartialEq<J::Item>,
    {
        self.into_iter().eq(other)
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

    pub fn into_rtail_and_head(mut self) -> (I, I::Item)
    where
        I: DoubleEndedIterator,
    {
        // SAFETY: `self` must be non-empty.
        let head = unsafe { self.items.next_back().unwrap_maybe_unchecked() };
        (self.items, head)
    }

    pub fn min_by<F>(self, f: F) -> I::Item
    where
        F: FnMut(&I::Item, &I::Item) -> Ordering,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.min_by(f).unwrap_maybe_unchecked() }
    }

    pub fn min_by_key<B, F>(self, mut f: F) -> I::Item
    where
        B: Ord,
        F: FnMut(&I::Item) -> B,
    {
        self.min_by(move |lhs, rhs| f(lhs).cmp(&f(rhs)))
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        self.min_by(Ord::cmp)
    }

    pub fn max_by<F>(self, f: F) -> I::Item
    where
        F: FnMut(&I::Item, &I::Item) -> Ordering,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.max_by(f).unwrap_maybe_unchecked() }
    }

    pub fn max_by_key<B, F>(self, mut f: F) -> I::Item
    where
        B: Ord,
        F: FnMut(&I::Item) -> B,
    {
        self.max_by(move |lhs, rhs| f(lhs).cmp(&f(rhs)))
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        self.max_by(Ord::cmp)
    }

    pub fn reduce<F>(self, f: F) -> I::Item
    where
        F: FnMut(I::Item, I::Item) -> I::Item,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.reduce(f).unwrap_maybe_unchecked() }
    }

    pub fn copied<'a, T>(self) -> Iterator1<Copied<I>>
    where
        T: 'a + Copy,
        I: Iterator<Item = &'a T>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::copied) }
    }

    pub fn cloned<'a, T>(self) -> Iterator1<Cloned<I>>
    where
        T: 'a + Clone,
        I: Iterator<Item = &'a T>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::cloned) }
    }

    pub fn enumerate(self) -> Iterator1<Enumerate<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::enumerate) }
    }

    pub fn map<T, F>(self, f: F) -> Iterator1<Map<I, F>>
    where
        F: FnMut(I::Item) -> T,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.map(f)) }
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
        unsafe { self.and_then_unchecked(I::cycle) }
    }

    pub fn chain<T>(self, chained: T) -> Iterator1<Chain<I, T::IntoIter>>
    where
        T: IntoIterator<Item = I::Item>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.chain(chained)) }
    }

    pub fn zip<T>(self, zipped: T) -> Iterator1<Zip<I, T::IntoIter>>
    where
        T: IntoIterator1<Item = I::Item>,
    {
        // SAFETY: Both input iterators are non-empty, and so this combinator function cannot
        //         reduce the cardinality to zero.
        unsafe { self.and_then_unchecked(move |items| items.zip(zipped)) }
    }

    pub fn step_by(self, step: usize) -> Iterator1<StepBy<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.step_by(step)) }
    }

    pub fn inspect<F>(self, f: F) -> Iterator1<Inspect<I, F>>
    where
        F: FnMut(&I::Item),
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.inspect(f)) }
    }

    pub fn peekable(self) -> Iterator1<Peekable<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::peekable) }
    }

    pub fn rev(self) -> Iterator1<Rev<I>>
    where
        I: DoubleEndedIterator,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::rev) }
    }

    pub fn flatten(self) -> Iterator1<Flatten<I>>
    where
        I::Item: IntoIterator1,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::flatten) }
    }

    pub fn flat_map<U, F>(self, f: F) -> Iterator1<FlatMap<I, U, F>>
    where
        F: FnMut(I::Item) -> U,
        U: IntoIterator1,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(|items| items.flat_map(f)) }
    }

    pub fn collect1<T>(self) -> T
    where
        T: FromIterator1<I::Item>,
    {
        T::from_iter1(self)
    }

    // `Iterator::unzip` relies on tuple implementations for `Extend`, but this isn't possible for
    // `Extend1`, since it is a morphism from `Self` to `NonEmpty<Self>` (it consumes `self` and
    // cannot produce a tuple of `NonEmpty` types as its output). `Itertor1::unzip` therefore
    // relies on these same tuple implementations of `Extend` and then constructs the `NonEmpty`
    // outputs.
    pub fn unzip<L, R, FromL, FromR>(self) -> (NonEmpty<FromL>, NonEmpty<FromR>)
    where
        I: Iterator<Item = (L, R)>,
        FromL: Default + Extend<L> + MaybeEmpty,
        FromR: Default + Extend<R> + MaybeEmpty,
    {
        let mut unzipped: (FromL, FromR) = Default::default();
        unzipped.extend(self);
        let (left, right) = unzipped;
        // SAFETY: `self` is non-empty and, given that, the `Extend` implementation over tuples
        //         produces two non-empty collections.
        unsafe {
            (
                NonEmpty::from_maybe_empty_unchecked(left),
                NonEmpty::from_maybe_empty_unchecked(right),
            )
        }
    }
}

#[cfg(feature = "itertools")]
#[cfg_attr(docsrs, doc(cfg(feature = "itertools")))]
impl<I> Iterator1<I>
where
    I: Iterator,
{
    pub fn minmax(self) -> MinMax<I::Item>
    where
        I::Item: PartialOrd,
    {
        match self.into_iter().minmax() {
            // SAFETY: `self` must be non-empty.
            MinMaxResult::NoElements => unsafe { safety::unreachable_maybe_unchecked() },
            MinMaxResult::OneElement(item) => MinMax::One(item),
            MinMaxResult::MinMax(min, max) => MinMax::Many((min, max)),
        }
    }

    pub fn map_into<T>(self) -> Iterator1<MapInto<I, T>>
    where
        I::Item: Into<T>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::map_into) }
    }

    pub fn map_ok<F, T, U, E>(self, f: F) -> Iterator1<MapOk<I, F>>
    where
        I: Iterator<Item = result::Result<T, E>>,
        F: FnMut(T) -> U,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.map_ok(f)) }
    }

    pub fn update<F>(self, f: F) -> Iterator1<Update<I, F>>
    where
        F: FnMut(&mut I::Item),
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.update(f)) }
    }

    pub fn dedup(self) -> Iterator1<Dedup<I>>
    where
        I::Item: PartialOrd,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::dedup) }
    }

    pub fn dedup_with_count(self) -> Iterator1<DedupWithCount<I>>
    where
        I::Item: PartialOrd,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::dedup_with_count) }
    }

    pub fn dedup_by<F>(self, f: F) -> Iterator1<DedupBy<I, F>>
    where
        F: FnMut(&I::Item, &I::Item) -> bool,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.dedup_by(f)) }
    }

    pub fn dedup_by_with_count<F>(self, f: F) -> Iterator1<DedupByWithCount<I, F>>
    where
        F: FnMut(&I::Item, &I::Item) -> bool,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.dedup_by_with_count(f)) }
    }

    pub fn with_position(self) -> Iterator1<WithPosition<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::with_position) }
    }

    pub fn first_and_then_pad_with<F>(self, min: usize, f: F) -> HeadAndTail<PadUsing<I, F>>
    where
        F: FnMut(usize) -> I::Item,
    {
        self.first_and_then(move |items| items.pad_using(min, f))
    }

    pub fn merge<T>(self, merged: T) -> Iterator1<Merge<I, T::IntoIter>>
    where
        I::Item: PartialOrd,
        T: IntoIterator<Item = I::Item>,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.merge(merged)) }
    }

    pub fn merge_by<T, F>(self, merged: T, f: F) -> Iterator1<MergeBy<I, T::IntoIter, F>>
    where
        T: IntoIterator<Item = I::Item>,
        F: FnMut(&I::Item, &I::Item) -> bool,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.merge_by(merged, f)) }
    }

    pub fn zip_longest<T>(self, zipped: T) -> Iterator1<ZipLongest<I, T::IntoIter>>
    where
        T: IntoIterator,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.zip_longest(zipped)) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn tee(self) -> (Iterator1<Tee<I>>, Iterator1<Tee<I>>)
    where
        I::Item: Clone,
    {
        let (xs, ys) = self.items.tee();
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe {
            (
                Iterator1::from_iter_unchecked(xs),
                Iterator1::from_iter_unchecked(ys),
            )
        }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn multipeek(self) -> Iterator1<MultiPeek<I>> {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::multipeek) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted(self) -> Iterator1<vec::IntoIter<I::Item>>
    where
        I::Item: Ord,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::sorted) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted_by<F>(self, f: F) -> Iterator1<vec::IntoIter<I::Item>>
    where
        F: FnMut(&I::Item, &I::Item) -> Ordering,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.sorted_by(f)) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted_by_key<K, F>(self, f: F) -> Iterator1<vec::IntoIter<I::Item>>
    where
        K: Ord,
        F: FnMut(&I::Item) -> K,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.sorted_by_key(f)) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn sorted_by_cached_key<K, F>(self, f: F) -> Iterator1<vec::IntoIter<I::Item>>
    where
        K: Ord,
        F: FnMut(&I::Item) -> K,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.sorted_by_cached_key(f)) }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn powerset(self) -> Iterator1<Powerset<I>>
    where
        I::Item: Clone,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::powerset) }
    }

    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn unique(self) -> Iterator1<Unique<I>>
    where
        I::Item: Clone + Eq + Hash,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(I::unique) }
    }

    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn unique_by<K, F>(self, f: F) -> Iterator1<UniqueBy<I, K, F>>
    where
        K: Eq + Hash,
        F: FnMut(&I::Item) -> K,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.unique_by(f)) }
    }

    pub fn find_or_first<F>(self, f: F) -> I::Item
    where
        F: FnMut(&I::Item) -> bool,
    {
        let item = self.items.find_or_first(f);
        // SAFETY: `self` must be non-empty.
        unsafe { item.unwrap_maybe_unchecked() }
    }

    pub fn find_or_last<F>(self, f: F) -> I::Item
    where
        F: FnMut(&I::Item) -> bool,
    {
        let item = self.items.find_or_last(f);
        // SAFETY: `self` must be non-empty.
        unsafe { item.unwrap_maybe_unchecked() }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<I> Iterator1<I>
where
    I: Iterator + ParallelBridge + Send,
    I::Item: Send,
{
    pub fn par_bridge(self) -> ParallelIterator1<rayon::iter::IterBridge<I>> {
        // SAFETY: The input iterator is non-empty and the conversion into a parallel iterator
        //         cannot reduce the cardinality of the iterator to zero.
        unsafe { ParallelIterator1::from_par_iter_unchecked(self.items.par_bridge()) }
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

impl<I> IntoIterator1 for Iterator1<I>
where
    I: Iterator,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ParallelIterator1<I> {
    items: I,
}

#[cfg(feature = "rayon")]
impl<I> ParallelIterator1<I> {
    /// # Safety
    ///
    /// `items` must yield one or more items from its [`IntoParallelIterator`] implementation.
    pub unsafe fn from_par_iter_unchecked<T>(items: T) -> Self
    where
        T: IntoParallelIterator<Iter = I>,
    {
        ParallelIterator1 {
            items: items.into_par_iter(),
        }
    }

    /// Maps the inner [`ParallelIterator`] with the given function without checking that the
    /// output is non-empty.
    ///
    /// # Safety
    ///
    /// The parallel iterator returned by the function `f` must yield one or more items (must never
    /// be empty).
    pub unsafe fn and_then_unchecked<J, F>(self, f: F) -> ParallelIterator1<J>
    where
        J: ParallelIterator,
        F: FnOnce(I) -> J,
    {
        unsafe { ParallelIterator1::from_par_iter_unchecked(f(self.items)) }
    }

    pub const fn as_par_iter(&self) -> &I {
        &self.items
    }
}

#[cfg(feature = "rayon")]
impl<I> ParallelIterator1<I>
where
    I: ParallelIterator,
{
    pub fn len(&self) -> NonZeroUsize
    where
        I: IndexedParallelIterator,
    {
        NonZeroUsize::clamped(self.items.len())
    }

    pub fn count(self) -> NonZeroUsize {
        // Though the count must be non-zero here, it may overflow to zero.
        NonZeroUsize::new(self.items.count())
            .expect("non-empty parallel iterator has zero items or overflow in count")
    }

    pub fn map<U, F>(self, f: F) -> ParallelIterator1<rayon::iter::Map<I, F>>
    where
        U: Send,
        F: Fn(I::Item) -> U + Send + Sync,
    {
        // SAFETY: This combinator function cannot reduce the cardinality of the iterator to zero.
        unsafe { self.and_then_unchecked(move |items| items.map(f)) }
    }

    pub fn min_by<F>(self, f: F) -> I::Item
    where
        F: Fn(&I::Item, &I::Item) -> Ordering + Send + Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.min_by(f).unwrap_maybe_unchecked() }
    }

    pub fn min_by_key<B, F>(self, f: F) -> I::Item
    where
        B: Ord,
        F: Fn(&I::Item) -> B + Send + Sync,
    {
        self.min_by(move |lhs, rhs| f(lhs).cmp(&f(rhs)))
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        self.min_by(Ord::cmp)
    }

    pub fn max_by<F>(self, f: F) -> I::Item
    where
        F: Fn(&I::Item, &I::Item) -> Ordering + Send + Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.max_by(f).unwrap_maybe_unchecked() }
    }

    pub fn max_by_key<B, F>(self, f: F) -> I::Item
    where
        B: Ord,
        F: Fn(&I::Item) -> B + Send + Sync,
    {
        self.max_by(move |lhs, rhs| f(lhs).cmp(&f(rhs)))
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        self.max_by(Ord::cmp)
    }

    pub fn reduce_with<F>(self, f: F) -> I::Item
    where
        F: Fn(I::Item, I::Item) -> I::Item + Send + Sync,
    {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.reduce_with(f).unwrap_maybe_unchecked() }
    }

    pub fn collect1<T>(self) -> T
    where
        T: FromParallelIterator1<I::Item>,
    {
        T::from_par_iter1(self)
    }

    pub fn collect_into_vec1(self, items: &mut Vec1<I::Item>)
    where
        I: IndexedParallelIterator,
    {
        // `self` is non-empty, so clearing and extending the underlying `Vec` from `self` cannot
        // violate the non-empty guarantee of `Vec1`.
        self.items.collect_into_vec(&mut items.items)
    }
}

#[cfg(feature = "rayon")]
impl<I> IntoParallelIterator for ParallelIterator1<I>
where
    I: ParallelIterator,
{
    type Iter = I;
    type Item = I::Item;

    fn into_par_iter(self) -> Self::Iter {
        self.items
    }
}

#[cfg(feature = "rayon")]
impl<I> IntoParallelIterator1 for ParallelIterator1<I>
where
    I: ParallelIterator,
{
    fn into_par_iter1(self) -> ParallelIterator1<Self::Iter> {
        self
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

pub fn rtail_and_head<I, T>(tail: I, head: T) -> RTailAndHead<I>
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

fn empty_or_into<T>(items: Option<T>) -> EmptyOrInto<T>
where
    T: IntoIterator,
{
    items.map(T::into_iter).into_iter().flatten()
}

#[cfg(test)]
pub mod harness {
    use core::iter::Peekable;
    use core::ops::RangeInclusive;
    use rstest::fixture;

    use crate::iter1::Iterator1;

    pub type Xs1 = Iterator1<Peekable<RangeInclusive<u8>>>;

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> Xs1 {
        Iterator1::try_from_iter(0..=end).unwrap_or_else(|_| panic!("range `0..={}` is empty", end))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::iter1::{IntoIterator1, ThenIterator1};
    use crate::slice1::{Slice1, slice1};
    #[cfg(feature = "alloc")]
    use crate::vec1::{Vec1, vec1};

    #[cfg(feature = "alloc")]
    #[rstest]
    fn unzip_iter1_into_vec1_then_eq() {
        let xs: (Vec1<u8>, Vec1<u8>) = [(0, 1); 4].into_iter1().unzip();
        assert_eq!(xs, (vec1![0; 4], vec1![1; 4]));
    }

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
        assert!(
            xs.into_iter()
                .or_non_empty(xs1)
                .eq(expected.iter1().copied())
        );
    }
}

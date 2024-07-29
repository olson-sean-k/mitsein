// TODO: Implement tests. Consider `rstest`.
// TODO: Provide a feature for integration with `std`-only APIs, namely `std::io`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(
    clippy::cast_lossless,
    clippy::checked_conversions,
    clippy::cloned_instead_of_copied,
    clippy::explicit_into_iter_loop,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::from_iter_instead_of_collect,
    clippy::if_not_else,
    clippy::manual_ok_or,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_else,
    clippy::unreadable_literal,
    clippy::unused_self
)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod segment;
mod serde;

pub mod array1;
pub mod array_vec1;
pub mod boxed1;
pub mod btree_map1;
pub mod btree_set1;
pub mod iter1;
pub mod slice1;
pub mod sync1;
pub mod vec1;
pub mod vec_deque1;

pub mod prelude {
    pub use crate::array1::Array1;
    pub use crate::iter1::{FromIterator1, IntoIterator1, IteratorExt as _, ThenIterator1};
    #[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
    pub use crate::sync1::{ArcSlice1Ext as _, WeakSlice1Ext as _};
    pub use crate::{Saturate, Saturated, Segmentation, Vacancy};
    #[cfg(feature = "alloc")]
    pub use {
        crate::boxed1::BoxedSlice1Ext as _,
        crate::btree_map1::OrOnlyExt as _,
        crate::vec1::{vec1, CowSlice1Ext as _, Vec1},
    };
}

use core::num::NonZeroUsize;
#[cfg(feature = "serde")]
use {
    ::serde::{Deserialize, Serialize},
    ::serde_derive::{Deserialize, Serialize},
};

#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};

pub use segment::{Segment, Segmentation, SegmentedBy};

trait NonZeroExt<T> {
    unsafe fn new_maybe_unchecked(n: T) -> Self;

    fn clamped(n: T) -> Self;
}

impl NonZeroExt<usize> for NonZeroUsize {
    #[cfg(all(not(miri), test))]
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap()
    }

    #[cfg(not(all(not(miri), test)))]
    #[inline(always)]
    unsafe fn new_maybe_unchecked(n: usize) -> Self {
        NonZeroUsize::new_unchecked(n)
    }

    fn clamped(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }
}

trait OptionExt<T> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    #[cfg(all(not(miri), test))]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap()
    }

    #[cfg(not(all(not(miri), test)))]
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap_unchecked()
    }
}

trait ResultExt<T, E> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    #[cfg(all(not(miri), test))]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        match self {
            Ok(value) => value,
            Err(_) => panic!("called `Result::unwrap_maybe_unchecked` on an `Err` value"),
        }
    }

    #[cfg(not(all(not(miri), test)))]
    #[inline(always)]
    unsafe fn unwrap_maybe_unchecked(self) -> T {
        self.unwrap_unchecked()
    }
}

pub trait Vacancy {
    fn vacancy(&self) -> usize;
}

pub trait Saturated<T>: Sized {
    type Remainder;

    fn saturated(items: T) -> (Self, Self::Remainder);
}

pub trait Saturate<T> {
    type Remainder;

    fn saturate(&mut self, items: T) -> Self::Remainder;
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            deserialize = "Self: TryFrom<Serde<T>, Error = EmptyError>, \
                           T: Clone + Deserialize<'de>,",
            serialize = "T: Clone + Serialize,",
        ),
        try_from = "Serde<T>",
        into = "Serde<T>",
    )
)]
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonEmpty<T>
where
    T: ?Sized,
{
    items: T,
}

impl<T> AsRef<T> for NonEmpty<T> {
    fn as_ref(&self) -> &T {
        &self.items
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Cardinality<O, M> {
    One(O),
    Many(M),
}

impl<O, M> Cardinality<O, M> {
    pub fn one(self) -> Option<O> {
        match self {
            Cardinality::One(one) => Some(one),
            _ => None,
        }
    }

    pub fn many(self) -> Option<M> {
        match self {
            Cardinality::Many(many) => Some(many),
            _ => None,
        }
    }

    pub fn map_one<U, F>(self, f: F) -> Cardinality<U, M>
    where
        F: FnOnce(O) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(f(one)),
            Cardinality::Many(many) => Cardinality::Many(many),
        }
    }

    pub fn map_many<U, F>(self, f: F) -> Cardinality<O, U>
    where
        F: FnOnce(M) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(one),
            Cardinality::Many(many) => Cardinality::Many(f(many)),
        }
    }
}

impl<T> Cardinality<T, T> {
    pub fn map<U, F>(self, f: F) -> Cardinality<U, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Cardinality::One(one) => Cardinality::One(f(one)),
            Cardinality::Many(many) => Cardinality::Many(f(many)),
        }
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec"))]
fn saturate_positional_vacancy<T, I>(destination: &mut T, source: I) -> I::IntoIter
where
    T: Extend<I::Item> + Vacancy,
    I: IntoIterator,
{
    let n = destination.vacancy();
    let mut source = source.into_iter();
    destination.extend(source.by_ref().take(n));
    source
}

macro_rules! with_literals {
    ($f:ident$(,)?) => {};
    ($f:ident, [$($N:literal $(,)?)+]$(,)?) => {
        $(
            $f!($N);
        )+
    };
}
pub(crate) use with_literals;

#[cfg(test)]
mod tests {}

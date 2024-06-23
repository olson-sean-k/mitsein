// TODO: Implement tests. Consider `rstest`.

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

mod serde;

pub mod array1;
pub mod boxed1;
pub mod btree_map1;
pub mod btree_set1;
pub mod iter1;
pub mod option1;
pub mod slice1;
pub mod vec1;
pub mod vec_deque1;

pub mod prelude {
    pub use crate::array1::Array1;
    #[cfg(feature = "alloc")]
    pub use crate::btree_map1::OrOnlyExt as _;
    pub use crate::iter1::{
        FromIterator1, IntoIterator1, IteratorExt as _, RemainderExt as _, Then1,
    };
    pub use crate::option1::OptionExt as _;
    pub use crate::NonZeroUsizeExt as _;
}

use core::num::NonZeroUsize;
#[cfg(feature = "serde")]
use {
    ::serde::{Deserialize, Serialize},
    ::serde_derive::{Deserialize, Serialize},
};

#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};

pub trait NonZeroUsizeExt {
    const ONE: Self;
}

impl NonZeroUsizeExt for NonZeroUsize {
    // SAFETY:
    const ONE: Self = unsafe { NonZeroUsize::new_unchecked(1) };
}

pub trait FnInto: FnOnce() -> Self::Into {
    type Into;

    fn call(self) -> Self::Into;
}

impl<T, F> FnInto for F
where
    F: FnOnce() -> T,
{
    type Into = T;

    fn call(self) -> Self::Into {
        (self)()
    }
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Arity<O, M> {
    One(O),
    Many(M),
}

impl<O, M> Arity<O, M> {
    pub fn one(self) -> Option<O> {
        match self {
            Arity::One(one) => Some(one),
            _ => None,
        }
    }

    pub fn many(self) -> Option<M> {
        match self {
            Arity::Many(many) => Some(many),
            _ => None,
        }
    }

    pub fn map_one<U, F>(self, f: F) -> Arity<U, M>
    where
        F: FnOnce(O) -> U,
    {
        match self {
            Arity::One(one) => Arity::One(f(one)),
            Arity::Many(many) => Arity::Many(many),
        }
    }

    pub fn map_many<U, F>(self, f: F) -> Arity<O, U>
    where
        F: FnOnce(M) -> U,
    {
        match self {
            Arity::One(one) => Arity::One(one),
            Arity::Many(many) => Arity::Many(f(many)),
        }
    }
}

impl<T> Arity<T, T> {
    pub fn map<U, F>(self, f: F) -> Arity<U, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Arity::One(one) => Arity::One(f(one)),
            Arity::Many(many) => Arity::Many(f(many)),
        }
    }
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

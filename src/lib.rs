// TODO: Do not depend on `std` (depend on `core` and `alloc` instead).
// TODO: Support serialization.
// TODO: Implement tests. Consider `rstest`.

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

pub mod array1;
pub mod iter1;
pub mod option1;
pub mod slice1;
pub mod vec1;

pub mod prelude {
    pub use crate::iter1::{FromIterator1, IntoIterator1, IteratorExt as _, RemainderExt as _};
    pub use crate::option1::OptionExt as _;
    pub use crate::NonZeroUsizeExt as _;
}

use std::num::NonZeroUsize;

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

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonEmpty<T>
where
    T: ?Sized,
{
    items: T,
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

macro_rules! with_non_zero_array_size_literals {
    ($f:ident$(,)?) => {
        $crate::with_literals!(
            $f,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ],
        );
    };
}
pub(crate) use with_non_zero_array_size_literals;

#[cfg(test)]
mod tests {}

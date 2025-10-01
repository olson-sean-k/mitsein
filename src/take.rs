#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

use core::fmt::{self, Debug, Formatter};

use crate::{Cardinality, MaybeEmpty, NonEmpty};

#[must_use]
pub struct TakeIfMany<'a, T, U, N = ()> {
    items: &'a mut NonEmpty<T>,
    index: N,
    many: fn(&mut NonEmpty<T>, N) -> U,
}

impl<'a, T, U, N> TakeIfMany<'a, T, U, N> {
    pub(crate) fn with(
        items: &'a mut NonEmpty<T>,
        index: N,
        many: fn(&mut NonEmpty<T>, N) -> U,
    ) -> Self {
        TakeIfMany { items, index, many }
    }
}

impl<'a, T, U, N> TakeIfMany<'a, T, U, N>
where
    T: MaybeEmpty,
{
    pub(crate) fn take_or_else<E, F>(self, one: F) -> Result<U, E>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> E,
    {
        let TakeIfMany { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => Err(one(items, index)),
            Cardinality::Many(_) => Ok((many)(items, index)),
        }
    }

    // It is tempting to use this function to implement `and_if` functions. However, this requires
    // knowledge of the position of the target item in its collection. For example, an `and_if`
    // function would not have enough information if implemented for `vec_deque1::PopIfMany`, since
    // items can be popped from both ends. Instead, this function is used to implement counterparts
    // to standard APIs like `Vec::pop_if` with bespoke functions on non-empty types, like
    // `Vec1::pop_if_many_and`.
    #[cfg(feature = "alloc")]
    pub(crate) fn take_if<F>(self, f: F) -> Option<U>
    where
        F: FnOnce(&mut NonEmpty<T>) -> bool,
    {
        let TakeIfMany { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => None,
            Cardinality::Many(_) => {
                if f(items) {
                    Some((many)(items, index))
                }
                else {
                    None
                }
            },
        }
    }

    pub fn or_else<E, F>(self, f: F) -> Result<U, E>
    where
        F: FnOnce() -> E,
    {
        self.take_or_else(|_, _| f())
    }

    pub fn or_none(self) -> Option<U> {
        self.take_or_else(|_, _| ()).ok()
    }

    pub fn or_false(self) -> bool {
        self.or_none().is_some()
    }
}

impl<'a, T, U, N> TakeIfMany<'a, T, Option<U>, N>
where
    T: MaybeEmpty,
{
    #[cfg(any(feature = "alloc", feature = "arrayvec"))]
    pub(crate) fn try_take_or_else<E, F>(self, one: F) -> Option<Result<U, E>>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> Option<E>,
    {
        let TakeIfMany { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => one(items, index).map(Err),
            Cardinality::Many(_) => (many)(items, index).map(Ok),
        }
    }
}

impl<T, U, N> Debug for TakeIfMany<'_, T, U, N>
where
    NonEmpty<T>: Debug,
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TakeIfMany")
            .field("items", &self.items)
            .field("index", &self.index)
            .field("many", &self.many)
            .finish()
    }
}

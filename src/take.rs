#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

use core::fmt::{self, Debug, Formatter};

use crate::{Cardinality, MaybeEmpty, NonEmpty};

#[must_use]
pub struct Take<'a, T, U, N = ()> {
    items: &'a mut NonEmpty<T>,
    index: N,
    many: fn(&mut NonEmpty<T>, N) -> U,
}

impl<'a, T, U, N> Take<'a, T, U, N> {
    pub(crate) fn with(
        items: &'a mut NonEmpty<T>,
        index: N,
        many: fn(&mut NonEmpty<T>, N) -> U,
    ) -> Self {
        Take { items, index, many }
    }
}

impl<'a, T, U, N> Take<'a, T, U, N>
where
    T: MaybeEmpty,
{
    pub(crate) fn take_or_else<E, F>(self, one: F) -> Result<U, E>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> E,
    {
        let Take { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => Err(one(items, index)),
            Cardinality::Many(_) => Ok((many)(items, index)),
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

impl<'a, T, U, N> Take<'a, T, Option<U>, N>
where
    T: MaybeEmpty,
{
    #[cfg(any(feature = "alloc", feature = "arrayvec"))]
    pub(crate) fn try_take_or_else<E, F>(self, one: F) -> Option<Result<U, E>>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> Option<E>,
    {
        let Take { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => one(items, index).map(Err),
            Cardinality::Many(_) => (many)(items, index).map(Ok),
        }
    }
}

impl<T, U, N> Debug for Take<'_, T, U, N>
where
    NonEmpty<T>: Debug,
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Take")
            .field("items", &self.items)
            .field("index", &self.index)
            .field("many", &self.many)
            .finish()
    }
}

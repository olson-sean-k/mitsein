#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;

use crate::{Cardinality, MaybeEmpty, NonEmpty, Vacancy};

#[derive(Debug)]
pub enum PutItem {}

#[derive(Debug)]
pub enum PutWith {}

#[derive(Debug)]
#[must_use]
pub struct PutOrLast<'a, T, U, N = (), B = PutItem>
where
    T: ?Sized,
{
    items: &'a mut T,
    index: N,
    item: U,
    vacancy: fn(&mut T, N, U),
    phantom: PhantomData<fn() -> B>,
}

impl<'a, T, U, N, B> PutOrLast<'a, T, U, N, B>
where
    T: ?Sized,
{
    pub(crate) fn put_with(items: &'a mut T, index: N, item: U, vacancy: fn(&mut T, N, U)) -> Self {
        PutOrLast {
            items,
            index,
            item,
            vacancy,
            phantom: PhantomData,
        }
    }
}

impl<'a, T, U, N, B> PutOrLast<'a, T, U, N, B>
where
    T: ?Sized,
{
    pub(crate) fn vacancy_or_else<E, F>(self, saturated: F) -> Result<(), E>
    where
        T: Vacancy,
        F: FnOnce(&'a mut T, N, U) -> E,
    {
        let PutOrLast {
            items,
            index,
            item,
            vacancy,
            ..
        } = self;
        match items.vacancy() {
            0 => Err(saturated(items, index, item)),
            _ => {
                (vacancy)(items, index, item);
                Ok(())
            },
        }
    }
}

#[must_use]
pub struct TakeOrOnly<'a, T, U, N = ()>
where
    T: ?Sized,
{
    items: &'a mut NonEmpty<T>,
    index: N,
    many: fn(&mut NonEmpty<T>, N) -> U,
}

impl<'a, T, U, N> TakeOrOnly<'a, T, U, N>
where
    T: ?Sized,
{
    pub(crate) fn take_with(
        items: &'a mut NonEmpty<T>,
        index: N,
        many: fn(&mut NonEmpty<T>, N) -> U,
    ) -> Self {
        TakeOrOnly { items, index, many }
    }
}

impl<'a, T, U, N> TakeOrOnly<'a, T, U, N>
where
    T: MaybeEmpty + ?Sized,
{
    pub(crate) fn many_or_else<E, F>(self, one: F) -> Result<U, E>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> E,
    {
        let TakeOrOnly { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => Err(one(items, index)),
            Cardinality::Many(_) => Ok((many)(items, index)),
        }
    }

    pub fn none(self) -> Option<U> {
        self.many_or_else(|_, _| ()).ok()
    }
}

impl<'a, T, U, N> TakeOrOnly<'a, T, Option<U>, N>
where
    T: MaybeEmpty + ?Sized,
{
    pub(crate) fn try_many_or_else<E, F>(self, one: F) -> Option<Result<U, E>>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> Option<E>,
    {
        let TakeOrOnly { items, index, many } = self;
        match items.cardinality() {
            Cardinality::One(_) => one(items, index).map(Err),
            Cardinality::Many(_) => (many)(items, index).map(Ok),
        }
    }
}

impl<'a, T, U, N> Debug for TakeOrOnly<'a, T, U, N>
where
    NonEmpty<T>: Debug,
    T: ?Sized,
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TakeOrOnly")
            .field("items", &self.items)
            .field("index", &self.index)
            .field("many", &self.many)
            .finish()
    }
}

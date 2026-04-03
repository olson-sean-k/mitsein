//! Subsets of collections by key.

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};

use crate::subset::SubsetFor;

#[derive(Clone, Copy, Debug)]
pub struct KeyNotFoundError<Q> {
    key: Q,
}

impl<Q> KeyNotFoundError<Q> {
    pub(crate) const fn from_key(key: Q) -> Self {
        KeyNotFoundError { key }
    }

    pub fn into_key(self) -> Q {
        self.key
    }

    pub fn take(self) -> (Q, KeyNotFoundError<()>) {
        let KeyNotFoundError { key } = self;
        (key, KeyNotFoundError::from_key(()))
    }

    pub fn take_and_drop(self) -> KeyNotFoundError<()> {
        self.take().1
    }
}

impl<Q> KeyNotFoundError<&'_ Q> {
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_owning(self) -> KeyNotFoundError<Q::Owned>
    where
        Q: ToOwned,
    {
        KeyNotFoundError::from_key(self.key.to_owned())
    }
}

impl<Q> Display for KeyNotFoundError<Q>
where
    Q: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "key {:?} not found", &self.key)
    }
}

impl<Q> Error for KeyNotFoundError<Q> where Q: Debug {}

pub trait ByKey<Q>: SubsetFor
where
    Q: ?Sized,
{
    // LINT: Though this type is quite complex, the indirection introduced by a type defintion is
    //       arguably less clear and a bit trickier to understand.
    #[allow(clippy::type_complexity)]
    fn except<'a>(
        &'a mut self,
        key: &'a Q,
    ) -> Result<ExceptKeySubset<'a, Self::Kind, Self::Target, Q>, KeyNotFoundError<&'a Q>>;
}

#[must_use]
pub struct ExceptKeySubset<'a, K, T, Q>
where
    K: ?Sized + SubsetFor<Target = T>,
    T: ?Sized,
    Q: ?Sized,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) key: &'a Q,
}

impl<'a, K, T, Q> ExceptKeySubset<'a, K, T, Q>
where
    K: ?Sized + SubsetFor<Target = T>,
    T: ?Sized,
    Q: ?Sized,
{
    #[cfg(feature = "alloc")]
    pub(crate) fn unchecked(items: &'a mut T, key: &'a Q) -> Self {
        ExceptKeySubset { items, key }
    }

    pub fn key(&self) -> &Q {
        self.key
    }
}

impl<K, T, Q> Debug for ExceptKeySubset<'_, K, T, Q>
where
    K: ?Sized + SubsetFor<Target = T>,
    T: Debug + ?Sized,
    Q: Debug + ?Sized,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExceptKeySubset")
            .field("items", &self.items)
            .field("key", &self.key)
            .finish()
    }
}

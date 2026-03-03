//! Subsets of non-empty collections by key.

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};

#[derive(Clone, Copy)]
pub struct KeyNotFoundError<Q>(pub Q);

impl<Q> KeyNotFoundError<Q> {
    pub fn into_key(self) -> Q {
        self.0
    }
}

impl<Q> KeyNotFoundError<&'_ Q> {
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_owning(self) -> KeyNotFoundError<Q::Owned>
    where
        Q: ToOwned,
    {
        KeyNotFoundError(self.0.to_owned())
    }
}

impl<Q> Debug for KeyNotFoundError<Q> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("KeyNotFoundError")
            .finish_non_exhaustive()
    }
}

impl<Q> Display for KeyNotFoundError<Q>
where
    Q: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "key not found: {:?}", self.0)
    }
}

impl<Q> Error for KeyNotFoundError<Q> where Q: Debug {}

/// A subset of a non-empty collection that exempts a key.
///
/// This is a very general type constructor: refer to more specific type definitions to see the
/// relevant APIs for a particular collection type. For example, see [`hash_set1::ExceptKeySubset`]
/// to see supported APIs for [`HashSet1`]. Every supported non-empty collection type has such a
/// subset type definition.
///
/// [`hash_set1::ExceptKeySubset`]: crate::hash_set1::ExceptKeySubset
/// [`HashSet1`]: crate::hash_set1::HashSet1
#[derive(Debug)]
#[must_use]
pub struct ExceptKeySubset<'a, T, Q>
where
    T: ?Sized,
    Q: ?Sized,
{
    pub(crate) items: &'a mut T,
    pub(crate) key: &'a Q,
}

impl<'a, T, Q> ExceptKeySubset<'a, T, Q>
where
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

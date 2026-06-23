//! Subsets of non-empty collections by key.

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

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
    pub(crate) fn unchecked(items: &'a mut T, key: &'a Q) -> Self {
        ExceptKeySubset { items, key }
    }

    pub fn key(&self) -> &Q {
        self.key
    }
}

#![cfg(feature = "serde")]
#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use core::fmt::{self, Display, Formatter};
use serde_derive::{Deserialize, Serialize};

use crate::NonEmpty;

#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Serde<T> {
    pub items: T,
}

impl<T> From<NonEmpty<T>> for Serde<T> {
    fn from(items: NonEmpty<T>) -> Self {
        Serde { items: items.items }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmptyError;

impl Display for EmptyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "non-empty collection has no items")
    }
}

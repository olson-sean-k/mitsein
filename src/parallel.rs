#![cfg(feature = "rayon")]
#![cfg_attr(docsrs, doc(cfg(feature = "rayon")))]

use core::fmt::{self, Debug, Formatter};

use crate::NonEmpty;

#[must_use]
#[repr(transparent)]
pub struct Parallel<'a, T> {
    pub(crate) items: &'a NonEmpty<T>,
}

impl<T> Debug for Parallel<'_, T>
where
    NonEmpty<T>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Parallel")
            .field("items", self.items)
            .finish()
    }
}

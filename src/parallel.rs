#![cfg(feature = "rayon")]
#![cfg_attr(docsrs, doc(cfg(feature = "rayon")))]

use core::fmt::{self, Debug, Formatter};

pub trait Parallelization {
    type Target: ?Sized;

    fn parallel(&self) -> Parallel<&'_ Self::Target>;
}

#[must_use]
#[repr(transparent)]
pub struct Parallel<T> {
    pub(crate) items: T,
}

impl<T> Debug for Parallel<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Parallel")
            .field("items", &self.items)
            .finish()
    }
}

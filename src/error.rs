#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};

pub(crate) const EMPTY_ERROR_MESSAGE: &str = "failed to construct non-empty collection: no items";

/// An error in which a non-empty value is expected but an empty value is observed.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct EmptyError<T> {
    items: T,
}

impl<T> EmptyError<T> {
    pub(crate) fn from_empty(items: T) -> Self {
        EmptyError { items }
    }

    /// Converts the error into the empty value.
    pub fn into_empty(self) -> T {
        self.items
    }

    pub(crate) fn map<U, F>(self, f: F) -> EmptyError<U>
    where
        F: FnOnce(T) -> U,
    {
        EmptyError::from_empty(f(self.items))
    }

    /// Takes the empty value out of the error, returning it and a unit error.
    pub fn take(self) -> (T, EmptyError<()>) {
        (self.items, EmptyError::from_empty(()))
    }

    /// Takes the empty value out of the error and immediately drops it, returning a unit error.
    pub fn take_and_drop(self) -> EmptyError<()> {
        EmptyError::from_empty(())
    }

    /// Converts the error to a reference to the empty value.
    pub fn as_empty(&self) -> &T {
        &self.items
    }
}

impl<T> EmptyError<&'_ T> {
    /// Maps the empty value from a borrowed type into its owned type.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_owning(self) -> EmptyError<T::Owned>
    where
        T: ToOwned,
    {
        EmptyError::from_empty(self.items.to_owned())
    }
}

impl<T> Debug for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("EmptyError").finish_non_exhaustive()
    }
}

impl<T> Display for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{EMPTY_ERROR_MESSAGE}")
    }
}

impl<T> Error for EmptyError<T> {}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Clone, Copy, Debug)]
pub struct KeyNotFoundError<Q> {
    key: Q,
}

#[cfg(feature = "alloc")]
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

#[cfg(feature = "alloc")]
impl<Q> KeyNotFoundError<&'_ Q> {
    pub fn into_owning(self) -> KeyNotFoundError<Q::Owned>
    where
        Q: ToOwned,
    {
        KeyNotFoundError::from_key(self.key.to_owned())
    }
}

#[cfg(feature = "alloc")]
impl<Q> Display for KeyNotFoundError<Q>
where
    Q: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "key {:?} not found", &self.key)
    }
}

#[cfg(feature = "alloc")]
impl<Q> Error for KeyNotFoundError<Q> where Q: Debug {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OutOfBoundsError<N> {
    Point(N),
    Range(N, N),
}

impl<N> Display for OutOfBoundsError<N>
where
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OutOfBoundsError::Point(point) => {
                write!(formatter, "point at {point:?} is out of bounds")
            },
            OutOfBoundsError::Range(start, end) => write!(
                formatter,
                "range from {start:?} to {end:?} is out of bounds"
            ),
        }
    }
}

impl<N> Error for OutOfBoundsError<N> where N: Debug {}

impl<N> From<N> for OutOfBoundsError<N> {
    fn from(point: N) -> Self {
        OutOfBoundsError::Point(point)
    }
}

impl<N> From<(N, N)> for OutOfBoundsError<N> {
    fn from(range: (N, N)) -> Self {
        let (start, end) = range;
        OutOfBoundsError::Range(start, end)
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RangeError<N> {
    OutOfBounds(OutOfBoundsError<N>),
    Unordered(UnorderedError<N>),
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> Display for RangeError<N>
where
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RangeError::OutOfBounds(error) => write!(formatter, "{error}"),
            RangeError::Unordered(error) => write!(formatter, "{error}"),
        }
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> Error for RangeError<N> where N: Debug {}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> From<OutOfBoundsError<N>> for RangeError<N> {
    fn from(error: OutOfBoundsError<N>) -> Self {
        RangeError::OutOfBounds(error)
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> From<UnorderedError<N>> for RangeError<N> {
    fn from(error: UnorderedError<N>) -> Self {
        RangeError::Unordered(error)
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UnorderedError<N>(pub N, pub N);

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> Display for UnorderedError<N>
where
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let UnorderedError(start, end) = self;
        write!(
            formatter,
            "range unordered: starts at {start:?} but ends at {end:?}"
        )
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<N> Error for UnorderedError<N> where N: Debug {}

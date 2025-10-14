#![cfg(any(feature = "alloc", feature = "arrayvec"))]

mod index;
mod item;
mod trim;

use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

pub use crate::segment::range::index::IndexRange;
#[cfg(feature = "alloc")]
pub use {crate::segment::range::item::ItemRange, crate::segment::range::trim::TrimRange};

#[cfg(feature = "alloc")]
pub(crate) use {
    crate::segment::range::item::OptionExt, crate::segment::range::trim::ResolveTrimRange,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RangeError<N> {
    OutOfBounds(OutOfBoundsError<N>),
    Unordered(UnorderedError<N>),
}

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

impl<N> Error for RangeError<N> where N: Debug {}

impl<N> From<OutOfBoundsError<N>> for RangeError<N> {
    fn from(error: OutOfBoundsError<N>) -> Self {
        RangeError::OutOfBounds(error)
    }
}

impl<N> From<UnorderedError<N>> for RangeError<N> {
    fn from(error: UnorderedError<N>) -> Self {
        RangeError::Unordered(error)
    }
}

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UnorderedError<N>(pub N, pub N);

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

impl<N> Error for UnorderedError<N> where N: Debug {}

// Though they are not intersections in the most strict sense, implementations of this trait must
// consider adjacencies as intersections.
pub trait Intersect<R>: Sized {
    type Output;
    type Error;

    fn intersect(self, range: R) -> Result<Self::Output, Self::Error>;
}

pub trait IntoRangeBounds<N>: RangeBounds<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>);
}

impl<N> IntoRangeBounds<N> for Range<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let Range { start, end } = self;
        (Bound::Included(start), Bound::Excluded(end))
    }
}

impl<N> IntoRangeBounds<N> for RangeFrom<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let RangeFrom { start } = self;
        (Bound::Included(start), Bound::Unbounded)
    }
}

impl<N> IntoRangeBounds<N> for RangeFull {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        (Bound::Unbounded, Bound::Unbounded)
    }
}

impl<N> IntoRangeBounds<N> for RangeInclusive<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let (start, end) = self.into_inner();
        (Bound::Included(start), Bound::Included(end))
    }
}

impl<N> IntoRangeBounds<N> for RangeTo<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let RangeTo { end } = self;
        (Bound::Unbounded, Bound::Excluded(end))
    }
}

impl<N> IntoRangeBounds<N> for RangeToInclusive<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let RangeToInclusive { end } = self;
        (Bound::Unbounded, Bound::Included(end))
    }
}

pub trait Project<T> {
    type Output;
    type Error;

    fn project(self, other: T) -> Result<Self::Output, Self::Error>;
}

#[cfg(feature = "alloc")]
pub fn ordered_range_bounds<N, R>(range: R) -> Result<R, R>
where
    N: Ord,
    R: RangeBounds<N>,
{
    use Bound::{Excluded, Included, Unbounded};

    match (range.start_bound(), range.end_bound()) {
        // Unlike the opposite bounds, `start == end` is considered unordered (rather than
        // empty).
        (Excluded(start), Included(end)) if start < end => Ok(range),
        (Included(start), Excluded(end) | Included(end)) if start <= end => Ok(range),
        (Unbounded, _) | (_, Unbounded) => Ok(range),
        // This arm includes ranges that are exclusive at both ends. It is not possible to
        // determine if such a range is properly ordered without more knowledge of the domain of
        // its index type. At time of writing, no such range is supported in `core`.
        _ => Err(range),
    }
}

pub(crate) const fn panic_start_underflow() -> ! {
    panic!("underflow in range start")
}

pub(crate) const fn panic_start_overflow() -> ! {
    panic!("overflow in range start")
}

pub(crate) const fn panic_end_underflow() -> ! {
    panic!("underflow in range end")
}

pub(crate) const fn panic_end_overflow() -> ! {
    panic!("overflow in range end")
}

pub(crate) const fn panic_len_underflow() -> ! {
    panic!("underflow in range length")
}

pub(crate) const fn panic_len_overflow() -> ! {
    panic!("overflow in range length")
}

pub(crate) const fn panic_index_out_of_bounds() -> ! {
    panic!("index out of bounds")
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use core::ops::{Bound, RangeBounds};
    use rstest::rstest;

    use crate::segment::range;

    #[rstest]
    #[case::empty(0i32..0)]
    #[case::one(0i32..=0)]
    #[case::one(0i32..1)]
    #[case::many(0i32..9)]
    fn ordered_range_bounds_is_ok<N, R>(#[case] range: R)
    where
        N: Ord,
        R: RangeBounds<N>,
    {
        assert!(range::ordered_range_bounds(range).is_ok());
    }

    // LINT: Ranges are intentionally reversed here to test that they are considered unordered.
    #[allow(clippy::reversed_empty_ranges)]
    #[rstest]
    #[case(2i32..0)]
    #[case('b'..='a')]
    // This range is in fact ordered. However, `range::ordered_range_bounds` does not have enough
    // information to determine this for arbitrary types.
    #[case::indeterminant((Bound::Excluded(0i32), Bound::Excluded(9i32)))]
    fn ordered_range_bounds_is_err<N, R>(#[case] range: R)
    where
        N: Ord,
        R: RangeBounds<N>,
    {
        assert!(range::ordered_range_bounds(range).is_err())
    }
}

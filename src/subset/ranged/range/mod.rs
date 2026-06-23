#![cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]

mod index;
mod item;
mod trim;

#[cfg(feature = "alloc")]
use core::ops::{Bound, RangeBounds};

pub use crate::subset::ranged::range::index::IndexRange;
#[cfg(feature = "alloc")]
pub use crate::subset::ranged::range::{item::ItemRange, trim::TrimRange};

#[cfg(feature = "alloc")]
pub(crate) use crate::subset::ranged::range::item::OptionExt;

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

    use crate::subset::ranged::range;

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

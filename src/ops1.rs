//! Non-empty bounded [ranges][`core::ops`].

// Unlike other non-empty types, ranges do not rely on `MaybeEmpty` implementations. One of the
// primary mechanisms provided by `MaybeEmpty` is `TakeIfMany`, which inspects the cardinality of
// the target collection, but that API is not used by range types.

use core::num::NonZeroUsize;
use core::ops::{Bound, Range as LegacyRange, RangeBounds, RangeInclusive as LegacyRangeInclusive};
use core::range::{Range, RangeInclusive};

use crate::cmp::UnsafeOrd;
use crate::iter1::{IntoIterator1, Iterator1, UnsafeStep};
use crate::{EmptyError, NonEmpty};

pub type Range1<T> = NonEmpty<Range<T>>;

impl<T> Range1<T> {
    /// # Safety
    ///
    /// `items` must be ordered and non-empty. For example, it is unsound to call this function
    /// with the range `Range { start: 0, end: 0}`.
    pub const unsafe fn from_range_unchecked(items: Range<T>) -> Self {
        NonEmpty { items }
    }

    pub fn into_range(self) -> Range<T> {
        self.items
    }

    pub const fn start(&self) -> &T {
        &self.items.start
    }

    pub const fn end(&self) -> &T {
        &self.items.end
    }

    pub fn iter1(&self) -> Iterator1<Range<T>>
    where
        Range<T>: Iterator<Item = T>,
        T: UnsafeStep,
    {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.clone()) }
    }

    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: PartialOrd<T> + ?Sized,
    {
        self.items.contains(item)
    }
}

impl Range1<usize> {
    pub fn zero_to_non_zero(end: NonZeroUsize) -> Self {
        // SAFETY: `end` is non-zero, so the half-open range is non-empty.
        unsafe {
            Range1::from_range_unchecked(Range {
                start: 0,
                end: end.get(),
            })
        }
    }
}

impl<T> From<Range1<T>> for Range<T> {
    fn from(items: Range1<T>) -> Self {
        items.items
    }
}

impl<T> From<Range1<T>> for LegacyRange<T> {
    fn from(items: Range1<T>) -> Self {
        items.items.into()
    }
}

impl<T> IntoIterator for Range1<T>
where
    Range<T>: IntoIterator<Item = T>,
{
    type Item = T;
    type IntoIter = <Range<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for Range1<T>
where
    Range<T>: IntoIterator<Item = T>,
    T: UnsafeStep,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> RangeBounds<T> for Range1<T> {
    fn start_bound(&self) -> Bound<&T> {
        self.items.start_bound()
    }

    fn end_bound(&self) -> Bound<&T> {
        self.items.end_bound()
    }
}

impl<T> RangeBounds<T> for Range1<&T> {
    fn start_bound(&self) -> Bound<&T> {
        self.items.start_bound()
    }

    fn end_bound(&self) -> Bound<&T> {
        self.items.end_bound()
    }
}

impl<T> TryFrom<Range<T>> for Range1<T>
where
    T: UnsafeOrd,
{
    type Error = EmptyError<Range<T>>;

    fn try_from(items: Range<T>) -> Result<Self, Self::Error> {
        if items.is_empty() {
            Err(EmptyError::from_empty(items))
        } else {
            Ok(NonEmpty { items })
        }
    }
}

impl<T> TryFrom<LegacyRange<T>> for Range1<T>
where
    T: UnsafeOrd,
{
    type Error = EmptyError<LegacyRange<T>>;

    fn try_from(items: LegacyRange<T>) -> Result<Self, Self::Error> {
        if items.is_empty() {
            Err(EmptyError::from_empty(items))
        } else {
            Ok(NonEmpty {
                items: items.into(),
            })
        }
    }
}

pub type RangeInclusive1<T> = NonEmpty<RangeInclusive<T>>;

impl<T> RangeInclusive1<T> {
    /// # Safety
    ///
    /// `items` must be ordered and non-empty. For example, it is unsound to call this function
    /// with the range `RangeInclusive { start: 9, last: 0 }`. Note that ordered [`RangeInclusive`]s are implicitly non-empty.
    pub const unsafe fn from_range_inclusive_unchecked(items: RangeInclusive<T>) -> Self {
        NonEmpty { items }
    }

    pub fn into_range_inclusive(self) -> RangeInclusive<T> {
        self.items
    }

    pub const fn start(&self) -> &T {
        &self.items.start
    }

    pub const fn last(&self) -> &T {
        &self.items.last
    }

    pub fn iter1(&self) -> Iterator1<RangeInclusive<T>>
    where
        RangeInclusive<T>: Iterator<Item = T>,
        T: UnsafeStep,
    {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.clone()) }
    }

    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: PartialOrd<T> + ?Sized,
    {
        self.items.contains(item)
    }
}

impl RangeInclusive1<usize> {
    pub fn zero_to(last: usize) -> Self {
        // SAFETY: The closed range starts at zero and can only end at an inclusive minimum of
        //         zero, and so is non-empty.
        unsafe {
            RangeInclusive1::from_range_inclusive_unchecked(RangeInclusive { start: 0, last })
        }
    }
}

impl From<Range1<usize>> for RangeInclusive1<usize> {
    fn from(items: Range1<usize>) -> Self {
        let Range { start, end } = items.items;
        // SAFETY: The half-open `Range1` is non-empty, and so `end` must be greater than `start`
        //         and therefore non-zero. This means that this subtraction (predecessor) need not
        //         be checked.
        unsafe {
            RangeInclusive1::from_range_inclusive_unchecked(RangeInclusive {
                start,
                last: end - 1,
            })
        }
    }
}

impl<T> From<RangeInclusive1<T>> for RangeInclusive<T> {
    fn from(items: RangeInclusive1<T>) -> Self {
        items.items
    }
}

impl<T> From<RangeInclusive1<T>> for LegacyRangeInclusive<T> {
    fn from(items: RangeInclusive1<T>) -> Self {
        items.items.into()
    }
}

impl<T> IntoIterator for RangeInclusive1<T>
where
    RangeInclusive<T>: IntoIterator<Item = T>,
{
    type Item = T;
    type IntoIter = <RangeInclusive<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for RangeInclusive1<T>
where
    RangeInclusive<T>: IntoIterator<Item = T>,
    T: UnsafeStep,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` is non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T> RangeBounds<T> for RangeInclusive1<T> {
    fn start_bound(&self) -> Bound<&T> {
        self.items.start_bound()
    }

    fn end_bound(&self) -> Bound<&T> {
        self.items.end_bound()
    }
}

impl<T> RangeBounds<T> for RangeInclusive1<&T> {
    fn start_bound(&self) -> Bound<&T> {
        self.items.start_bound()
    }

    fn end_bound(&self) -> Bound<&T> {
        self.items.end_bound()
    }
}

impl<T> TryFrom<RangeInclusive<T>> for RangeInclusive1<T>
where
    T: UnsafeOrd,
{
    type Error = EmptyError<RangeInclusive<T>>;

    fn try_from(items: RangeInclusive<T>) -> Result<Self, Self::Error> {
        if items.is_empty() {
            Err(EmptyError::from_empty(items))
        } else {
            Ok(NonEmpty { items })
        }
    }
}

impl<T> TryFrom<LegacyRangeInclusive<T>> for RangeInclusive1<T>
where
    T: UnsafeOrd,
{
    type Error = EmptyError<LegacyRangeInclusive<T>>;

    fn try_from(items: LegacyRangeInclusive<T>) -> Result<Self, Self::Error> {
        if items.is_empty() {
            Err(EmptyError::from_empty(items))
        } else {
            Ok(NonEmpty {
                items: items.into(),
            })
        }
    }
}

#[macro_export]
macro_rules! range1 {
    ($start:literal .. $end:literal) => {{
        $crate::ops1::range1!(@closed $start, @open $end)
    }};
    ($start:literal .. $end:tt) => {{
        $crate::ops1::range1!(@closed $start, @open $end)
    }};
    ($start:tt .. $end:literal) => {{
        $crate::ops1::range1!(@closed $start, @open $end)
    }};
    ($start:tt .. $end:tt) => {{
        $crate::ops1::range1!(@closed $start, @open $end)
    }};

    ($start:literal ..= $end:literal) => {{
        $crate::ops1::range1!(@closed $start, @closed $end)
    }};
    ($start:literal ..= $end:tt) => {{
        $crate::ops1::range1!(@closed $start, @closed $end)
    }};
    ($start:tt ..= $end:literal) => {{
        $crate::ops1::range1!(@closed $start, @closed $end)
    }};
    ($start:tt ..= $end:tt) => {{
        $crate::ops1::range1!(@closed $start, @closed $end)
    }};

    (@closed $start:expr, @open $end:expr) => {{
        const {
            const fn congruent_unsafe_ord_bounds<T>(_start: &T, _end: &T)
            where
                T: $crate::cmp::UnsafeOrd,
            {
            }

            // LINT: To support `..` and `..=` syntax, other matchers of this macro match against
            //       token trees, which often require parentheses that raise `unused_parens` when
            //       in expressions (here). For example, in `range1!(0..(N + 1))`, the expression
            //       `N + 1` must be delimited by parentheses to match.
            #[allow(unused_parens)]
            let start = $start;
            #[allow(unused_parens)]
            let end = $end;
            congruent_unsafe_ord_bounds(&start, &end);
            assert!(start < end);

            // SAFETY: `start` is less than `end` (per the above assertion) and so the range is
            //         non-empty.
            unsafe { $crate::ops1::Range1::from_range_unchecked(core::range::Range{start, end}) }
        }
    }};
    (@closed $start:expr, @closed $last:expr) => {{
        const {
            const fn congruent_unsafe_ord_bounds<T>(_start: &T, _last: &T)
            where
                T: $crate::cmp::UnsafeOrd,
            {
            }

            // LINT: To support `..` and `..=` syntax, other matchers of this macro match against
            //       token trees, which often require parentheses that raise `unused_parens` when
            //       in expressions (here). For example, in `range1!(0..(N + 1))`, the expression
            //       `N + 1` must be delimited by parentheses to match.
            #[allow(unused_parens)]
            let start = $start;
            #[allow(unused_parens)]
            let last = $last;
            congruent_unsafe_ord_bounds(&start, &last);
            assert!(start <= last);

            // SAFETY: `start` is less than or equal to `end` (per the above assertion) and so the
            //         range is non-empty.
            unsafe { $crate::ops1::RangeInclusive1::from_range_inclusive_unchecked(core::range::RangeInclusive { start, last }) }
        }
    }};
}
pub use range1;

#[cfg(test)]
mod tests {
    use core::ops::{Range, RangeInclusive};
    use rstest::rstest;

    use crate::ops1::{Range1, RangeInclusive1};
    #[cfg(feature = "alloc")]
    use {
        crate::iter1::IntoIterator1,
        crate::slice1::{Slice1, slice1},
        crate::vec1::Vec1,
    };

    #[rstest]
    #[should_panic]
    #[case::empty(0..0)]
    #[should_panic]
    #[case::empty(42..42)]
    #[allow(clippy::reversed_empty_ranges)]
    #[should_panic]
    #[case::unordered(1..0)]
    #[case::one(0..1)]
    #[case::one(42..43)]
    #[case::many(0..10)]
    fn unwrap_range1_from_range_then_ok_or_panic(#[case] range: Range<usize>) {
        let _ = Range1::try_from(range).unwrap();
    }

    #[rstest]
    #[allow(clippy::reversed_empty_ranges)]
    #[should_panic]
    #[case::unordered(1..=0)]
    #[case::one(0..=0)]
    #[case::one(42..=42)]
    #[case::many(0..=9)]
    fn unwrap_range_inclusive1_from_range_inclusive_then_ok_or_panic(
        #[case] range: RangeInclusive<usize>,
    ) {
        let _ = RangeInclusive1::try_from(range).unwrap();
    }

    #[cfg(feature = "alloc")]
    #[rstest]
    #[case::one(range1!(0..1), slice1![0])]
    #[case::one(range1!(42..43), slice1![42])]
    #[case::many(range1!(0..10), slice1![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    fn collect_range1_into_vec1_then_eq(
        #[case] range: Range1<usize>,
        #[case] expected: &Slice1<usize>,
    ) {
        let xs: Vec1<_> = range.into_iter1().collect1();
        assert_eq!(xs.as_slice1(), expected);
    }

    #[cfg(feature = "alloc")]
    #[rstest]
    #[case::one(range1!(0..=0), slice1![0])]
    #[case::one(range1!(42..=42), slice1![42])]
    #[case::many(range1!(0..=9), slice1![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    fn collect_range_inclusive1_into_vec1_then_eq(
        #[case] range: RangeInclusive1<usize>,
        #[case] expected: &Slice1<usize>,
    ) {
        let xs: Vec1<_> = range.into_iter1().collect1();
        assert_eq!(xs.as_slice1(), expected);
    }
}

mod _compile_fail_tests {
    /// ```compile_fail
    /// let xs = mitsein::ops1::range1!(0..0);
    /// ```
    #[doc(hidden)]
    const fn _empty_bounds_then_exclusive_range1_macro_compilation_fails() {}

    /// ```compile_fail
    /// let xs = mitsein::ops1::range1!(0u64..1u32);
    /// ```
    #[doc(hidden)]
    const fn _incongruent_bounds_then_exclusive_range1_macro_compilation_fails() {}

    /// ```compile_fail
    /// let xs = mitsein::ops1::range1!(0u64..=1u32);
    /// ```
    #[doc(hidden)]
    const fn _incongruent_bounds_then_inclusive_range1_macro_compilation_fails() {}

    /// ```compile_fail
    /// let xs = mitsein::ops1::range1!(1..0);
    /// ```
    #[doc(hidden)]
    const fn _unordered_bounds_then_exclusive_range1_macro_compilation_fails() {}

    /// ```compile_fail
    /// let xs = mitsein::ops1::range1!(1..=0);
    /// ```
    #[doc(hidden)]
    const fn _unordered_bounds_then_inclusive_range1_macro_compilation_fails() {}
}

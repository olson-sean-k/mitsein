use core::borrow::{Borrow, BorrowMut};
use core::num::NonZeroUsize;

use crate::iter1::IntoIterator1;
use crate::slice1::Slice1;
#[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
use crate::sync1::ArcSlice1;
#[cfg(feature = "alloc")]
use crate::vec1::Vec1;

#[diagnostic::on_unimplemented(
    note = "due to a technical limitation, `Array1` is not yet implemented for all non-empty array \
            types"
)]
pub trait Array1:
    AsMut<Slice1<Self::Item>>
    + AsRef<Slice1<Self::Item>>
    + Borrow<Slice1<Self::Item>>
    + BorrowMut<Slice1<Self::Item>>
    + IntoIterator1
    + Sized
{
    const N: NonZeroUsize;

    #[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
    #[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", target_has_atomic = "ptr"))))]
    fn into_arc_slice1(self) -> ArcSlice1<Self::Item>;

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    fn into_vec1(self) -> Vec1<Self::Item> {
        self.into_iter1().collect1()
    }

    fn as_slice1(&self) -> &Slice1<Self::Item>;

    fn as_mut_slice1(&mut self) -> &mut Slice1<Self::Item>;
}

macro_rules! with_non_zero_array_size_literals {
    ($f:ident$(,)?) => {
        $crate::with_literals!(
            $f,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ],
        );
    };
}
pub(crate) use with_non_zero_array_size_literals;

/// # Safety
macro_rules! impl_array1_for_array {
    ($N:literal) => {
        impl<T> $crate::array1::Array1 for [T; $N] {
            // SAFETY:
            const N: core::num::NonZeroUsize =
                unsafe { core::num::NonZeroUsize::new_unchecked($N) };

            #[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
            #[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", target_has_atomic = "ptr"))))]
            fn into_arc_slice1(self) -> $crate::sync1::ArcSlice1<Self::Item> {
                use $crate::sync1::ArcSlice1Ext as _;

                $crate::sync1::ArcSlice1::from_array1(self)
            }

            fn as_slice1(&self) -> &$crate::slice1::Slice1<T> {
                self.as_ref()
            }

            fn as_mut_slice1(&mut self) -> &mut $crate::slice1::Slice1<T> {
                self.as_mut()
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_array1_for_array);

/// # Safety
macro_rules! impl_as_mut_for_array {
    ($N:literal) => {
        impl<T> core::convert::AsMut<$crate::slice1::Slice1<T>> for [T; $N] {
            fn as_mut(&mut self) -> &mut $crate::slice1::Slice1<T> {
                // SAFETY:
                unsafe { $crate::slice1::Slice1::from_mut_slice_unchecked(self.as_mut_slice()) }
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_as_mut_for_array);

/// # Safety
macro_rules! impl_as_ref_for_array {
    ($N:literal) => {
        impl<T> core::convert::AsRef<$crate::slice1::Slice1<T>> for [T; $N] {
            fn as_ref(&self) -> &$crate::slice1::Slice1<T> {
                // SAFETY:
                unsafe { $crate::slice1::Slice1::from_slice_unchecked(self.as_slice()) }
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_as_ref_for_array);

/// # Safety
macro_rules! impl_borrow_for_array {
    ($N:literal) => {
        impl<T> core::borrow::Borrow<$crate::slice1::Slice1<T>> for [T; $N] {
            fn borrow(&self) -> &$crate::slice1::Slice1<T> {
                // SAFETY:
                unsafe { $crate::slice1::Slice1::from_slice_unchecked(self.as_slice()) }
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_borrow_for_array);

/// # Safety
macro_rules! impl_borrow_mut_for_array {
    ($N:literal) => {
        impl<T> core::borrow::BorrowMut<$crate::slice1::Slice1<T>> for [T; $N] {
            fn borrow_mut(&mut self) -> &mut $crate::slice1::Slice1<T> {
                // SAFETY:
                unsafe { $crate::slice1::Slice1::from_mut_slice_unchecked(self.as_mut_slice()) }
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_borrow_mut_for_array);

/// # Safety
macro_rules! impl_into_iterator1_for_array {
    ($N:literal) => {
        impl<T> $crate::iter1::IntoIterator1 for [T; $N] {
            fn into_iter1(self) -> $crate::iter1::Iterator1<Self::IntoIter> {
                // SAFETY:
                unsafe { $crate::iter1::Iterator1::from_iter_unchecked(self) }
            }
        }
    };
}
with_non_zero_array_size_literals!(impl_into_iterator1_for_array);

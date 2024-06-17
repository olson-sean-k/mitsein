use core::num::NonZeroUsize;

use crate::iter1::IntoIterator1;
use crate::slice1::Slice1;
#[cfg(feature = "alloc")]
use crate::vec1::Vec1;

pub trait Array1:
    AsMut<Slice1<Self::Item>> + AsRef<Slice1<Self::Item>> + IntoIterator1 + Sized
{
    const N: NonZeroUsize;

    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    fn into_vec1(self) -> Vec1<Self::Item> {
        self.into_iter1().collect()
    }

    fn as_slice1(&self) -> &Slice1<Self::Item>;

    fn as_mut_slice1(&mut self) -> &mut Slice1<Self::Item>;
}

macro_rules! impl_array1_for_array {
    ($N:literal) => {
        impl<T> $crate::array1::Array1 for [T; $N] {
            // SAFETY:
            const N: core::num::NonZeroUsize =
                unsafe { core::num::NonZeroUsize::new_unchecked($N) };

            fn as_slice1(&self) -> &$crate::slice1::Slice1<T> {
                self.as_ref()
            }

            fn as_mut_slice1(&mut self) -> &mut $crate::slice1::Slice1<T> {
                self.as_mut()
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_array1_for_array);

macro_rules! impl_as_mut_for_array {
    ($N:literal) => {
        impl<T> core::convert::AsMut<$crate::slice1::Slice1<T>> for [T; $N] {
            fn as_mut(&mut self) -> &mut $crate::slice1::Slice1<T> {
                $crate::slice1::Slice1::from_mut_slice_unchecked(self.as_mut_slice())
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_as_mut_for_array);

macro_rules! impl_as_ref_for_array {
    ($N:literal) => {
        impl<T> core::convert::AsRef<$crate::slice1::Slice1<T>> for [T; $N] {
            fn as_ref(&self) -> &$crate::slice1::Slice1<T> {
                $crate::slice1::Slice1::from_slice_unchecked(self.as_slice())
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_as_ref_for_array);

macro_rules! impl_into_iterator1_for_array {
    ($N:literal) => {
        impl<T> $crate::iter1::IntoIterator1 for [T; $N] {
            fn into_iter1(self) -> $crate::iter1::Iterator1<Self::IntoIter> {
                $crate::iter1::Iterator1::from_iter_unchecked(self)
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_into_iterator1_for_array);

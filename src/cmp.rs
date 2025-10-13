//! Comparison and ordering extensions.

/// Types that can be used in APIs where consistent total ordering is required for memory safety.
///
/// # Safety
///
/// Types that implement this trait must exhibit consistent behavior as described for [`Ord`].
/// Unlike [`Ord`] however, **inconsistent implementations of [`Ord`] and this trait are unsound**.
/// Bounds on this trait indicate that inconsistent [`Ord`] implementations are not memory safe.
/// For example, [`BTreeSet1::split_off_tail`] requires consistent ordering to maintain its
/// non-empty guarantee.
///
/// [`BTreeSet1::split_off_tail`]: crate::btree_set1::BTreeSet1::split_off_tail
#[diagnostic::on_unimplemented(
    message = "`{Self}` may not implement consistent total ordering",
    label = "types used here must implement consistent total ordering for soundness",
    note = "there may be alternative collection APIs for types that do not implement this trait"
)]
pub unsafe trait UnsafeOrd: Ord {}

// SAFETY: The implementations of `UnsafeOrd` in this module trust that the `Ord` implementations
//         of the given types from `core`, `alloc`, etc. conform to the safety requirements of
//         `UnsafeOrd`. Moreover, these `Ord` implementations are very unlikely to change and are
//         even less likely to change in such a way that they are non-conformant with `UnsafeOrd`.

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod alloc {
    use crate::cmp::UnsafeOrd;

    unsafe impl<T> UnsafeOrd for alloc::sync::Arc<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for alloc::boxed::Box<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for alloc::collections::btree_set::BTreeSet<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for alloc::borrow::Cow<'_, T> where T: Clone + UnsafeOrd {}
    unsafe impl UnsafeOrd for alloc::ffi::CString {}
    unsafe impl<T> UnsafeOrd for alloc::rc::Rc<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for alloc::string::String {}
    unsafe impl<T> UnsafeOrd for alloc::vec::Vec<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for alloc::collections::vec_deque::VecDeque<T> where T: UnsafeOrd {}

    unsafe impl<T> UnsafeOrd for crate::btree_set1::BTreeSet1<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for crate::string1::String1 {}
    unsafe impl<T> UnsafeOrd for crate::vec1::Vec1<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for crate::vec_deque1::VecDeque1<T> where T: UnsafeOrd {}
}

#[cfg(feature = "arrayvec")]
#[cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]
mod array_vec {
    use crate::array1::Array1;
    use crate::cmp::UnsafeOrd;

    unsafe impl<T, const N: usize> UnsafeOrd for arrayvec::ArrayVec<T, N> where T: UnsafeOrd {}

    unsafe impl<T, const N: usize> UnsafeOrd for crate::array_vec1::ArrayVec1<T, N>
    where
        [T; N]: Array1,
        T: UnsafeOrd,
    {
    }
}

// `Cell` and `RefCell` are intentionally absent here and do not implement `UnsafeOrd`. Interior
// mutability is incompatible with the safety requirements of `UnsafeOrd`.
mod core {
    use crate::cmp::UnsafeOrd;

    unsafe impl UnsafeOrd for () {}
    unsafe impl UnsafeOrd for bool {}
    unsafe impl UnsafeOrd for char {}
    unsafe impl UnsafeOrd for i8 {}
    unsafe impl UnsafeOrd for i16 {}
    unsafe impl UnsafeOrd for i32 {}
    unsafe impl UnsafeOrd for i64 {}
    unsafe impl UnsafeOrd for i128 {}
    unsafe impl UnsafeOrd for isize {}
    unsafe impl UnsafeOrd for str {}
    unsafe impl UnsafeOrd for u8 {}
    unsafe impl UnsafeOrd for u16 {}
    unsafe impl UnsafeOrd for u32 {}
    unsafe impl UnsafeOrd for u64 {}
    unsafe impl UnsafeOrd for u128 {}
    unsafe impl UnsafeOrd for usize {}

    unsafe impl<T> UnsafeOrd for *const T where T: ?Sized + UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for *mut T where T: ?Sized + UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for &'_ T where T: ?Sized + UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for &'_ mut T where T: ?Sized + UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for [T] where T: UnsafeOrd {}
    unsafe impl<T, const N: usize> UnsafeOrd for [T; N] where T: UnsafeOrd {}

    unsafe impl UnsafeOrd for core::ffi::CStr {}
    unsafe impl UnsafeOrd for core::time::Duration {}
    unsafe impl UnsafeOrd for core::convert::Infallible {}
    unsafe impl UnsafeOrd for core::net::Ipv4Addr {}
    unsafe impl UnsafeOrd for core::net::Ipv6Addr {}
    unsafe impl<T> UnsafeOrd for core::mem::ManuallyDrop<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for core::ptr::NonNull<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for core::num::NonZeroI8 {}
    unsafe impl UnsafeOrd for core::num::NonZeroI16 {}
    unsafe impl UnsafeOrd for core::num::NonZeroI32 {}
    unsafe impl UnsafeOrd for core::num::NonZeroI64 {}
    unsafe impl UnsafeOrd for core::num::NonZeroI128 {}
    unsafe impl UnsafeOrd for core::num::NonZeroIsize {}
    unsafe impl UnsafeOrd for core::num::NonZeroU8 {}
    unsafe impl UnsafeOrd for core::num::NonZeroU16 {}
    unsafe impl UnsafeOrd for core::num::NonZeroU32 {}
    unsafe impl UnsafeOrd for core::num::NonZeroU64 {}
    unsafe impl UnsafeOrd for core::num::NonZeroU128 {}
    unsafe impl UnsafeOrd for core::num::NonZeroUsize {}
    unsafe impl<T> UnsafeOrd for Option<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for core::marker::PhantomData<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for core::marker::PhantomPinned {}
    unsafe impl<T> UnsafeOrd for core::pin::Pin<T>
    where
        T: core::ops::Deref,
        T::Target: UnsafeOrd,
    {
    }
    unsafe impl<T> UnsafeOrd for core::task::Poll<T> where T: UnsafeOrd {}
    unsafe impl<T, E> UnsafeOrd for Result<T, E>
    where
        T: UnsafeOrd,
        E: UnsafeOrd,
    {
    }
    unsafe impl<T> UnsafeOrd for core::cmp::Reverse<T> where T: UnsafeOrd {}
    unsafe impl<T> UnsafeOrd for core::num::Saturating<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for core::net::SocketAddrV4 {}
    unsafe impl UnsafeOrd for core::net::SocketAddrV6 {}
    unsafe impl UnsafeOrd for core::any::TypeId {}
    unsafe impl<T> UnsafeOrd for core::num::Wrapping<T> where T: UnsafeOrd {}

    unsafe impl<T> UnsafeOrd for crate::slice1::Slice1<T> where T: UnsafeOrd {}
    unsafe impl UnsafeOrd for crate::str1::Str1 {}

    macro_rules! impl_unsafe_ord_for_tuple {
        (($($T:ident $(,)?)+) $(,)?) => {
            unsafe impl<$($T,)+> $crate::cmp::UnsafeOrd for ($($T,)+)
            where
                $(
                    $T: UnsafeOrd,
                )+
            {
            }
        };
    }
    crate::with_tuples!(impl_unsafe_ord_for_tuple, (T1, T2, T3, T4, T5, T6, T7, T8));
}

#[cfg(feature = "smallvec")]
#[cfg_attr(docsrs, doc(cfg(feature = "smallvec")))]
mod small_vec {
    use crate::cmp::UnsafeOrd;

    unsafe impl<A> UnsafeOrd for smallvec::SmallVec<A>
    where
        A: smallvec::Array,
        A::Item: UnsafeOrd,
    {
    }

    unsafe impl<A> UnsafeOrd for crate::small_vec1::SmallVec1<A>
    where
        A: smallvec::Array,
        A::Item: UnsafeOrd,
    {
    }
}

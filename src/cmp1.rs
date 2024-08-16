/// # Safety
pub unsafe trait Ord1: Ord {}

// SAFETY: The implementations of `Ord1` in this module trust that the `Ord` implementations of the
//         given types from `core`, `alloc`, etc. conform to the safety requirements of `Ord1`.
//         Moreover, these `Ord` implementations are very unlikely to change and are even less
//         likely to change in such a way that they are non-conformant with `Ord1`.

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod alloc {
    use crate::cmp1::Ord1;

    unsafe impl<T> Ord1 for alloc::sync::Arc<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for alloc::boxed::Box<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for alloc::collections::btree_set::BTreeSet<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for alloc::borrow::Cow<'_, T> where T: Clone + Ord1 {}
    unsafe impl Ord1 for alloc::ffi::CString {}
    unsafe impl<T> Ord1 for alloc::rc::Rc<T> where T: Ord1 {}
    unsafe impl Ord1 for alloc::string::String {}
    unsafe impl<T> Ord1 for alloc::vec::Vec<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for alloc::collections::vec_deque::VecDeque<T> where T: Ord1 {}

    unsafe impl<T> Ord1 for crate::btree_set1::BTreeSet1<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for crate::vec1::Vec1<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for crate::vec_deque1::VecDeque1<T> where T: Ord1 {}
}

#[cfg(feature = "arrayvec")]
#[cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]
mod array_vec {
    use crate::array1::Array1;
    use crate::cmp1::Ord1;

    unsafe impl<T, const N: usize> Ord1 for arrayvec::ArrayVec<T, N> where T: Ord1 {}

    unsafe impl<T, const N: usize> Ord1 for crate::array_vec1::ArrayVec1<T, N>
    where
        [T; N]: Array1,
        T: Ord1,
    {
    }
}

// `Cell` and `RefCell` are intentionally absent here and do not implement `Ord1`. Interior
// mutability is incompatible with the safety requirements of `Ord1`.
mod core {
    use crate::cmp1::Ord1;

    unsafe impl Ord1 for () {}
    unsafe impl Ord1 for bool {}
    unsafe impl Ord1 for char {}
    unsafe impl Ord1 for i8 {}
    unsafe impl Ord1 for i16 {}
    unsafe impl Ord1 for i32 {}
    unsafe impl Ord1 for i64 {}
    unsafe impl Ord1 for i128 {}
    unsafe impl Ord1 for isize {}
    unsafe impl Ord1 for str {}
    unsafe impl Ord1 for u8 {}
    unsafe impl Ord1 for u16 {}
    unsafe impl Ord1 for u32 {}
    unsafe impl Ord1 for u64 {}
    unsafe impl Ord1 for u128 {}
    unsafe impl Ord1 for usize {}

    unsafe impl<T> Ord1 for *const T where T: Ord1 {}
    unsafe impl<T> Ord1 for *mut T where T: Ord1 {}
    unsafe impl<'a, T> Ord1 for &'a T where T: Ord1 {}
    unsafe impl<'a, T> Ord1 for &'a mut T where T: Ord1 {}
    unsafe impl<T> Ord1 for [T] where T: Ord1 {}
    unsafe impl<T, const N: usize> Ord1 for [T; N] where T: Ord1 {}

    unsafe impl Ord1 for core::ffi::CStr {}
    unsafe impl Ord1 for core::time::Duration {}
    unsafe impl Ord1 for core::convert::Infallible {}
    unsafe impl Ord1 for core::net::Ipv4Addr {}
    unsafe impl Ord1 for core::net::Ipv6Addr {}
    unsafe impl<T> Ord1 for core::mem::ManuallyDrop<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for core::ptr::NonNull<T> where T: Ord1 {}
    unsafe impl Ord1 for core::num::NonZeroI8 {}
    unsafe impl Ord1 for core::num::NonZeroI16 {}
    unsafe impl Ord1 for core::num::NonZeroI32 {}
    unsafe impl Ord1 for core::num::NonZeroI64 {}
    unsafe impl Ord1 for core::num::NonZeroI128 {}
    unsafe impl Ord1 for core::num::NonZeroIsize {}
    unsafe impl Ord1 for core::num::NonZeroU8 {}
    unsafe impl Ord1 for core::num::NonZeroU16 {}
    unsafe impl Ord1 for core::num::NonZeroU32 {}
    unsafe impl Ord1 for core::num::NonZeroU64 {}
    unsafe impl Ord1 for core::num::NonZeroU128 {}
    unsafe impl Ord1 for core::num::NonZeroUsize {}
    unsafe impl<T> Ord1 for Option<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for core::marker::PhantomData<T> where T: Ord1 {}
    unsafe impl Ord1 for core::marker::PhantomPinned {}
    unsafe impl<T> Ord1 for core::pin::Pin<T>
    where
        T: core::ops::Deref,
        T::Target: Ord1,
    {
    }
    unsafe impl<T> Ord1 for core::task::Poll<T> where T: Ord1 {}
    unsafe impl<T, E> Ord1 for Result<T, E>
    where
        T: Ord1,
        E: Ord1,
    {
    }
    unsafe impl<T> Ord1 for core::cmp::Reverse<T> where T: Ord1 {}
    unsafe impl<T> Ord1 for core::num::Saturating<T> where T: Ord1 {}
    unsafe impl Ord1 for core::net::SocketAddrV4 {}
    unsafe impl Ord1 for core::net::SocketAddrV6 {}
    unsafe impl Ord1 for core::any::TypeId {}
    unsafe impl<T> Ord1 for core::num::Wrapping<T> where T: Ord1 {}

    macro_rules! impl_ord1_for_tuple {
        (($($T:ident $(,)?)+) $(,)?) => {
            unsafe impl<$($T,)+> $crate::cmp1::Ord1 for ($($T,)+)
            where
                $(
                    $T: Ord1,
                )+
            {
            }
        };
    }
    crate::with_tuples!(impl_ord1_for_tuple, (T1, T2, T3, T4, T5, T6, T7, T8));
}

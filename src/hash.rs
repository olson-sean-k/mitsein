//! Hashing extensions.

use ::core::hash::Hash;

// SAFETY: The implementations of `UnsafeHash` in this module trust that the `Eq` and `Hash`
//         implementations of the given types from `core`, `alloc`, etc. conform to the safety
//         requirements of `UnsafeHash`. Moreover, these `Eq` and `Hash` implementations are very
//         unlikely to change and are even less likely to change in such a way that they are
//         non-conformant with `UnsafeHash`.

/// Types that can be used in APIs where consistent equivalence w.r.t. hashing is required for
/// memory safety.
///
/// # Safety
///
/// Types that implement this trait must exhibit consistent [`Eq`] behavior as described for
/// [`Hash`]. Unlike [`Eq`] and [`Hash`] however, **inconsistent implementations of this trait are
/// unsound**. Bounds on this trait indicate that inconsistent [`Eq`] and [`Hash`] implementations
/// are not memory safe. For example, [`HashSet1::except`] requires consistent equivalence in its
/// operations to maintain the non-empty guarantee.
///
/// [`HashSet1::except`]: crate::except::ByKey::except
#[diagnostic::on_unimplemented(
    message = "`{Self}` may not implement consistent hashing and equivalence",
    label = "types used here must implement consistent equivalence w.r.t. hashing for soundness"
)]
pub unsafe trait UnsafeHash: Eq + Hash {}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod alloc {
    use crate::hash::UnsafeHash;

    unsafe impl<T> UnsafeHash for alloc::sync::Arc<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for alloc::boxed::Box<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for alloc::collections::btree_set::BTreeSet<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for alloc::borrow::Cow<'_, T> where T: Clone + UnsafeHash {}
    unsafe impl UnsafeHash for alloc::ffi::CString {}
    unsafe impl<T> UnsafeHash for alloc::rc::Rc<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for alloc::string::String {}
    unsafe impl<T> UnsafeHash for alloc::vec::Vec<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for alloc::collections::vec_deque::VecDeque<T> where T: UnsafeHash {}

    unsafe impl<T> UnsafeHash for crate::btree_set1::BTreeSet1<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for crate::string1::String1 {}
    unsafe impl<T> UnsafeHash for crate::vec1::Vec1<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for crate::vec_deque1::VecDeque1<T> where T: UnsafeHash {}
}

#[cfg(feature = "arrayvec")]
#[cfg_attr(docsrs, doc(cfg(feature = "arrayvec")))]
mod array_vec {
    use crate::array1::Array1;
    use crate::hash::UnsafeHash;

    unsafe impl<T, const N: usize> UnsafeHash for arrayvec::ArrayVec<T, N> where T: UnsafeHash {}

    unsafe impl<T, const N: usize> UnsafeHash for crate::array_vec1::ArrayVec1<T, N>
    where
        [T; N]: Array1,
        T: UnsafeHash,
    {
    }
}

// `Cell` and `RefCell` are intentionally absent here and do not implement `UnsafeHash`. Interior
// mutability is incompatible with the safety requirements of `UnsafeHash`.
mod core {
    use crate::hash::UnsafeHash;

    unsafe impl UnsafeHash for () {}
    unsafe impl UnsafeHash for bool {}
    unsafe impl UnsafeHash for char {}
    unsafe impl UnsafeHash for i8 {}
    unsafe impl UnsafeHash for i16 {}
    unsafe impl UnsafeHash for i32 {}
    unsafe impl UnsafeHash for i64 {}
    unsafe impl UnsafeHash for i128 {}
    unsafe impl UnsafeHash for isize {}
    unsafe impl UnsafeHash for str {}
    unsafe impl UnsafeHash for u8 {}
    unsafe impl UnsafeHash for u16 {}
    unsafe impl UnsafeHash for u32 {}
    unsafe impl UnsafeHash for u64 {}
    unsafe impl UnsafeHash for u128 {}
    unsafe impl UnsafeHash for usize {}

    unsafe impl<T> UnsafeHash for *const T where T: ?Sized + UnsafeHash {}
    unsafe impl<T> UnsafeHash for *mut T where T: ?Sized + UnsafeHash {}
    unsafe impl<T> UnsafeHash for &'_ T where T: ?Sized + UnsafeHash {}
    unsafe impl<T> UnsafeHash for &'_ mut T where T: ?Sized + UnsafeHash {}
    unsafe impl<T> UnsafeHash for [T] where T: UnsafeHash {}
    unsafe impl<T, const N: usize> UnsafeHash for [T; N] where T: UnsafeHash {}

    unsafe impl<B, C> UnsafeHash for core::ops::ControlFlow<B, C>
    where
        B: UnsafeHash,
        C: UnsafeHash,
    {
    }
    unsafe impl UnsafeHash for core::ffi::CStr {}
    unsafe impl<T> UnsafeHash for core::mem::Discriminant<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for core::time::Duration {}
    unsafe impl UnsafeHash for core::convert::Infallible {}
    unsafe impl UnsafeHash for core::net::Ipv4Addr {}
    unsafe impl UnsafeHash for core::net::Ipv6Addr {}
    unsafe impl<T> UnsafeHash for core::mem::ManuallyDrop<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for core::ptr::NonNull<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for core::num::NonZeroI8 {}
    unsafe impl UnsafeHash for core::num::NonZeroI16 {}
    unsafe impl UnsafeHash for core::num::NonZeroI32 {}
    unsafe impl UnsafeHash for core::num::NonZeroI64 {}
    unsafe impl UnsafeHash for core::num::NonZeroI128 {}
    unsafe impl UnsafeHash for core::num::NonZeroIsize {}
    unsafe impl UnsafeHash for core::num::NonZeroU8 {}
    unsafe impl UnsafeHash for core::num::NonZeroU16 {}
    unsafe impl UnsafeHash for core::num::NonZeroU32 {}
    unsafe impl UnsafeHash for core::num::NonZeroU64 {}
    unsafe impl UnsafeHash for core::num::NonZeroU128 {}
    unsafe impl UnsafeHash for core::num::NonZeroUsize {}
    unsafe impl<T> UnsafeHash for Option<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for core::marker::PhantomData<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for core::marker::PhantomPinned {}
    unsafe impl<T> UnsafeHash for core::pin::Pin<T>
    where
        T: core::ops::Deref,
        T::Target: UnsafeHash,
    {
    }
    unsafe impl<T> UnsafeHash for core::task::Poll<T> where T: UnsafeHash {}
    unsafe impl<N> UnsafeHash for core::ops::Range<N> where N: UnsafeHash {}
    unsafe impl<N> UnsafeHash for core::ops::RangeFrom<N> where N: UnsafeHash {}
    unsafe impl<N> UnsafeHash for core::ops::RangeInclusive<N> where N: UnsafeHash {}
    unsafe impl<N> UnsafeHash for core::ops::RangeToInclusive<N> where N: UnsafeHash {}
    unsafe impl<T, E> UnsafeHash for Result<T, E>
    where
        T: UnsafeHash,
        E: UnsafeHash,
    {
    }
    unsafe impl<T> UnsafeHash for core::cmp::Reverse<T> where T: UnsafeHash {}
    unsafe impl<T> UnsafeHash for core::num::Saturating<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for core::net::SocketAddrV4 {}
    unsafe impl UnsafeHash for core::net::SocketAddrV6 {}
    unsafe impl UnsafeHash for core::any::TypeId {}
    unsafe impl<T> UnsafeHash for core::num::Wrapping<T> where T: UnsafeHash {}

    unsafe impl<T> UnsafeHash for crate::slice1::Slice1<T> where T: UnsafeHash {}
    unsafe impl UnsafeHash for crate::str1::Str1 {}

    macro_rules! impl_unsafe_hash_for_tuple {
        (($T:ident,) $(,)?) => {
            #[cfg_attr(docsrs, doc(fake_variadic))]
            unsafe impl<$T> $crate::hash::UnsafeHash for ($T,)
            where
                $T: UnsafeHash,
            {
            }
        };
        (($($T:ident $(,)?)+) $(,)?) => {
            #[cfg_attr(docsrs, doc(hidden))]
            unsafe impl<$($T,)+> $crate::hash::UnsafeHash for ($($T,)+)
            where
                $(
                    $T: UnsafeHash,
                )+
            {
            }
        };
    }
    crate::with_tuples!(impl_unsafe_hash_for_tuple, (T1, T2, T3, T4, T5, T6, T7, T));
}

#[cfg(feature = "smallvec")]
#[cfg_attr(docsrs, doc(cfg(feature = "smallvec")))]
mod small_vec {
    use crate::hash::UnsafeHash;

    unsafe impl<A> UnsafeHash for smallvec::SmallVec<A>
    where
        A: smallvec::Array,
        A::Item: UnsafeHash,
    {
    }

    unsafe impl<A> UnsafeHash for crate::small_vec1::SmallVec1<A>
    where
        A: smallvec::Array,
        A::Item: UnsafeHash,
    {
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
mod std {
    use crate::hash::UnsafeHash;

    unsafe impl UnsafeHash for std::time::Instant {}
    unsafe impl UnsafeHash for std::ffi::OsStr {}
    unsafe impl UnsafeHash for std::ffi::OsString {}
    unsafe impl UnsafeHash for std::time::SystemTime {}
    unsafe impl UnsafeHash for std::thread::ThreadId {}
}

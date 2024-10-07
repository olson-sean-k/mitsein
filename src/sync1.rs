#![cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
#![cfg_attr(docsrs, doc(cfg(all(feature = "alloc", target_has_atomic = "ptr"))))]

use alloc::sync::{Arc, Weak};

use crate::array1::Array1;
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _};
use crate::iter1::{FromIterator1, IntoIterator1};
use crate::safety::ResultExt as _;
use crate::slice1::Slice1;
use crate::vec1::{CowSlice1, CowSlice1Ext as _, Vec1};

pub type ArcSlice1<T> = Arc<Slice1<T>>;

pub trait ArcSlice1Ext<T>: Sized {
    /// # Safety
    unsafe fn from_arc_slice_unchecked(items: Arc<[T]>) -> Self;

    fn try_from_arc_slice(items: Arc<[T]>) -> Result<Self, Arc<[T]>>;

    fn from_array1<const N: usize>(items: [T; N]) -> Self
    where
        [T; N]: Array1;

    fn from_boxed_slice1(items: BoxedSlice1<T>) -> Self;

    fn from_cow_slice1(items: CowSlice1<T>) -> Self
    where
        T: Clone;

    fn try_into_arc_array<const N: usize>(self) -> Result<Arc<[T; N]>, Self>;

    fn into_arc_slice(self) -> Arc<[T]>;

    fn as_slice1(&self) -> &Slice1<T>;
}

impl<T> ArcSlice1Ext<T> for ArcSlice1<T> {
    unsafe fn from_arc_slice_unchecked(items: Arc<[T]>) -> Self {
        let items = Arc::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input slice is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `[T]` and
        //         `Slice1<T>` have the same representation (`Slice1<T>` is `repr(transparent)`).
        //         Moreover, the allocator only requires that the memory location and layout are
        //         the same when deallocating, so dropping the transmuted `Arc` is sound.
        Arc::from_raw(items as *const Slice1<T>)
    }

    fn try_from_arc_slice(items: Arc<[T]>) -> Result<Self, Arc<[T]>> {
        match items.len() {
            0 => Err(items),
            // SAFETY:
            _ => Ok(unsafe { ArcSlice1::from_arc_slice_unchecked(items) }),
        }
    }

    fn from_array1<const N: usize>(items: [T; N]) -> Self
    where
        [T; N]: Array1,
    {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items)) }
    }

    fn from_boxed_slice1(items: BoxedSlice1<T>) -> Self {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.into_boxed_slice())) }
    }

    fn from_cow_slice1(items: CowSlice1<T>) -> Self
    where
        T: Clone,
    {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.into_cow_slice())) }
    }

    fn try_into_arc_array<const N: usize>(self) -> Result<Arc<[T; N]>, Self> {
        if self.len().get() == N {
            // SAFETY:
            Ok(unsafe { self.into_arc_slice().try_into().unwrap_maybe_unchecked() })
        }
        else {
            Err(self)
        }
    }

    fn into_arc_slice(self) -> Arc<[T]> {
        let items = Arc::into_raw(self);
        // SAFETY: This transmutation is safe, because `[T]` and `Slice1<T>` have the same
        //         representation (`Slice1<T>` is `repr(transparent)`). Moreover, the allocator
        //         only requires that the memory location and layout are the same when
        //         deallocating, so dropping the transmuted `Arc` is sound.
        unsafe { Arc::from_raw(items as *mut [T]) }
    }

    fn as_slice1(&self) -> &Slice1<T> {
        // SAFETY:
        unsafe { Slice1::from_slice_unchecked(self.items.as_ref()) }
    }
}

impl<'a, T> From<&'a Slice1<T>> for ArcSlice1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.as_slice())) }
    }
}

impl<T> From<Vec1<T>> for ArcSlice1<T> {
    fn from(items: Vec1<T>) -> Self {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.items)) }
    }
}

impl<T> FromIterator1<T> for ArcSlice1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY:
        unsafe { ArcSlice1::from_arc_slice_unchecked(items.into_iter1().collect()) }
    }
}

pub type WeakSlice1<T> = Weak<Slice1<T>>;

pub trait WeakSlice1Ext<T>: Sized {
    /// # Safety
    unsafe fn from_weak_slice_unchecked(items: Weak<[T]>) -> Self;
}

impl<T> WeakSlice1Ext<T> for WeakSlice1<T> {
    unsafe fn from_weak_slice_unchecked(items: Weak<[T]>) -> Self {
        let items = Weak::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input slice is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `[T]` and
        //         `Slice1<T>` have the same representation (`Slice1<T>` is `repr(transparent)`).
        //         Moreover, the allocator only requires that the memory location and layout are
        //         the same when deallocating, so dropping the transmuted `Weak` is sound.
        Weak::from_raw(items as *const Slice1<T>)
    }
}

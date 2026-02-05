//! Non-empty [synchronized][`sync`] collections.
//!
//! [`sync`]: alloc::sync

#![cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
#![cfg_attr(docsrs, doc(cfg(all(feature = "alloc", target_has_atomic = "ptr"))))]

use alloc::sync::{Arc, Weak};

use crate::array1::Array1;
use crate::borrow1::{CowSlice1, CowSlice1Ext as _, CowStr1, CowStr1Ext as _};
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _, BoxedStr1, BoxedStr1Ext as _};
use crate::iter1::{FromIterator1, IntoIterator1};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::vec1::Vec1;
use crate::{EmptyError, MaybeEmpty};

pub type ArcSlice1<T> = Arc<Slice1<T>>;

pub trait ArcSlice1Ext<T>: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Arc::<T>::from([])`][`Arc::from`].
    ///
    /// [`Arc::from`]: alloc::sync::Arc::from
    unsafe fn from_arc_slice_unchecked(items: Arc<[T]>) -> Self;

    fn try_from_arc_slice(items: Arc<[T]>) -> Result<Self, EmptyError<Arc<[T]>>>;

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
        unsafe { Arc::from_raw(items as *const Slice1<T>) }
    }

    fn try_from_arc_slice(items: Arc<[T]>) -> Result<Self, EmptyError<Arc<[T]>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { ArcSlice1::from_arc_slice_unchecked(items) }),
        }
    }

    fn from_array1<const N: usize>(items: [T; N]) -> Self
    where
        [T; N]: Array1,
    {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items)) }
    }

    fn from_boxed_slice1(items: BoxedSlice1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.into_boxed_slice())) }
    }

    fn from_cow_slice1(items: CowSlice1<T>) -> Self
    where
        T: Clone,
    {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.into_cow_slice())) }
    }

    fn try_into_arc_array<const N: usize>(self) -> Result<Arc<[T; N]>, Self> {
        match self.into_arc_slice().try_into() {
            Ok(items) => Ok(items),
            // SAFETY: `self` and therefore `items` are non-empty.
            Err(items) => Err(unsafe { ArcSlice1::from_arc_slice_unchecked(items) }),
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
        self.as_ref()
    }
}

impl<'a, T> From<&'a Slice1<T>> for ArcSlice1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.as_slice())) }
    }
}

impl<T> From<Vec1<T>> for ArcSlice1<T> {
    fn from(items: Vec1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(Arc::from(items.items)) }
    }
}

impl<T> FromIterator1<T> for ArcSlice1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { ArcSlice1::from_arc_slice_unchecked(items.into_iter().collect()) }
    }
}

pub type WeakSlice1<T> = Weak<Slice1<T>>;

pub trait WeakSlice1Ext<T>: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Arc::downgrade(&Arc::<T>::from([]))`][`Arc::downgrade`].
    ///
    /// [`Arc::downgrade`]: alloc::sync::Arc::downgrade
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
        unsafe { Weak::from_raw(items as *const Slice1<T>) }
    }
}

pub type ArcStr1 = Arc<Str1>;

pub trait ArcStr1Ext: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Arc::<str>::from("")`][`Arc::from`].
    ///
    /// [`Arc::from`]: alloc::sync::Arc::from
    unsafe fn from_arc_str_unchecked(items: Arc<str>) -> Self;

    fn try_from_arc_str(items: Arc<str>) -> Result<Self, EmptyError<Arc<str>>>;

    fn from_boxed_str1(items: BoxedStr1) -> Self;

    fn from_cow_str1(items: CowStr1) -> Self;

    fn into_arc_str(self) -> Arc<str>;

    fn as_str1(&self) -> &Str1;
}

impl ArcStr1Ext for ArcStr1 {
    unsafe fn from_arc_str_unchecked(items: Arc<str>) -> Self {
        let items = Arc::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input string is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `str` and
        //         `Str1` have the same representation (`Str1` is `repr(transparent)`). Moreover,
        //         the allocator only requires that the memory location and layout are the same
        //         when deallocating, so dropping the transmuted `Arc` is sound.
        unsafe { Arc::from_raw(items as *const Str1) }
    }

    fn try_from_arc_str(items: Arc<str>) -> Result<Self, EmptyError<Arc<str>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { ArcStr1::from_arc_str_unchecked(items) }),
        }
    }

    fn from_boxed_str1(items: BoxedStr1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcStr1::from_arc_str_unchecked(Arc::from(items.into_boxed_str())) }
    }

    fn from_cow_str1(items: CowStr1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { ArcStr1::from_arc_str_unchecked(Arc::from(items.into_cow_str())) }
    }

    fn into_arc_str(self) -> Arc<str> {
        let items = Arc::into_raw(self);
        // SAFETY: This transmutation is safe, because `str` and `Str1` have the same
        //         representation (`Str1` is `repr(transparent)`). Moreover, the allocator only
        //         requires that the memory location and layout are the same when deallocating, so
        //         dropping the transmuted `Arc` is sound.
        unsafe { Arc::from_raw(items as *mut str) }
    }

    fn as_str1(&self) -> &Str1 {
        self.as_ref()
    }
}

pub type WeakStr1 = Weak<Str1>;

pub trait WeakStr1Ext: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Arc::downgrade(&Arc::<str>::from(""))`][`Arc::downgrade`].
    ///
    /// [`Arc::downgrade`]: alloc::sync::Arc::downgrade
    unsafe fn from_weak_str_unchecked(items: Weak<str>) -> Self;
}

impl WeakStr1Ext for WeakStr1 {
    unsafe fn from_weak_str_unchecked(items: Weak<str>) -> Self {
        let items = Weak::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input string is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `str` and
        //         `Str1` have the same representation (`Str1` is `repr(transparent)`). Moreover,
        //         the allocator only requires that the memory location and layout are the same
        //         when deallocating, so dropping the transmuted `Weak` is sound.
        unsafe { Weak::from_raw(items as *const Str1) }
    }
}

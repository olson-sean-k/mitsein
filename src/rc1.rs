//! Non-empty [reference-counted][`rc`] collections.
//!
//! [`rc`]: alloc::rc

#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::rc::{Rc, Weak};

use crate::array1::Array1;
use crate::borrow1::{CowSlice1, CowSlice1Ext as _, CowStr1, CowStr1Ext as _};
use crate::boxed1::{BoxedSlice1, BoxedSlice1Ext as _, BoxedStr1, BoxedStr1Ext as _};
use crate::iter1::{FromIterator1, IntoIterator1};
use crate::slice1::Slice1;
use crate::str1::Str1;
use crate::vec1::Vec1;
use crate::{EmptyError, MaybeEmpty};

pub type RcSlice1<T> = Rc<Slice1<T>>;

pub trait RcSlice1Ext<T>: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Rc::<T>::from([])`][`Rc::from`].
    ///
    /// [`Rc::from`]: alloc::rc::Rc::from
    unsafe fn from_rc_slice_unchecked(items: Rc<[T]>) -> Self;

    fn try_from_rc_slice(items: Rc<[T]>) -> Result<Self, EmptyError<Rc<[T]>>>;

    fn from_array1<const N: usize>(items: [T; N]) -> Self
    where
        [T; N]: Array1;

    fn from_boxed_slice1(items: BoxedSlice1<T>) -> Self;

    fn from_cow_slice1(items: CowSlice1<T>) -> Self
    where
        T: Clone;

    fn try_into_rc_array<const N: usize>(self) -> Result<Rc<[T; N]>, Self>;

    fn into_rc_slice(self) -> Rc<[T]>;

    fn as_slice1(&self) -> &Slice1<T>;
}

impl<T> RcSlice1Ext<T> for RcSlice1<T> {
    unsafe fn from_rc_slice_unchecked(items: Rc<[T]>) -> Self {
        let items = Rc::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input slice is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `[T]` and
        //         `Slice1<T>` have the same representation (`Slice1<T>` is `repr(transparent)`).
        //         Moreover, the allocator only requires that the memory location and layout are
        //         the same when deallocating, so dropping the transmuted `Rc` is sound.
        unsafe { Rc::from_raw(items as *const Slice1<T>) }
    }

    fn try_from_rc_slice(items: Rc<[T]>) -> Result<Self, EmptyError<Rc<[T]>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { RcSlice1::from_rc_slice_unchecked(items) }),
        }
    }

    fn from_array1<const N: usize>(items: [T; N]) -> Self
    where
        [T; N]: Array1,
    {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(Rc::from(items)) }
    }

    fn from_boxed_slice1(items: BoxedSlice1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(Rc::from(items.into_boxed_slice())) }
    }

    fn from_cow_slice1(items: CowSlice1<T>) -> Self
    where
        T: Clone,
    {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(Rc::from(items.into_cow_slice())) }
    }

    fn try_into_rc_array<const N: usize>(self) -> Result<Rc<[T; N]>, Self> {
        match self.into_rc_slice().try_into() {
            Ok(items) => Ok(items),
            // SAFETY: `self` and therefore `items` must be non-empty.
            Err(items) => Err(unsafe { RcSlice1::from_rc_slice_unchecked(items) }),
        }
    }

    fn into_rc_slice(self) -> Rc<[T]> {
        let items = Rc::into_raw(self);
        // SAFETY: This transmutation is safe, because `[T]` and `Slice1<T>` have the same
        //         representation (`Slice1<T>` is `repr(transparent)`). Moreover, the allocator
        //         only requires that the memory location and layout are the same when
        //         deallocating, so dropping the transmuted `Rc` is sound.
        unsafe { Rc::from_raw(items as *mut [T]) }
    }

    fn as_slice1(&self) -> &Slice1<T> {
        self.as_ref()
    }
}

impl<'a, T> From<&'a Slice1<T>> for RcSlice1<T>
where
    T: Clone,
{
    fn from(items: &'a Slice1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(Rc::from(items.as_slice())) }
    }
}

impl<T> From<Vec1<T>> for RcSlice1<T> {
    fn from(items: Vec1<T>) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(Rc::from(items.items)) }
    }
}

impl<T> FromIterator1<T> for RcSlice1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { RcSlice1::from_rc_slice_unchecked(items.into_iter().collect()) }
    }
}

pub type WeakSlice1<T> = Weak<Slice1<T>>;

pub trait WeakSlice1Ext<T>: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Rc::downgrade(&Rc::<T>::from([]))`][`Rc::downgrade`].
    ///
    /// [`Rc::downgrade`]: alloc::rc::Rc::downgrade
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

pub type RcStr1 = Rc<Str1>;

pub trait RcStr1Ext: Sized {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with
    /// [`Rc::<str>::from("")`][`Rc::from`].
    ///
    /// [`Rc::from`]: alloc::rc::Rc::from
    unsafe fn from_rc_str_unchecked(items: Rc<str>) -> Self;

    fn try_from_rc_str(items: Rc<str>) -> Result<Self, EmptyError<Rc<str>>>;

    fn from_boxed_str1(items: BoxedStr1) -> Self;

    fn from_cow_str1(items: CowStr1) -> Self;

    fn into_rc_str(self) -> Rc<str>;

    fn as_str1(&self) -> &Str1;
}

impl RcStr1Ext for RcStr1 {
    unsafe fn from_rc_str_unchecked(items: Rc<str>) -> Self {
        let items = Rc::into_raw(items);
        // SAFETY: Client code is responsible for asserting that the input string is non-empty (and
        //         so this function is unsafe). This transmutation is safe, because `str` and
        //         `Str1` have the same representation (`Str1` is `repr(transparent)`). Moreover,
        //         the allocator only requires that the memory location and layout are the same
        //         when deallocating, so dropping the transmuted `Rc` is sound.
        unsafe { Rc::from_raw(items as *const Str1) }
    }

    fn try_from_rc_str(items: Rc<str>) -> Result<Self, EmptyError<Rc<str>>> {
        match items.as_ref().cardinality() {
            None => Err(EmptyError::from_empty(items)),
            // SAFETY: `items` is non-empty.
            _ => Ok(unsafe { RcStr1::from_rc_str_unchecked(items) }),
        }
    }

    fn from_boxed_str1(items: BoxedStr1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcStr1::from_rc_str_unchecked(Rc::from(items.into_boxed_str())) }
    }

    fn from_cow_str1(items: CowStr1) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { RcStr1::from_rc_str_unchecked(Rc::from(items.into_cow_str())) }
    }

    fn into_rc_str(self) -> Rc<str> {
        let items = Rc::into_raw(self);
        // SAFETY: This transmutation is safe, because `str` and `Str1` have the same
        //         representation (`Str1` is `repr(transparent)`). Moreover, the allocator only
        //         requires that the memory location and layout are the same when deallocating, so
        //         dropping the transmuted `Rc` is sound.
        unsafe { Rc::from_raw(items as *mut str) }
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
    /// [`Rc::downgrade(&Rc::<str>::from(""))`][`Rc::downgrade`].
    ///
    /// [`Rc::downgrade`]: alloc::rc::Rc::downgrade
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

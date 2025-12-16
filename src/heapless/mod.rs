//! Non-empty [`heapless`] types.
//!
//! ## Collections
//!
//! This module provides the following non-empty [`heapless`] collections:
//!
//! - [`Vec1`][`vec1`]
//!
//! [`heapless`]: [`::heapless`]

#![cfg(feature = "heapless")]
#![cfg_attr(docsrs, doc(cfg(feature = "heapless")))]

pub mod vec1;

// This macro mirrors `crate::impl_partial_eq_for_non_empty`, but supports storage type parameters
// for non-empty `heapless` types. This macro can replace the other, but it very complex and that
// complexity is specific to `heapless`, so it is relegated here instead.
macro_rules! impl_partial_eq_for_non_empty {
    (
        [$(for $ritem:ident $(,$rstorage:ident as $rtrait:ident)? $(,)?)? in $rhs:ty]
        ==
        [$(for $litem:ident $(,$lstorage:ident as $ltrait:ident)? $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem, $($rstorage,)?)? $($litem $(, $lstorage)?)?>
        ::core::cmp::PartialEq<$rhs> for $lhs
        where
            $(
                $litem: ::core::cmp::PartialEq<$ritem>,
                $($lstorage: ?Sized + $ltrait<$litem>,)?
            )?
            $(
                $($rstorage: ?Sized + $rtrait<$ritem>,)?
            )?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, &rhs.items)
            }
        }
    };
    (
        [$(for $ritem:ident $(,const $rn:ident: usize)? $(,)?)? in $rhs:ty]
        <=
        [$(for $litem:ident $(,$lstorage:ident as $ltrait:ident)? $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem,)? $($litem $(, $lstorage)?,)? $($(const $rn: usize)?)?>
        ::core::cmp::PartialEq<$rhs> for $lhs
        where
            $(
                $litem: ::core::cmp::PartialEq<$ritem>,
                $($lstorage: ?Sized + $ltrait<$litem>,)?
            )?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, rhs)
            }
        }
    };
    (
        [$(for $ritem:ident $(,$rstorage:ident as $rtrait:ident)? $(,)?)? in $rhs:ty]
        =>
        [$(for $litem:ident $(,const $ln:ident: usize)? $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem, $($rstorage,)?)? $($litem,)? $($(const $ln: usize)?)?>
        ::core::cmp::PartialEq<$rhs> for $lhs
        where
            $($litem: ::core::cmp::PartialEq<$ritem>,)?
            $(
                $($rstorage: ?Sized + $rtrait<$ritem>,)?
            )?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(self, &rhs.items)
            }
        }
    };
}
pub(crate) use impl_partial_eq_for_non_empty;

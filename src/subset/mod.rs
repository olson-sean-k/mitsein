//! Views into subsets of non-empty collections.

// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound.
//         Subset APIs interact directly with item removal and so bugs here may break the non-empty
//         guarantee. In particular, features of internal range types like contact and projection
//         must be correct.

mod keyed;
mod ranged;

#[cfg(feature = "alloc")]
pub use crate::subset::keyed::*;
#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
pub use crate::subset::{
    ranged::range::{OutOfBoundsError, RangeError, UnorderedError},
    ranged::*,
};

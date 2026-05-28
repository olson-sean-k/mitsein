//! Views into subsets of non-empty collections.

// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound.
//         Subset APIs interact directly with item removal and so bugs here may break the non-empty
//         guarantee. In particular, range types, intersection, and projection must be correct.

mod ordered;
mod unordered;

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
pub use crate::subset::ordered::*;
#[cfg(feature = "alloc")]
pub use crate::subset::unordered::*;

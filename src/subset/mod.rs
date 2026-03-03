//! Views into subsets of non-empty collections.
//!
//! This module provides types that isolate strict subsets of [`NonEmpty`] collections. These view
//! types support mass-removal operations like `clear` and `drain`. By first isolating a strict
//! subset, removals can be performed against a subset without emptying the source collection.
//! Subsets do not support _additive_ writes like `push` and `insert`.
//!
//! # Ranges
//!
//! [`OnlyRangeSubset`] isolates a subset of an ordered [`NonEmpty`] collection covered by a range.
//! Such a subset can be constructed explicitly from [standard range types][`core::range`] or
//! nominally from predefined ranges (namely a tail or reverse-tail).
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, -3];
//!
//! // Construct an explicit subset over the range `1..` in `xs`. This is fallible.
//! let xss = xs.only(1..).unwrap();
//! assert_eq!(xss.as_slice(), &[1, -3]);
//!
//! // Construct a nominal subset over the tail of `xs`. This is infallible.
//! let xss = xs.tail();
//! assert_eq!(xss.as_slice(), &[1, -3]);
#![doc = "```"]
//!
//! Most [`OnlyRangeSubset`] types can be further subdivided after construction.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, -3];
//!
//! // Clear the reverse-tail of the tail of `xs`. This first isolates `[1, -3]` and then `[1]`.
//! xs.tail().rtail().clear();
//! assert_eq!(xs.as_slice(), &[0, -3]);
#![doc = "```"]
//!
//! Ranges work both positionally (by index) or relationally (by item). For example, [`Vec1`] orders
//! its items positionally while [`BTreeSet1`] orders its items relationally via the [`Ord`]
//! implementation.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::btree_set1::BTreeSet1;
//! use mitsein::prelude::*;
//!
//! let mut xs = BTreeSet1::from_iter1([-7i64, -3, 0i64, 1, 9]);
//!
//! xs.only(-3..=1).unwrap().clear();
//! assert_eq!(xs, BTreeSet1::from_iter1([-7, 9]));
#![doc = "```"]
//!
//! # Keys
//!
//! TBD.
//!
//! [`BTreeSet1`]: crate::btree_set1::BTreeSet1
//! [`NonEmpty`]: crate::NonEmpty
//! [`Vec1`]: crate::vec1::Vec1

// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound.
//         Subset APIs interact directly with item removal and so bugs here may break the non-empty
//         guarantee. In particular, range types, intersection, and projection must be correct.

mod ordered;
mod unordered;

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
pub use crate::subset::ordered::*;
#[cfg(feature = "alloc")]
pub use crate::subset::unordered::*;

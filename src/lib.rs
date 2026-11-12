//! Mitsein provides strongly typed APIs for non-empty collections, slices, iterators, and more.
//! Expressing the non-empty guarantee in the type system produces more ergonomic, less error-prone,
//! and better performing APIs when modeling systems in which emptiness is not a valid state.
//!
//! Many problem domains fundamentally require a representation of one or more items. However, code
//! written against standard types in these situations must accomodate emptiness where it does not
//! belong. This has some unfortunate consequences:
//!
//! - Emptiness is a persistent edge case that can easily and subtly introduce bugs.
//! - Checking for emptiness introduces branches and panics that aren't strictly necessary.
//! - APIs must aggressively validate data and expose tangential errors regarding emptiness.
//!
//! Non-empty types resolve these issues by encapsulating the representation of one or more items.
//! Consider the following function written against the standard [slice][`type@slice`] type:
#![doc = ""]
#![cfg_attr(feature = "std", doc = "```rust")]
#![cfg_attr(not(feature = "std"), doc = "```rust,ignore")]
//! pub fn print_route_endpoints(waypoints: &[String]) -> Result<(), ()> {
//!     let first = waypoints.first().ok_or(())?;
//!     let last = waypoints.last().ok_or(())?;
//!     println!("{first} -> {last}");
//!     Ok(())
//! }
#![doc = "```"]
//!
//! Note the calls to `ok_or` and that the `print_route_endpoints` function is fallible. Because a
//! [slice][`type@slice1] may be empty, the `first` and `last` functions must return an [`Option`]
//! and code must handle this as an error condition here. If a non-empty [`Slice1`] is used instead,
//! then the non-empty requirement is expressed in the API and this error condition no longer exists
//! in the function:
#![doc = ""]
#![cfg_attr(feature = "std", doc = "```rust")]
#![cfg_attr(not(feature = "std"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! pub fn print_route_endpoints(waypoints: &Slice1<String>) {
//!     let first = waypoints.first();
//!     let last = waypoints.last();
//!     println!("{first} -> {last}");
//! }
#![doc = "```"]
//!
//! Non-empty types are also useful as fields in data structures. See the [Examples](#examples)
//! section for more.
//!
//! # Nomenclature
//!
//! Non-empty APIs are typically named by appending `1` to the names of their maybe-empty
//! counterparts. For example, the non-empty counterpart to [`Vec`] is named [`Vec1`] and its
//! non-empty counterpart to `iter` is named [`iter1`][`Vec1::iter1`]. The term "non-empty" is also
//! used where "1" is either not possible or especially unclear in context, such as
//! [`or_non_empty`][`ThenIterator1::or_non_empty`].
//!
//! Non-empty types have a cardinality that describes the quantity of items that they contain: one
//! or many. Here, "many" means more than one item. This concept is represented by the
//! [`Cardinality`] type and is featured heavily in APIs that may remove a single item from a
//! non-empty collection. These APIs typically have an `_if_many` suffix in their name that
//! indicates an important predicate: an item cannot be taken out of a non-empty collection if its
//! cardinality is one. For example, [`Vec1::pop_if_many`] is the counterpart to [`Vec::pop`] and
//! returns a [`Cardinality`] type. The [`One`] and [`Many`] variants of [`Cardinality`] are part of
//! the [prelude][`prelude`].
//!
//! # Structure
//!
//! Modules in this crate reflect corresponding modules in [`core`], [`alloc`], and [`std`]. For
//! example, the `alloc::vec` module has a non-empty counterpart in `mitsein::vec1`. Similarly,
//! [`str`] is exported in `core::str` and its non-empty counterpart [`Str1`] is exported in
//! `mitsein::str1`.
//!
//! Unlike [`alloc`] however, there is no intermediate `collections` module in Mitsein. Instead, all
//! collection modules are exported directly in the crate root (or an integration module; see
//! [`heapless`]). For example, [`BTreeSet`] is exported in `alloc::collections::btree_set`, but its
//! non-empty counterpart [`BTreeSet1`] is exported more directly in `mitsein::btree_set1`.
//!
//! Some generic types are reused extensively, in particular [`NonEmpty`] and [`Cardinality`]. Each
//! reuse of these types features a dedicated type alias. For example, the non-empty [`Vec1`] type
//! is an alias for `NonEmpty<Vec<_>>`. Because of this reuse, the API documentation for these types
//! can be extremely noisy. **Refer to the documentation for their more specific type aliases
//! instead to see the relevant APIs.**
//!
//! # Non-Empty Types
//!
//! Non-empty types present APIs that somewhat differ from their maybe-empty counterparts and
//! reflect the non-empty guarantee. Just like standard types, some essential non-empty types are
//! included in the [prelude][`prelude`], such as [`Vec1`] and the [`vec1!`] macro.
//!
//! ## Collections
//!
//! This crate provides the following non-empty collections and supporting APIs:
//!
//! - [`ArrayVec1`][`array_vec1`]
//! - [`BTreeMap1`][`btree_map1`]
//! - [`BTreeSet1`][`btree_set1`]
//! - [`HashSet1`][`hash_set1`]
//! - [`IndexMap1`][`index_map1`]
//! - [`IndexSet1`][`index_set1`]
//! - [`SmallVec1`][`small_vec1`]
//! - [`String1`][`string1`]
//! - [`Vec1`][`mod@vec1`]
//! - [`VecDeque1`][`vec_deque1`]
//! - [`heapless`](crate::heapless#collections)
//!
//! Non-empty collections are represented with the generic [`NonEmpty`] type. These types are
//! exported as type aliases in their respective modules.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = Vec1::from_head_and_tail(0i64, [1, 2, 3]);
//! while let Many(_) = xs.pop_if_many() {}
//!
//! assert_eq!(xs.as_slice(), &[0]);
#![doc = "```"]
//!
//! ## Slices
//!
//! Like collections, non-empty slices are represented with the generic [`NonEmpty`] type and the
//! [`Slice1`] and [`Str1`] type aliases. These types are unsized and so are accessed via references
//! just like standard slices.
//!
//! ```rust
//! use mitsein::prelude::*;
//!
//! let xs = slice1![0i64, 1, 2, 3];
//! let sum = xs.first() + xs.last(); // This is infallible.
//!
//! assert_eq!(sum, 3);
//! ```
//!
//! See the [`slice1`][`mod@slice1`] and [`str1`][`mod@str1`] modules.
//!
//! ## Smart Pointers
//!
//! This crate provides the following non-empty smart pointers for slices and strings:
//!
//! - [`ArcSlice1`][`ArcSlice1Ext`]
//! - [`ArcStr1`][`ArcStr1Ext`]
//! - [`BoxedSlice1`][`BoxedSlice1Ext`]
//! - [`BoxedStr1`][`BoxedStr1Ext`]
//! - [`CowSlice1`][`CowSlice1Ext`]
//! - [`CowStr1`][`CowStr1Ext`]
//! - [`RcSlice1`][`RcSlice1Ext`]
//! - [`RcStr1`][`RcStr1Ext`]
//!
//! Each of these type aliases has an accompanying extension trait for operations and conversions
//! that take advantage of the non-empty guarantee. For example, [`ArcSlice1Ext`] provides APIs for
//! non-empty slices in an [`Arc`]. Some collection types like [`Vec1`] support conversions from and
//! into these smart pointers.
//!
//! ## Iterators
//!
//! Non-empty iterators are provided by the generic [`Iterator1`] type and directly support
//! combinator functions that cannot discard all items (e.g., [`map`][`Iterator1::map`]). Non-empty
//! collections and views naturally support iteration, collection, etc. via [`Iterator1`].
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let xs = vec1![0i32, 1, 2];
//! let ys: Vec1<_> = xs.iter1().copied().map(|x| x + 1).collect1();
//!
//! assert_eq!(ys.as_slice(), &[1, 2, 3]);
#![doc = "```"]
//!
//! Non-empty iterators have supporting traits that bridge between maybe-empty [`Iterator`] types
//! and non-empty [`Iterator1`] types.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let xs = vec![0i32, 1, 2];
//! // `or_non_empty` either constructs an `Iterator1` from `self` if it is non-empty or instead
//! // constructs an `Iterator1` from its non-empty parameter (`[42]`).
//! let xs: Vec1<_> = xs.into_iter().skip(3).or_non_empty([42]).collect1();
//!
//! let ys = Vec::new();
//! // `extend1` converts a maybe-empty type into a non-empty type.
//! let ys: Vec1<_> = ys.extend1([0i32]);
//!
//! assert_eq!(xs.as_slice(), &[42]);
//! assert_eq!(ys.as_slice(), &[0]);
#![doc = "```"]
//!
//! When the `rayon` feature is enabled, the [`ParallelIterator1`] type implements parallel and
//! non-empty iterators.
//!
//! See the [`iter1`] module.
//!
//! ## Arrays
//!
//! Because primitive arrays must contain initialized items at capacity in safe code, the only
//! empty array types are `[_; 0]`. The [`Array1`] trait is implemented for arrays with a non-zero
//! cardinality and provides conversions and operations that take advantage of the non-empty
//! guarantee of such arrays.
//!
//! ```rust
//! use mitsein::prelude::*;
//!
//! let mut xs = [0i64, 1, 2, 3];
//! let x = xs.as_mut_slice1().first_mut(); // This is infallible.
//! *x = 4;
//!
//! assert_eq!(xs.as_slice(), &[4, 1, 2, 3]);
//! ```
//!
//! At time of writing, it is not possible to implement [`Array1`] for any and all non-empty array
//! types, so this trait is only implemented for arrays with one to 64 items.
//!
//! See the [`array1`] module.
//!
//! ## Ranges
//!
//! This crate provides the following non-empty bounded range types:
//!
//! - [`Range1`]
//! - [`RangeInclusive1`]
//!
//! Like collections, non-empty ranges are represented with the generic [`NonEmpty`] type. Non-empty
//! range types implement both the [`IntoIterator`] and [`IntoIterator1`] traits. Non-empty ranges
//! can be used to index slices and types that can be trivially coerced or converted to slices. The
//! output of slicing by a non-empty range is a [`Slice1`].
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//! use mitsein::range1::range1;
//!
//! let mut xs = vec![0i64, 1, 2, 3];
//! let ys = &mut xs[range1![3..4]]; // `ys` is a `Slice1`.
//! let y = ys.first_mut(); // This is infallible.
//! *y = 42;
//!
//! assert_eq!(xs.as_slice(), &[0, 1, 2, 42]);
#![doc = "```"]
//!
//! See the [`range1`][`mod@crate::range1`] module.
//!
//! # Subsets
//!
//! Subset APIs construct views over a strict subset of items in a non-empty collection. A subset
//! can freely mutate and remove the items in its collection by first asserting that one or more
//! items in the collection are not a part of the subset and are therefore never affected. This
//! allows subsets to present more typical removal APIs that need not repeatedly check that a
//! removal does not violate the non-empty guarantee. Subsets are one of the most efficient ways to
//! mass-remove and drain items from non-empty collections.
//!
//! Subsets can be constructed from ranges over ordered collections. These ranges can be explicit
//! (e.g., from standard range types) or nominal, such as the tail of a collection.
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, 2, 3, 4];
//! xs.tail().clear(); // Efficiently clears the tail of `xs`.
//!
//! assert_eq!(xs.as_slice(), &[0]);
#![doc = "```"]
//!
//! Subsets can also be constructed from keys. Unordered collections only support this kind of
//! subset.
#![doc = ""]
#![cfg_attr(feature = "std", doc = "```rust")]
#![cfg_attr(not(feature = "std"), doc = "```rust,ignore")]
//! use mitsein::hash_set1::HashSet1;
//! use mitsein::prelude::*;
//!
//! let mut xs = HashSet1::<_>::from_iter1([0i64, 1, 2, 3, 4]);
//! xs.except(&0).unwrap().clear(); // Efficiently clears all but `0` from `xs`.
//!
//! assert_eq!(xs, HashSet1::from_one(0));
#![doc = "```"]
//!
//! See the [`subset`] module.
//!
//! # Examples
//!
//! The following examples demonstrate how and why non-empty types can be used in situations where
//! emptiness is not a valid state.
//!
//! ## Aggregating an Iterator
//!
//! Aggregation generally requires one or more items. Consider the following operation written
//! against the standard [`Vec`] type:
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! let mut xs = vec![0i64, 1, -3, 2];
//! let absmax = xs.into_iter().map(i64::abs).max().expect("no xs!");
//!
//! assert_eq!(3, absmax);
#![doc = "```"]
//!
//! Note the `expect` call. Because a [`Vec`] and its iterators may be empty, the `max` function
//! must return an [`Option`] here. Code must handle this as an error condition if this is
//! unexpected. If a non-empty [`Vec1`] is used instead, then this error condition disappears:
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! let mut xs = vec1![0i64, 1, -3, 2];
//! let absmax = xs.into_iter1().map(i64::abs).max();
//!
//! assert_eq!(3, absmax);
#![doc = "```"]
//!
//! ## Computing a Bounding Box
//!
//! Consider a representation of an [axis-aligned bounding
//! box](https://en.wikipedia.org/wiki/Bounding_volume) `Aabb`. An `Aabb` can be constructed from
//! one or more points in space by computing the minimum and maximum points. An implementation may
//! look something like this:
#![doc = ""]
#![cfg_attr(all(feature = "alloc", feature = "itertools"), doc = "```rust")]
#![cfg_attr(
    not(all(feature = "alloc", feature = "itertools")),
    doc = "```rust,ignore"
)]
//! // This provides `minmax`, which computes both the minimum and maximum in a single pass.
//! use itertools::Itertools;
//!
//!# pub type Point = [f32; 3];
//!# pub struct Aabb;
//!#
//! impl Aabb {
//!     pub fn from_points<I>(points: I) -> Option<Self>
//!     where
//!         I: IntoIterator<Item = Point>,
//!     {
//!         // `into_option` returns `None` if there are no items.
//!         points.into_iter().minmax().into_option().map(|minmax| {
//!#             let _ = minmax;
//!             Aabb { /* ... */ }
//!         })
//!     }
//! }
#![doc = "```"]
//!
//! In the above example, `Aabb::from_points` returns `Option<Self>`. This is unfortunate, because
//! it expresses a cardinality error concerning the input `points` as an error constructing an
//! `Aabb`. Note too that `minmax` and `into_option` must examine (indirectly or otherwise) the
//! cardinality of `points` regardless of whether or not the caller has already done so.
//!
//! An implementation using non-empty types may instead look something like this:
#![doc = ""]
#![cfg_attr(all(feature = "alloc", feature = "itertools"), doc = "```rust")]
#![cfg_attr(
    not(all(feature = "alloc", feature = "itertools")),
    doc = "```rust,ignore"
)]
//! use mitsein::prelude::*;
//!
//!# pub type Point = [f32; 3];
//!# pub struct Aabb;
//!#
//! impl Aabb {
//!     pub fn from_points<I>(points: I) -> Self
//!     where
//!         // This is a bound on `IntoIterator1` (note the `1`), the non-empty counterpart of
//!         // `IntoIterator`.
//!         I: IntoIterator1<Item = Point>,
//!     {
//!         // This non-empty `minmax` function is an integration with `itertools`.
//!         let minmax = points.into_iter1().minmax();
//!#         let _ = minmax;
//!         Aabb { /* ... */ }
//!     }
//! }
#![doc = "```"]
//!
//! The bound on [`IntoIterator1`] captures the requirement for one or more points in the
//! `from_points` API. This relieves this function from this tangential error condition and avoids
//! redundant checks for emptiness.
//!
//! ## Representing an AST
//!
//! Consider an [abstract syntax tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) for
//! a simple programming language. The AST must model literals (like `5`) and blocks (like
//! `{ ... }`). Crucially, blocks cannot be empty.
//!
//! A minimal representation using standard maybe-empty types may look like this:
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! pub enum AstNode {
//!     Literal(i64),
//!     Block(Vec<AstNode>),
//! }
#![doc = "```"]
//!
//! This representation isn't quite right though: as written, a block consists of zero or more AST
//! nodes, but a block cannot be empty. This pushes validation and error handling into code where it
//! does not belong, which can easily lead to redundant checks and logic errors.
//!
//! Non-empty types strictly and correctly represent the intended relationship here:
#![doc = ""]
#![cfg_attr(feature = "alloc", doc = "```rust")]
#![cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
//! use mitsein::prelude::*;
//!
//! pub enum AstNode {
//!     Literal(i64),
//!     // `Vec1` guarantees that the block consists of at least one `AstNode`.
//!     Block(Vec1<AstNode>),
//! }
#![doc = "```"]
//!
//! # Integrations and Feature Flags
//!
//! Mitsein supports `no_std` environments and provides feature flags for integrating as needed
//! with [`alloc`] and [`std`]. By default, the `std` feature is enabled for complete support of
//! the standard library.
//!
//! The following table summarizes supported feature flags and integrations.
//!
//! | Feature     | Also Enables | Default | Crate         | Description                                                    |
//! |-------------|--------------|---------|---------------|----------------------------------------------------------------|
//! | `alloc`     |              | Yes     | `alloc`       | Non-empty collections that allocate, like `Vec1`.              |
//! | `arbitrary` | `std`        | No      | [`arbitrary`] | Construction of arbitrary non-empty collections.               |
//! | `arrayvec`  |              | No      | [`arrayvec`]  | Non-empty implementations of [`arrayvec`] types.               |
//! | `either`    |              | No      | [`either`]    | Non-empty iterator implementation for `Either`.                |
//! | `heapless`  |              | No      | [`heapless`]  | Non-empty implementations of [`heapless`] types.               |
//! | `indexmap`  | `alloc`      | No      | [`indexmap`]  | Non-empty implementations of [`indexmap`] types.               |
//! | `itertools` | `either`     | No      | [`itertools`] | Combinators from [`itertools`] for `Iterator1`.                |
//! | `rayon`     | `std`        | No      | [`rayon`]     | Parallel operations for non-empty types.                       |
//! | `schemars`  | `alloc`      | No      | [`schemars`]  | JSON schema generation for non-empty types.                    |
//! | `serde`     |              | No      | [`serde`]     | De/serialization of non-empty collections with [`serde`].      |
//! | `smallvec`  | `alloc`      | No      | [`smallvec`]  | Non-empty implementations of [`smallvec`] types.               |
//! | `std`       | `alloc`      | Yes     | `std`         | Non-empty hashing collections and integrations with `std::io`. |
//!
//! # Misleading API Documentation for Dereferenced Methods
//!
//! At time of writing, `rustdoc` ignores type parameters in the "Methods from `Deref<Target = _>`"
//! section. For types that implement `Deref<Target = NonEmpty<_>>`, **the API documentation may be
//! misleading and list all methods of [`NonEmpty`] rather than only the methods available for the
//! given type parameter.**
//!
//! This is a problem for types that dereference to a [`NonEmpty`] type, such as [`Vec1`]. Because
//! [`Vec1`] dereferences to [`Slice1`] (a.k.a. `NonEmpty<[_]>`), its dereferenced method
//! documentation may incorrectly list all [`NonEmpty`] methods for any and all type parameters
//! (when only methods for **slice type parameters** are actually available).
//!
//! See [this `rustdoc` bug](https://github.com/rust-lang/rust/issues/24686).
//!
//! [`Arc`]: alloc::sync::Arc
//! [`ArcSlice1Ext`]: crate::sync1::ArcSlice1Ext
//! [`ArcStr1Ext`]: crate::sync1::ArcStr1Ext
//! [`Array1`]: crate::array1::Array1
//! [`arbitrary`]: https://crates.io/crates/arbitrary
//! [`arrayvec`]: https://crates.io/crates/arrayvec
//! [`BoxedSlice1Ext`]: crate::boxed1::BoxedSlice1Ext
//! [`BoxedStr1Ext`]: crate::boxed1::BoxedStr1Ext
//! [`BTreeSet`]: alloc::collections::btree_set::BTreeSet
//! [`BTreeSet1`]: crate::btree_set1::BTreeSet1
//! [`CowSlice1Ext`]: crate::borrow1::CowSlice1Ext
//! [`CowStr1Ext`]: crate::borrow1::CowStr1Ext
//! [`either`]: https://crates.io/crates/either
//! [`heapless`]: https://crates.io/crates/heapless
//! [`indexmap`]: https://crates.io/crates/indexmap
//! [`IntoIterator1`]: crate::iter1::IntoIterator1
//! [`Iterator1`]: crate::iter1::Iterator1
//! [`Iterator1::map`]: crate::iter1::Iterator1::map
//! [`itertools`]: https://crates.io/crates/itertools
//! [`ParallelIterator1`]: crate::iter1::ParallelIterator1
//! [`Range1`]: crate::range1::Range1
//! [`RangeInclusive1`]: crate::range1::RangeInclusive1
//! [`rayon`]: https://crates.io/crates/rayon
//! [`RcSlice1Ext`]: crate::rc1::RcSlice1Ext
//! [`RcStr1Ext`]: crate::rc1::RcStr1Ext
//! [`schemars`]: https://crates.io/crates/schemars
//! [`serde`]: https://crates.io/crates/serde
//! [`Slice1`]: crate::slice1::Slice1
//! [`smallvec`]: https://crates.io/crates/smallvec
//! [`Str1`]: crate::str1::Str1
//! [`ThenIterator1::or_non_empty`]: crate::iter1::ThenIterator1::or_non_empty
//! [`Vec`]: alloc::vec::Vec
//! [`Vec::pop`]: alloc::vec::Vec::pop
//! [`Vec1`]: crate::vec1::Vec1
//! [`Vec1::iter1`]: crate::vec1::Vec1::iter1
//! [`Vec1::pop_if_many`]: crate::vec1::Vec1::pop_if_many

// TODO: At time of writing, it is not possible to specify or enable features required for
//       documentation examples without explicitly applying `doc` attributes. These attributes harm
//       the legibility of non-rendered documentation. Migrate this to a cleaner mechanism when
//       possible.

// SAFETY: This crate implements non-empty collections, slices, and iterators. This non-empty
//         invariant is critical to memory safety and soundness, because these implementations use
//         unsafe code to omit branches and checks that are unnecessary if and only if the
//         invariant holds.
//
//         The non-empty invariant is pervasive: code within `unsafe` blocks tends to be
//         uninteresting and is very unlikely to be the source of memory safety bugs. It is trivial
//         and more likely for safe code to break this invariant! Sometimes maintaining this
//         invariant can be subtle, such as interactions with `drain` and `Ord`.
//
//         Most unsafe code in this crate falls into the following basic categories:
//
//           1. The code implements a `NonEmpty` type and therefore assumes that `self` is indeed
//              non-empty.
//           2. The code converts into or from another `NonEmpty` type and so assumes that the
//              input is non-empty.
//           3. The code explicitly checks the invariant or constructs non-empty data.
//
//         Unsafe code in categories (1) and (2) relies on the implementation and public APIs of
//         `NonEmpty` types. Unsafe code in category (3) relies on much more local checks or
//         enforcement of the non-empty invariant.
//
//         This crate must trust undocumented or unpromised behavior of foreign code. For example,
//         `Iterator1` combinators assume that implementations in `core` and `itertools` function
//         in a particular way. If `Map` were ever to discard an item, then `Iterator1::map` would
//         be unsound, for example. As another example, `SwapDrainOnlyRangeSubset` relies on details
//         of the `Vec::drain` implementation. For the overwhelming majority of this code, a change
//         in behavior that this crate relies upon for memory safety would be a major bug in the
//         upstream package and is very unlikely to occur.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(
    clippy::cast_lossless,
    clippy::checked_conversions,
    clippy::cloned_instead_of_copied,
    clippy::explicit_into_iter_loop,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::if_not_else,
    clippy::manual_ok_or,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_else,
    clippy::unreadable_literal,
    clippy::unused_self
)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod safety;
mod schemars;
mod serde;

pub mod array1;
pub mod array_vec1;
pub mod borrow1;
pub mod boxed1;
pub mod btree_map1;
pub mod btree_set1;
pub mod cmp;
pub mod hash;
pub mod hash_set1;
pub mod heapless;
pub mod index_map1;
pub mod index_set1;
pub mod iter1;
pub mod range1;
pub mod rc1;
pub mod slice1;
pub mod small_vec1;
pub mod str1;
pub mod string1;
pub mod subset;
pub mod sync1;
pub mod vec1;
pub mod vec_deque1;

mod sealed {
    use crate::Cardinality;

    /// A collection or slice type that may be empty and can be adapted by [`NonEmpty`].
    ///
    /// # Safety
    ///
    /// The implementation of this trait determines whether or not a collection or view is empty.
    /// This query is used to construct non-empty types, and an inconsistent implementation is
    /// unsound. In particular, it is unsound for [`MaybeEmpty::cardinality`] implementations to
    /// return [`Some`] when `Self` is empty.
    ///
    /// [`NonEmpty`]: crate::NonEmpty
    pub unsafe trait MaybeEmpty {
        /// Gets the [cardinality][`Cardinality`] of the collection.
        ///
        /// The cardinality of a maybe-empty collection (e.g., [`Vec`]) describes the number of
        /// items that it contains and is one of three things:
        ///
        /// - Zero (represented as [`None`]).
        /// - One (represented as `Some(Cardinality::One(_))`).
        /// - Many (represented as `Some(Cardinality::Many(_))`).
        ///
        /// [`Vec`]: alloc::vec::Vec
        fn cardinality(&self) -> Option<Cardinality<(), ()>>;
    }

    unsafe impl<T> MaybeEmpty for &'_ T
    where
        T: MaybeEmpty + ?Sized,
    {
        fn cardinality(&self) -> Option<Cardinality<(), ()>> {
            <T as MaybeEmpty>::cardinality(*self)
        }
    }

    unsafe impl<T> MaybeEmpty for &'_ mut T
    where
        T: MaybeEmpty + ?Sized,
    {
        fn cardinality(&self) -> Option<Cardinality<(), ()>> {
            <T as MaybeEmpty>::cardinality(*self)
        }
    }
}
use crate::sealed::*;

pub mod prelude {
    //! Re-exports of recommended APIs and extension traits for glob imports.

    pub use crate::array1::Array1;
    #[cfg(feature = "indexmap")]
    pub use crate::index_map1::OrOnlyEntryExt as _;
    #[cfg(feature = "either")]
    pub use crate::iter1::EitherExt as _;
    #[cfg(all(feature = "itertools", feature = "std"))]
    pub use crate::iter1::GroupingMapExt as _;
    #[cfg(feature = "itertools")]
    pub use crate::iter1::ItertoolsExt as _;
    pub use crate::iter1::{
        Drain1, Extend1, FromIterator1, IntoIterator1, IteratorExt as _, ThenIterator1 as _,
    };
    #[cfg(feature = "rayon")]
    pub use crate::iter1::{FromParallelIterator1, IntoParallelIterator1};
    pub use crate::slice1::{Slice1, SliceExt as _, slice1};
    pub use crate::str1::Str1;
    #[cfg(all(feature = "alloc", target_has_atomic = "ptr"))]
    pub use crate::sync1::{
        ArcSlice1Ext as _, ArcStr1Ext as _, WeakSlice1Ext as _, WeakStr1Ext as _,
    };
    pub use crate::{Many, One};
    #[cfg(feature = "alloc")]
    pub use {
        crate::borrow1::{CowSlice1Ext as _, CowStr1Ext as _},
        crate::boxed1::{BoxedSlice1Ext as _, BoxedStr1Ext as _},
        crate::btree_map1::OrOnlyEntryExt as _,
        crate::rc1::{RcSlice1Ext as _, RcStr1Ext as _, WeakSlice1Ext as _, WeakStr1Ext as _},
        crate::string1::String1,
        crate::vec1::{Vec1, vec1},
    };
}

#[cfg(feature = "alloc")]
use alloc::borrow::ToOwned;
use core::cmp::Ordering;
use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
use core::mem;
use core::num::NonZeroUsize;

pub use Cardinality::{Many, One};

const EMPTY_ERROR_MESSAGE: &str = "failed to construct non-empty value: no items";

/// Extension methods for [`MaybeEmpty`] types.
trait MaybeEmptyExt: MaybeEmpty {
    /// Applies a mapping function from `Self` to `T` if the collection is non-empty.
    ///
    /// # Errors
    ///
    /// Returns an [`EmptyError`] if `Self` is empty.
    fn map_non_empty<T, F>(self, f: F) -> Result<T, EmptyError<Self>>
    where
        Self: Sized,
        F: FnOnce(Self) -> T;

    /// Returns `true` if the collection is empty.
    fn is_empty(&self) -> bool;
}

// This blanket implementation of an extension trait prevents different behaviors of
// `map_non_empty` for different `MaybeEmpty` types (as opposed to a default implementation in
// `MaybeEmpty`, for example). This is important for memory safety, because `FromMaybeEmpty` relies
// on this behavior with unsafe code.
impl<T> MaybeEmptyExt for T
where
    T: MaybeEmpty,
{
    fn map_non_empty<U, F>(self, f: F) -> Result<U, EmptyError<Self>>
    where
        Self: Sized,
        F: FnOnce(Self) -> U,
    {
        if self.is_empty() {
            Err(EmptyError(self))
        }
        else {
            Ok(f(self))
        }
    }

    fn is_empty(&self) -> bool {
        self.cardinality().is_none()
    }
}

/// Extension methods for [`NonZero`] types.
///
/// [`NonZero`]: core::num::NonZero
trait NonZeroExt<T> {
    /// Constructs a non-zero type by clamping the inner value to the range `1..`, effectively
    /// mapping zero to one.
    fn clamped(n: T) -> Self;
}

impl NonZeroExt<usize> for NonZeroUsize {
    fn clamped(n: usize) -> Self {
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }
}

/// Conversions from maybe-empty types to non-empty types.
trait FromMaybeEmpty<T>: Sized
where
    T: MaybeEmpty + Sized,
{
    /// Converts `items` into the non-empty type `Self` if `items` is non-empty.
    ///
    /// # Errors
    ///
    /// Returns an [`EmptyError`] if `items` is empty.
    fn try_from_maybe_empty(items: T) -> Result<Self, EmptyError<T>> {
        items.map_non_empty(|items| {
            // SAFETY: The `map_non_empty` function only executes this code if `items` is
            //         non-empty.
            unsafe { Self::from_maybe_empty_unchecked(items) }
        })
    }

    /// Converts `items` into the non-empty type `Self` **without checking if `items` is
    /// non-empty**.
    ///
    /// # Safety
    ///
    /// `items` must be non-empty. See [`MaybeEmpty::cardinality`].
    unsafe fn from_maybe_empty_unchecked(items: T) -> Self;
}

/// An error in which a non-empty value is expected but an empty value is observed.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct EmptyError<T>(pub T);

impl<T> EmptyError<T> {
    /// Converts the error into the empty value.
    pub fn into_empty(self) -> T {
        self.0
    }

    fn map<U, F>(self, f: F) -> EmptyError<U>
    where
        F: FnOnce(T) -> U,
    {
        EmptyError(f(self.0))
    }
}

impl<T> EmptyError<&'_ T> {
    /// Maps the empty value from a borrowed type into its owned type.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    pub fn into_owning(self) -> EmptyError<T::Owned>
    where
        T: ToOwned,
    {
        self.map(T::to_owned)
    }
}

impl<T> Debug for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("EmptyError").finish_non_exhaustive()
    }
}

impl<T> Display for EmptyError<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{EMPTY_ERROR_MESSAGE}")
    }
}

// There are no bounds on `T` here (nor the `Debug` and `Display` implementations). The empty value
// is provided less for reporting and more for handling and flexibility, so it is not formatted into
// error text like in some other error types.
impl<T> Error for EmptyError<T> {}

/// A collection, slice, or range type that must contain one or more items (is never empty).
///
/// `NonEmpty` is an adapter that guarantees (barring unsafe code) that a value is non-empty and so
/// has one or more items. For example, `NonEmpty<Vec<_>>` represents a non-empty [`Vec`] with APIs
/// that reflect this non-empty guarantee.
///
/// This is a very general type: refer to more specific type aliases to see the relevant APIs for a
/// particular non-empty type. For example, see [`vec1::Vec1`] for supported non-empty [`Vec`] APIs.
/// Every non-empty collection has such a type alias.
///
/// Non-empty arrays and iterators are **not** represented with `NonEmpty`: see [`Array1`] and
/// [`Iterator1`].
///
/// [`Array1`]: crate::array1::Array1
/// [`Iterator1`]: crate::iter1::Iterator1
/// [`Vec`]: alloc::vec::Vec
/// [`vec1::Vec1`]: crate::vec1::Vec1
#[derive(Clone, Copy, Hash)]
#[repr(transparent)]
pub struct NonEmpty<T>
where
    T: ?Sized,
{
    items: T,
}

impl<T> NonEmpty<T>
where
    T: Sized,
{
    /// Applies a function `f` over a mutable reference to the inner maybe-empty value.
    ///
    /// # Errors
    ///
    /// Returns an [`EmptyError`] if the function `f` empties the inner value.
    ///
    /// # Examples
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// let xs = vec1![0i32, 1, 2, 3];
    /// assert!(xs.and_then_try(|xs| xs.clear()).is_err());
    #[doc = "```"]
    pub fn and_then_try<F>(self, f: F) -> Result<Self, EmptyError<T>>
    where
        // A bound on `FromMaybeEmpty` would be more direct, but that trait is not part of the
        // public API. `TryFrom` is used instead, since it is implemented in terms of
        // `FromMaybeEmpty`.
        Self: TryFrom<T, Error = EmptyError<T>>,
        F: FnOnce(&mut T),
    {
        let NonEmpty { mut items } = self;
        f(&mut items);
        <NonEmpty<T> as TryFrom<T>>::try_from(items)
    }
}

#[cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
impl<T> NonEmpty<T>
where
    T: MaybeEmpty + ?Sized,
{
    /// Gets the [cardinality][`Cardinality`] of the `NonEmpty`.
    fn cardinality(&self) -> Cardinality<(), ()> {
        match self.items.cardinality() {
            // SAFETY: `self.items` is non-empty.
            None => unsafe { safety::unreachable_maybe_unchecked() },
            Some(cardinality) => cardinality,
        }
    }

    /// Gets a mutable reference to the adapted value as a [`Cardinality`].
    #[cfg(feature = "alloc")]
    fn as_cardinality_items_mut(&mut self) -> Cardinality<&mut T, &mut T> {
        match self.cardinality() {
            One(_) => One(&mut self.items),
            Many(_) => Many(&mut self.items),
        }
    }
}

impl<T> AsRef<T> for NonEmpty<T>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        &self.items
    }
}

impl<T> Eq for NonEmpty<T>
where
    Self: PartialEq,
    T: Eq + ?Sized,
{
}

impl<T> FromMaybeEmpty<T> for NonEmpty<T>
where
    T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: T) -> Self {
        NonEmpty { items }
    }
}

impl<'a, T> FromMaybeEmpty<&'a T> for &'a NonEmpty<T>
where
    T: ?Sized,
    &'a T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: &'a T) -> Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `T` and
        //         `NonEmpty<T>` are the same.
        unsafe { mem::transmute::<&'_ T, &'_ NonEmpty<T>>(items) }
    }
}

impl<'a, T> FromMaybeEmpty<&'a mut T> for &'a mut NonEmpty<T>
where
    T: ?Sized,
    &'a mut T: MaybeEmpty,
{
    unsafe fn from_maybe_empty_unchecked(items: &'a mut T) -> Self {
        // SAFETY: `NonEmpty` is `repr(transparent)`, so the representations of `T` and
        //         `NonEmpty<T>` are the same.
        unsafe { mem::transmute::<&'_ mut T, &'_ mut NonEmpty<T>>(items) }
    }
}

impl<T> Ord for NonEmpty<T>
where
    Self: PartialOrd,
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.items, &other.items)
    }
}

impl<T, U> PartialEq<NonEmpty<U>> for NonEmpty<T>
where
    T: PartialEq<U> + ?Sized,
    U: ?Sized,
{
    fn eq(&self, other: &NonEmpty<U>) -> bool {
        PartialEq::eq(&self.items, &other.items)
    }
}

impl<T, U> PartialOrd<NonEmpty<U>> for NonEmpty<T>
where
    T: PartialOrd<U> + ?Sized,
    U: ?Sized,
{
    fn partial_cmp(&self, other: &NonEmpty<U>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.items, &other.items)
    }
}

/// Non-empty cardinality.
///
/// `Cardinality` describes and associates data with a non-empty cardinality: one or many. Some
/// behaviors of non-empty types are predicated on cardinality. In particular, it is not possible to
/// take the only remaining item out of a non-empty collection, and so APIs that take an item like
/// [`Vec1::pop_if_many`] return [a `Cardinality` type][`OrOnly`]. Similarly, non-empty map types
/// like [`BTreeMap1`] use `Cardinality` types in [their entry APIs][`OccupiedEntry`].
///
/// [`BTreeMap1`]: crate::btree_map1::BTreeMap1
/// [`OccupiedEntry`]: crate::btree_map1::OccupiedEntry
/// [`OrOnly`]: crate::vec1::OrOnly
/// [`Vec1::pop_if_many`]: crate::vec1::Vec1::pop_if_many
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Cardinality<O, M> {
    /// Exactly one item.
    One(O),
    /// More than one item.
    Many(M),
}

impl<O, M> Cardinality<O, M> {
    /// Converts the cardinality into an `Option<O>`, discarding the [`Many`] value, if any.
    pub fn one(self) -> Option<O> {
        match self {
            One(one) => Some(one),
            _ => None,
        }
    }

    /// Converts the cardinality into an `Option<M>`, discarding the [`One`] value, if any.
    pub fn many(self) -> Option<M> {
        match self {
            Many(many) => Some(many),
            _ => None,
        }
    }

    /// Maps a `Cardinality<O, M>` to `Cardinality<U, M>` by applying a function to the [`One`]
    /// value.
    pub fn map_one<U, F>(self, f: F) -> Cardinality<U, M>
    where
        F: FnOnce(O) -> U,
    {
        match self {
            One(one) => One(f(one)),
            Many(many) => Many(many),
        }
    }

    /// Maps a `Cardinality<O, M>` to `Cardinality<O, U>` by applying a function to the [`Many`]
    /// value.
    pub fn map_many<U, F>(self, f: F) -> Cardinality<O, U>
    where
        F: FnOnce(M) -> U,
    {
        match self {
            One(one) => One(one),
            Many(many) => Many(f(many)),
        }
    }

    /// Returns `true` if the `Cardinality` is [`One`], otherwise `false`.
    pub fn is_one(&self) -> bool {
        matches!(self, One(_))
    }

    /// Returns `true` if the `Cardinality` is [`Many`], otherwise `false`.
    pub fn is_many(&self) -> bool {
        matches!(self, Many(_))
    }
}

impl<T> Cardinality<T, T> {
    /// Maps a `Cardinality<T, T>` to `Cardinality<U, U>` by applying a function to the inner
    /// value.
    pub fn map<U, F>(self, f: F) -> Cardinality<U, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            One(one) => One(f(one)),
            Many(many) => Many(f(many)),
        }
    }
}

impl<T> Cardinality<&mut T, T> {
    /// Gets the [`Many`] value or replaces the [`One`] reference with `item`.
    ///
    /// When taking an item out of a non-empty collection, this function gets the taken item or, if
    /// there is only one remaining item, replaces it.
    ///
    /// # Examples
    ///
    /// Implementing a pop (a.k.a. drop) operation in a postfix calculator:
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// pub struct PostfixCalculator {
    ///     // There must always be a computation, so this is non-empty.
    ///     stack: Vec1<f64>,
    ///     // ...
    /// }
    ///
    /// impl PostfixCalculator {
    ///     pub fn pop(&mut self) -> f64 {
    ///         self.stack
    ///             .pop_if_many()
    ///             .or_replace_only(0.0)
    ///     }
    ///
    ///     // ...
    /// }
    #[doc = "```"]
    pub fn or_replace_only(self, item: T) -> T {
        self.or_else_replace_only(|| item)
    }

    /// Gets the [`Many`] value or replaces the [`One`] reference with the output of the given
    /// function.
    ///
    /// When taking an item out of a non-empty collection, this function gets the taken item or, if
    /// there is only one remaining item, replaces it.
    ///
    /// # Examples
    ///
    /// Implementing a back operation in a UI screen navigator:
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// #[derive(Default)]
    /// pub enum Screen {
    ///     #[default]
    ///     Home,
    ///     // ...
    /// }
    ///
    /// pub struct Navigator {
    ///     // There must always be an active `Screen`, so this is non-empty.
    ///     screens: Vec1<Screen>,
    ///     // ...
    /// }
    ///
    /// impl Navigator {
    ///     pub fn back(&mut self) -> Screen {
    ///         self.screens
    ///             .pop_if_many()
    ///             .or_else_replace_only(Screen::default)
    ///     }
    /// }
    #[doc = "```"]
    pub fn or_else_replace_only<F>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        match self {
            One(only) => mem::replace(only, f()),
            Many(item) => item,
        }
    }
}

macro_rules! with_literals {
    ($f:ident$(,)?) => {};
    ($f:ident, [$($N:literal $(,)?)+]$(,)?) => {
        $(
            $f!($N);
        )+
    };
}
pub(crate) use with_literals;

macro_rules! with_tuples {
    ($f:ident, ($head:ident $(,)?) $(,)?) => {
        $f!(($head,));
    };
    ($f:ident, ($head:ident, $($tail:ident $(,)?)+) $(,)?) => {
        $f!(($head, $($tail,)+));
        $crate::with_tuples!($f, ($($tail,)+));
    };
}
pub(crate) use with_tuples;

// This macro is similar to `with_tuples`, but accepts a zipped tuple and unzips it into two
// congruent tuples that are then forwarded to the given macro named by `f`. This is necessary to
// work around a limitation: given the tuples from `with_tuples`, it is not possible to hygenically
// concatenate or otherwise form a distinct but congruent item for a given element `T`.
//
// Accepting a zipped tuple ensures congruency between the two unzipped tuples forwarded to the
// given macro.
//
// See the `iter1::impl_from_iterator1_for_tuple` macro.
macro_rules! with_unzipped_tuples {
    ($f:ident, (($headX:ident, $headY:ident $(,)?) $(,)?) $(,)?) => {
        $f!(($headX,), ($headY,));
    };
    (
        $f:ident,
        (($headX:ident, $headY:ident $(,)?), $(($tailX:ident, $tailY:ident $(,)?) $(,)?)+)
        $(,)?
    ) => {
        $f!(($headX, $($tailX,)+), ($headY, $($tailY,)+));
        $crate::with_unzipped_tuples!($f, ($(($tailX, $tailY),)+));
    };
}
pub(crate) use with_unzipped_tuples;

macro_rules! impl_partial_eq_for_non_empty {
    (
        [$(for $ritem:ident $(,)?)? in $rhs:ty]
        ==
        [$(for $litem:ident $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem,)? $($litem)?> ::core::cmp::PartialEq<$rhs> for $lhs
        where
            $($litem: ::core::cmp::PartialEq<$ritem>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, &rhs.items)
            }
        }
    };
    (
        [$(for $ritem:ident $(,const $rn:ident: usize)? $(,)?)? in $rhs:ty]
        <=
        [$(for $litem:ident $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem,)? $($litem,)? $($(const $rn: usize)?)?> ::core::cmp::PartialEq<$rhs> for $lhs
        where
            $($litem: ::core::cmp::PartialEq<$ritem>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(&self.items, rhs)
            }
        }
    };
    (
        [$(for $ritem:ident $(,)?)? in $rhs:ty]
        =>
        [$(for $litem:ident $(,const $ln:ident: usize)? $(,)?)? in $lhs:ty] $(,)?
    ) => {
        impl<$($ritem,)? $($litem,)? $($(const $ln: usize)?)?> ::core::cmp::PartialEq<$rhs> for $lhs
        $(where
            $litem: ::core::cmp::PartialEq<$ritem>,)?
        {
            fn eq(&self, rhs: &$rhs) -> bool {
                ::core::cmp::PartialEq::eq(self, &rhs.items)
            }
        }
    };
}
pub(crate) use impl_partial_eq_for_non_empty;

#[cfg(all(test, feature = "alloc"))]
pub mod harness {
    pub trait KeyValueRef {
        type Cloned;

        fn cloned(&self) -> Self::Cloned;
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }

    impl<'a, K, V> KeyValueRef for (&'a K, &'a mut V)
    where
        K: Clone,
        V: Clone,
    {
        type Cloned = (K, V);

        fn cloned(&self) -> Self::Cloned {
            (self.0.clone(), self.1.clone())
        }
    }
}

**Mitsein** is a Rust library that provides strongly typed APIs for non-empty
collections and views, including iterators, slices, vectors, and much more.

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/mitsein-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/mitsein)
[![docs.rs](https://img.shields.io/badge/docs.rs-mitsein-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/mitsein)
[![crates.io](https://img.shields.io/crates/v/mitsein.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/mitsein)

## Basic Usage

Allocating a `Vec1` from one or more items (infallibly):

```rust
use mitsein::prelude::*;

let xs = Vec1::from_one(0i32);
let xs = Vec1::from([0i32, 1, 2]);
let xs = Vec1::from_head_and_tail(0i32, [1, 2]);

let xs: Vec1<_> = [0i32].into_iter1().collect1();

let xs = vec1![0i32, 1, 2];
```

Allocating a `Vec1` from zero or more items (fallibly):

```rust
use mitsein::prelude::*;

let ys = vec![0i32, 1, 2];

let xs = Vec1::try_from(ys).unwrap();
let xs = Vec1::try_from(&[0i32, 1, 2]).unwrap();
let xs = Vec1::try_from_iter([0i32, 1, 2]).unwrap();

let xs: Vec1<_> = [0i32].into_iter().try_collect1().unwrap();
```

Mapping over items in a `Vec1`:

```rust
use mitsein::prelude::*;

let xs = Vec1::from([0i32, 1, 2]);
let ys: Vec1<_> = xs.into_iter1().map(|x| x + 1).collect1();
```

Removing items from a `Vec1`:

```rust
use mitsein::prelude::*;

let mut xs = Vec1::from([0i32, 1, 2]);
while let Ok(item) = xs.pop_or().get_only() { ... }

let mut xs = Vec1::from([0i32, 1, 2]);
xs.tail().clear();
```

Bridging between `Iterator` and `Iterator1`:

```rust
use mitsein::iter1;
use mitsein::prelude::*;

let xs = iter1::head_and_tail(0i32, [1, 2]);
let xs: Vec1<_> = xs.into_iter().skip(3).or_non_empty([3]).collect1();
assert_eq!(xs.as_slice(), &[3]);
```

## Features and Comparisons

**Non-empty iterator APIs separate concerns using familiar patterns and
syntax.** Mitsein need not expose combinatorial sets of inherent iterator-like
functions in non-empty collections. For example, the [`vec1`] crate supports map
operations over its `Vec1` type via the `Vec1::mapped`, `Vec1::mapped_ref`, and
`Vec1::mapped_mut` functions. Mitsein instead exposes map operations via
`Iterator1`, which can support any non-empty view or collection with a more
typical API (i.e., `Iterator1::map`).

Non-empty slice APIs enable borrowing and copy-on-write, so **Mitsein supports
the standard `Cow` type**, unlike other non-empty implementations like the
[`nonempty`] and [`vec1`] crates.

**Items are stored consistently in Mitsein.** No head item is allocated
differently. For example, the [`nonempty`] crate directly exposes a head item
that is, unlike tail items, **not** allocated on the heap. This precludes views
and slicing and can have surprising performance implications when items are
non-trivial to copy or clone.

**Non-empty collection APIs that exhibit different behavior from their
counterparts have distinct names and APIs in Mitsein.** For example, the
[`vec1`] crate presents `Vec1::pop` and `Vec1::remove`, which have different
behavior than `Vec::pop` and `Vec::remove`. Mitsein presents explicit APIs like
`Vec1::pop_or`, which return a proxy type that determines what to do when only
one item remains. This leads to more explicit expressions like
`xs.pop_or().get_only()`, `xs.pop_or().none()`, and
`xs.remove_or(1).else_replace_only(|| 0)`.

**Mitsein separates many non-empty error concerns into a segmentation API.**
Segments span a range in a collection and support the insertion and removal of
items. Non-empty collections can be segmented prior to removals, which
consolidates error conditions: the segment can be freely manipulated without
checks or errors after it has been constructed. The [`nonempty`] and [`nunny`]
crates have limited or no support for removals while the [`vec1`] crate only
supports fallible removals directly against `Vec1`.

```rust
use mitsein::prelude::*;

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().clear();
assert_eq!(xs.as_slice(), &[0i32]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().rtail().swap_drain(..); // `swap_drain` is the counterpart to `drain`.
assert_eq!(xs.as_slice(), &[0i32, 4]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.segment(..3).retain(|x| *x % 2 != 0);
assert_eq!(xs.as_slice(), &[1i32, 3, 4]);
```

**Mitsein provides comprehensive coverage of ordered collections and container
APIs in `core` and `alloc`.** This notably includes `slice`, `str`, `BTreeMap`,
`BTreeSet`, `Box`, and `Arc`. Non-empty types also implement standard traits
like their counterparts. The [`nonempty`] and [`vec1`] crates lack support for
primitive types like `slice` and collections other than `Vec`.

Mitsein is a `no_std` library and both `alloc` and `std` are optional.
**Non-empty slices, iterators, and arrays can be used in contexts where OS
features or allocation are not available.** This also includes the integration
with [`arrayvec`][`arrayvec`].

## Memory Safety

**Mitsein uses unsafe code.** Some of this unsafe code is used to support
`repr(transparent)` and other unsafe conversions, but the majority is used to
avoid unnecessary branching. For example, given the non-empty guarantee, it
_shouldn't_ be necessary for the `Slice1::first` function to check that a first
item is actually present. Omitting this code is great, but it also means that
there are opportunities for undefined behavior and unsound APIs in Mitsein. Of
course, the authors strive to prevent this; issues and pull requests are
welcome!

The nature of unsafe code is also somewhat unusual in Mitsein. The overwhelming
majority of unsafe code is uninteresting and **not responsible for maintaining
invariants**. Audits are best focused on **safe** code that affects non-empty
invariants instead.

Branching is toggled in the `safety` module. The presence of items in non-empty
types is asserted when executing tests, though not in the context of
[Miri][`miri`]. These conditional checks use the nomenclature
`_maybe_unchecked`.

## Integrations and Cargo Features

Mitsein provides some optional features and integrations via the following Cargo
features.

| Feature     | Default | Primary Dependency | Description                                               |
|-------------|---------|--------------------|-----------------------------------------------------------|
| `alloc`     | No      | `alloc`            | Non-empty collections that allocate, like `Vec1`.         |
| `arbitrary` | No      | [`arbitrary`]      | Construction of arbitrary non-empty collections.          |
| `arrayvec`  | No      | [`arrayvec`]       | Non-empty implementations of [`arrayvec`] types.          |
| `indexmap`  | No      | [`indexmap`]       | Non-empty implementations of [`indexmap`] types.          |
| `itertools` | No      | [`itertools`]      | Combinators from [`itertools`] for `Iterator1`.           |
| `rayon`     | No      | [`rayon`]          | Parallel operations for non-empty types.                  |
| `serde`     | No      | [`serde`]          | De/serialization of non-empty collections with [`serde`]. |
| `smallvec`  | No      | [`smallvec`]       | Non-empty implementations of [`smallvec`] types.          |
| `std`       | Yes     | `std`              | Integrations with `std::io`.                              |

[`arbitrary`]: https://crates.io/crates/arbitrary
[`arrayvec`]: https://crates.io/crates/arrayvec
[`indexmap`]: https://crates.io/crates/indexmap
[`itertools`]: https://crates.io/crates/itertools
[`miri`]: https://github.com/rust-lang/miri
[`nonempty`]: https://crates.io/crates/nonempty
[`nunny`]: https://crates.io/crates/nunny
[`rayon`]: https://crates.io/crates/rayon
[`serde`]: https://crates.io/crates/serde
[`smallvec`]: https://crates.io/crates/smallvec
[`vec1`]: https://crates.io/crates/vec1

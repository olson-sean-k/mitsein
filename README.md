**Mitsein** is a Rust library that provides strongly typed APIs for non-empty
collections and views, including (but not limited to) iterators, slices, and
vectors.

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/mitsein-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/mitsein)
[![docs.rs](https://img.shields.io/badge/docs.rs-mitsein-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/mitsein)
[![crates.io](https://img.shields.io/crates/v/mitsein.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/mitsein)

## Basic Usage

Allocating a `Vec1` from one or more items (infallibly):

```rust
use mitsein::vec1::{vec1, Vec1};

let xs = Vec1::from_one(0i32);
let xs = Vec1::from([0i32, 1, 2]);
let xs = Vec1::from_head_and_tail(0i32, [1, 2]);

let xs = vec1![0i32, 1, 2];
```

Allocating a `Vec1` from zero or more items (fallibly):

```rust
use mitsein::prelude::*;

let ys = vec![0i32, 1, 2];

let xs = Vec1::try_from(ys)?;
let xs = Vec1::try_from(&[0i32, 1, 2])?;
let xs = Vec1::try_from_iter([0i32, 1, 2])?;
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
while let Ok(item) = xs.pop_or_get_only() { ... }

xs.tail().clear();
```

Bridging between `Iterator` and `Iterator1`:

```rust
use mitsein::iter1;
use mitsein::prelude::*;

let xs = iter1::head_and_tail(0i32, [1, 2]);
let xs: Vec1<_> = xs.into_iter().skip(3).or_non_empty([3]).collect1();
assert_eq!(xs.as_slice(), &[3]);

let xs = Vec1::from([0i32, 1, 2]);
let ys: Vec1<_> = xs
    .iter1()
    .all_and(|x| *x == 0)
    .if_not_and_then_remainder(|| {
        eprintln!("non-zero");
    })
    .or_one(&0i32)
    .collect1();
assert_eq!(ys.as_slice(), &[&2]);
```

## Features and Comparisons

By providing non-empty APIs over both collections **and** views (i.e., slices
and iterators), **Mitsein separates concerns just like standard collection and
iterator APIs**. Unlike many other non-empty collection implementations, Mitsein
need not expose combinatorial sets of inherent iterator-like functions in
collections for each pair of receiver and operation. For example, the [`vec1`]
crate supports map operations over its `Vec1` type via the `Vec1::mapped`,
`Vec1::mapped_ref`, and `Vec1::mapped_mut` functions. Mitsein instead exposes
map operations via `Iterator1`, which can support any non-empty view or
collection with a more typical API.

Non-empty view APIs also enable borrowing, so **Mitsein collection types support
copy-on-write via the standard `Cow` type**, unlike many other non-empty
collection implementations.

Non-empty iterator APIs are largely compatible with standard iterators, allowing
non-empty collections to interact ergonomically with collections of zero or more
items. **Mitsein views and collections use more familiar syntax for mapping,
taking, chaining, etc.**

**Items are stored contiguously or otherwise consistently in Mitsein**. This
means that no head item is allocated diffently. For example, the [`nonempty`]
crate directly exposes a head item that is, unlike tail items, **not** allocated
on the heap. This can potentially cause surprising behavior or poor performance
when the item type is large, for example.

Non-empty APIs in collections that exhibit different behavior are distinct from
counterparts in Mitsein. For example, the [`vec1`] crate presents `Vec1::pop`
and `Vec1::remove`, which may be unclear in context. Mitsein instead presents
`Vec1::pop_or_get_only` and `Vec1::remove_or_get_only`.

**Mitsein separates many non-empty error concerns into a segmentation API**. The
segmentation API provides a view (called a segment) into collections that
supports **topological** mutations (unlike slices, for example). This works well
for non-empty collections, which can be segmented prior to otherwise fallible
operations. The [`nonempty`] and [`nunny`] crates have limited (or no) support
for removals while the [`vec1`] crate provides fallible but bespoke
counterparts.

```rust
use mitsein::prelude::*;

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().clear();
assert_eq!(xs.as_slice(), &[0i32]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().rtail().drain(..);
assert_eq!(xs.as_slice(), &[0i32, 4]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.segment(1..).truncate(2);
assert_eq!(xs.as_slice(), &[0i32, 1, 2]);
```

**Mitsein provides complete coverage of ordered collections and heap container
APIs in `core` and `alloc`**. This notably includes `slice`, `BTreeMap`,
`BTreeSet`, `Box`, and `Arc`. The [`nonempty`] and [`vec1`] crates lack support
for primitive types like `slice` and positional collections in `alloc` beyond
`Vec`.

Mitsein is a `no_std` library and `alloc` is optional. **Non-empty slices,
iterators, and arrays can be used in embedded contexts, or any other environment
where OS features or allocation are not available.** Integration with
[`arrayvec`][`arrayvec`] also functions in `no_std` environments.

## Cargo Features

Mitsein provides some optional features and integrations via the following Cargo
features.

| Feature     | Default | Dependencies            | Description                                               |
|-------------|---------|-------------------------|-----------------------------------------------------------|
| `alloc`     | No      | `alloc`                 | Non-empty collections that allocate, like `Vec1`.         |
| `arrayvec`  | No      | `arrayvec`              | Non-empty implementation of [`ArrayVec`][`arrayvec`].     |
| `itertools` | No      | `itertools`             | Combinators from [`itertools`] for `Iterator1`.           |
| `serde`     | No      | `serde`, `serde_derive` | De/serialization of non-empty collections with [`serde`]. |
| `std`       | Yes     | `std`                   | Integrations with `std::io`.                              |

Some features enable other crate and dependency features. For example, `std`
enables `alloc` and both of these features enable similar features in optional
dependencies when applicable.

[`arrayvec`]: https://crates.io/crates/arrayvec
[`itertools`]: https://crates.io/crates/itertools
[`nonempty`]: https://crates.io/crates/nonempty
[`nunny`]: https://crates.io/crates/nunny
[`serde`]: https://crates.io/crates/serde
[`vec1`]: https://crates.io/crates/vec1

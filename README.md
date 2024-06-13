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

let xs = Vec1::from_item(0i32);
let xs = Vec1::from([0i32, 1, 2]);
let xs = Vec1::from_head_and_tail(0i32, [1, 2]);

let xs = vec1![0i32, 1, 2];
```

Allocating a `Vec1` from zero or more items (fallibly):

```rust
use mitsein::vec1::Vec1;

let ys = vec![0i32, 1, 2];

let xs = Vec1::try_from(ys)?;
let xs = Vec1::try_from(&[0i32, 1, 2])?;
let xs = Vec1::try_from_iter([0i32, 1, 2])?;
```

Mapping over items in `Vec1`s:

```rust
use mitsein::prelude::*;
use mitsein::vec1::Vec1;

let xs = Vec1::from([0i32, 1, 2]);
let ys: Vec1<_> = xs.into_iter1().map(|x| x + 1).collect();
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
non-empty collections to interact ergonomically with collection of zero or more
items. **Mitsein views and collections use more familiar syntax for mapping,
taking, chaining, etc.**

**Items are stored contiguously or otherwise in a homogeneous manner in
Mitsein**. This means that no head item is allocated diffently. For example, the
[`nonempty`] crate directly exposes a head item that is, unlike tail items,
**not** allocated on the heap. This can potentially cause surprising or poor
performance when the item type is large, as the head item is stack allocated.

[`nonempty`]: https://crates.io/crates/nonempty
[`vec1`]: https://crates.io/crates/vec1

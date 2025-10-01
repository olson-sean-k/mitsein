**Mitsein** (mɪtsaɪ̯n | _mitt-zign_) is a Rust library that provides strongly
typed APIs for non-empty collections and views, including iterators, slices,
vectors, and much more.

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
while let Ok(item) = xs.pop_if_many().or_get_only() { ... }

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

### Separation of Concerns

Mitsein separates concerns into dedicated APIs much like standard types. This
persists the non-empty guarantee across types and supports familiar patterns and
syntax.

Non-empty iterators support any non-empty collection or view and mirror their
counterparts. For example, the [`vec1`] crate supports map operations over its
`Vec1` type via bespoke `Vec1::mapped`, `Vec1::mapped_ref`, and
`Vec1::mapped_mut` functions. Mitsein instead exposes map operations via
`Iterator1::map`, which supports a variety of types and receivers just like
`Iterator` types do.

```rust
use mitsein::prelude::*;

let xs = Vec1::from([0i32, 1, 2, 3, 4]);
let ys: Vec1<_> = xs.into_iter1().map(|x| x * 2).collect1();
```

Mitsein provides a segmentation API, which isolates a range within a collection
that supports insertions and removals. Non-empty collections can be segmented
prior to removals, which consolidates error conditions: a segment can be freely
manipulated without checks or errors after its construction.

```rust
use mitsein::prelude::*;

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().clear();
assert_eq!(xs.as_slice(), &[0i32]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.tail().rtail().swap_drain(..); // `swap_drain` is the counterpart to `drain`.
assert_eq!(xs.as_slice(), &[0i32, 4]);

let mut xs = Vec1::from([0i32, 1, 2, 3, 4]);
xs.segment(..3).unwrap().retain(|x| *x % 2 != 0);
assert_eq!(xs.as_slice(), &[1i32, 3, 4]);
```

Non-empty slice APIs enable borrowing and copy-on-write, so Mitsein supports
the standard `Cow` type, unlike many other non-empty implementations like the
[`nonempty`] and [`vec1`] crates. Extension traits provide seemless conversions
for these types.

```rust
use mitsein::borrow1::CowSlice1;
use mitsein::prelude::*;

let xs = CowSlice1::from(slice1![0i32, 1, 2, 3, 4]);
let xs = xs.into_arc_slice1();
```

### Consistent Storage

Items are stored consistently in Mitsein. No head item is allocated differently.
For example, the [`nonempty`] crate directly exposes a head item that is, unlike
tail items, **not** allocated on the heap. This precludes views and slicing and
can have surprising performance implications when items are non-trivial to copy
or clone.

Non-empty collections are defined by the transparent `NonEmpty` type
constructor, so the representation of a non-empty collection is always exactly
the same as its counterpart. For example, `Vec1<T>` is a type definition for
`NonEmpty<Vec<T>>` and has the same representation as `Vec<T>`.

### Explicitness

Non-empty collection APIs that exhibit different behavior from their
counterparts are distinct in Mitsein. For example, functions that take items out
of collections like `Vec1::pop_if_many` have a `_if_many` suffix and  return a
proxy type rather than an `Option`. This leads to more explicit and distinct
expressions like `xs.pop_if_many().or_get_only()` and
`xs.remove_if_many(1).or_else_replace_only(|| 0)`.

Similarly, operations that have additional constraints or otherwise cannot be
directly supported by non-empty collections are separated into [segmentation
APIs](#separation-of-concerns), such as `vec1::Segment::swap_drain`.

### Comprehensiveness

Mitsein provides comprehensive coverage of ordered collections and container
APIs in `core` and `alloc`. This notably includes `slice`, `str`, `BTreeMap`,
`BTreeSet`, `Box`, and `Arc`. Non-empty types also implement standard traits
like their counterparts. The [`nonempty`] and [`vec1`] crates lack support for
primitive types like `slice` and collections other than `Vec`.

Non-empty counterparts for iterators and many standard conversions, such as
between `Vec1` and `Arc1`, are also provided. These are crucial for maintaining
the non-empty invariant when manipulating non-empty types.

Mitsein is a `no_std` library and both `alloc` and `std` are optional.
**Non-empty slices, iterators, and arrays can be used in contexts where OS
features or allocation are not available.** This also includes the integration
with [`arrayvec`]. There are also optional integrations with foundational and
popular crates like [`itertools`], [`rayon`], and [`serde`].

## Memory Safety

**Mitsein uses unsafe code.** Some of this unsafe code is used to support
`repr(transparent)` and other unsafe conversions, but most is used to avoid
unnecessary branching. For example, given the non-empty guarantee, it
_shouldn't_ be necessary for the `Slice1::first` function to check that a first
item is actually present. Omitting this check when unnecessary is great, but it
also means that there are opportunities for undefined behavior and unsound APIs
in Mitsein. Of course, the authors strive to prevent this; issues and pull
requests are welcome!

The nature of unsafe code is also somewhat unusual in Mitsein. The overwhelming
majority of unsafe code is uninteresting and **not responsible for maintaining
invariants**. Audits are best focused on **safe** code that affects non-empty
invariants instead.

Checking is toggled in the `safety` module. The presence of items in non-empty
types is asserted in tests builds, but not in non-test builds (nor in the
context of [Miri][`miri`]). APIs that interact with these  conditional checks
use the nomenclature "maybe unchecked", such as `unwrap_maybe_unchecked`.

## Integrations and Feature Flags

Mitsein provides some optional features and integrations via the following
feature flags.

| Feature     | Also Enables | Default | Crate         | Description                                               |
|-------------|--------------|---------|---------------|-----------------------------------------------------------|
| `alloc`     |              | No      | `alloc`       | Non-empty collections that allocate, like `Vec1`.         |
| `arbitrary` | `std`        | No      | [`arbitrary`] | Construction of arbitrary non-empty collections.          |
| `arrayvec`  |              | No      | [`arrayvec`]  | Non-empty implementations of [`arrayvec`] types.          |
| `either`    |              | No      | [`either`]    | Non-empty iterator implementation for `Either`.           |
| `indexmap`  | `alloc`      | No      | [`indexmap`]  | Non-empty implementations of [`indexmap`] types.          |
| `itertools` | `either`     | No      | [`itertools`] | Combinators from [`itertools`] for `Iterator1`.           |
| `rayon`     | `std`        | No      | [`rayon`]     | Parallel operations for non-empty types.                  |
| `schemars`  | `alloc`      | No      | [`schemars`]  | JSON schema generation for non-empty types.               |
| `serde`     |              | No      | [`serde`]     | De/serialization of non-empty collections with [`serde`]. |
| `smallvec`  | `alloc`      | No      | [`smallvec`]  | Non-empty implementations of [`smallvec`] types.          |
| `std`       | `alloc`      | Yes     | `std`         | Integrations with `std::io`.                              |

[`arbitrary`]: https://crates.io/crates/arbitrary
[`arrayvec`]: https://crates.io/crates/arrayvec
[`either`]: https://crates.io/crates/either
[`indexmap`]: https://crates.io/crates/indexmap
[`itertools`]: https://crates.io/crates/itertools
[`miri`]: https://github.com/rust-lang/miri
[`nonempty`]: https://crates.io/crates/nonempty
[`nunny`]: https://crates.io/crates/nunny
[`rayon`]: https://crates.io/crates/rayon
[`schemars`]: https://crates.io/crates/schemars
[`serde`]: https://crates.io/crates/serde
[`smallvec`]: https://crates.io/crates/smallvec
[`vec1`]: https://crates.io/crates/vec1

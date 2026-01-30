# Contributing to Mitsein

Thanks for considering contributions! Issues and pull requests are welcome. The
following provides some information and guidance about contributing.

Mitsein is distributed under the [MIT license]. Contributions must adhere to this
license.

## Issues

Issues concerning feature requests, bugs, questions, and other discussion are
appreciated! Please be sure to search for existing issues that may have already
addressed a concern before opening a new issue.

## Code Changes

In general, please refer to existing code for guidance on style and workflow.

### Toolchains and Components

Mitsein targets stable Rust. See the crate manifest for the [MSRV]. However, a
nightly toolchain is used for formatting (via [`rustfmt`]) and memory safety
analysis (via [Miri]). At a minimum, both a stable and nightly toolchain are
necessary to develop Mitsein, and can be installed with the following [`rustup`]
commands:

```shell
rustup toolchain install stable
rustup toolchain install nightly
```

When making certain kinds of changes, it can be useful to install and run [Miri]
and [`cargo-hack`] locally. These can be installed with the following commands:

```shell
rustup +nightly component add miri
cargo +stable install cargo-hack --locked
```

These additional tools are not strictly necessary and are both periodically
executed by GitHub workflows. [`cargo-hack`] is recommended when changes affect
conditional compilation against feature flags, because it can be easy to break
specific combinations of interdependent features.

### Tests

New APIs and features should be developed with unit tests. Mitsein uses
[`rstest`] for fixtures and cases. Many modules have a `harness` where fixtures
and other shared test code concerning that module are defined. For example, the
`vec1` module has the following sub-module structure for unit tests:

```rust
#[cfg(test)]
pub mod harness {
    ...

    #[fixture]
    pub fn xs1(...) -> Vec1<_> { ... }
}

#[cfg(test)]
mod tests {
    ...
    use crate::vec1::harness::{self, xs1};

    #[rstest]
    fn this_then_that(xs1: Vec1<_>) { ... }
}
```

Unit test functions feature descriptive names with the general form of
`{this}_then_{that}`. Specific and long names are encouraged for unit test
functions. If a test fails, it is ideally easy to determine what happened and
contrast that with what was expected to happen. Similarly, [`rstest`] cases
should have names where applicable. Consider this `vec1` unit test, for example:

```rust
#[rstest]
#[case::tail(harness::xs1(3), 1..)]
#[case::rtail(harness::xs1(3), ..3)]
#[case::middle(harness::xs1(9), 4..8)]
fn retain_none_from_vec1_segment_then_segment_is_empty<R>(
    #[case] mut xs1: Vec1<u8>,
    #[case] range: R,
) where
    ...
{
    ...
}
```

### Formatting

Code in Mitsein is formatted using [`rustfmt`] from the nightly channel:

```shell
cargo +nightly fmt
```

Note that Mitsein uses some non-standard formatting rules. See
[`rustfmt.toml`](https://github.com/olson-sean-k/mitsein/blob/master/rustfmt.toml).

[`cargo-hack`]: https://github.com/taiki-e/cargo-hack
[Miri]: https://github.com/rust-lang/miri
[MIT license]: https://opensource.org/license/mit
[MSRV]: https://doc.rust-lang.org/cargo/reference/rust-version.html
[`rstest`]: https://docs.rs/rstest/latest/rstest/
[`rustfmt`]: https://github.com/rust-lang/rustfmt

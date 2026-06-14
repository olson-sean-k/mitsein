default: build

analyze:
    cargo +nightly miri test --all-features

build:
    cargo build --all-features

clean:
    cargo clean

document:
    RUSTDOCFLAGS="--cfg=docsrs" cargo +nightly doc -p mitsein --no-deps --all-features

feature-powerset:
    RUSTFLAGS="-D warnings" cargo hack check --tests --feature-powerset --skip default

format:
    cargo +nightly fmt

lint:
    cargo clippy --all-features --all-targets -- -D clippy::all

test:
    cargo test --all-features

use core::ops::Bound;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ItemRange<N> {
    start: Bound<N>,
    end: Bound<N>,
}

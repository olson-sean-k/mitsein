// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound. The
//         segmentation APIs interact with non-empty collections and bugs here may break the
//         non-empty invariant. In particular, range types, intersection, and projection must be
//         correct.

#![cfg(any(feature = "alloc", feature = "arrayvec"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "arrayvec"))))]

pub mod range;

use core::fmt::{self, Debug, Formatter};
use core::ops::RangeBounds;

// The ranges constructed by these trait functions are `'static`. For relational ranges over
// non-trivial types like `Vec` (for example, in a `BTreeSet<Vec<isize>>`), these ranges can be
// expensive to construct this way. However, this trait is primarily used to implement
// `Segment::intersect` and friends, which construct a `Segment`. Because `Segment`s hold an
// exclusive reference to their target, they cannot also hold shared references into keys in that
// target, so these ranges must clone the data in that case. At time of writing, this is the only
// use case, so this trait punts on borrowing indices into the target.
pub trait Ranged {
    type NominalRange;

    fn all(&self) -> Self::NominalRange;

    fn tail(&self) -> Self::NominalRange;

    fn rtail(&self) -> Self::NominalRange;
}

pub trait SegmentedOver: Sized {
    // This type is used in the output of `SegmentedBy::segment` and related functions of
    // `Segmentation`. Though not strictly necessary, it prevents sub-segmentation from
    // constructing arbitrarily nested types in outputs.
    type Kind: SegmentedOver<Target = Self::Target>;
    type Target: Ranged;
}

// TODO: Support segmenting over a query type `Q` borrowed from a key or owned index type `K`. Note
//       that any segment obtained via a type `Q` must always index with this type (not `K`),
//       because the `Ord` implementations can disagree and expose items outside of the segment's
//       range. This can be unsound, such as removing an item from a non-empty collection that is
//       out of the segment's range.
//
//       Note that this may interact poorly with the `tail` and `rtail` APIs, because they must
//       decide on `Q` and so any such segment must always use this `Q`.
// This trait implements `segment` rather than `Segmentation` so that implementors can apply
// arbitrary bounds to `R` while `Segmentation::segment` can lower those bounds into the function
// (rather than the trait).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be segmented by the type `{R}`",
    label = "segmented by the type `{R}` here",
    note = "positional collections are typically segmented by ranges over `usize`",
    note = "relational collections are typically segmented by ranges over the item type"
)]
pub trait SegmentedBy<N, R>: SegmentedOver
where
    N: ?Sized,
    R: RangeBounds<N>,
{
    type Range;

    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target, Self::Range>;
}

pub trait Segmentation: SegmentedOver {
    type Tail;

    fn segment<N, R>(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target, Self::Range>
    where
        Self: SegmentedBy<N, R>,
        N: ?Sized,
        R: RangeBounds<N>,
    {
        SegmentedBy::segment(self, range)
    }

    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Tail>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target, Self::Tail>;
}

pub struct Segment<'a, K, T, R>
where
    K: SegmentedOver<Target = T>,
    T: Ranged + ?Sized,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: R,
}

impl<K, T, R> Debug for Segment<'_, K, T, R>
where
    K: SegmentedOver<Target = T>,
    K::Target: Debug,
    T: Debug + Ranged,
    R: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Segment")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

impl<K, T, R> SegmentedOver for Segment<'_, K, T, R>
where
    K: SegmentedOver<Target = T>,
    T: Ranged,
{
    type Kind = K;
    type Target = K::Target;
}

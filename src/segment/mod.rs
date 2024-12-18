// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound. The
//         segmentation APIs interact with non-empty collections and bugs here may break the
//         non-empty invariant. In particular, range types, intersection, and projection must be
//         correct.

#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

pub mod range;

use core::fmt::{self, Debug, Formatter};

pub trait Indexer {
    type Index;
}

pub trait Ranged {
    type Range: Indexer;

    fn range(&self) -> Self::Range;

    fn tail(&self) -> Self::Range;

    fn rtail(&self) -> Self::Range;
}

pub trait SegmentedOver: Sized {
    type Target: Ranged;
    // This type is used in the output of `SegmentedBy::segment` and related functions of
    // `Segmentation`. Though not strictly necessary, it prevents sub-segmentation from
    // constructing arbitrarily nested types in outputs.
    type Kind: SegmentedOver<Target = Self::Target>;
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
pub trait SegmentedBy<R>: SegmentedOver {
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target>;
}

pub trait Segmentation: SegmentedOver {
    fn segment<R>(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target>
    where
        Self: SegmentedBy<R>,
    {
        SegmentedBy::segment(self, range)
    }

    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target>;
}

pub struct Segment<'a, K, T>
where
    K: SegmentedOver<Target = T>,
    T: Ranged,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: <K::Target as Ranged>::Range,
}

impl<K, T> Debug for Segment<'_, K, T>
where
    K: SegmentedOver<Target = T>,
    K::Target: Debug,
    <K::Target as Ranged>::Range: Debug,
    T: Debug + Ranged,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Segment")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

impl<K, T> SegmentedOver for Segment<'_, K, T>
where
    K: SegmentedOver<Target = T>,
    T: Ranged,
{
    type Target = K::Target;
    type Kind = K;
}

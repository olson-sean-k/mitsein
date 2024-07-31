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

pub trait Segmented {
    type Kind: Segmented<Target = Self::Target>;
    type Target: Ranged;
}

// TODO: Support segmenting over a query type `Q` borrowed from a key or owned index type `K`. Note
//       that any segment obtained via a type `Q` must always index with this type (not `K`),
//       because the `Ord` implementations can disagree and expose items outside of the segment's
//       range. This can be unsound, such as removing an item from a non-empty collection that is
//       out of the segment's range.
// This trait implements `segment` rather than `Segmentation` so that implementors can apply
// arbitrary bounds to `R` while `Segmentation::segment` can lower those bounds into the function
// (rather than the trait).
pub trait SegmentedBy<R>: Segmented {
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target>;
}

pub trait Segmentation: Segmented {
    fn segment<R>(&mut self, range: R) -> Segment<'_, Self::Kind, Self::Target>
    where
        Self: SegmentedBy<R>,
    {
        SegmentedBy::segment(self, range)
    }

    fn tail(&mut self) -> Segment<'_, Self::Kind, Self::Target>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind, Self::Target>;
}

// TODO: The input type parameter `T` ought not be necessary here, but, at time of writing, the
//       following `impl`s are considered duplicate definitions (E0592):
//
//       impl<'a, K, T> Segment<'a, K>
//       where
//           K: Segmented<Target = Vec<T>>,
//       {
//           fn noop(&self) {}
//       }
//
//       impl<'a, K, T> Segment<'a, K>
//       where
//           K: Segmented<Target = VecDeque<T>>,
//       {
//           fn noop(&self) {}
//       }
//
//       Note that the associated `Target` types in the bounds on `K` differ. There is no `K` that
//       can satisfy both of these bounds, but `rustc` cannot yet reason about this. Remove this
//       redundant input type parameter if and when such `impl`s no longer conflict.
pub struct Segment<'a, K, T>
where
    K: Segmented<Target = T> + ?Sized,
    T: Ranged + ?Sized,
{
    pub(crate) items: &'a mut T,
    pub(crate) range: <K::Target as Ranged>::Range,
}

impl<'a, K, T> Debug for Segment<'a, K, T>
where
    K: Segmented<Target = T> + ?Sized,
    <K::Target as Ranged>::Range: Debug,
    T: Debug + Ranged + ?Sized,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Segment")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

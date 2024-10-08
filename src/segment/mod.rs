// SAFETY: Though this module contains no unsafe code, an incorrect implementation is unsound. The
//         segmentation APIs interact with non-empty collections and bugs here may break the
//         non-empty invariant. In particular, range types, intersection, and projection must be
//         correct.

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

pub trait SegmentedOver {
    type Kind: SegmentedOver<Target = Self::Target>;
    type Target: Ranged + ?Sized;
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
    fn segment(&mut self, range: R) -> Segment<'_, Self::Kind>;
}

pub trait Segmentation: SegmentedOver {
    fn segment<R>(&mut self, range: R) -> Segment<'_, Self::Kind>
    where
        Self: SegmentedBy<R>,
    {
        SegmentedBy::segment(self, range)
    }

    fn tail(&mut self) -> Segment<'_, Self::Kind>;

    fn rtail(&mut self) -> Segment<'_, Self::Kind>;
}

pub struct Segment<'a, K>
where
    K: SegmentedOver + ?Sized,
{
    pub(crate) items: &'a mut K::Target,
    pub(crate) range: <K::Target as Ranged>::Range,
}

impl<'a, K> Debug for Segment<'a, K>
where
    K: SegmentedOver + ?Sized,
    K::Target: Debug,
    <K::Target as Ranged>::Range: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Segment")
            .field("items", &self.items)
            .field("range", &self.range)
            .finish()
    }
}

impl<'a, K> SegmentedOver for Segment<'a, K>
where
    K: SegmentedOver,
{
    type Kind = K;
    type Target = K::Target;
}

// TODO: The forwarding type constructors implemented by this macro ought not be necessary, but at
//       time of writing the following `impl`s are considered duplicate definitions (E0592):
//
//       impl<'a, K, T> Segment<'a, K>
//       where
//           K: Segmentation<Target = Vec<T>>,
//       {
//           fn noop(&self) {}
//       }
//
//       impl<'a, K, T> Segment<'a, K>
//       where
//           K: Segmentation<Target = VecDeque<T>>,
//       {
//           fn noop(&self) {}
//       }
//
//       Note that the associated `Target` types in the bounds on `K` differ. There is no `K` that
//       can satisfy both of these bounds, but `rustc` cannot yet reason about this. Instead, a
//       bespoke forwarding type constructor is composed with `K` for each `Target` type so that
//       such `impl`s remain disjoint by effectively lifting the `Target` type into the input type
//       parameter `K`.
//
//       Remove this macro and the forwarding types when possible.
// LINT: This macro is unused when segmentation is not implemented. Segmentation is implemented
//       when any of the `alloc` or `arrayvec` features are enabled.
#[cfg_attr(
    not(any(feature = "alloc", feature = "arrayvec")),
    expect(unused_macros)
)]
macro_rules! impl_target_forward_type_and_definition {
    (
        for <$($ts:ident $(,)?)+ $([$(const $cis:ident: $cts:ty $(,)?)+])? $(,)?>
        $(where $($bt:ident: $bb:tt $(+ $bbs:tt)* $(,)?)+)? => $target:ident,
        $forward:ident,
        $(#[$($attrs:tt)*])*
        $segment:ident $(,)?
    ) => {
        mod segment_forward_segmentation_ {
            // The `$target` must be imported into the parent scope.
            use super::*;

            pub struct $forward<K>(core::marker::PhantomData<fn() -> K>, core::convert::Infallible);

            impl<K_, $($ts,)+ $($(const $cis: $cts,)+)?>
            $crate::segment::SegmentedOver for $forward<K_>
            where
                K_: $crate::segment::SegmentedOver<Target = $target::<$($ts,)+ $($($cis,)+)?>>,
                $($($bt: $bb $(+ $bbs)*,)+)?
            {
                type Kind = Self;
                type Target = $target<$($ts,)+ $($($cis,)+)?>;
            }
        }
        use self::segment_forward_segmentation_::$forward;

        $(#[$($attrs)*])*
        pub type $segment<'a, K> = Segment<'a, $forward<K>>;
    }
}
// LINT: This macro is unused when segmentation is not implemented. Segmentation is implemented
//       when any of the `alloc` or `arrayvec` features are enabled.
#[cfg_attr(
    not(any(feature = "alloc", feature = "arrayvec")),
    expect(unused_imports)
)]
pub(crate) use impl_target_forward_type_and_definition;

//! A non-empty [`IndexSet`][`set`].

#![cfg(feature = "indexmap")]
#![cfg_attr(docsrs, doc(cfg(feature = "indexmap")))]

use core::borrow::Borrow;
use core::fmt::{self, Debug, Formatter};
use core::hash::{BuildHasher, Hash};
use core::num::NonZeroUsize;
use core::ops::{BitAnd, BitOr, BitXor, RangeBounds, Sub};
use indexmap::set::{self, IndexSet};
use indexmap::Equivalent;
#[cfg(feature = "std")]
use std::hash::RandomState;

#[cfg(feature = "std")]
use crate::array1::Array1;
use crate::iter1::{self, Extend1, FromIterator1, IntoIterator1, Iterator1};
#[cfg(feature = "rayon")]
use crate::parallel;
use crate::safety::{NonZeroExt as _, OptionExt as _};
use crate::segment::range::{self, PositionalRange, Project};
use crate::segment::{self, Ranged, Segmentation, SegmentedBy, SegmentedOver};
use crate::take;
use crate::{FromMaybeEmpty, MaybeEmpty, NonEmpty};

impl<T, S> Extend1<T> for IndexSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend_non_empty<I>(mut self, items: I) -> IndexSet1<T, S>
    where
        I: IntoIterator1<Item = T>,
    {
        self.extend(items);
        // SAFETY: The input iterator `items` is non-empty and `extend` either pushes one or more
        //         items or panics, so `self` must be non-empty here.
        unsafe { IndexSet1::from_index_set_unchecked(self) }
    }
}

unsafe impl<T, S> MaybeEmpty for IndexSet<T, S> {
    fn cardinality(&self) -> Option<crate::Cardinality<(), ()>> {
        match self.len() {
            0 => None,
            1 => Some(crate::Cardinality::One(())),
            _ => Some(crate::Cardinality::Many(())),
        }
    }
}

impl<T, S> Ranged for IndexSet<T, S> {
    type Range = PositionalRange;

    fn range(&self) -> Self::Range {
        From::from(0..self.len())
    }

    fn tail(&self) -> Self::Range {
        From::from(1..self.len())
    }

    fn rtail(&self) -> Self::Range {
        From::from(0..self.len().saturating_sub(1))
    }
}

impl<T, S> Segmentation for IndexSet<T, S> {
    fn tail(&mut self) -> Segment<'_, Self, T, S> {
        Segmentation::segment(self, Ranged::tail(self))
    }

    fn rtail(&mut self) -> Segment<'_, Self, T, S> {
        Segmentation::segment(self, Ranged::rtail(self))
    }
}

impl<T, S, R> SegmentedBy<R> for IndexSet<T, S>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self, T, S> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T, S> SegmentedOver for IndexSet<T, S> {
    type Kind = Self;
    type Target = Self;
}

type TakeOr<'a, T, S, U, N = ()> = take::TakeOr<'a, IndexSet<T, S>, U, N>;

pub type PopOr<'a, T, S> = TakeOr<'a, T, S, T>;

pub type DropRemoveOr<'a, 'q, T, S, Q> = TakeOr<'a, T, S, bool, &'q Q>;

pub type TakeRemoveOr<'a, 'q, T, S, Q> = TakeOr<'a, T, S, Option<T>, &'q Q>;

impl<'a, T, S, U, N> TakeOr<'a, T, S, U, N> {
    pub fn only(self) -> Result<U, &'a T> {
        self.take_or_else(|items, _| items.first())
    }
}

impl<'a, T, S, Q> TakeOr<'a, T, S, bool, &'_ Q>
where
    T: Borrow<Q>,
    S: BuildHasher,
    Q: Equivalent<T> + Hash + ?Sized,
{
    pub fn get(self) -> Result<bool, Option<&'a T>> {
        self.take_or_else(|items, query| items.get(query))
    }
}

impl<'a, T, S, Q> TakeOr<'a, T, S, Option<T>, &'_ Q>
where
    T: Borrow<Q>,
    S: BuildHasher,
    Q: Equivalent<T> + Hash + ?Sized,
{
    pub fn get(self) -> Option<Result<T, &'a T>> {
        self.try_take_or_else(|items, query| items.get(query))
    }
}

#[cfg(feature = "std")]
pub type IndexSet1<T, S = RandomState> = NonEmpty<IndexSet<T, S>>;

#[cfg(not(feature = "std"))]
pub type IndexSet1<T, S> = NonEmpty<IndexSet<T, S>>;

impl<T, S> IndexSet1<T, S> {
    /// # Safety
    ///
    /// `items` must be non-empty. For example, it is unsound to call this function with the
    /// immediate output of [`IndexSet::new()`][`IndexSet::new`].
    ///
    /// [`IndexSet::new`]: indexmap::set::IndexSet::new
    pub unsafe fn from_index_set_unchecked(items: IndexSet<T, S>) -> Self {
        FromMaybeEmpty::from_maybe_empty_unchecked(items)
    }

    pub fn into_index_set(self) -> IndexSet<T, S> {
        self.items
    }

    pub fn split_off_tail(&mut self) -> IndexSet<T, S>
    where
        S: Clone,
    {
        self.items.split_off(1)
    }

    pub fn pop_or(&mut self) -> PopOr<'_, T, S>
    where
        T: Eq + Hash,
    {
        // SAFETY: `with` executes this closure only if `self` contains more than one item.
        TakeOr::with(self, (), |items, _| unsafe {
            items.items.pop().unwrap_maybe_unchecked()
        })
    }

    pub fn first(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.first().unwrap_maybe_unchecked() }
    }

    pub fn last(&self) -> &T {
        // SAFETY: `self` must be non-empty.
        unsafe { self.items.last().unwrap_maybe_unchecked() }
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY: `self` must be non-empty.
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn iter1(&self) -> Iterator1<set::Iter<'_, T>> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    pub fn parallel(&self) -> Parallel<'_, T, S>
    where
        T: Sync,
        S: Sync,
    {
        Parallel { items: self }
    }

    pub const fn as_index_set(&self) -> &IndexSet<T, S> {
        &self.items
    }
}

impl<T, S> IndexSet1<T, S>
where
    S: BuildHasher,
{
    pub fn get<Q>(&self, query: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        self.items.get(query)
    }

    pub fn swap_remove_or<'a, 'q, Q>(&'a mut self, query: &'q Q) -> DropRemoveOr<'a, 'q, T, S, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| items.items.swap_remove(query))
    }

    pub fn swap_take_or<'a, 'q, Q>(&'a mut self, query: &'q Q) -> TakeRemoveOr<'a, 'q, T, S, Q>
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        TakeOr::with(self, query, |items, query| items.items.swap_take(query))
    }

    pub fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Equivalent<T> + Hash + ?Sized,
    {
        self.items.contains(item)
    }
}

impl<T, S> IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    pub fn from_one(item: T) -> Self
    where
        S: Default,
    {
        iter1::one(item).collect1()
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        S: Default,
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        S: Default,
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn append<R, SR>(&mut self, items: R)
    where
        R: Into<IndexSet<T, SR>>,
    {
        self.items.append(&mut items.into())
    }

    pub fn insert(&mut self, item: T) -> bool {
        self.items.insert(item)
    }

    pub fn replace(&mut self, item: T) -> Option<T> {
        self.items.replace(item)
    }

    pub fn difference<'a, R, SR>(&'a self, other: &'a R) -> set::Difference<'a, T, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.difference(other.as_ref())
    }

    pub fn symmetric_difference<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> set::SymmetricDifference<'a, T, S, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.symmetric_difference(other.as_ref())
    }

    pub fn intersection<'a, R, SR>(&'a self, other: &'a R) -> set::Intersection<'a, T, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.intersection(other.as_ref())
    }

    pub fn union<'a, R, SR>(&'a self, other: &'a R) -> Iterator1<set::Union<'a, T, S>>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: 'a + BuildHasher,
    {
        // SAFETY: `self` must be non-empty and `BTreeSet::union` cannot reduce the cardinality of
        //         its inputs.
        unsafe { Iterator1::from_iter_unchecked(self.items.union(other.as_ref())) }
    }

    pub fn is_disjoint<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.is_disjoint(other.as_ref())
    }

    pub fn is_subset<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.is_subset(other.as_ref())
    }

    pub fn is_superset<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher,
    {
        self.items.is_superset(other.as_ref())
    }
}

impl<R, T, S> BitAnd<&'_ R> for &'_ IndexSet1<T, S>
where
    R: AsRef<IndexSet<T, S>>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn bitand(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() & rhs.as_ref()
    }
}

impl<R, T, S> BitOr<&'_ R> for &'_ IndexSet1<T, S>
where
    R: AsRef<IndexSet<T, S>>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet1<T, S>;

    fn bitor(self, rhs: &'_ R) -> Self::Output {
        // SAFETY: `self` must be non-empty and `IndexSet::bitor` cannot reduce the cardinality of
        //         its inputs.
        unsafe { IndexSet1::from_index_set_unchecked(self.as_index_set() | rhs.as_ref()) }
    }
}

impl<R, T, S> BitXor<&'_ R> for &'_ IndexSet1<T, S>
where
    R: AsRef<IndexSet<T, S>>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn bitxor(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() ^ rhs.as_ref()
    }
}

impl<T, S> Debug for IndexSet1<T, S>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T, S> Extend<T> for IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

#[cfg(feature = "std")]
impl<T, const N: usize> From<[T; N]> for IndexSet1<T, RandomState>
where
    [T; N]: Array1,
    T: Eq + Hash,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY: `items` is non-empty.
        unsafe { IndexSet1::from_index_set_unchecked(IndexSet::from(items)) }
    }
}

impl<T, S> From<IndexSet1<T, S>> for IndexSet<T, S> {
    fn from(items: IndexSet1<T, S>) -> Self {
        items.items
    }
}

impl<T, S> FromIterator1<T> for IndexSet1<T, S>
where
    T: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY: `items` is non-empty.
        unsafe { IndexSet1::from_index_set_unchecked(items.into_iter().collect()) }
    }
}

impl<T, S> IntoIterator for IndexSet1<T, S> {
    type Item = T;
    type IntoIter = set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T, S> IntoIterator1 for IndexSet1<T, S> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY: `self` must be non-empty.
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, S> Segmentation for IndexSet1<T, S> {
    fn tail(&mut self) -> Segment<'_, Self, T, S> {
        Segmentation::segment(self, Ranged::tail(&self.items))
    }

    fn rtail(&mut self) -> Segment<'_, Self, T, S> {
        Segmentation::segment(self, Ranged::rtail(&self.items))
    }
}

impl<T, S, R> SegmentedBy<R> for IndexSet1<T, S>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, Self, T, S> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T, S> SegmentedOver for IndexSet1<T, S> {
    type Target = IndexSet<T, S>;
    type Kind = Self;
}

impl<R, T, S> Sub<&'_ R> for &'_ IndexSet1<T, S>
where
    R: AsRef<IndexSet<T, S>>,
    T: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Output = IndexSet<T, S>;

    fn sub(self, rhs: &'_ R) -> Self::Output {
        self.as_index_set() - rhs.as_ref()
    }
}

impl<T, S> TryFrom<IndexSet<T, S>> for IndexSet1<T, S> {
    type Error = IndexSet<T, S>;

    fn try_from(items: IndexSet<T, S>) -> Result<Self, Self::Error> {
        FromMaybeEmpty::try_from_maybe_empty(items)
    }
}

#[cfg(all(feature = "rayon", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub type Parallel<'a, T, S = RandomState> = parallel::Parallel<'a, IndexSet<T, S>>;

#[cfg(all(feature = "rayon", not(feature = "std")))]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
pub type Parallel<'a, T, S> = parallel::Parallel<'a, IndexSet<T, S>>;

#[cfg(feature = "rayon")]
impl<T, S> Parallel<'_, T, S>
where
    T: Eq + Hash + Sync,
    S: BuildHasher + Sync,
{
    pub fn difference<'a, R, SR>(&'a self, other: &'a R) -> set::rayon::ParDifference<'a, T, S, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_difference(other.as_ref())
    }

    pub fn symmetric_difference<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> set::rayon::ParSymmetricDifference<'a, T, S, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_symmetric_difference(other.as_ref())
    }

    pub fn intersection<'a, R, SR>(
        &'a self,
        other: &'a R,
    ) -> set::rayon::ParIntersection<'a, T, S, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_intersection(other.as_ref())
    }

    // TODO: Implement parallel non-empty iterators and use them here.
    pub fn union<'a, R, SR>(&'a self, other: &'a R) -> set::rayon::ParUnion<'a, T, S, SR>
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: 'a + BuildHasher + Sync,
    {
        self.items.items.par_union(other.as_ref())
    }

    pub fn is_disjoint<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_is_disjoint(other.as_ref())
    }

    pub fn is_subset<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_is_subset(other.as_ref())
    }

    pub fn is_superset<R, SR>(&self, other: &R) -> bool
    where
        R: AsRef<IndexSet<T, SR>>,
        SR: BuildHasher + Sync,
    {
        self.items.items.par_is_superset(other.as_ref())
    }
}

#[cfg(feature = "std")]
pub type Segment<'a, K, T, S = RandomState> = segment::Segment<'a, K, IndexSet<T, S>>;

#[cfg(not(feature = "std"))]
pub type Segment<'a, K, T, S> = segment::Segment<'a, K, IndexSet<T, S>>;

// TODO: It should be possible to safely implement `swap_drain` for segments over `IndexSet1`. The
//       `IndexSet::drain` iterator immediately culls its indices but then defers to `vec::Drain`
//       for removing buckets. `IndexSet::swap_indices` can be used much like `slice::swap` here.
impl<K, T, S> Segment<'_, K, T, S>
where
    K: SegmentedOver<Target = IndexSet<T, S>>,
{
    pub fn clear(&mut self) {
        self.items.drain(self.range.get_and_clear_from_end());
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }
}

impl<K, T, S> Segmentation for Segment<'_, K, T, S>
where
    K: SegmentedOver<Target = IndexSet<T, S>>,
{
    fn tail(&mut self) -> Segment<'_, K, T, S> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> Segment<'_, K, T, S> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<K, T, S, R> SegmentedBy<R> for Segment<'_, K, T, S>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: SegmentedBy<R> + SegmentedOver<Target = IndexSet<T, S>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> Segment<'_, K, T, S> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

#[cfg(all(test, feature = "std"))]
pub mod harness {
    use rstest::fixture;

    use crate::index_set1::IndexSet1;
    use crate::iter1::{self, FromIterator1};

    #[fixture]
    pub fn xs1(#[default(4)] end: u8) -> IndexSet1<u8> {
        IndexSet1::from_iter1(iter1::harness::xs1(end))
    }
}

#[cfg(test)]
mod tests {}

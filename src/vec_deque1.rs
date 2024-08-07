#![cfg(feature = "alloc")]
#![cfg_attr(docsrs, doc(cfg(feature = "alloc")))]

use alloc::collections::vec_deque::{self, VecDeque};
use core::cmp::Ordering;
use core::fmt::{self, Debug, Formatter};
use core::iter;
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{Index, IndexMut, RangeBounds};

use crate::array1::Array1;
use crate::iter1::{self, FromIterator1, IntoIterator1, Iterator1, Saturate};
use crate::segment::range::{self, PositionalRange, Project, ProjectionExt as _};
use crate::segment::{self, Ranged, Segment, Segmentation, SegmentedOver};
#[cfg(feature = "serde")]
use crate::serde::{EmptyError, Serde};
use crate::slice1::Slice1;
use crate::{NonEmpty, NonZeroExt as _, OptionExt as _, Vacancy};

segment::impl_target_forward_type_and_definition!(
    for <T> => VecDeque,
    VecDequeTarget,
    VecDequeSegment,
);

impl<T> Ranged for VecDeque<T> {
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

impl<T, I> Saturate<I> for VecDeque<T>
where
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<T> Segmentation for VecDeque<T> {
    fn tail(&mut self) -> VecDequeSegment<'_, Self> {
        Segment::intersect(self, &Ranged::tail(self))
    }

    fn rtail(&mut self) -> VecDequeSegment<'_, Self> {
        Segment::intersect(self, &Ranged::rtail(self))
    }
}

impl<T, R> segment::SegmentedBy<R> for VecDeque<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecDequeSegment<'_, Self> {
        Segment::intersect(self, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for VecDeque<T> {
    type Kind = VecDequeTarget<Self>;
    type Target = Self;
}

impl<T> Vacancy for VecDeque<T> {
    fn vacancy(&self) -> usize {
        self.capacity() - self.len()
    }
}

pub type VecDeque1<T> = NonEmpty<VecDeque<T>>;

impl<T> VecDeque1<T> {
    /// # Safety
    pub const unsafe fn from_vec_deque_unchecked(items: VecDeque<T>) -> Self {
        VecDeque1 { items }
    }

    pub fn from_one(item: T) -> Self {
        iter1::one(item).collect1()
    }

    pub fn from_one_with_capacity(item: T, capacity: usize) -> Self {
        let mut items = VecDeque::with_capacity(capacity);
        items.push_back(item);
        // SAFETY:
        unsafe { VecDeque1::from_vec_deque_unchecked(items) }
    }

    pub fn from_head_and_tail<I>(head: T, tail: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::head_and_tail(head, tail).collect1()
    }

    pub fn from_tail_and_head<I>(tail: I, head: T) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter1::tail_and_head(tail, head).collect1()
    }

    pub fn into_vec_deque(self) -> VecDeque<T> {
        self.items
    }

    fn many_or_else<'a, U, M, O>(&'a mut self, many: M, one: O) -> Result<T, U>
    where
        M: FnOnce(&'a mut VecDeque<T>) -> T,
        O: FnOnce(&'a mut VecDeque<T>) -> U,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => Err(one(&mut self.items)),
            _ => Ok(many(&mut self.items)),
        }
    }

    fn try_many_or_else<'a, U, M, O>(&'a mut self, many: M, one: O) -> Option<Result<T, U>>
    where
        M: FnOnce(&'a mut VecDeque<T>) -> Option<T>,
        O: FnOnce(&'a mut VecDeque<T>) -> Option<U>,
    {
        match self.items.len() {
            0 => unreachable!(),
            1 => one(&mut self.items).map(Err),
            _ => many(&mut self.items).map(Ok),
        }
    }

    fn many_or_get_only<F>(&mut self, f: F) -> Result<T, &T>
    where
        F: FnOnce(&mut VecDeque<T>) -> T,
    {
        // SAFETY:
        self.many_or_else(f, |items| unsafe { items.front().unwrap_maybe_unchecked() })
    }

    fn many_or_replace_only_with<F, R>(&mut self, f: F, replace: R) -> Result<T, T>
    where
        F: FnOnce(&mut VecDeque<T>) -> T,
        R: FnOnce() -> T,
    {
        // SAFETY:
        self.many_or_else(f, move |items| unsafe {
            mem::replace(items.get_mut(0).unwrap_maybe_unchecked(), replace())
        })
    }

    fn try_many_or_get<F>(&mut self, index: usize, f: F) -> Option<Result<T, &T>>
    where
        F: FnOnce(&mut VecDeque<T>) -> Option<T>,
    {
        self.try_many_or_else(f, move |items| items.get(index))
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        self.items.shrink_to(capacity)
    }

    pub fn shrink_to_fit(&mut self) {
        self.items.shrink_to_fit()
    }

    pub fn make_contiguous(&mut self) -> &mut Slice1<T> {
        // SAFETY:
        unsafe { Slice1::from_mut_slice_unchecked(self.items.make_contiguous()) }
    }

    pub fn rotate_left(&mut self, n: usize) {
        self.items.rotate_left(n)
    }

    pub fn rotate_right(&mut self, n: usize) {
        self.items.rotate_right(n)
    }

    pub fn split_off_tail(&mut self) -> VecDeque<T> {
        self.items.split_off(1)
    }

    pub fn append<R>(&mut self, items: R)
    where
        R: Into<VecDeque<T>>,
    {
        self.items.append(&mut items.into())
    }

    pub fn push_front(&mut self, item: T) {
        self.items.push_front(item)
    }

    pub fn push_back(&mut self, item: T) {
        self.items.push_back(item)
    }

    pub fn pop_front_or_get_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_get_only(|items| unsafe { items.pop_front().unwrap_maybe_unchecked() })
    }

    pub fn pop_front_or_replace_only(&mut self, replacement: T) -> Result<T, T> {
        self.pop_front_or_replace_only_with(move || replacement)
    }

    pub fn pop_front_or_replace_only_with<F>(&mut self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        // SAFETY:
        self.many_or_replace_only_with(
            |items| unsafe { items.pop_front().unwrap_maybe_unchecked() },
            f,
        )
    }

    pub fn pop_back_or_get_only(&mut self) -> Result<T, &T> {
        // SAFETY:
        self.many_or_get_only(|items| unsafe { items.pop_back().unwrap_maybe_unchecked() })
    }

    pub fn pop_back_or_replace_only(&mut self, replacement: T) -> Result<T, T> {
        self.pop_back_or_replace_only_with(move || replacement)
    }

    pub fn pop_back_or_replace_only_with<F>(&mut self, f: F) -> Result<T, T>
    where
        F: FnOnce() -> T,
    {
        // SAFETY:
        self.many_or_replace_only_with(
            |items| unsafe { items.pop_back().unwrap_maybe_unchecked() },
            f,
        )
    }

    pub fn insert(&mut self, index: usize, item: T) {
        self.items.insert(index, item)
    }

    pub fn remove_or_get_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_get(index, move |items| items.remove(index))
    }

    pub fn remove_or_replace_only(&mut self, index: usize, replacement: T) -> Option<Result<T, T>> {
        self.remove_or_replace_only_with(index, move || replacement)
    }

    pub fn remove_or_replace_only_with<F>(&mut self, index: usize, f: F) -> Option<Result<T, T>>
    where
        F: FnOnce() -> T,
    {
        self.try_many_or_else(
            move |items| items.remove(index),
            move |items| {
                items
                    .get_mut(index)
                    .map(move |only| mem::replace(only, f()))
            },
        )
    }

    pub fn swap_remove_front_or_get_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_get(index, move |items| items.swap_remove_front(index))
    }

    pub fn swap_remove_back_or_get_only(&mut self, index: usize) -> Option<Result<T, &T>> {
        self.try_many_or_get(index, move |items| items.swap_remove_back(index))
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    pub fn len(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.len()) }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_maybe_unchecked(self.items.capacity()) }
    }

    pub fn front(&self) -> &T {
        // SAFETY:
        unsafe { self.items.front().unwrap_maybe_unchecked() }
    }

    pub fn front_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.front_mut().unwrap_maybe_unchecked() }
    }

    pub fn back(&self) -> &T {
        // SAFETY:
        unsafe { self.items.back().unwrap_maybe_unchecked() }
    }

    pub fn back_mut(&mut self) -> &mut T {
        // SAFETY:
        unsafe { self.items.back_mut().unwrap_maybe_unchecked() }
    }

    pub fn iter1(&self) -> Iterator1<vec_deque::Iter<'_, T>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.iter()) }
    }

    pub fn iter1_mut(&mut self) -> Iterator1<vec_deque::IterMut<'_, T>> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items.iter_mut()) }
    }

    pub const fn as_vec_deque(&self) -> &VecDeque<T> {
        &self.items
    }
}

impl<T> Debug for VecDeque1<T>
where
    T: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.items.iter()).finish()
    }
}

impl<T> Extend<T> for VecDeque1<T> {
    fn extend<I>(&mut self, extension: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(extension)
    }
}

impl<T, const N: usize> From<[T; N]> for VecDeque1<T>
where
    [T; N]: Array1,
{
    fn from(items: [T; N]) -> Self {
        // SAFETY:
        unsafe { VecDeque1::from_vec_deque_unchecked(VecDeque::from(items)) }
    }
}

impl<T> From<VecDeque1<T>> for VecDeque<T> {
    fn from(items: VecDeque1<T>) -> Self {
        items.items
    }
}

impl<T> FromIterator1<T> for VecDeque1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>,
    {
        // SAFETY:
        unsafe { VecDeque1::from_vec_deque_unchecked(items.into_iter1().collect()) }
    }
}

impl<T, I> Index<I> for VecDeque1<T>
where
    VecDeque<T>: Index<I>,
{
    type Output = <VecDeque<T> as Index<I>>::Output;

    fn index(&self, at: I) -> &Self::Output {
        self.items.index(at)
    }
}

impl<T, I> IndexMut<I> for VecDeque1<T>
where
    VecDeque<T>: IndexMut<I>,
{
    fn index_mut(&mut self, at: I) -> &mut Self::Output {
        self.items.index_mut(at)
    }
}

impl<T> IntoIterator for VecDeque1<T> {
    type Item = T;
    type IntoIter = vec_deque::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<T> IntoIterator1 for VecDeque1<T> {
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        // SAFETY:
        unsafe { Iterator1::from_iter_unchecked(self.items) }
    }
}

impl<T, I> Saturate<I> for VecDeque1<T>
where
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<T> Segmentation for VecDeque1<T> {
    fn tail(&mut self) -> VecDequeSegment<'_, Self> {
        let range = Ranged::tail(&self.items);
        Segment::intersect_strict_subset(&mut self.items, &range)
    }

    fn rtail(&mut self) -> VecDequeSegment<'_, Self> {
        let range = Ranged::rtail(&self.items);
        Segment::intersect_strict_subset(&mut self.items, &range)
    }
}

impl<T, R> segment::SegmentedBy<R> for VecDeque1<T>
where
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecDequeSegment<'_, Self> {
        Segment::intersect_strict_subset(&mut self.items, &range::ordered_range_offsets(range))
    }
}

impl<T> SegmentedOver for VecDeque1<T> {
    type Kind = VecDequeTarget<Self>;
    type Target = VecDeque<T>;
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<T> TryFrom<Serde<VecDeque<T>>> for VecDeque1<T> {
    type Error = EmptyError;

    fn try_from(serde: Serde<VecDeque<T>>) -> Result<Self, Self::Error> {
        VecDeque1::try_from(serde.items).map_err(|_| EmptyError)
    }
}

impl<T> TryFrom<VecDeque<T>> for VecDeque1<T> {
    type Error = VecDeque<T>;

    fn try_from(items: VecDeque<T>) -> Result<Self, Self::Error> {
        match items.len() {
            0 => Err(items),
            // SAFETY:
            _ => Ok(unsafe { VecDeque1::from_vec_deque_unchecked(items) }),
        }
    }
}

impl<T> Vacancy for VecDeque1<T> {
    fn vacancy(&self) -> usize {
        self.items.vacancy()
    }
}

impl<'a, K, T> VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
{
    pub fn split_off(&mut self, at: usize) -> VecDeque<T> {
        let at = self.range.project(&at).expect_in_bounds();
        let range = From::from(at..self.range.end);
        let items = self.items.drain(range).collect();
        self.range = range;
        items
    }

    pub fn resize(&mut self, len: usize, fill: T)
    where
        T: Clone,
    {
        self.resize_with(len, move || fill.clone())
    }

    pub fn resize_with<F>(&mut self, len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let basis = self.len();
        if len > basis {
            let n = len - basis;
            self.extend(iter::repeat_with(f).take(n))
        }
        else {
            self.truncate(len)
        }
    }

    pub fn truncate(&mut self, len: usize) {
        let basis = self.len();
        if len < basis {
            let n = basis - len;
            self.items.drain((self.range.end - n)..self.range.end);
            self.range.take_from_end(n);
        }
    }

    pub fn push_front(&mut self, item: T) {
        self.items.insert(self.range.start, item);
        self.range.put_from_end(1);
    }

    pub fn push_back(&mut self, item: T) {
        self.items.insert(self.range.end, item);
        self.range.put_from_end(1);
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            self.items
                .remove(self.range.start)
                .inspect(|_| self.range.take_from_end(1))
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.range.is_empty() {
            None
        }
        else {
            self.items
                .remove(self.range.end - 1)
                .inspect(|_| self.range.take_from_end(1))
        }
    }

    pub fn insert(&mut self, index: usize, item: T) {
        let index = self.range.project(&index).expect_in_bounds();
        self.items.insert(index, item);
        self.range.put_from_end(1);
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        self.items
            .remove(index)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_front(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        self.items.swap(index, self.range.start);
        self.items
            .remove(self.range.start)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn swap_remove_back(&mut self, index: usize) -> Option<T> {
        let index = self.range.project(&index).ok()?;
        let swapped = self.range.end - 1;
        self.items.swap(index, swapped);
        self.items
            .remove(swapped)
            .inspect(|_| self.range.take_from_end(1))
    }

    pub fn clear(&mut self) {
        self.items.drain(self.range.get_and_clear_from_end());
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn iter(&self) -> vec_deque::Iter<'_, T> {
        VecDeque::range(self.items, self.range)
    }

    pub fn iter_mut(&mut self) -> vec_deque::IterMut<'_, T> {
        self.items.range_mut(self.range)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, K, T> Eq for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
    T: Eq,
{
}

impl<'a, K, T> Extend<T> for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
{
    fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let n = self.items.len();
        // Split off the remainder beyond the segment to avoid spurious inserts and copying. This
        // comes at the cost of a necessary allocation and bulk copy, which isn't great when
        // extending from a small number of items with a small remainder.
        let tail = self.items.split_off(self.range.end);
        self.items.extend(items);
        self.items.extend(tail);
        let n = self.items.len() - n;
        self.range.put_from_end(n);
    }
}

// TODO: At time of writing, this implementation conflicts with the `Extend` implementation above
//       (E0119). However, `T` does not generalize `&'i T` here, because the associated `Target`
//       type is the same (`Vec<T>`) in both implementations (and a reference would be added to all
//       `T`)! This appears to be a limitation rather than a true conflict.
//
// impl<'a, 'i, K, T> Extend<&'i T> for VecDequeSegment<'a, K>
// where
//     K: Segmentation<Target = VecDeque<T>>,
//     T: 'i + Copy,
// {
//     fn extend<I>(&mut self, items: I)
//     where
//         I: IntoIterator<Item = &'i T>,
//     {
//         self.extend(items.into_iter().copied())
//     }
// }

impl<'a, K, T> Ord for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<'a, K, T> PartialEq<Self> for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
    T: PartialEq<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<'a, K, T> PartialOrd<Self> for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
    T: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<'a, K, T, I> Saturate<I> for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
    I: IntoIterator<Item = T>,
{
    fn saturate(&mut self, items: I) -> I::IntoIter {
        iter1::saturate_positional_vacancy(self, items)
    }
}

impl<'a, K, T> Segmentation for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
{
    fn tail(&mut self) -> VecDequeSegment<'_, K> {
        let range = self.project(&(1..));
        Segment::intersect(self.items, &range)
    }

    fn rtail(&mut self) -> VecDequeSegment<'_, K> {
        let range = self.project(&(..self.len().saturating_sub(1)));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T, R> segment::SegmentedBy<R> for VecDequeSegment<'a, K>
where
    PositionalRange: Project<R, Output = PositionalRange>,
    K: segment::SegmentedBy<R> + SegmentedOver<Target = VecDeque<T>>,
    R: RangeBounds<usize>,
{
    fn segment(&mut self, range: R) -> VecDequeSegment<'_, K> {
        let range = self.project(&range::ordered_range_offsets(range));
        Segment::intersect(self.items, &range)
    }
}

impl<'a, K, T> Vacancy for VecDequeSegment<'a, K>
where
    K: SegmentedOver<Target = VecDeque<T>>,
{
    fn vacancy(&self) -> usize {
        self.items.vacancy()
    }
}

#[cfg(test)]
mod tests {
    use crate::vec_deque1::VecDeque1;
    use crate::Segmentation;

    #[test]
    fn segmentation() {
        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.tail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[0]);

        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.tail().tail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[0, 1]);

        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.tail().rtail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[0, 3]);

        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.rtail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[3]);

        let mut xs = VecDeque1::from([0i32]);
        assert_eq!(xs.tail().len(), 0);
        xs.tail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[0]);

        let mut xs = VecDeque1::from([0i32]);
        assert_eq!(xs.rtail().len(), 0);
        xs.rtail().clear();
        assert_eq!(xs.make_contiguous().as_slice(), &[0]);

        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.tail().pop_front();
        assert_eq!(xs.make_contiguous().as_slice(), &[0, 2, 3]);

        let mut xs = VecDeque1::from([0i32, 1, 2, 3]);
        xs.segment(1..1).pop_front();
        assert_eq!(xs.make_contiguous().as_slice(), &[0, 1, 2, 3]);
    }
}

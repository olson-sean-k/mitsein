#![cfg(feature = "alloc")]

use core::borrow::Borrow;
use core::ops::{Bound, RangeBounds};

use crate::segment::range::{IntoRangeBounds, UnorderedError};

pub trait OptionExt<N> {
    fn contains<Q>(&self, key: &Q) -> bool
    where
        N: Borrow<Q> + Ord,
        Q: Ord + ?Sized;
}

impl<N> OptionExt<N> for Option<ItemRange<N>> {
    fn contains<Q>(&self, key: &Q) -> bool
    where
        N: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.as_ref().is_some_and(|range| range.contains(key))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ItemRange<N> {
    start: Bound<N>,
    end: Bound<N>,
}

impl<N> ItemRange<N> {
    pub(crate) fn unchecked(start: Bound<N>, end: Bound<N>) -> Self {
        ItemRange { start, end }
    }

    pub fn bounded(start: N, end: N) -> Result<Self, UnorderedError<N>>
    where
        N: Ord,
    {
        if start <= end {
            Ok(ItemRange::unchecked(
                Bound::Included(start),
                Bound::Excluded(end),
            ))
        }
        else {
            Err(UnorderedError(start, end))
        }
    }

    pub fn empty_at(at: N) -> Self
    where
        N: Clone,
    {
        ItemRange::unchecked(Bound::Included(at.clone()), Bound::Excluded(at))
    }

    pub fn retain_in_range<'a, F>(&'a self, mut f: F) -> impl 'a + FnMut(&N) -> bool
    where
        N: Ord,
        F: 'a + FnMut(&N) -> bool,
    {
        let mut by_key_value = self.retain_key_value_in_range(move |key, _| f(key));
        move |item| by_key_value(item, &mut ())
    }

    pub fn retain_key_value_in_range<'a, T, F>(
        &'a self,
        mut f: F,
    ) -> impl 'a + FnMut(&N, &mut T) -> bool
    where
        N: Ord,
        F: 'a + FnMut(&N, &mut T) -> bool,
    {
        move |key, value| {
            if self.contains(key) {
                f(key, value)
            }
            else {
                true
            }
        }
    }

    pub fn as_ref(&self) -> ItemRange<&N> {
        ItemRange::unchecked(self.start.as_ref(), self.end.as_ref())
    }

    pub fn borrow<Q>(&self) -> ItemRange<&Q>
    where
        N: Borrow<Q>,
        Q: ?Sized,
    {
        ItemRange::unchecked(
            self.start.as_ref().map(N::borrow),
            self.end.as_ref().map(N::borrow),
        )
    }

    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        N: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        RangeBounds::contains(&self.borrow::<Q>(), key)
    }
}

impl<N> IntoRangeBounds<N> for ItemRange<N> {
    fn into_bounds(self) -> (Bound<N>, Bound<N>) {
        let ItemRange { start, end } = self;
        (start, end)
    }
}

impl<N> ItemRange<&'_ N> {
    pub fn cloned(&self) -> ItemRange<N>
    where
        N: Clone,
    {
        ItemRange::unchecked(self.start.cloned(), self.end.cloned())
    }
}

impl<N> RangeBounds<N> for ItemRange<N> {
    fn start_bound(&self) -> Bound<&N> {
        self.start.as_ref()
    }

    fn end_bound(&self) -> Bound<&N> {
        self.end.as_ref()
    }
}

impl<N> RangeBounds<N> for ItemRange<&'_ N>
where
    N: ?Sized,
{
    fn start_bound(&self) -> Bound<&N> {
        self.start
    }

    fn end_bound(&self) -> Bound<&N> {
        self.end
    }
}

use std::cmp;
use std::iter::{
    Chain, Cloned, Copied, Cycle, Enumerate, Inspect, Map, Peekable, Rev, StepBy, Take,
};
use std::num::NonZeroUsize;
use std::option;

pub trait IteratorExt: Iterator {
    fn one_or_more(self) -> Remainder<Self>
    where
        Self: Sized;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn one_or_more(self) -> Remainder<Self> {
        Iterator1::try_from_iter(self).ok()
    }
}

pub trait FromIterator1<T> {
    fn from_iter1<I>(items: I) -> Self
    where
        I: IntoIterator1<Item = T>;
}

// TODO: This should work, but `rustc` incorrectly believes that this is a coherence error (E0119).
//       The error claims that downstream crates may implement `FromIterator` for collection types
//       in this crate, but they cannot implement foreign traits for foreign types!
//
// impl<T, U> FromIterator1<U> for T
// where
//     T: FromIterator<U>,
// {
//     fn from_iter1<I>(items: I) -> Self
//     where
//         I: IntoIterator1<Item = U>,
//     {
//         items.into_iter1().into_iter().collect()
//     }
// }

pub trait IntoIterator1: IntoIterator {
    fn into_iter1(self) -> Iterator1<Self::IntoIter>;
}

impl<I> IntoIterator1 for Iterator1<I>
where
    I: Iterator,
{
    fn into_iter1(self) -> Iterator1<Self::IntoIter> {
        self
    }
}

pub type Remainder<I> = Option<Iterator1<Peekable<I>>>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Iterator1<I> {
    items: I,
}

impl<I> Iterator1<I> {
    pub(crate) fn from_iter_unchecked<T>(items: T) -> Self
    where
        T: IntoIterator<IntoIter = I>,
    {
        Iterator1 {
            items: items.into_iter(),
        }
    }
}

impl<I> Iterator1<I>
where
    I: Iterator,
{
    pub fn try_from_iter<T>(items: T) -> Result<Iterator1<Peekable<I>>, Peekable<I>>
    where
        T: IntoIterator<IntoIter = I>,
    {
        let mut items = items.into_iter().peekable();
        match items.peek() {
            Some(_) => Ok(Iterator1::from_iter_unchecked(items)),
            _ => Err(items),
        }
    }

    #[inline(always)]
    fn zero_or_more<T, F>(mut self, f: F) -> (T, Remainder<I>)
    where
        F: FnOnce(&mut I) -> T,
    {
        let output = f(&mut self.items);
        (output, Iterator1::try_from_iter(self.items).ok())
    }

    #[inline(always)]
    fn one_or_more<J, F>(self, f: F) -> Iterator1<J>
    where
        J: Iterator,
        F: FnOnce(I) -> J,
    {
        Iterator1::from_iter_unchecked(f(self.items))
    }

    pub fn size_hint(&self) -> (NonZeroUsize, Option<NonZeroUsize>) {
        let (lower, upper) = self.items.size_hint();
        // SAFETY:
        unsafe {
            (
                NonZeroUsize::new_unchecked(cmp::max(1, lower)),
                upper.and_then(|upper| match upper {
                    0 => None,
                    upper => Some(NonZeroUsize::new_unchecked(upper)),
                }),
            )
        }
    }

    pub fn len(&self) -> NonZeroUsize
    where
        I: ExactSizeIterator,
    {
        NonZeroUsize::new(self.items.len()).expect("non-empty iterator has zero items")
    }

    pub fn as_iter(&self) -> &I {
        &self.items
    }

    pub fn count(self) -> NonZeroUsize {
        // SAFETY:
        unsafe { NonZeroUsize::new_unchecked(self.items.count()) }
    }

    pub fn last(self) -> I::Item {
        // SAFETY:
        unsafe { self.items.last().unwrap_unchecked() }
    }

    pub fn min(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.min().unwrap_unchecked() }
    }

    pub fn max(self) -> I::Item
    where
        I::Item: Ord,
    {
        // SAFETY:
        unsafe { self.items.max().unwrap_unchecked() }
    }

    pub fn reduce<F>(self, f: F) -> I::Item
    where
        F: FnMut(I::Item, I::Item) -> I::Item,
    {
        // SAFETY:
        unsafe { self.items.reduce(f).unwrap_unchecked() }
    }

    pub fn any<F>(self, f: F) -> (bool, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.zero_or_more(move |items| items.any(f))
    }

    pub fn all<F>(self, f: F) -> (bool, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.zero_or_more(move |items| items.all(f))
    }

    pub fn nth(self, n: usize) -> (Option<I::Item>, Remainder<I>) {
        self.zero_or_more(move |items| items.nth(n))
    }

    pub fn find<F>(self, f: F) -> (Option<I::Item>, Remainder<I>)
    where
        F: FnMut(&I::Item) -> bool,
    {
        self.zero_or_more(move |items| items.find(f))
    }

    pub fn find_map<T, F>(self, f: F) -> (Option<T>, Remainder<I>)
    where
        F: FnMut(I::Item) -> Option<T>,
    {
        self.zero_or_more(move |items| items.find_map(f))
    }

    pub fn position<F>(self, f: F) -> (Option<usize>, Remainder<I>)
    where
        F: FnMut(I::Item) -> bool,
    {
        self.zero_or_more(move |items| items.position(f))
    }

    pub fn rposition<F>(self, f: F) -> (Option<usize>, Remainder<I>)
    where
        I: DoubleEndedIterator + ExactSizeIterator,
        F: FnMut(I::Item) -> bool,
    {
        self.zero_or_more(move |items| items.rposition(f))
    }

    pub fn copied<'a, T>(self) -> Iterator1<Copied<I>>
    where
        T: 'a + Copy,
        I: Iterator<Item = &'a T>,
    {
        self.one_or_more(I::copied)
    }

    pub fn cloned<'a, T>(self) -> Iterator1<Cloned<I>>
    where
        T: 'a + Clone,
        I: Iterator<Item = &'a T>,
    {
        self.one_or_more(I::cloned)
    }

    pub fn enumerate(self) -> Iterator1<Enumerate<I>> {
        self.one_or_more(I::enumerate)
    }

    pub fn map<T, F>(self, f: F) -> Iterator1<Map<I, F>>
    where
        F: FnMut(I::Item) -> T,
    {
        self.one_or_more(move |items| items.map(f))
    }

    pub fn take(self, n: NonZeroUsize) -> Iterator1<Take<I>> {
        self.one_or_more(move |items| items.take(n.into()))
    }

    pub fn cycle(self) -> Iterator1<Cycle<I>>
    where
        I: Clone,
    {
        self.one_or_more(I::cycle)
    }

    pub fn chain<J>(self, chained: J) -> Iterator1<Chain<I, J::IntoIter>>
    where
        J: IntoIterator<Item = I::Item>,
    {
        self.one_or_more(move |items| items.chain(chained))
    }

    pub fn step_by(self, step: usize) -> Iterator1<StepBy<I>> {
        self.one_or_more(move |items| items.step_by(step))
    }

    pub fn inspect<F>(self, f: F) -> Iterator1<Inspect<I, F>>
    where
        F: FnMut(&I::Item),
    {
        self.one_or_more(move |items| items.inspect(f))
    }

    pub fn peekable(self) -> Iterator1<Peekable<I>> {
        self.one_or_more(I::peekable)
    }

    pub fn rev(self) -> Iterator1<Rev<I>>
    where
        I: DoubleEndedIterator,
    {
        self.one_or_more(I::rev)
    }

    pub fn collect<T>(self) -> T
    where
        T: FromIterator1<I::Item>,
    {
        T::from_iter1(self)
    }
}

impl<I> Iterator1<Peekable<I>>
where
    I: Iterator,
{
    pub fn peek(&mut self) -> &I::Item {
        // SAFETY:
        unsafe { self.items.peek().unwrap_unchecked() }
    }
}

impl<I> IntoIterator for Iterator1<I>
where
    I: Iterator,
{
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> Self::IntoIter {
        self.items
    }
}

pub fn one<T>(item: T) -> Iterator1<option::IntoIter<T>> {
    Iterator1::from_iter_unchecked(Some(item))
}

pub fn one_or_more<T, I>(head: T, tail: I) -> Iterator1<impl Iterator<Item = T>>
where
    I: IntoIterator<Item = T>,
{
    Iterator1::from_iter_unchecked(Some(head).into_iter().chain(tail))
}

macro_rules! impl_into_iterator1_for_array {
    ($N:literal) => {
        impl<T> $crate::iter1::IntoIterator1 for [T; $N] {
            fn into_iter1(self) -> Iterator1<Self::IntoIter> {
                $crate::iter1::Iterator1::from_iter_unchecked(self)
            }
        }
    };
}
crate::with_non_zero_array_size_literals!(impl_into_iterator1_for_array);

#[cfg(test)]
mod tests {}

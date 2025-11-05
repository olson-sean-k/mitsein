#![cfg(any(feature = "arrayvec", feature = "alloc"))]
#![cfg_attr(docsrs, doc(cfg(any(feature = "arrayvec", feature = "alloc"))))]

use core::fmt::{self, Debug, Formatter};
use core::mem::{self, ManuallyDrop, MaybeUninit};

use crate::{Cardinality, MaybeEmpty, NonEmpty};

pub type FnMany<T, U, N> = fn(&mut NonEmpty<T>, N) -> U;

struct Target<'a, T, N> {
    items: &'a mut NonEmpty<T>,
    index: N,
}

impl<T, N> Debug for Target<'_, T, N>
where
    NonEmpty<T>: Debug,
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Target")
            .field("items", &self.items)
            .field("index", &self.index)
            .finish()
    }
}

pub struct TakeIfMany<'a, T, U, N = ()>
where
    T: MaybeEmpty,
{
    // This field wraps `Target` in both `ManuallyDrop` and `MaybeUninit`.
    //
    // `ManuallyDrop` not only prevents dropping the `Target` without explicit code, but also
    // provides `ManuallyDrop::take`, which uses `ptr::read` to provide a shallow copy of the
    // `Target` and to essentially move the `Target` out through a mutable reference.
    //
    // `MaybeUninit` prevents the compiler from making assumptions about `Target`. In particular,
    // using `ManuallyDrop::take` constructs a mutable alias of the `Target::items` field. Without
    // `MaybeUninit`, this is undefined behavior, because the compiler would assume that these
    // fields refer to different data when they actually alias the same data.
    target: ManuallyDrop<MaybeUninit<Target<'a, T, N>>>,
    // This field need not be a part of `Target`, because it is `Copy`. There's no need to copy it
    // through `ManuallyDrop::take` and `ptr::read`.
    many: FnMany<T, U, N>,
}

impl<'a, T, U, N> TakeIfMany<'a, T, U, N>
where
    T: MaybeEmpty,
{
    pub(crate) fn with(items: &'a mut NonEmpty<T>, index: N, many: FnMany<T, U, N>) -> Self {
        TakeIfMany {
            target: ManuallyDrop::new(MaybeUninit::new(Target { items, index })),
            many,
        }
    }

    fn read_and_forget<O, F>(mut self, f: F) -> O
    where
        F: FnOnce(Target<'a, T, N>, FnMany<T, U, N>) -> O,
    {
        let many = self.many;
        // SAFETY: `target` must be initialized and must not have been taken before this call.
        //         Moreover, `Drop::drop` must not be called after taking `target`, because it also
        //         takes `target`. This is safe here, because `Drop::drop` cannot have been called
        //         before reaching this code and `self` is forgotten (via `mem::forget`) below.
        //         `target` is always initialized through `TakeIfMany::with`.
        //
        //         Note too that `mem::forget` must be reached after taking `target` here. For
        //         example, a panic just before `mem::forget` would cause `Drop::drop` to take
        //         `target` a second time and therefore a double-drop. It is important no fallible
        //         operations occur between taking and forgetting!
        let target = unsafe { ManuallyDrop::take(&mut self.target).assume_init() };
        mem::forget(self);
        // This may panic, but this is okay, because `self` has already been forgotten.
        f(target, many)
    }

    pub(crate) fn take_or_else<E, F>(self, one: F) -> Result<U, E>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> E,
    {
        self.read_and_forget(|Target { items, index }, many| match items.cardinality() {
            Cardinality::One(_) => Err(one(items, index)),
            Cardinality::Many(_) => Ok((many)(items, index)),
        })
    }

    // It is tempting to use this function to implement `and_if` functions. However, this requires
    // knowledge of the position of the target item in its collection. For example, an `and_if`
    // function would not have enough information if implemented for `vec_deque1::PopIfMany`, since
    // items can be popped from both ends. Instead, this function is used to implement counterparts
    // to standard APIs like `Vec::pop_if` with bespoke functions on non-empty types, like
    // `Vec1::pop_if_many_and`.
    #[cfg(feature = "alloc")]
    pub(crate) fn take_if<F>(self, f: F) -> Option<U>
    where
        F: FnOnce(&mut NonEmpty<T>) -> bool,
    {
        self.read_and_forget(|Target { items, index }, many| match items.cardinality() {
            Cardinality::One(_) => None,
            Cardinality::Many(_) => {
                if f(items) {
                    Some((many)(items, index))
                }
                else {
                    None
                }
            },
        })
    }

    pub fn or_else<E, F>(self, f: F) -> Result<U, E>
    where
        F: FnOnce() -> E,
    {
        self.take_or_else(|_, _| f())
    }

    pub fn or_none(self) -> Option<U> {
        self.take_or_else(|_, _| ()).ok()
    }

    pub fn or_false(self) -> bool {
        self.or_none().is_some()
    }
}

impl<'a, T, U, N> TakeIfMany<'a, T, Option<U>, N>
where
    T: MaybeEmpty,
{
    #[cfg(any(feature = "alloc", feature = "arrayvec"))]
    pub(crate) fn try_take_or_else<E, F>(self, one: F) -> Option<Result<U, E>>
    where
        F: FnOnce(&'a mut NonEmpty<T>, N) -> Option<E>,
    {
        self.read_and_forget(|Target { items, index }, many| match items.cardinality() {
            Cardinality::One(_) => one(items, index).map(Err),
            Cardinality::Many(_) => (many)(items, index).map(Ok),
        })
    }
}

impl<T, U, N> Debug for TakeIfMany<'_, T, U, N>
where
    NonEmpty<T>: Debug,
    T: MaybeEmpty,
    N: Debug,
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TakeIfMany")
            .field("target", &self.target)
            .field("many", &self.many)
            .finish()
    }
}

impl<T, U, N> Drop for TakeIfMany<'_, T, U, N>
where
    T: MaybeEmpty,
{
    fn drop(&mut self) {
        // SAFETY: `target` must be initialized and must not have been taken before this call.
        //         `TakeIfMany::read_and_forget` also takes `target`, but it also forgets `self`
        //         (via `mem::forget`) to prevent this code from being reached. Note too that the
        //         contents of `ManuallyDrop` are never dropped, so there is no double-drop, even
        //         if `target`'s fields are not `Copy`. `target` is always initialized through
        //         `TakeIfMany::with`.
        let target = unsafe { ManuallyDrop::take(&mut self.target).assume_init() };
        if let Cardinality::Many(_) = target.items.cardinality() {
            (self.many)(target.items, target.index);
        }
    }
}

// Some tests in this module may not be very interesting in the conventional sense, but are very
// interesting for analysis with Miri.
#[cfg(all(test, feature = "alloc"))]
mod tests {
    use rstest::rstest;

    use crate::vec1::Vec1;
    use crate::vec1::harness::xs1;

    #[rstest]
    #[should_panic]
    fn remove_if_many_out_of_bounds_from_vec1_with_proxy_fn_then_panics(mut xs1: Vec1<u8>) {
        let _ = xs1.remove_if_many(8).or_get();
    }

    #[rstest]
    #[should_panic]
    fn remove_if_many_out_of_bounds_from_vec1_without_proxy_fn_then_panics(mut xs1: Vec1<u8>) {
        xs1.remove_if_many(8);
    }

    #[rstest]
    fn take_from_vec1_without_proxy_fn_then_take_if_many_eagerly_drops(mut xs1: Vec1<u8>) {
        xs1.pop_if_many();
        assert_eq!(xs1.as_slice(), &[0, 1, 2, 3]);
        xs1.remove_if_many(0);
        assert_eq!(xs1.as_slice(), &[1, 2, 3]);
    }
}

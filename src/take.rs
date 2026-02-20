#![cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless"))]
#![cfg_attr(
    docsrs,
    doc(cfg(any(feature = "alloc", feature = "arrayvec", feature = "heapless")))
)]

use core::fmt::{self, Debug, Formatter};
use core::mem::{self, MaybeUninit};

use crate::{Cardinality, MaybeEmpty, NonEmpty};

pub type FnMany<T, U, N> = fn(&mut NonEmpty<T>, N) -> U;

struct Target<'a, T, N>
where
    T: ?Sized,
{
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

/// A proxy for an operation that takes an item out of a [`NonEmpty`] collection.
///
/// This is a very general type constructor: refer to more specific type definitions to see the
/// relevant APIs for a particular non-empty collection. For example, see [`vec1::RemoveIfMany`] to
/// see supported APIs for [`vec1::Vec1::remove_if_many`].
///
/// `TakeIfMany` is returned by `_if_many` operations of [`NonEmpty`] types, like
/// [`vec1::Vec1::pop_if_many`] and [`vec1::Vec1::remove_if_many`]. An item can only be taken out
/// of a [`NonEmpty`] collection if it contains many items ([more than one][`Cardinality`]).
///
/// `TakeIfMany` provides method chaining that can react to the result of taking an item out of a
/// [`NonEmpty`] collection, typically by mapping over the taken item or by operating on the only
/// remaining item left in the collection. A taken item can only be accessed using these methods.
/// If none of these methods are used, then `TakeIfMany`'s [`Drop`] implementation takes and drops
/// any item.
///
/// # Examples
///
/// Remove an item from a [`Vec1`]:
#[doc = ""]
#[cfg_attr(feature = "alloc", doc = "```rust")]
#[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
/// use mitsein::prelude::*;
///
/// let mut xs = vec1![0i64, 1, -3];
/// xs.remove_if_many(0);
///
/// assert_eq!(&[1, -3], xs.as_slice());
#[doc = "```"]
///
/// Pop items from a [`Vec1`] in a loop via [`or_none`]:
#[doc = ""]
#[cfg_attr(feature = "alloc", doc = "```rust")]
#[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
/// use mitsein::prelude::*;
///
/// let mut xs = vec1![0i64, 1, -3];
/// while let Some(x) = xs.pop_if_many().or_none() {
///     // ...
/// }
///
/// assert_eq!(&0, xs.first());
#[doc = "```"]
///
/// Pop an item from a [`BTreeMap1`] or replace the only item with its [`Default`]:
#[doc = ""]
#[cfg_attr(feature = "alloc", doc = "```rust")]
#[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
/// use mitsein::btree_map1::BTreeMap1;
///
/// let mut xs = BTreeMap1::from_one(("key", 42i64));
/// let x = xs.pop_first_if_many().or_else_replace_only(i64::default);
///
/// assert_eq!((&"key", &0), xs.first_key_value());
/// assert_eq!(Err(42), x);
#[doc = "```"]
///
/// [`BTreeMap1`]: crate::btree_map1::BTreeMap1
/// [`Cardinality::Many`]: crate::Cardinality::Many
/// [`or_none`]: crate::take::TakeIfMany::or_none
/// [`vec1::RemoveIfMany`]: crate::vec1::RemoveIfMany
/// [`vec1::Vec1::pop_if_many`]: crate::vec1::Vec1::pop_if_many
/// [`vec1::Vec1::remove_if_many`]: crate::vec1::Vec1::remove_if_many
/// [`Vec1`]: crate::vec1::Vec1
pub struct TakeIfMany<'a, T, U, N = ()>
where
    T: MaybeEmpty + ?Sized,
{
    // This field wraps `Target` in `MaybeUninit`.
    //
    // `MaybeUninit` prevents dropping the `Target` without explicit code and also prevents the
    // compiler from making assumptions about `Target`. In particular, using
    // `MaybeUninit::assume_init_read` constructs a mutable alias of the `Target::items` field.
    // Without `MaybeUninit`, this is undefined behavior, because the compiler would assume that
    // `TakeIfMany::target` is valid and that `Target::items` refers to different data even though
    // these fields actually alias the same data.
    target: MaybeUninit<Target<'a, T, N>>,
    // This field need not be a part of `Target`, because it is `Copy`. There's no need to copy it
    // through `MaybeUninit::assume_init_read`.
    many: FnMany<T, U, N>,
}

impl<'a, T, U, N> TakeIfMany<'a, T, U, N>
where
    T: MaybeEmpty + ?Sized,
{
    pub(crate) fn with(items: &'a mut NonEmpty<T>, index: N, many: FnMany<T, U, N>) -> Self {
        TakeIfMany {
            target: MaybeUninit::new(Target { items, index }),
            many,
        }
    }

    /// Reads the [`Target`] and then [forgets][`mem::forget] `self`, passing the [`Target`] and
    /// [`FnMany`] to the given function.
    ///
    /// This function consumes the `TakeIfMany` and passes its deconstruction to the given function
    /// where the [`Target`] and [`FnMany`] can be dropped safely. Critically, `self` is forgotten,
    /// so the [`Drop`] implementation for `TakeIfMany` is bypassed when calling this function.
    /// This is the fundamental primitive for `TakeIfMany` methods.
    fn read_and_forget<O, F>(self, f: F) -> O
    where
        F: FnOnce(Target<'a, T, N>, FnMany<T, U, N>) -> O,
    {
        let many = self.many;
        // SAFETY: `target` must be initialized and must not have been read before this call.
        //         Moreover, `Drop::drop` must not be called after reading `target`, because it
        //         also reads `target`. This is safe here, because `Drop::drop` cannot have been
        //         called before reaching this code and `self` is forgotten (via `mem::forget`)
        //         below. `target` is always initialized through `TakeIfMany::with`.
        //
        //         Note too that `mem::forget` must be reached after reading `target` here. For
        //         example, a panic just before `mem::forget` would cause `Drop::drop` to take
        //         `target` a second time and therefore a double-drop. It is important no fallible
        //         operations occur between taking and forgetting!
        let target = unsafe { self.target.assume_init_read() };
        mem::forget(self);
        // This may panic, but this is okay, because `self` has already been forgotten.
        f(target, many)
    }

    /// Executes either the [`FnMany`] or the given function against the [`Target`].
    ///
    /// When the target collection contains many items, the [`FnMany`] is executed and its output
    /// is returned as [`Ok`]. Otherwise, the given function is executed and its output is returned
    /// as [`Err`].
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
    /// Executes the [`FnMany`] against the [`Target`] if the collection has many items and the
    /// given predicate returns `true`.
    ///
    /// When the target collection contains many items, the given function is executed and, if it
    /// returns `true`, then the [`FnMany`] is subsequently executed and its output is returned as
    /// [`Some`]. If the target collection contains only one item or the given function returns
    /// `false`, then [`None`] is returned.
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

    /// Returns the taken item as [`Ok`] or, if the item cannot be taken, the output of the given
    /// function as [`Err`].
    ///
    /// # Examples
    ///
    /// Pop an item from a [`Vec1`] or construct a [`Default`]:
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// let mut xs = vec1![42i64];
    /// let x = match xs.pop_if_many().or_else(i64::default) {
    ///     Ok(x) | Err(x) => x,
    /// };
    ///
    /// assert_eq!(&[42], xs.as_slice());
    /// assert_eq!(0, x);
    #[doc = "```"]
    ///
    /// [`Vec1`]: crate::vec1::Vec1
    pub fn or_else<E, F>(self, f: F) -> Result<U, E>
    where
        F: FnOnce() -> E,
    {
        self.take_or_else(|_, _| f())
    }

    /// Returns the taken item as [`Some`] or otherwise returns [`None`].
    ///
    /// # Examples
    ///
    /// Pop items from a [`Vec1`] in a loop:
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// let mut xs = vec1![0i64, 1, 2, 3, 4];
    /// while let Some(x) = xs.pop_if_many().or_none() {
    ///     // ...
    /// }
    ///
    /// assert_eq!(&0, xs.first());
    #[doc = "```"]
    ///
    /// [`Vec1`]: crate::vec1::Vec1
    pub fn or_none(self) -> Option<U> {
        self.take_or_else(|_, _| ()).ok()
    }

    /// Returns `true` if the item is taken or otherwise `false`.
    ///
    /// # Examples
    ///
    /// Pop an item from a [`Vec1`] or print an error:
    #[doc = ""]
    #[cfg_attr(feature = "alloc", doc = "```rust")]
    #[cfg_attr(not(feature = "alloc"), doc = "```rust,ignore")]
    /// use mitsein::prelude::*;
    ///
    /// fn pop_or_print(xs: &mut Vec1<i64>) {
    ///     if !xs.pop_if_many().or_false() {
    ///         eprintln!("failed to pop item from non-empty `Vec`");
    ///     }
    /// }
    #[doc = "```"]
    ///
    /// [`Vec1`]: crate::vec1::Vec1
    pub fn or_false(self) -> bool {
        self.or_none().is_some()
    }
}

impl<'a, T, U, N> TakeIfMany<'a, T, Option<U>, N>
where
    T: MaybeEmpty + ?Sized,
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
    T: MaybeEmpty + ?Sized,
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
    T: MaybeEmpty + ?Sized,
{
    fn drop(&mut self) {
        // SAFETY: `target` must be initialized and must not have been read before this call.
        //         `TakeIfMany::read_and_forget` also reads `target`, but it also forgets `self`
        //         (via `mem::forget`) to prevent this code from being reached. Note too that the
        //         contents of `MaybeUninit` are never dropped, so there is no double-drop, even if
        //         `target`'s fields are not `Copy`. `target` is always initialized through
        //         `TakeIfMany::with`.
        let target = unsafe { self.target.assume_init_read() };
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
    fn take_from_vec1_without_proxy_fn_then_take_if_many_immediately_drops_and_takes(
        mut xs1: Vec1<u8>,
    ) {
        xs1.pop_if_many();
        assert_eq!(xs1.as_slice(), &[0, 1, 2, 3]);
        xs1.remove_if_many(0);
        assert_eq!(xs1.as_slice(), &[1, 2, 3]);
    }
}

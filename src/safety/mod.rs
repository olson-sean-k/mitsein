#[cfg(all(not(miri), test))]
mod checked;
#[cfg(not(all(not(miri), test)))]
mod unchecked;

pub trait NonZeroExt<T> {
    unsafe fn new_maybe_unchecked(n: T) -> Self;
}

pub trait OptionExt<T> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

pub trait ResultExt<T, E> {
    unsafe fn unwrap_maybe_unchecked(self) -> T;
}

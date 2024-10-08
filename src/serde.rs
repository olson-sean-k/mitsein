#![cfg(feature = "serde")]
#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use core::fmt::{self, Display, Formatter};
use serde_derive::{Deserialize, Serialize};

use crate::NonEmpty;

const EMPTY_ERROR_MESSAGE: &str = "failed to deserialize non-empty collection: no items";

#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Serde<T> {
    pub items: T,
}

impl<T> From<NonEmpty<T>> for Serde<T> {
    fn from(items: NonEmpty<T>) -> Self {
        Serde { items: items.items }
    }
}

impl<T, U> TryFrom<Serde<U>> for NonEmpty<T>
where
    NonEmpty<T>: TryFrom<U>,
{
    type Error = EmptyError;

    fn try_from(serde: Serde<U>) -> Result<Self, Self::Error> {
        NonEmpty::try_from(serde.items).map_err(|_| EmptyError)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmptyError;

impl Display for EmptyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", EMPTY_ERROR_MESSAGE)
    }
}

#[cfg(all(test, any(feature = "alloc", feature = "arrayvec")))]
pub mod harness {
    use core::fmt::Debug;
    use rstest::fixture;
    use serde::{Deserialize, Serialize};
    use serde_test::{self, Token};

    use crate::serde::EMPTY_ERROR_MESSAGE;

    #[fixture]
    pub fn sequence(#[default(5)] len: u8) -> impl Iterator<Item = Token> {
        Some(Token::Seq {
            len: Some(usize::from(len)),
        })
        .into_iter()
        .chain((0..len).map(Token::U8))
        .chain(Some(Token::SeqEnd))
    }

    #[cfg(feature = "alloc")]
    #[fixture]
    pub fn map(#[default(5)] len: u8) -> impl Iterator<Item = Token> {
        use crate::btree_map1;

        Some(Token::Map {
            len: Some(usize::from(len)),
        })
        .into_iter()
        .chain((0..len).flat_map(|key| [Token::U8(key), Token::Char(btree_map1::harness::VALUE)]))
        .chain(Some(Token::MapEnd))
    }

    pub fn assert_into_and_from_tokens_eq<T, N>(items: T, tokens: impl IntoIterator<Item = Token>)
    where
        for<'de> T: Debug + Deserialize<'de> + PartialEq + Serialize,
        N: AsRef<[Token]> + FromIterator<Token>,
    {
        let tokens: N = tokens.into_iter().collect();
        serde_test::assert_tokens(&items, tokens.as_ref());
    }

    pub fn assert_deserialize_error_eq_empty_error<T, N>(tokens: impl IntoIterator<Item = Token>)
    where
        for<'de> T: Debug + Deserialize<'de> + PartialEq + Serialize,
        N: AsRef<[Token]> + FromIterator<Token>,
    {
        let tokens: N = tokens.into_iter().collect();
        serde_test::assert_de_tokens_error::<T>(tokens.as_ref(), EMPTY_ERROR_MESSAGE);
    }
}

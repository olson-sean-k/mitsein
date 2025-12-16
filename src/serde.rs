#![cfg(feature = "serde")]
#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use core::fmt::Display;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::NonEmpty;

// A blanket implementation like this for reference types (i.e., `&'_ NonEmpty<T>`) is not
// possible, because it causes infinite recursion when the compiler attempts to query the type `T`
// and substitutes `NonEmpty<T>`. However, the number of such implementations for standard
// maybe-empty types is very small, so more bespoke implementations are used for non-empty types
// instead. See `Slice1` and `Str1`.
impl<'de, T> Deserialize<'de> for NonEmpty<T>
where
    Self: TryFrom<T>,
    <Self as TryFrom<T>>::Error: Display,
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use ::serde::de::Error;

        let items = T::deserialize(deserializer)?;
        NonEmpty::try_from(items).map_err(D::Error::custom)
    }
}

impl<T> Serialize for NonEmpty<T>
where
    T: Serialize + ?Sized,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.items.serialize(serializer)
    }
}

#[cfg(all(
    test,
    any(feature = "alloc", feature = "arrayvec", feature = "heapless")
))]
pub mod harness {
    use core::fmt::Debug;
    use rstest::fixture;
    use serde::{Deserialize, Serialize};
    use serde_test::{self, Token};

    use crate::EMPTY_ERROR_MESSAGE;

    #[cfg(feature = "alloc")]
    #[fixture]
    pub fn borrowed_bytes(
        #[default(&[0, 1, 2, 3, 4])] bytes: &'static [u8],
    ) -> impl Iterator<Item = Token> {
        Some(Token::BorrowedBytes(bytes)).into_iter()
    }

    #[cfg(feature = "alloc")]
    #[fixture]
    pub fn borrowed_str(
        #[default(&"non-empty")] string: &'static str,
    ) -> impl Iterator<Item = Token> {
        Some(Token::BorrowedStr(string)).into_iter()
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
    pub fn assert_ref_from_tokens_eq<'de, T>(items: &'de T, tokens: &'de [Token])
    where
        T: ?Sized,
        &'de T: Debug + Deserialize<'de> + PartialEq,
    {
        serde_test::assert_de_tokens(&items, tokens);
    }

    #[cfg(feature = "alloc")]
    pub fn assert_into_tokens_eq<T, N>(items: T, tokens: impl IntoIterator<Item = Token>)
    where
        T: Debug + PartialEq + Serialize,
        N: AsRef<[Token]> + FromIterator<Token>,
    {
        let tokens: N = tokens.into_iter().collect();
        serde_test::assert_ser_tokens(&items, tokens.as_ref());
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

#![cfg(feature = "rkyv")]
#![cfg_attr(docsrs, doc(cfg(feature = "rkyv")))]

use rkyv::rancor::Fallible;
use rkyv::{Archive, Deserialize, Place, Portable, Serialize};

use crate::NonEmpty;

// SAFETY: `NonEmpty<T>` is `#[repr(transparent)]` over `T`, so it has identical layout.
// `Portable` when `T` is `Portable`.
unsafe impl<T: Portable> Portable for NonEmpty<T> {}

impl<T: Archive> Archive for NonEmpty<T> {
    type Archived = NonEmpty<T::Archived>;
    type Resolver = T::Resolver;

    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        // SAFETY: `NonEmpty<T::Archived>` is `#[repr(transparent)]` over `T::Archived`, so the
        // place can be safely cast.
        let out_inner = unsafe { out.cast_unchecked::<T::Archived>() };
        self.items.resolve(resolver, out_inner);
    }
}

impl<T, S> Serialize<S> for NonEmpty<T>
where
    T: Serialize<S>,
    S: Fallible + ?Sized,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        self.items.serialize(serializer)
    }
}

impl<T, D> Deserialize<NonEmpty<T>, D> for NonEmpty<T::Archived>
where
    T: Archive,
    T::Archived: Deserialize<T, D>,
    D: Fallible + ?Sized,
{
    fn deserialize(&self, deserializer: &mut D) -> Result<NonEmpty<T>, D::Error> {
        self.items
            .deserialize(deserializer)
            .map(|items| NonEmpty { items })
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use rkyv::rancor;

    use crate::vec1::Vec1;

    #[test]
    fn roundtrip_vec1() {
        let original = Vec1::from_head_and_tail(1u8, [2, 3, 4]);
        let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
        let archived = unsafe { rkyv::access_unchecked::<rkyv::Archived<Vec1<u8>>>(&bytes) };
        assert_eq!(archived.items.len(), 4);
        assert_eq!(archived.items[0], 1);
        assert_eq!(archived.items[3], 4);
    }

    #[test]
    fn roundtrip_cardinality() {
        use crate::{ArchivedCardinality, Cardinality};

        let original = Cardinality::<u32, u32>::Many(42);
        let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
        let archived =
            unsafe { rkyv::access_unchecked::<rkyv::Archived<Cardinality<u32, u32>>>(&bytes) };
        match archived {
            ArchivedCardinality::Many(v) => assert_eq!(*v, 42),
            _ => panic!("expected Many variant"),
        }
    }
}

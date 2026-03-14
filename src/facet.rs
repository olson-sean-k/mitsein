#![cfg(feature = "facet")]
#![cfg_attr(docsrs, doc(cfg(feature = "facet")))]

use facet::Facet;

use crate::NonEmpty;

// SAFETY: `NonEmpty<T>` is `#[repr(transparent)]` over `T`. The shape correctly describes it as a
// transparent struct with a single field at offset 0, pointing to `T`'s shape.
unsafe impl<'facet, T> Facet<'facet> for NonEmpty<T>
where
    T: Facet<'facet> + 'facet,
{
    const SHAPE: &'static facet::Shape = &const {
        facet::ShapeBuilder::for_sized::<NonEmpty<T>>("NonEmpty")
            .ty(facet::Type::User(facet::UserType::Struct(
                facet::StructType {
                    repr: facet::Repr::transparent(),
                    kind: facet::StructKind::Struct,
                    fields: &const {
                        [facet::FieldBuilder::new("items", facet::shape_of::<T>, 0).build()]
                    },
                },
            )))
            .inner(<T as Facet>::SHAPE)
            .build()
    };
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;
    use facet::Facet;

    use crate::{EmptyError, NonEmpty};

    #[test]
    fn non_empty_vec_has_shape() {
        let shape = <NonEmpty<Vec<u8>> as Facet>::SHAPE;
        assert_eq!(shape.type_identifier, "NonEmpty");
        assert!(shape.inner.is_some());
    }

    #[test]
    fn empty_error_has_shape() {
        let shape = <EmptyError<u32> as Facet>::SHAPE;
        assert_eq!(shape.type_identifier, "EmptyError");
    }
}

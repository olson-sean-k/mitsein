#![cfg(feature = "alloc")]

pub trait ResolveTrimRange<R> {
    fn resolve_trim_range(&self, range: TrimRange) -> R;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TrimRange {
    pub tail: usize,
    pub rtail: usize,
}

impl TrimRange {
    pub const ALL: Self = TrimRange { tail: 0, rtail: 0 };
    pub const TAIL1: Self = TrimRange { tail: 1, rtail: 0 };
    pub const RTAIL1: Self = TrimRange { tail: 0, rtail: 1 };

    pub const fn tail(self) -> Self {
        let TrimRange { tail, rtail } = self;
        TrimRange {
            tail: match tail.checked_add(1) {
                Some(tail) => tail,
                _ => self::panic_tail_overflow(),
            },
            rtail,
        }
    }

    pub const fn rtail(self) -> Self {
        let TrimRange { tail, rtail } = self;
        TrimRange {
            tail,
            rtail: match rtail.checked_add(1) {
                Some(rtail) => rtail,
                _ => self::panic_rtail_overflow(),
            },
        }
    }

    pub const fn is_all(&self) -> bool {
        self.tail == 0 && self.rtail == 0
    }
}

const fn panic_tail_overflow() -> ! {
    panic!("overflow in trim range tail")
}

const fn panic_rtail_overflow() -> ! {
    panic!("overflow in trim range rtail")
}

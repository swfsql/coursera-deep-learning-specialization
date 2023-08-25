pub trait StackRightN: Sized {
    // fn stack1(self) -> <Self as StackRight<1>>::Output
    // where
    //     Self: StackRight<1>,
    // {
    //     StackRight::<1>::stack(self)
    // }

    // fn stack2(self) -> <Self as StackRight<2>>::Output
    // where
    //     Self: StackRight<2>,
    // {
    //     StackRight::<2>::stack(self)
    // }

    fn stack3(self) -> <Self as StackRight<3>>::Output
    where
        Self: StackRight<3>,
    {
        StackRight::<3>::stack(self)
    }

    fn stack4(self) -> <Self as StackRight<4>>::Output
    where
        Self: StackRight<4>,
    {
        StackRight::<4>::stack(self)
    }

    fn stack5(self) -> <Self as StackRight<5>>::Output
    where
        Self: StackRight<5>,
    {
        StackRight::<5>::stack(self)
    }

    fn stack6(self) -> <Self as StackRight<6>>::Output
    where
        Self: StackRight<6>,
    {
        StackRight::<6>::stack(self)
    }

    fn stack7(self) -> <Self as StackRight<7>>::Output
    where
        Self: StackRight<7>,
    {
        StackRight::<7>::stack(self)
    }

    fn stack8(self) -> <Self as StackRight<8>>::Output
    where
        Self: StackRight<8>,
    {
        StackRight::<8>::stack(self)
    }

    fn stack9(self) -> <Self as StackRight<9>>::Output
    where
        Self: StackRight<9>,
    {
        StackRight::<9>::stack(self)
    }
}

impl<T: Sized + IsTuple> StackRightN for T {}
pub trait IsTuple {}
impl<A> IsTuple for (A,) {}
impl<A, B> IsTuple for (A, B) {}

pub trait StackRight<const N: usize> {
    type Output;
    fn stack(self) -> Self::Output;
}

impl<F0> StackRight<1> for (F0,) {
    type Output = Self;

    fn stack(self) -> Self::Output {
        (self.0,)
    }
}

impl<F0, F1> StackRight<2> for (F0, F1) {
    type Output = Self;

    fn stack(self) -> Self::Output {
        (self.0, self.1)
    }
}

impl<F0, F1, F2> StackRight<3> for (F0, F1, F2) {
    type Output = (F0, (F1, F2));

    fn stack(self) -> Self::Output {
        (self.0, (self.1, self.2))
    }
}

impl<F0, F1, F2, F3> StackRight<4> for (F0, F1, F2, F3) {
    type Output = (F0, (F1, (F2, F3)));

    fn stack(self) -> Self::Output {
        (self.0, (self.1, (self.2, self.3)))
    }
}

impl<F0, F1, F2, F3, F4> StackRight<5> for (F0, F1, F2, F3, F4) {
    type Output = (F0, (F1, (F2, (F3, F4))));

    fn stack(self) -> Self::Output {
        (self.0, (self.1, (self.2, (self.3, self.4))))
    }
}

impl<F0, F1, F2, F3, F4, F5> StackRight<6> for (F0, F1, F2, F3, F4, F5) {
    type Output = (F0, (F1, (F2, (F3, (F4, F5)))));

    fn stack(self) -> Self::Output {
        (self.0, (self.1, (self.2, (self.3, (self.4, self.5)))))
    }
}

impl<F0, F1, F2, F3, F4, F5, F6> StackRight<7> for (F0, F1, F2, F3, F4, F5, F6) {
    type Output = (F0, (F1, (F2, (F3, (F4, (F5, F6))))));

    fn stack(self) -> Self::Output {
        (
            self.0,
            (self.1, (self.2, (self.3, (self.4, (self.5, self.6))))),
        )
    }
}

impl<F0, F1, F2, F3, F4, F5, F6, F7> StackRight<8> for (F0, F1, F2, F3, F4, F5, F6, F7) {
    type Output = (F0, (F1, (F2, (F3, (F4, (F5, (F6, F7)))))));

    fn stack(self) -> Self::Output {
        (
            self.0,
            (
                self.1,
                (self.2, (self.3, (self.4, (self.5, (self.6, self.7))))),
            ),
        )
    }
}

impl<F0, F1, F2, F3, F4, F5, F6, F7, F8> StackRight<9> for (F0, F1, F2, F3, F4, F5, F6, F7, F8) {
    type Output = (F0, (F1, (F2, (F3, (F4, (F5, (F6, (F7, F8))))))));

    fn stack(self) -> Self::Output {
        (
            self.0,
            (
                self.1,
                (
                    self.2,
                    (self.3, (self.4, (self.5, (self.6, (self.7, self.8))))),
                ),
            ),
        )
    }
}

pub trait FlatRightN: Sized {
    fn flat1(self) -> <Self as FlatRight<1>>::Output
    where
        Self: FlatRight<1>,
    {
        FlatRight::<1>::flat(self)
    }

    fn flat2(self) -> <Self as FlatRight<2>>::Output
    where
        Self: FlatRight<2>,
    {
        FlatRight::<2>::flat(self)
    }

    fn flat3(self) -> <Self as FlatRight<3>>::Output
    where
        Self: FlatRight<3>,
    {
        FlatRight::<3>::flat(self)
    }

    fn flat4(self) -> <Self as FlatRight<4>>::Output
    where
        Self: FlatRight<4>,
    {
        FlatRight::<4>::flat(self)
    }

    fn flat5(self) -> <Self as FlatRight<5>>::Output
    where
        Self: FlatRight<5>,
    {
        FlatRight::<5>::flat(self)
    }

    fn flat6(self) -> <Self as FlatRight<6>>::Output
    where
        Self: FlatRight<6>,
    {
        FlatRight::<6>::flat(self)
    }

    fn flat7(self) -> <Self as FlatRight<7>>::Output
    where
        Self: FlatRight<7>,
    {
        FlatRight::<7>::flat(self)
    }

    fn flat8(self) -> <Self as FlatRight<8>>::Output
    where
        Self: FlatRight<8>,
    {
        FlatRight::<8>::flat(self)
    }

    fn flat9(self) -> <Self as FlatRight<9>>::Output
    where
        Self: FlatRight<9>,
    {
        FlatRight::<9>::flat(self)
    }
}

impl<T: Sized> FlatRightN for T {}

pub trait FlatRight<const N: usize> {
    type Output;
    fn flat(self) -> Self::Output;
}

impl<F0> FlatRight<1> for (F0,) {
    type Output = Self;

    fn flat(self) -> Self::Output {
        (self.0,)
    }
}

impl<F0, F1> FlatRight<2> for (F0, F1) {
    type Output = Self;

    fn flat(self) -> Self::Output {
        (self.0, self.1)
    }
}

impl<F0, F1, F2> FlatRight<3> for (F0, (F1, F2)) {
    type Output = (F0, F1, F2);

    fn flat(self) -> Self::Output {
        (self.0, self.1 .0, self.1 .1)
    }
}

impl<F0, F1, F2, F3> FlatRight<4> for (F0, (F1, (F2, F3))) {
    type Output = (F0, F1, F2, F3);

    fn flat(self) -> Self::Output {
        (self.0, self.1 .0, self.1 .1 .0, self.1 .1 .1)
    }
}

impl<F0, F1, F2, F3, F4> FlatRight<5> for (F0, (F1, (F2, (F3, F4)))) {
    type Output = (F0, F1, F2, F3, F4);

    fn flat(self) -> Self::Output {
        (
            self.0,
            self.1 .0,
            self.1 .1 .0,
            self.1 .1 .1 .0,
            self.1 .1 .1 .1,
        )
    }
}

impl<F0, F1, F2, F3, F4, F5> FlatRight<6> for (F0, (F1, (F2, (F3, (F4, F5))))) {
    type Output = (F0, F1, F2, F3, F4, F5);

    fn flat(self) -> Self::Output {
        (
            self.0,
            self.1 .0,
            self.1 .1 .0,
            self.1 .1 .1 .0,
            self.1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1,
        )
    }
}

impl<F0, F1, F2, F3, F4, F5, F6> FlatRight<7> for (F0, (F1, (F2, (F3, (F4, (F5, F6)))))) {
    type Output = (F0, F1, F2, F3, F4, F5, F6);

    fn flat(self) -> Self::Output {
        (
            self.0,
            self.1 .0,
            self.1 .1 .0,
            self.1 .1 .1 .0,
            self.1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1,
        )
    }
}

impl<F0, F1, F2, F3, F4, F5, F6, F7> FlatRight<8> for (F0, (F1, (F2, (F3, (F4, (F5, (F6, F7))))))) {
    type Output = (F0, F1, F2, F3, F4, F5, F6, F7);

    fn flat(self) -> Self::Output {
        (
            self.0,
            self.1 .0,
            self.1 .1 .0,
            self.1 .1 .1 .0,
            self.1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1 .1,
        )
    }
}

impl<F0, F1, F2, F3, F4, F5, F6, F7, F8> FlatRight<9>
    for (F0, (F1, (F2, (F3, (F4, (F5, (F6, (F7, F8))))))))
{
    type Output = (F0, F1, F2, F3, F4, F5, F6, F7, F8);

    fn flat(self) -> Self::Output {
        (
            self.0,
            self.1 .0,
            self.1 .1 .0,
            self.1 .1 .1 .0,
            self.1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1 .1 .0,
            self.1 .1 .1 .1 .1 .1 .1 .1,
        )
    }
}

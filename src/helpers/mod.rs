pub mod _burn;
pub mod _candle;
pub mod _dfdx;

use std::fmt::Display;

use float_cmp::ApproxEq;

pub trait Approx<D>: Copy {
    fn approx<MARGIN>(self, rhs: Self, margin: MARGIN) -> bool
    where
        D: ApproxEq,
        MARGIN: Into<<f32 as ApproxEq>::Margin> + Clone,
        <D as ApproxEq>::Margin: From<MARGIN>;
}

impl<D> Approx<D> for D
where
    D: Copy + ApproxEq + Display,
{
    fn approx<MARGIN>(self, rhs: Self, margin: MARGIN) -> bool
    where
        MARGIN: Into<<D as ApproxEq>::Margin> + Clone,
        <D as ApproxEq>::Margin: From<MARGIN>,
    {
        if !self.approx_eq(rhs, margin) {
            println!("(lhr, rhs) = ({self}, {rhs})");
            false
        } else {
            true
        }
    }
}

impl<D, const N: usize> Approx<D> for [D; N]
where
    D: Copy + ApproxEq + Display,
{
    fn approx<MARGIN>(self, rhs: Self, margin: MARGIN) -> bool
    where
        MARGIN: Into<<D as ApproxEq>::Margin> + Clone,
        <D as ApproxEq>::Margin: From<MARGIN>,
    {
        for (nl, nr) in self.into_iter().zip(rhs) {
            if !nl.approx_eq(nr, margin.clone()) {
                println!("(lhs, rhs) = ({nl}, {nr})");
                return false;
            }
        }
        true
    }
}

impl<D, const N: usize, const M: usize> Approx<D> for [[D; M]; N]
where
    D: Copy + ApproxEq + Display,
{
    fn approx<MARGIN>(self, rhs: Self, margin: MARGIN) -> bool
    where
        MARGIN: Into<<D as ApproxEq>::Margin> + Clone,
        <D as ApproxEq>::Margin: From<MARGIN>,
    {
        for (nl, nr) in self.into_iter().zip(rhs) {
            for (ml, mr) in nl.into_iter().zip(nr) {
                if !ml.approx_eq(mr, margin.clone()) {
                    println!("(lhs, rhs) = ({ml}, {mr})");
                    return false;
                }
            }
        }
        true
    }
}

impl<D, const N: usize, const M: usize, const O: usize> Approx<D> for [[[D; O]; M]; N]
where
    D: Copy + ApproxEq + Display,
{
    fn approx<MARGIN>(self, rhs: Self, margin: MARGIN) -> bool
    where
        MARGIN: Into<<D as ApproxEq>::Margin> + Clone,
        <D as ApproxEq>::Margin: From<MARGIN>,
    {
        for (nl, nr) in self.into_iter().zip(rhs) {
            for (ml, mr) in nl.into_iter().zip(nr) {
                for (ol, or) in ml.into_iter().zip(mr) {
                    if !ol.approx_eq(or, margin.clone()) {
                        println!("(lhs, rhs) = ({ol}, {or})");
                        return false;
                    }
                }
            }
        }
        true
    }
}

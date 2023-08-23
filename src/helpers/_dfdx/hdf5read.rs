use super::{Device, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Shape, Tensor, TensorFrom};
use dfdx::shapes::ConstShape;
use ndarray::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};

pub trait Hdf5Read {
    fn tensor_from_hdf5<E, Rank>(
        self,
        file: &hdf5::File,
        name: &str,
    ) -> anyhow::Result<Tensor<Rank, E, Device>>
    where
        E: hdf5::H5Type + dfdx::dtypes::Unit,
        Rank: ConstShape + Shape + RankToArray<{ <Rank as Shape>::NUM_DIMS }>;
}

impl Hdf5Read for &Device {
    fn tensor_from_hdf5<E, Rank>(
        self,
        file: &hdf5::File,
        name: &str,
    ) -> anyhow::Result<Tensor<Rank, E, Device>>
    where
        E: hdf5::H5Type + dfdx::dtypes::Unit,
        Rank: ConstShape + Shape + RankToArray<{ <Rank as Shape>::NUM_DIMS }>,
    {
        let data = file.dataset(name)?.read::<E, Rank::Ix>()?;
        assert_eq!(data.shape(), Rank::to_array());

        if data.is_standard_layout() {
            Ok(self.tensor(data.into_raw_vec()))
        } else {
            Ok(self.tensor(data.into_iter().collect::<Vec<_>>()))
        }
    }
}

pub trait RankToArray<const N: usize> {
    type Ix: ndarray::Dimension;
    fn to_array() -> [usize; N];
}

impl RankToArray<0> for () {
    type Ix = Ix0;
    fn to_array() -> [usize; 0] {
        []
    }
}

impl<const M0: usize> RankToArray<1> for Rank1<M0> {
    type Ix = Ix1;
    fn to_array() -> [usize; 1] {
        [M0]
    }
}

impl<const M0: usize, const M1: usize> RankToArray<2> for Rank2<M0, M1> {
    type Ix = Ix2;
    fn to_array() -> [usize; 2] {
        [M0, M1]
    }
}

impl<const M0: usize, const M1: usize, const M2: usize> RankToArray<3> for Rank3<M0, M1, M2> {
    type Ix = Ix3;
    fn to_array() -> [usize; 3] {
        [M0, M1, M2]
    }
}

impl<const M0: usize, const M1: usize, const M2: usize, const M3: usize> RankToArray<4>
    for Rank4<M0, M1, M2, M3>
{
    type Ix = Ix4;
    fn to_array() -> [usize; 4] {
        [M0, M1, M2, M3]
    }
}

impl<const M0: usize, const M1: usize, const M2: usize, const M3: usize, const M4: usize>
    RankToArray<5> for Rank5<M0, M1, M2, M3, M4>
{
    type Ix = Ix5;
    fn to_array() -> [usize; 5] {
        [M0, M1, M2, M3, M4]
    }
}

impl<
        const M0: usize,
        const M1: usize,
        const M2: usize,
        const M3: usize,
        const M4: usize,
        const M5: usize,
    > RankToArray<6> for Rank6<M0, M1, M2, M3, M4, M5>
{
    type Ix = Ix6;
    fn to_array() -> [usize; 6] {
        [M0, M1, M2, M3, M4, M5]
    }
}

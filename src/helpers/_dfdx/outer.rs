use super::Device_;
use dfdx::tensor_ops::{BroadcastTo, PermuteTo};
use dfdx::{
    shapes::{Rank1, Rank2},
    tensor::Tensor,
};

/// Outer-Product.
pub trait OuterProduct<Rhs> {
    type Output;
    fn outer(self, rhs: Rhs) -> Self::Output;
}

impl<const M: usize, const N: usize> OuterProduct<Tensor<Rank1<N>, f32, Device_>>
    for Tensor<Rank1<M>, f32, Device_>
{
    type Output = Tensor<Rank2<M, N>, f32, Device_>;
    /// Implements the outer-product between two 1D Tensors.
    fn outer(self, rhs: Tensor<Rank1<N>, f32, Device_>) -> Self::Output {
        let lhs = self.broadcast::<Rank2<M, N>, _>();
        let rhs = rhs.broadcast::<Rank2<N, M>, _>();
        let rhs = rhs.permute::<Rank2<M, N>, _>();
        lhs * rhs
    }
}

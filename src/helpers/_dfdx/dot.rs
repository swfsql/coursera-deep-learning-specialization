use super::Device;
use dfdx::tensor_ops::BroadcastTo;
use dfdx::{
    shapes::{Rank1, Rank2, Rank3, Shape},
    tensor::{AsArray, Tensor},
};

/// Dot-Product.
pub trait Dot<Rhs> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

// TODO: add missing case (set of cases) of N-D Tensor dot M-D Tensor, for M>=2.
// numpy docs says "it is a sum product over the last axis of a and the second-to-last axis of b".

impl<const M: usize> Dot<Tensor<Rank1<M>, f32, Device>> for Tensor<Rank1<M>, f32, Device> {
    type Output = f32;
    /// Implements the dot-product between two 1D Tensors of the same length.
    ///
    /// Inner Product between the two tensors.
    fn dot(self, rhs: Tensor<Rank1<M>, f32, Device>) -> Self::Output {
        use dfdx::tensor_ops::SumTo;
        // element-wise product
        (self * rhs)
            // sum
            .sum()
            // extract scalar from the 0-Dimension tensor
            .array()
    }
}

impl<const M: usize, const N: usize, const O: usize> Dot<Tensor<Rank2<N, O>, f32, Device>>
    for Tensor<Rank2<M, N>, f32, Device>
{
    type Output = Tensor<Rank2<M, O>, f32, Device>;
    /// Implements the dot-product between two 2D Tensors.
    ///
    /// Matrix Multiplication between the two tensors (matmul).
    fn dot(self, rhs: Tensor<Rank2<N, O>, f32, Device>) -> Self::Output {
        dfdx::tensor_ops::TryMatMul::matmul(self, rhs)
    }
}

impl<S: Shape> Dot<f32> for Tensor<S, f32, Device> {
    type Output = Tensor<S, f32, Device>;
    /// Implements the dot-product between a Tensor and a scalar.
    ///
    /// The scalar multiplies element-wise.
    fn dot(self, rhs: f32) -> Self::Output {
        self * rhs
    }
}

impl<S: Shape> Dot<Tensor<S, f32, Device>> for f32 {
    type Output = Tensor<S, f32, Device>;
    /// Implements the dot-product between a scalar and a Tensor.
    ///
    /// The scalar multiplies element-wise.
    fn dot(self, rhs: Tensor<S, f32, Device>) -> Self::Output {
        rhs * self
    }
}

impl<const M: usize, const N: usize> Dot<Tensor<Rank1<N>, f32, Device>>
    for Tensor<Rank2<M, N>, f32, Device>
{
    type Output = Tensor<Rank1<M>, f32, Device>;
    /// Implements the dot-product between a 2D Tensor and a 1D Tensor.
    ///
    /// The 1D Tensor is broadcasted to match the dimensions of the 2D Tensor, then
    /// the two Tensors are multiplied element-wise, then the last dimension is reduced
    /// by a sum of it's elements.
    fn dot(self, rhs: Tensor<Rank1<N>, f32, Device>) -> Self::Output {
        use dfdx::tensor_ops::SumTo;
        let rhs = rhs.broadcast::<Rank2<M, N>, _>();
        let mul = self * rhs;
        mul.sum()
    }
}

impl<const M: usize, const N: usize, const O: usize> Dot<Tensor<Rank1<O>, f32, Device>>
    for Tensor<Rank3<M, N, O>, f32, Device>
{
    type Output = Tensor<Rank2<M, N>, f32, Device>;
    /// Implements the dot-product between a 3D Tensor and a 1D Tensor.
    ///
    /// The 1D Tensor is broadcasted to match the dimensions of the 3D Tensor, then
    /// the two Tensors are multiplied element-wise, then the last dimension is reduced
    /// by a sum of it's elements.
    fn dot(self, rhs: Tensor<Rank1<O>, f32, Device>) -> Self::Output {
        use dfdx::tensor_ops::SumTo;
        let rhs = rhs.broadcast::<Rank3<M, N, O>, _>();
        let mul = self * rhs;
        mul.sum()
    }
}
// Note: it's hard to not make each implementation by hand (or macro), because the dimension is reduced.
// at least, I don't know how to do that.

#[test]
fn test_dot() {
    use dfdx::tensor::TensorFrom;

    let dev = super::device();

    // 1D dot 1D
    let x1: Tensor<Rank1<2>, f32, Device> = dev.tensor([2., 3.]);
    let x2: Tensor<Rank1<2>, f32, Device> = dev.tensor([2., 1.]);
    assert_eq!(x1.dot(x2), 7.);

    // 2D dot 2D
    let x1: Tensor<Rank2<2, 2>, f32, Device> = dev.tensor([[1., 0.], [0., 1.]]);
    let x2: Tensor<Rank2<2, 2>, f32, Device> = dev.tensor([[4., 1.], [2., 2.]]);
    assert_eq!(x1.dot(x2).array(), [[4., 1.], [2., 2.]]);
}

pub mod dot;
pub mod flat_right;
pub mod hdf5read;
pub mod outer;
pub mod stack_right;

#[cfg(not(feature = "cuda"))]
pub type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
pub type Device = dfdx::tensor::Cuda;

pub use dfdx::{
    dtypes::Dtype,
    prelude::Storage,
    shapes::Axis,
    shapes::{
        Axes, Axes2, Axes3, Axes4, Axes5, Axes6, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6,
        Shape,
    },
    tensor::{AsArray, OnesTensor, SampleTensor, Tensor, TensorFrom, ZerosTensor},
    tensor_ops::{
        BroadcastTo, ChooseFrom, MeanTo, PermuteTo, RealizeTo, ReshapeTo, SumTo, TryGe, TryGt,
    },
};
pub use dot::Dot;
pub use flat_right::FlatRightN;
pub use hdf5read::Hdf5Read;
pub use outer::OuterProduct;
pub use stack_right::StackRightN;

pub type TensorF32<Rank> = Tensor<Rank, f32, Device>;
pub type TensorU8<Rank> = Tensor<Rank, u8, Device>;

#[cfg(not(feature = "cuda"))]
pub fn device() -> Device {
    Device::seed_from_u64(0)
}

#[cfg(feature = "cuda")]
pub fn device() -> Device {
    Device::try_build(0, 0).unwrap()
}

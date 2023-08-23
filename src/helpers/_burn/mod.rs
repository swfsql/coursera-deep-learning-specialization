#[cfg(not(feature = "wgpu"))]
use burn_ndarray::NdArrayBackend;

#[cfg(not(feature = "wgpu"))]
pub type _Backend<Float> = NdArrayBackend<Float>;

#[cfg(not(feature = "wgpu"))]
pub use burn_ndarray::NdArrayDevice as Device;

#[cfg(feature = "wgpu")]
use burn_wgpu::{AutoGraphicsApi, WgpuBackend};

#[cfg(feature = "wgpu")]
pub type _Backend<Float> = WgpuBackend<AutoGraphicsApi, Float, i32>;

#[cfg(feature = "wgpu")]
pub use burn_wgpu::WgpuDevice as Device;

pub use burn::tensor::{backend::Backend, Tensor};

#[cfg(not(feature = "wgpu"))]
pub fn device() -> Device {
    Device::Cpu
}

#[cfg(feature = "wgpu")]
pub fn device() -> Device {
    Device::BestAvailable
}

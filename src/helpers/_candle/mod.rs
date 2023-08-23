pub use candle_core::{Device, Tensor};

pub fn device() -> anyhow::Result<Device> {
    Ok(Device::cuda_if_available(0)?)
}

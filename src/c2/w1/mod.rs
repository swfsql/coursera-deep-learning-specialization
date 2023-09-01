pub mod pa_01_init;
pub mod pa_02_regularization;
pub mod pa_03_gradient_check;
pub mod util;

pub mod prelude {
    #[allow(unused_imports)]
    pub(crate) use super::pa_01_init::layerc2;
    pub use super::pa_01_init::{HeInit, LayerInitialization, NormalInit, ZeroInit};
    pub use super::pa_02_regularization::{Dropout, DropoutSeed, FrobeniusReg};
}

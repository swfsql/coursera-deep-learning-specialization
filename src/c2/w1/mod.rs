pub mod pa_01_init;
pub mod pa_02_regularization;
pub mod util;

pub mod prelude {
    #[allow(unused_imports)]
    pub(crate) use super::pa_01_init::layerc2;
    pub use super::pa_01_init::{HeInit, LayerInitialization, NormalInit, ZeroInit};
}

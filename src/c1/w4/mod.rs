pub mod pa_01_deep_nn;
pub mod pa_02_deep_nn;

pub mod prelude {
    #[allow(unused_imports)]
    pub(crate) use super::pa_01_deep_nn::layerc1;
    pub use super::pa_01_deep_nn::{
        non_zero, Cache, CleanupGrads, CostSetup, DownUpGrads, Downward, DownwardA, DownwardZ,
        DownwardZA, Dzdx, Grads, LastA, Layer, Linear, MLogistical, MSquared, Mda, Mdz, ReLU,
        Sigmoid, SplitHead, Tanh, UpdateParameters, UpwardAZ, UpwardDownZUpA, UpwardJA, UpwardZwb,
        Wb, WrapA, A, B, W, X, Y, Z,
    };
    pub use super::pa_02_deep_nn::LayersSetup;
}

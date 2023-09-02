//! Initialization
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Initialization/Initialization.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Initialization.ipynb
//!
//! I recommend first watching all of C2W1.

#![allow(unused_imports)]

use crate::c1::w4::prelude::*;
use crate::c2::w1::util::pa_01::*;
use crate::helpers::{Approx, _dfdx::*};
use dfdx::prelude::*;

/// C02W01PA01 Part 1 - Neural Network Model.
mod _1 {
    use super::*;

    pub trait LayerInitialization: Sized {
        fn init_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            device: &Device_,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A>;

        fn init_layer_wb<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            device: &Device_,
        ) -> Layer<FEATLEN, NODELEN, Z, A>
        where
            Z: Default,
            A: Default,
        {
            self.init_layer(device, Z::default(), A::default())
        }

        fn reinit_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            layer: Layer<FEATLEN, NODELEN, Z, A>,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A>;

        fn reinit_layer_wb<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            layer: Layer<FEATLEN, NODELEN, Z, A>,
        ) -> Layer<FEATLEN, NODELEN, Z, A>
        where
            Z: Default,
            A: Default,
        {
            self.reinit_layer(layer, Z::default(), A::default())
        }
    }

    /// A macro similar to layerc1 except that the initialization information must be informed for every layer.
    #[allow(unused_macros)]
    #[allow(unused_attributes)]
    #[macro_export]
    macro_rules! layerc2 {
            // implicit layers creation, hidden layers are Linear_>Relu, the last is Linear_>Sigmoid
            ($dev:expr, [$head_layer_node:literal, $($tail_layers_nodes:literal $tail_inits:expr),*]) => {
                // separates the feature and forward to another macro call
                $crate::layerc2!($dev, auto, $head_layer_node, [$($tail_layers_nodes $tail_inits),*])
            };

            // explicit single layer creation
            ($dev:expr, $features:literal, $z:expr=>$a:expr => $layer_nodes:literal $init:expr) => {
                {
                    // returns the layer
                    type _Layer<const FEATLEN: usize, const NODELEN: usize, Z, A> = $crate::c1::w4::pa_01_deep_nn::Layer<FEATLEN, NODELEN, Z, A>;
                    use $crate::c2::w1::pa_01_init::LayerInitialization as _Init;
                    let _layer: _Layer<$features, $layer_nodes, _, _> = _Init::init_layer($init, $dev, $z, $a);
                    _layer
                }
            };

            // implicit hidden layer creation, all Linear_>Relu
            ($dev:expr, auto, $node_features:literal, [$head_layer_node:literal $head_init:expr, $($tail_layers_nodes:literal $tail_inits:expr),*]) => {
                (
                    {
                        // creates a single implicit layer
                        let _linear = $crate::c1::w4::pa_01_deep_nn::Linear_::default();
                        let _relu = $crate::c1::w4::pa_01_deep_nn::ReLU::default();
                        $crate::layerc2!($dev, $node_features, _linear=>_relu => $head_layer_node $head_init)
                    },
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    $crate::layerc2!($dev, auto, $head_layer_node, [$($tail_layers_nodes $tail_inits),*])
                )
            };

            // explicit hidden layer creation
            ($dev:expr, $node_features:literal, $z:expr=>$a:expr => [$head_layer_node:literal $head_init:expr, $($tail_layers_nodes:literal $tail_inits:expr),*] $($other:tt)*) => {
                (
                    // creates a single layer
                    $crate::layerc2!($dev, $node_features, $z=>$a => $head_layer_node $head_init),
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    $crate::layerc2!($dev, $head_layer_node, $z=>$a => [$($tail_layers_nodes $tail_inits),*] $($other)*)
                )
            };

            // implicit last layer creation, Linear_>Sigmoid
            ($dev:expr, auto, $node_features:literal, [$last_layer_node:literal $last_init:expr]) => {
                {
                    // creates a single implicit layer (which is the last)
                    let _linear = $crate::c1::w4::pa_01_deep_nn::Linear_::default();
                    let _sigmoid = $crate::c1::w4::pa_01_deep_nn::Sigmoid::default();
                    $crate::layerc2!($dev, $node_features, _linear=>_sigmoid => $last_layer_node $last_init)
                }
            };

            // explicit last layer creation (with no continuation)
            ($dev:expr, $node_features:literal, $z:expr=>$a:expr => [$last_layer_node:literal $last_init:expr]) => {
                // returns the layer
                $crate::layerc2!($dev, $node_features, $z=>$a => $last_layer_node $last_init)
            };

            // explicit "last" layer creation (with a continuation)
            ($dev:expr, $node_features:literal, $z:expr=>$a:expr => [$last_layer_node:literal $last_init:expr] $($other:tt)*) => {
                (
                    // returns the layer
                    $crate::layerc2!($dev, $node_features, $z=>$a => $last_layer_node $last_init),
                    // makes a brand new macro call on whatever arguments remains,
                    // forwarding the last "feature" information
                    $crate::layerc2!($dev, $last_layer_node $($other)*)
                )
            };
        }
    pub(crate) use layerc2;
}
pub(crate) use _1::layerc2;
pub use _1::LayerInitialization;

/// C02W01PA01 Part 2 - Zero Initialization.
pub mod _2 {
    use super::*;

    /// Initializes w and b to zero.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZeroInit;

    impl LayerInitialization for ZeroInit {
        /// Initializes w and b to zero.
        fn init_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            device: &Device_,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            Layer::with(device.zeros(), device.zeros(), z, a)
        }

        /// Re-initializes w and b to zero.
        fn reinit_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            mut layer: Layer<FEATLEN, NODELEN, Z, A>,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            layer.w.fill_with_zeros();
            layer.b.fill_with_zeros();
            layer.z = z;
            layer.a = a;
            layer
        }
    }

    #[test]
    fn test_zero_init() {
        let dev = &device();
        use ZeroInit as Zero;

        let layers = layerc2!(dev, [3, 2 Zero, 1 Zero]);
        assert_eq!(layers.0.w.array(), [[0.; 3]; 2]);
        assert_eq!(layers.0.b.array(), [[0.; 1]; 2]);
        assert_eq!(layers.1.w.array(), [[0.; 2]; 1]);
        assert_eq!(layers.1.b.array(), [[0.; 1]; 1]);
    }

    #[test]
    fn test_zero_train() {
        use ZeroInit as Zero;

        let dev = &device();
        let train_x = dev.tensor(XTRAIN);
        let train_y = dev.tensor(YTRAIN);

        let layers = layerc2!(dev, [2, 10 Zero, 5 Zero, 1 Zero]);
        let mut opt = GradientDescend::new(1e-2);
        let mut cost_setup = MLogistical;
        // 400 training steps is enough
        let layers = layers.train(
            train_x.clone(),
            train_y.clone(),
            &mut cost_setup,
            &mut opt,
            400,
            100,
        );
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            // the cost is constant
            .approx(std::f32::consts::LN_2, (1e-7, 0)));

        // train accuracy
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert_eq!(accuracy.array(), [50.]);
        }

        // test accuracy
        {
            let test_x = dev.tensor(XTEST);
            let test_y = dev.tensor(YTEST);
            let yhat = layers.clone().predict(test_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert_eq!(accuracy.array(), [50.]);
        }
    }
}
pub use _2::ZeroInit;

/// C02W01PA01 Part 3 - Random Initialization.
pub mod _3 {
    use super::*;

    /// Initializes w to scaled samples from a normal distribution and initializes b to zero.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct NormalInit(f32);

    impl LayerInitialization for NormalInit {
        /// Initializes w to scaled samples from a normal distribution and initializes b to zero.
        fn init_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            device: &Device_,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            Layer::with(device.sample_normal() * self.0, device.zeros(), z, a)
        }

        /// Re-initializes w to scaled samples from a normal distribution and initializes b to zero.
        fn reinit_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            mut layer: Layer<FEATLEN, NODELEN, Z, A>,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            layer.w.fill_with_distr(rand_distr::StandardNormal);
            layer.w = layer.w * self.0;
            layer.b.fill_with_zeros();
            layer.z = z;
            layer.a = a;
            layer
        }
    }

    #[test]
    fn test_rand_init() {
        let dev = &device();

        let normal10 = NormalInit(10.);
        let layers = layerc2!(dev, [3, 2 normal10, 1 normal10]);
        assert_eq!(
            layers.0.w.array(),
            [
                [7.12813, 8.583315, -24.362438],
                [1.6334426, -12.750102, 12.87171]
            ]
        );
        assert_eq!(layers.0.b.array(), [[0.; 1]; 2]);
        assert_eq!(layers.1.w.array(), [[-14.814075, 6.1259484]]);
        assert_eq!(layers.1.b.array(), [[0.; 1]; 1]);
    }

    #[test]
    fn test_rand_train() {
        let dev = &device();
        let train_x = dev.tensor(XTRAIN);
        let train_y = dev.tensor(YTRAIN);

        let normal10 = NormalInit(10.);
        let layers = layerc2!(dev, [2, 10 normal10, 5 normal10, 1 normal10]);
        let mut opt = GradientDescend::new(1e-2);
        let mut cost_setup = MLogistical;
        // 400 training steps is enough
        let layers = layers.train(
            train_x.clone(),
            train_y.clone(),
            &mut cost_setup,
            &mut opt,
            400,
            100,
        );
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            // the model got stuck and the cost is high (bad luck for the random's seed)
            .approx(33.985054, (1e-2, 0)));

        // train accuracy (still bad because the model got stuck)
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert_eq!(accuracy.array(), [50.666664]);
        }

        // test accuracy (still bad because the model got stuck)
        {
            let test_x = dev.tensor(XTEST);
            let test_y = dev.tensor(YTEST);
            let yhat = layers.clone().predict(test_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert_eq!(accuracy.array(), [49.0]);
        }
    }
}
pub use _3::NormalInit;

/// C02W01PA01 Part 4 - He Initialization.
pub mod _4 {
    use super::*;

    /// Initializes w to scaled samples from a normal distribution (scaled as recommended for ReLU activations by He),
    /// and initializes b to zero.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct HeInit(pub f32);

    impl HeInit {
        pub fn sigmoid() -> Self {
            HeInit(1. / 2f32.sqrt())
        }
    }

    impl LayerInitialization for HeInit {
        /// Initializes w to scaled samples from a normal distribution (scaled as recommended for ReLU activations by He),
        /// and initializes b to zero.
        fn init_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            device: &Device_,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            Layer::with(
                device.sample_normal() * self.0 * (2. / (FEATLEN as f32)).sqrt(),
                device.zeros(),
                z,
                a,
            )
        }

        /// Re-initializes w to scaled samples from a normal distribution (scaled as recommended for ReLU activations by He),
        /// and initializes b to zero.
        fn reinit_layer<const FEATLEN: usize, const NODELEN: usize, Z, A>(
            self,
            mut layer: Layer<FEATLEN, NODELEN, Z, A>,
            z: Z,
            a: A,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            layer.w.fill_with_distr(rand_distr::StandardNormal);
            layer.w = layer.w * self.0 * (2. / (FEATLEN as f32)).sqrt();
            layer.b.fill_with_zeros();
            layer.z = z;
            layer.a = a;
            layer
        }
    }

    #[test]
    fn test_he_init() {
        let dev = &device();

        let he = HeInit(1.);
        let layers = layerc2!(dev, [2, 4 he, 1 he]);
        assert_eq!(
            layers.0.w.array(),
            [
                [0.712813, 0.85833144],
                [-2.4362438, 0.16334426],
                [-1.2750102, 1.287171],
                [-1.4814075, 0.61259484]
            ]
        );
        assert_eq!(layers.0.b.array(), [[0.; 1]; 4]);
        assert_eq!(
            layers.1.w.array(),
            [[0.43685743, 1.229541, 1.0871886, -1.9944816]]
        );
        assert_eq!(layers.1.b.array(), [[0.; 1]; 1]);
    }

    #[test]
    fn test_he_train() {
        let dev = &device();
        let train_x = dev.tensor(XTRAIN);
        let train_y = dev.tensor(YTRAIN);

        let he = HeInit(1.);
        let layers = layerc2!(dev, [2, 10 he, 5 he, 1 he]);
        let mut opt = GradientDescend::new(1e-2);
        let mut cost_setup = MLogistical;
        // 400 training steps is enough
        let layers = layers.train(
            train_x.clone(),
            train_y.clone(),
            &mut cost_setup,
            &mut opt,
            15_000,
            1000,
        );
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            .approx(0.09247275, (1e-4, 0)));

        // train accuracy
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert!(accuracy.array().approx([98.], (2., 0)));
        }

        // test accuracy (still bad because the model got stuck)
        {
            let test_x = dev.tensor(XTEST);
            let test_y = dev.tensor(YTEST);
            let yhat = layers.clone().predict(test_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert!(accuracy.array().approx([96.], (2., 0)));
        }
    }
}
pub use _4::HeInit;

/// C02W01PA01 Part 5 - Conclusions.
mod _5 {
    // no content
}

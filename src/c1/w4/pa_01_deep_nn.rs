//! Building your Deep Neural Network: Step by Step
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Neural%20Networks%20and%20Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step.ipynb
//!
//! I recommend first watching all of C1W4.

#[allow(unused_imports)]
use crate::helpers::{Approx, _dfdx::*};

/// C01W04PA01 Part 1 - Packages.
mod _1 {
    // no content
}

/// C01W04PA01 Part 2 - Outline of the Assignment.
mod _2 {
    // no content
}

/// C01W04PA01 Part 3 - Initialization.
pub mod _3 {
    use super::*;

    /// C01W04PA01 Section 1 - 2-Layer Neural Network.
    pub mod _1 {
        use super::*;

        pub type W<const NODELEN: usize, const FEATLEN: usize> = TensorF32<Rank2<NODELEN, FEATLEN>>;
        pub type B<const NODELEN: usize> = TensorF32<Rank2<NODELEN, 1>>;

        #[derive(Clone, Debug)]
        pub struct Layer<const FEATLEN: usize, const NODELEN: usize, Z = Linear, A = ReLU> {
            /// Rows for nodes, columns for features.
            pub w: W<NODELEN, FEATLEN>,

            /// Rows for nodes.
            // for the bias, the implied feature is a single constant 1.0 value.
            pub b: B<NODELEN>,

            /// Calculation function z(features, w, b).
            pub z: Z,

            /// Activation function a(z).
            pub a: A,
        }

        impl<const FEATLEN: usize, const NODELEN: usize, Z, A> Layer<FEATLEN, NODELEN, Z, A> {
            pub fn uniform(device: &Device, z: Z, a: A) -> Self {
                let w: W<NODELEN, FEATLEN> = device.sample_uniform();
                let b: B<NODELEN> = device.zeros();
                Self::with(w, b, z, a)
            }
            pub fn uniform_wb(device: &Device) -> Self
            where
                Z: Default,
                A: Default,
            {
                let w: W<NODELEN, FEATLEN> = device.sample_uniform();
                let b: B<NODELEN> = device.zeros();
                Self::with_wb(w, b)
            }
            pub fn normal(device: &Device, z: Z, a: A) -> Self {
                let w: W<NODELEN, FEATLEN> = device.sample_normal();
                let b: B<NODELEN> = device.zeros();
                Self::with(w, b, z, a)
            }
            pub fn normal_wb(device: &Device) -> Self
            where
                Z: Default,
                A: Default,
            {
                let w: W<NODELEN, FEATLEN> = device.sample_normal();
                let b: B<NODELEN> = device.zeros();
                Self::with_wb(w, b)
            }
            pub fn with_wb(w: W<NODELEN, FEATLEN>, b: B<NODELEN>) -> Self
            where
                Z: Default,
                A: Default,
            {
                Self::with(w, b, Z::default(), A::default())
            }
            pub fn with(w: W<NODELEN, FEATLEN>, b: B<NODELEN>, z: Z, a: A) -> Self {
                Self { w, b, z, a }
            }
            pub fn from_values(
                device: &Device,
                w: [[f32; FEATLEN]; NODELEN],
                b: [[f32; 1]; NODELEN],
                z: Z,
                a: A,
            ) -> Self {
                let w: W<NODELEN, FEATLEN> = device.tensor(w);
                let b: B<NODELEN> = device.tensor(b);
                Self::with(w, b, z, a)
            }
        }

        /// Weight and Bias calculation.
        ///
        /// `z(w, features, b) = w * features + b`.
        #[derive(Clone, Debug, PartialEq, Default)]
        pub struct Linear;

        /// Activation calculation.
        ///
        /// `a(z) = 0 if z < 0`.
        /// `a(z) = 1 if z >= 0`.
        #[derive(Clone, Debug, PartialEq, Default)]
        pub struct ReLU;

        /// Activation calculation.
        ///
        /// `a(z) = σ(z) = 1 / (1 + e^-z)`
        #[derive(Clone, Debug, PartialEq, Default)]
        pub struct Sigmoid;

        /// Activation calculation.
        ///
        /// `a(z) = tanh(z) = (e^z - e^-z) / (e^z + e^-z)`
        #[derive(Clone, Debug, PartialEq, Default)]
        pub struct Tanh;

        #[test]
        fn example() {
            let dev = device();
            let _model = (
                Layer::<3, 2, Linear, ReLU>::normal(&dev, Linear, ReLU),
                Layer::<2, 1, Linear, Sigmoid>::normal(&dev, Linear, Sigmoid),
            );

            // (nothing to assert)
        }
    }
    pub use _1::{Layer, Linear, ReLU, Sigmoid, Tanh, B, W};

    /// C01W04PA01 Section 2 - L-Layer Neural Network.
    pub mod _2 {
        use super::*;

        /// Creates one or more layers from a scaled normal distribution.  
        /// The weights are further scaled to the inverse sqrt of the number of features.  
        /// The biases are set to zero.
        ///
        /// Example with implicit `Linear`>`Relu` hidden layers and with a final `Linear`>`Sigmoid` final layer
        /// ```rust
        /// # use coursera_exercises::layerc1;
        /// # use coursera_exercises::c1::w4::pa_01_deep_nn::{Linear, ReLU, Sigmoid, Layer};
        /// # use coursera_exercises::helpers::_dfdx::*;
        /// # let dev = device();
        /// let s = layerc1!(&dev, 1., [1, 1]);
        /// let rs = layerc1!(&dev, 1., [1, 1, 1]);
        /// let rrs = layerc1!(&dev, 1., [1, 1, 1, 1]).flat3();
        /// ```
        ///
        /// Example where all functions are explicit:
        /// ```rust
        /// # use coursera_exercises::layerc1;
        /// # use coursera_exercises::c1::w4::pa_01_deep_nn::{Linear, ReLU, Sigmoid, Layer};
        /// # use coursera_exercises::helpers::_dfdx::*;
        /// # let dev = device();
        /// let s = layerc1!(&dev, 1., 1, Linear > Sigmoid => [1]);
        /// let rs = layerc1!(&dev, 1., 1, Linear > ReLU => [1], Linear > Sigmoid => [1]);
        /// let rrs = layerc1!(&dev, 1., 1, Linear > ReLU => [1, 1], Linear > Sigmoid => [1]).flat3();
        /// ```
        #[allow(unused_macros)]
        #[allow(unused_attributes)]
        #[macro_export]
        macro_rules! layerc1 {
            // implicit layers creation, hidden layers are Linear>Relu, the last is Linear>Sigmoid
            ($dev:expr, $scalar:expr, [$head_layer_node:literal, $($tail_layers_nodes:literal),*]) => {
                // separates the feature and forward to another macro call
                $crate::layerc1!($dev, auto, $scalar, $head_layer_node, [$($tail_layers_nodes),*])
            };

            // explicit single layer creation
            ($dev:expr, $scalar:expr, $features:literal, $z:expr=>$a:expr => $layer_nodes:literal) => {
                {
                    // returns the layer
                    type _Layer<const FEATLEN: usize, const NODELEN: usize, Z, A> = $crate::c1::w4::pa_01_deep_nn::Layer<FEATLEN, NODELEN, Z, A>;
                    let mut _layer = _Layer::<$features, $layer_nodes, _, _>::normal($dev, $z, $a);
                    _layer.w = _layer.w * ($scalar as f32) / ($features as f32).sqrt();
                    _layer
                }
            };

            // implicit hidden layer creation, all Linear>Relu
            ($dev:expr, auto, $scalar:expr, $node_features:literal, [$head_layer_node:literal, $($tail_layers_nodes:literal),*]) => {
                (
                    {
                        // creates a single implicit layer
                        let _linear = $crate::c1::w4::pa_01_deep_nn::Linear::default();
                        let _relu = $crate::c1::w4::pa_01_deep_nn::ReLU::default();
                        $crate::layerc1!($dev, $scalar, $node_features, _linear=>_relu => $head_layer_node)
                    },
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    $crate::layerc1!($dev, auto, $scalar, $head_layer_node, [$($tail_layers_nodes),*])
                )
            };

            // explicit hidden layer creation
            ($dev:expr, $scalar:expr, $node_features:literal, $z:expr=>$a:expr => [$head_layer_node:literal, $($tail_layers_nodes:literal),*] $($other:tt)*) => {
                (
                    // creates a single layer
                    $crate::layerc1!($dev, $scalar, $node_features, $z=>$a => $head_layer_node),
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    $crate::layerc1!($dev, $scalar, $head_layer_node, $z=>$a => [$($tail_layers_nodes),*] $($other)*)
                )
            };

            // implicit last layer creation, Linear>Sigmoid
            ($dev:expr, auto, $scalar:expr, $node_features:literal, [$last_layer_node:literal]) => {
                {
                    // creates a single implicit layer (which is the last)
                    let _linear = $crate::c1::w4::pa_01_deep_nn::Linear;
                    let _sigmoid = $crate::c1::w4::pa_01_deep_nn::Sigmoid;
                    $crate::layerc1!($dev, $scalar, $node_features, _linear=>_sigmoid => $last_layer_node)
                }
            };

            // explicit last layer creation (with no continuation)
            ($dev:expr, $scalar:expr, $node_features:literal, $z:expr=>$a:expr => [$last_layer_node:literal]) => {
                // returns the layer
                $crate::layerc1!($dev, $scalar, $node_features, $z=>$a => $last_layer_node)
            };

            // explicit "last" layer creation (with a continuation)
            ($dev:expr, $scalar:expr, $node_features:literal, $z:expr=>$a:expr => [$last_layer_node:literal] $($other:tt)*) => {
                (
                    // returns the layer
                    $crate::layerc1!($dev, $scalar, $node_features, $z=>$a => $last_layer_node),
                    // makes a brand new macro call on whatever arguments remains,
                    // forwarding the last "feature" information
                    $crate::layerc1!($dev, $scalar, $last_layer_node $($other)*)
                )
            };
        }
        pub(crate) use layerc1;

        #[test]
        fn test_layers() {
            let dev = device();
            let rs = layerc1!(&dev, 1., [5, 4, 3]);
            let _r1: Layer<5, 4, Linear, ReLU> = rs.0;
            let _s2: Layer<4, 3, Linear, Sigmoid> = rs.1;

            // (nothing to assert)

            // note: I also created some helpers for tuple flattening, such as in:
            let (_, _, _) = layerc1!(&dev, 1., [1, 1, 1, 1]).flat3();
            // without it:
            let (_, (_, _)) = layerc1!(&dev, 1., [1, 1, 1, 1]);

            // Note: for this lesson I decided to call "forward" as "downward", and
            // "backward" as "upward".
            //
            // notice the type of the generated layers:
            let (_, (_, (_, (_, (_, _))))) = layerc1!(&dev, 1., [1, 1, 1, 1, 1, 1, 1]);
            // this is similar to being on the top of some stairs, and then you
            // throw the data "downwards" and it goes kicking down the starts.
            // Then you start pulling it "upwards" (imagine you have a rod),
            // then it interacts back with each stair on the way back up.

            // but ofc the dfdx library has a better interface for using models
            // (and for z functions and activation functions), but for this lesson I
            // decided to see where the abstractions that I thought could take me.
        }
    }
    pub(crate) use _2::layerc1;
}
#[allow(unused_imports)]
pub(crate) use _3::layerc1;
pub use _3::{Layer, Linear, ReLU, Sigmoid, Tanh, B, W};

/// C01W04PA01 Part 4 - Forward Propagation Module.
pub mod _4 {
    use super::*;

    /// C01W04PA01 Section 1 - Linear Forward.
    pub mod _1 {
        use super::*;

        pub type X<const FEATLEN: usize, const SETLEN: usize> = TensorF32<Rank2<FEATLEN, SETLEN>>;
        pub type Z<const NODELEN: usize, const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

        pub trait DownwardZ<const FEATLEN: usize, const NODELEN: usize>: Sized {
            fn downward_z<const SETLEN: usize>(self, x: X<FEATLEN, SETLEN>) -> Z<NODELEN, SETLEN>;
        }

        /// Any `Linear` layer can calculate z.
        impl<const FEATLEN: usize, const NODELEN: usize, A> DownwardZ<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            fn downward_z<const SETLEN: usize>(self, x: X<FEATLEN, SETLEN>) -> Z<NODELEN, SETLEN> {
                self.w.dot(x)
                    + self
                        .b
                        .reshape::<Rank1<NODELEN>>()
                        .broadcast::<Rank2<NODELEN, SETLEN>, _>()
            }
        }

        #[test]
        fn test_downward_z() {
            let dev = &device();
            let linear = layerc1!(dev, 1., [3, 1]);
            let x: X<3, 2> = dev.sample_normal();
            let z = linear.downward_z(x);
            assert!(z.array().approx([[-0.15656424, -2.1278346,]], (1e-6, 0)));
        }
    }
    pub use _1::{DownwardZ, X, Z};

    /// C01W04PA01 Section 2 - Linear-Activation Forward.
    pub mod _2 {
        use super::*;
        pub type A<const NODELEN: usize, const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

        pub trait DownwardA<const FEATLEN: usize, const NODELEN: usize>: Sized {
            fn downward_a<const SETLEN: usize>(
                &mut self,
                z: Z<NODELEN, SETLEN>,
            ) -> A<NODELEN, SETLEN>;
        }

        /// Any layer can activate with sigmoid.
        impl<const FEATLEN: usize, const NODELEN: usize, ZF> DownwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, ZF, Sigmoid>
        {
            fn downward_a<const SETLEN: usize>(
                &mut self,
                z: Z<NODELEN, SETLEN>,
            ) -> A<NODELEN, SETLEN> {
                z.sigmoid()
            }
        }

        /// Any layer can activate with ReLU.
        impl<const FEATLEN: usize, const NODELEN: usize, ZF> DownwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, ZF, ReLU>
        {
            fn downward_a<const SETLEN: usize>(
                &mut self,
                z: Z<NODELEN, SETLEN>,
            ) -> A<NODELEN, SETLEN> {
                z.relu()
            }
        }

        /// Any layer can activate with Tanh.
        impl<const FEATLEN: usize, const NODELEN: usize, ZF> DownwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, ZF, Tanh>
        {
            fn downward_a<const SETLEN: usize>(
                &mut self,
                z: Z<NODELEN, SETLEN>,
            ) -> A<NODELEN, SETLEN> {
                z.tanh()
            }
        }

        pub trait DownwardZA<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize>:
            Sized
        {
            type Output: WrapA<NODELEN, SETLEN>;
            fn downward_za(&mut self, x: X<FEATLEN, SETLEN>) -> Self::Output;
        }

        /// Allows `Linear` layers to make the z->a downward calculation.
        impl<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, A>
            DownwardZA<FEATLEN, NODELEN, SETLEN> for Layer<FEATLEN, NODELEN, Linear, A>
        where
            Self: Clone + DownwardA<FEATLEN, NODELEN>,
        {
            type Output = Cache<NODELEN, SETLEN>;
            fn downward_za(&mut self, x: X<FEATLEN, SETLEN>) -> Cache<NODELEN, SETLEN> {
                let z = self.clone().downward_z(x);
                let a = self.downward_a(z.clone());
                Cache { a }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Cache<const NODELEN: usize, const SETLEN: usize> {
            /// Activation function. Rows for nodes, columns for train/test set size.
            pub a: A<NODELEN, SETLEN>,
        }

        // (note: this is useful later on)
        /// Helper trait to access cache info.
        pub trait WrapA<const NODELEN: usize, const SETLEN: usize> {
            fn ref_a(&self) -> &A<NODELEN, SETLEN>;
            fn from_a(a: A<NODELEN, SETLEN>) -> Self;
            fn set_a(&mut self, a: A<NODELEN, SETLEN>);
        }

        impl<const NODELEN: usize, const SETLEN: usize> WrapA<NODELEN, SETLEN> for Cache<NODELEN, SETLEN> {
            fn ref_a(&self) -> &A<NODELEN, SETLEN> {
                &self.a
            }
            fn from_a(a: A<NODELEN, SETLEN>) -> Self {
                Self { a }
            }
            fn set_a(&mut self, a: A<NODELEN, SETLEN>) {
                self.a = a;
            }
        }

        #[test]
        fn test_downward() {
            let dev = &device();
            let x: X<3, 2> = dev.sample_normal();

            let mut sigmoid = layerc1!(dev, 1., 3, Linear => Sigmoid => [1]);
            let cache = sigmoid.downward_za(x.clone());
            assert!(cache.a.array().approx([[0.1271824, 0.44590577]], (1e-7, 0)));

            let mut relu = layerc1!(dev, 1., 3, Linear => ReLU => [1]);
            let cache = relu.downward_za(x);
            assert!(cache.a.array().approx([[0.6293237, 0.0]], (1e-6, 0)));
        }
    }
    pub use _2::{Cache, DownwardA, DownwardZA, WrapA, A};

    /// C01W04PA01 Section 3 - L-Layer Model.
    pub mod _3 {
        use super::*;

        /// Multiple downward_za calls between many adjacent layers.
        pub trait Downward<const SETLEN: usize>: Sized {
            /// The input set for a single layer, or the first input of a stack of layers.
            type Input;
            /// The Cache for a single layer, or stack of caches of multiple layers.
            type Output;

            /// Makes the downward call for a single layer, or make many calls for a stack of layers.
            fn downward<CostType>(
                &mut self,
                x: Self::Input,
                cost_setup: &mut CostType,
            ) -> Self::Output
            where
                CostType: CostSetup;
        }

        /// Downward call for a single layer.
        impl<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A> Downward<SETLEN>
            for Layer<FEATLEN, NODELEN, Z, A>
        where
            Layer<FEATLEN, NODELEN, Z, A>: DownwardZA<FEATLEN, NODELEN, SETLEN>,
        {
            type Input = X<FEATLEN, SETLEN>;
            type Output = (<Self as DownwardZA<FEATLEN, NODELEN, SETLEN>>::Output, ());

            fn downward<CostType>(
                &mut self,
                x: Self::Input,
                cost_setup: &mut CostType,
            ) -> Self::Output
            where
                CostType: CostSetup,
            {
                // make any necessary changes to the cost structure
                cost_setup.downward(self);

                (self.downward_za(x), ())
            }
        }

        /// Downward call for one (or recursively more) pair of layers.
        impl<Lower, const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A>
            Downward<SETLEN> for (Layer<FEATLEN, NODELEN, Z, A>, Lower)
        where
            Layer<FEATLEN, NODELEN, Z, A>: DownwardZA<FEATLEN, NODELEN, SETLEN>,
            Lower: Downward<SETLEN, Input = X<NODELEN, SETLEN>>,
        {
            type Input = X<FEATLEN, SETLEN>;
            type Output = (
                <Layer<FEATLEN, NODELEN, Z, A> as DownwardZA<FEATLEN, NODELEN, SETLEN>>::Output,
                Lower::Output,
            );

            fn downward<CostType>(
                &mut self,
                x: Self::Input,
                cost_setup: &mut CostType,
            ) -> Self::Output
            where
                CostType: CostSetup,
            {
                // make any necessary changes to the cost structure
                cost_setup.downward(&self.0);

                let current_cache = self.0.downward_za(x);
                let x_lower: X<NODELEN, SETLEN> = current_cache.ref_a().clone();
                (
                    current_cache,
                    self.1.downward::<CostType>(x_lower, cost_setup),
                )
            }
        }

        #[test]
        fn test_l_layers_downward() {
            let dev = &device();
            let x: X<5, 4> = dev.sample_normal();
            let mut layers = layerc1!(dev, 1., [5, 4, 3, 1]);
            let caches = layers.downward(x, &mut MLogistical::default());
            assert!(caches
                .last_a()
                .array()
                .approx([[0.5010462, 0.57347196, 0.5333355, 0.5255862,],], (1e-6, 0)));
        }
    }
    pub use _3::Downward;
}
pub use _4::{Cache, Downward, DownwardA, DownwardZ, DownwardZA, WrapA, A, X, Z};

/// C01W04PA01 Part 5 - Cost Function.
pub mod _5 {
    use super::*;
    pub type Y<const NODELEN: usize, const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

    pub trait CostSetup {
        /// M * Cost between the generated prediction and the given expected values.
        fn mcost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank1<NODELEN>>;

        /// Cost between the generated prediction and the given expected values.
        fn cost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank0>;

        /// Intermediate cost calculation during the downward pass.  
        /// That calculation can make use of w and b.
        ///
        /// By default, there is no intermediate calculation.
        fn downward<const NODELEN: usize, const FEATLEN: usize, Z, A>(
            &mut self,
            _layer: &Layer<FEATLEN, NODELEN, Z, A>,
        ) {
        }

        /// Optional direct additive term for ∂J/∂w and ∂J/∂b.
        ///
        /// Ie. For when not only each (w, b) affects J by going Z->A, but if they also additively and directly affects J,
        /// then their gradients have some additive calculation.
        ///
        /// By default, the J has no additive direct effect from w and b during the downward pass
        /// and so there is no additive term for the upward pass.
        fn direct_upward_dwdb<
            const NODELEN: usize,
            const FEATLEN: usize,
            const SETLEN: usize,
            Z,
            A,
        >(
            &self,
            _layer: &Layer<FEATLEN, NODELEN, Z, A>,
        ) -> Option<Wb<NODELEN, FEATLEN>> {
            None
        }

        fn update_params<const NODELEN: usize, const FEATLEN: usize, Z, A>(
            &self,
            layer: Layer<FEATLEN, NODELEN, Z, A>,
            gradient: Grads<NODELEN, FEATLEN>,
        ) -> Layer<FEATLEN, NODELEN, Z, A>;

        fn refresh_cost(&mut self) {}
    }

    /// Logistical cost function.
    ///
    /// cost function m*J(L) = sum (L)
    /// loss function L(ŷ,y) = -(y*log(ŷ) + (1-y)log(1-ŷ))
    ///
    /// Note that the cost function is multiplied by `m` (SETLEN).
    #[derive(Clone, Debug)]
    pub struct MLogistical {
        pub learning_rate: f32,
    }

    impl Default for MLogistical {
        fn default() -> Self {
            Self { learning_rate: 1. }
        }
    }

    impl MLogistical {
        pub fn new(learning_rate: f32) -> Self {
            Self { learning_rate }
        }
    }

    #[allow(clippy::let_and_return)]
    /// Logistical cost.
    impl CostSetup for MLogistical {
        /// M * Cost function m * J = sum (L).
        fn mcost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank1<NODELEN>> {
            // loss function L = -(y*log(ŷ) + (1-y)log(1-ŷ))
            let l1 = expect
                .clone()
                .dot((non_zero(predict.clone()).ln()).permute::<_, Axes2<1, 0>>());
            let l2 = expect.clone().negate() + 1.;
            let l3 = non_zero(predict.negate() + 1.).ln();
            let l = (l1 + l2.dot(l3.permute::<_, Axes2<1, 0>>())).negate();

            // M * cost function m * J = sum (L)
            l.sum::<Rank1<NODELEN>, Axis<1>>()
        }

        /// Cost function J = sum (L) / m.
        fn cost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank0> {
            let l = self.mcost(expect, predict);
            // cost function J = sum (L) / m
            l.sum::<Rank0, _>() / (SETLEN as f32)
        }

        fn update_params<const NODELEN: usize, const FEATLEN: usize, Z, A>(
            &self,
            mut layer: Layer<FEATLEN, NODELEN, Z, A>,
            gradient: Grads<NODELEN, FEATLEN>,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            layer.w = layer.w - gradient.dw * self.learning_rate;
            layer.b = layer.b - (gradient.db * self.learning_rate).reshape::<Rank2<NODELEN, 1>>();
            layer
        }
    }

    /// Squared cost function.
    ///
    /// cost function m*J(L) = sum (L)
    /// loss function L(ŷ,y) = (ŷ-y)²
    ///
    /// Note that the cost function is multiplied by `m` (SETLEN).
    #[derive(Clone, Debug)]
    pub struct MSquared {
        pub learning_rate: f32,
    }

    impl Default for MSquared {
        fn default() -> Self {
            Self { learning_rate: 1. }
        }
    }

    /// Squared cost.
    impl CostSetup for MSquared {
        /// M * Cost function m * J = sum (L).
        fn mcost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank1<NODELEN>> {
            // loss function L = (ŷ-y)²
            let l = (predict - expect).square();

            // M * cost function m * J = sum (L)
            l.sum::<Rank1<NODELEN>, Axis<1>>()
        }

        /// Cost function J = sum (L) / m.
        fn cost<const NODELEN: usize, const SETLEN: usize>(
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank0> {
            let l = self.mcost(expect, predict);

            // cost function J = sum (L) / m
            l.sum::<Rank0, _>() / (SETLEN as f32)
        }

        fn update_params<const NODELEN: usize, const FEATLEN: usize, Z, A>(
            &self,
            mut layer: Layer<FEATLEN, NODELEN, Z, A>,
            gradient: Grads<NODELEN, FEATLEN>,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            layer.w = layer.w - gradient.dw * self.learning_rate;
            layer.b = layer.b - (gradient.db * self.learning_rate).reshape::<Rank2<NODELEN, 1>>();
            layer
        }
    }

    #[test]
    fn test_cost() {
        let dev = &device();
        let y = dev.tensor([[1., 1., 0.]]);
        let cache = Cache {
            a: dev.tensor([[0.8, 0.9, 0.4]]),
        };
        let mut cost_setup = MLogistical::default();
        assert_eq!((cost_setup.cost(y, cache.a)).array(), 0.27977654);
    }
}
pub use _5::{CostSetup, MLogistical, MSquared, Y};

/// C01W04PA01 Part 6 - Backward Propagation Module.
pub mod _6 {
    use super::*;

    /// C01W04PA01 Section 1 - Linear Backward.
    pub mod _1 {
        use super::*;

        /// ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
        pub trait UpwardDownZUpA<const FEATLEN: usize>: Sized {
            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            type Output<const SETLEN: usize>;

            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            fn upward_dzdx<const SETLEN: usize>(
                self,
                up_cache: Cache<FEATLEN, SETLEN>,
            ) -> Self::Output<SETLEN>;
        }

        /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
        pub type Dzdx<const NODELEN: usize, const FEATLEN: usize> =
            TensorF32<Rank2<NODELEN, FEATLEN>>;

        /// Any lower layer with a Linear z function can calculate dzda[up] = ∂z[down]/∂a[up].
        impl<const FEATLEN: usize, const NODELEN: usize, A> UpwardDownZUpA<FEATLEN>
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            type Output<const SETLEN: usize> = Dzdx<NODELEN, FEATLEN>;
            // note: SETLEN is not used but it could be for other formulas (other derivates).

            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up] = w[down].
            fn upward_dzdx<const SETLEN: usize>(
                self,
                _up_cache: Cache<FEATLEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                self.w
            }
        }

        /// mda[up] = m * ∂J/∂a[up] = ∂z[down]/∂a[up] * ∂J/∂z[down] = dzdx * mdz[down].
        pub type Mda<const FEATLEN: usize, const SETLEN: usize> = TensorF32<Rank2<FEATLEN, SETLEN>>;

        /// mda[up] = m * ∂J/∂a[up] = ∂z[down]/∂a[up] * ∂J/∂z[down] = dzdx * mdz[down].
        pub fn upward_up_mda<
            const DOWN_NODELEN: usize,
            const UP_NODELEN: usize,
            const SETLEN: usize,
        >(
            // cache: Cache<NODELEN, SETLEN>,
            dzda: Dzdx<DOWN_NODELEN, UP_NODELEN>,
            down_mdz: Mdz<DOWN_NODELEN, SETLEN>,
        ) -> Mda<UP_NODELEN, SETLEN> {
            dzda.permute::<_, Axes2<1, 0>>().dot(down_mdz)
        }

        /// m * ∂z/∂w, m * ∂z/∂b.
        pub trait UpwardZwb<const FEATLEN: usize, const NODELEN: usize>: Sized {
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            type Output;
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            fn upward_dw_db<const SETLEN: usize>(
                self,
                upper_cache: Cache<FEATLEN, SETLEN>,
                mdz: Mdz<NODELEN, SETLEN>,
            ) -> Self::Output;
        }

        /// Any `Linear` layer can calculate dw = ∂z/∂w, db = ∂z/∂b.
        impl<const FEATLEN: usize, const NODELEN: usize, A> UpwardZwb<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            type Output = (W<NODELEN, FEATLEN>, TensorF32<Rank1<NODELEN>>);

            /// ∂z/∂w = upper_cache.a  
            /// dw = ∂J/∂w = ∂z/∂w * ∂J/∂z = upper_cache.a * mdz / m.
            ///
            /// ∂z/∂b = 1  
            /// db = ∂J/∂b = ∂z/∂b * ∂J/∂z = mdz / m.
            fn upward_dw_db<const SETLEN: usize>(
                self,
                upper_cache: Cache<FEATLEN, SETLEN>,
                mdz: Mdz<NODELEN, SETLEN>,
            ) -> Self::Output {
                let dw =
                    mdz.clone().dot(upper_cache.a.permute::<_, Axes2<1, 0>>()) / (SETLEN as f32);
                let db = mdz.sum::<Rank1<NODELEN>, _>() / (SETLEN as f32);
                (dw, db)
            }
        }

        #[test]
        fn test_upward_dw_db_and_mdaup() {
            let dev = &device();

            let mdz: Mdz<3, 4> = dev.sample_normal();
            let up_a: A<5, 4> = dev.sample_normal();
            let w: W<3, 5> = dev.sample_normal();
            let b: B<3> = dev.sample_normal();

            let up_cache = Cache { a: up_a };

            // note: only the 'Linear' part of the layer matters here
            let l: Layer<5, 3, Linear, Sigmoid> = Layer::with_wb(w, b);
            let (dw, db) = l.clone().upward_dw_db(up_cache.clone(), mdz.clone());
            let dzda = l.upward_dzdx(up_cache);
            let up_mda = upward_up_mda(dzda, mdz);

            assert!(dw.array().approx(
                [
                    [0.5509606, 0.7776225, 0.5967806, -0.8766387, 0.9333939],
                    [1.0130866, 0.8946871, 0.7919213, -0.49577445, 2.1065006],
                    [0.52858984, 0.558856, 0.15449898, -0.5000571, -1.5426211]
                ],
                (1e-6, 0)
            ));
            assert!(db
                .array()
                .approx([-0.17543876, -0.21416298, 0.26838428], (1e-7, 0)));
            assert!(up_mda.array().approx(
                [
                    [0.6519548, 1.6380557, -3.6384397, 0.21903485],
                    [-0.42141977, 2.2432818, -1.5973558, -0.7400213],
                    [-1.2011051, 0.9337286, 3.688091, -2.1032374],
                    [-0.32980642, -5.403079, 2.1622324, 3.678501],
                    [1.9665232, -2.6421018, 1.3423123, 0.16766931]
                ],
                (1e-6, 0)
            ));
        }
    }
    pub use _1::{upward_up_mda, Dzdx, Mda, UpwardDownZUpA, UpwardZwb};

    /// C01W04PA01 Section 2 - Linear-Activation Backward.
    pub mod _2 {
        use super::*;

        /// mdz = m * ∂J/∂z.
        pub type Mdz<const NODELEN: usize, const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

        /// m * ∂a/∂z.
        pub trait UpwardAZ<const NODELEN: usize>: Sized {
            /// mdz = m * ∂J/∂z.
            type Output<const SETLEN: usize>;
            /// mdz = m * ∂J/∂z.
            fn upward_mdz<const SETLEN: usize>(
                &mut self,
                cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN>;
        }

        /// Any layer with a Sigmoid activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> UpwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Sigmoid>
        {
            /// mdz = cache.a * (1 - cache.a) * mda.
            type Output<const SETLEN: usize> = Mdz<NODELEN, SETLEN>;

            /// ∂a/∂z = σ'(z) = e^-z / (1 + e^-z)² = σ(z) * (1 - σ(z)) = cache.a * (1 - cache.a).
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = cache.a * (1 - cache.a) * mda
            fn upward_mdz<const SETLEN: usize>(
                &mut self,
                cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                let _a_neg = cache.a.clone().negate();
                cache.a * (_a_neg + 1.) * mda
            }
        }

        /// Any layer with a Tanh activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> UpwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Tanh>
        {
            /// mdz = (1 - cache.a²) * mda.
            type Output<const SETLEN: usize> = Mdz<NODELEN, SETLEN>;

            /// ∂a/∂z = tanh'(z) = 1 - z² = 1 - cache.a².
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = (1 - cache.a²) * mda.
            fn upward_mdz<const SETLEN: usize>(
                &mut self,
                cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                (cache.a.square().negate() + 1.) * mda
            }
        }

        /// Any layer with a ReLU activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> UpwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, ReLU>
        {
            /// mdz.
            type Output<const SETLEN: usize> = Mdz<NODELEN, SETLEN>;

            /// ∂a/∂z = relu'(z) = {1 if z > 0; 0 if z <= 0} = {1 if cache.a > 0; 0 if cache.a <= 0}
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = {1 if cache.a > 0; 0 if cache.a <= 0} * mda
            fn upward_mdz<const SETLEN: usize>(
                &mut self,
                cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                let dev = cache.a.device();
                let mask = cache.a.gt(0.);
                let zeros: Mdz<NODELEN, SETLEN> = dev.zeros();
                mask.choose(mda, zeros)
            }
        }

        #[test]
        fn test_upward_mdz() {
            let dev = &device();

            let mda: Mda<1, 2> = dev.sample_normal();
            let w: W<1, 3> = dev.sample_normal();
            let b: B<1> = dev.sample_normal();
            let up_cache = Cache {
                a: dev.sample_normal(),
            };
            let z: Z<1, 2> = dev.sample_normal();

            // sigmoid layer test
            {
                // note that we should already have calculated σ(z) and stored it in cache.a during the downward pass:
                let cache = Cache {
                    a: z.clone().sigmoid(),
                };
                let l: Layer<3, 1, Linear, Sigmoid> = Layer::with_wb(w.clone(), b.clone());
                let mdz = l.clone().upward_mdz(cache.clone(), mda.clone());
                let (dw, db) = l.clone().upward_dw_db(up_cache.clone(), mdz.clone());
                let dzda = l.upward_dzdx(up_cache.clone());
                let up_mda = upward_up_mda(dzda, mdz);

                assert_eq!(
                    up_mda.array(),
                    [
                        [-0.3602831, -0.2700555,],
                        [0.024156112, 0.01810657,],
                        [-0.18855447, -0.14133377,],
                    ]
                );
                assert_eq!(dw.array(), [[-0.075585924, 0.14205638, -0.042644225,],]);
                assert_eq!(db.array(), [0.1293669]);
            }

            // relu layer test
            {
                // note: this layer is not "above" nor "below" the sigmoid one,
                // they are isolated tests but just using the same data

                // note that we should already have calculated relu(z) and stored it in cache.a during the downward pass:
                let cache = Cache {
                    a: z.clone().relu(),
                };
                let l: Layer<3, 1, Linear, ReLU> = Layer::with_wb(w.clone(), b.clone());
                let mdz = l.clone().upward_mdz(cache.clone(), mda.clone());
                let (dw, db) = l.clone().upward_dw_db(up_cache.clone(), mdz.clone());
                let dzda = l.upward_dzdx(up_cache.clone());
                let up_mda = upward_up_mda(dzda, mdz);

                assert_eq!(
                    up_mda.array(),
                    [[-0.0, -2.0911047,], [0.0, 0.14020352,], [-0.0, -1.0943813,],]
                );
                assert_eq!(dw.array(), [[0.2629047, 0.7462477, -1.2105147,],]);
                assert_eq!(db.array(), [0.42916572]);
            }
        }
    }
    pub use _2::{Mdz, UpwardAZ};

    /// C01W04PA01 Section 3 - L-Model Backward.
    pub mod _3 {
        use super::*;

        /// m * ∂J/∂a.
        pub trait UpwardJA<const NODELEN: usize, const SETLEN: usize>: Sized {
            /// mda = m * ∂J/∂a.
            type Output;

            /// mda = m * ∂J/∂a.
            fn upward_mda(
                &self,
                expect: A<NODELEN, SETLEN>,
                predict: A<NODELEN, SETLEN>,
            ) -> Self::Output;
        }

        #[allow(clippy::let_and_return)]
        /// Avoids the values of being zero.
        pub fn non_zero<const ROWS: usize, const COLS: usize>(
            values: TensorF32<Rank2<ROWS, COLS>>,
        ) -> TensorF32<Rank2<ROWS, COLS>> {
            let close_to_zero = values.device().tensor([[1e-38f32; COLS]; ROWS]);

            let select_positive = values.ge(0.);
            let positive_values = values.clone().maximum(close_to_zero.clone());
            let values = select_positive.choose(positive_values, values);

            let select_negative = values.le(0.);
            let negative_values = values.clone().minimum(close_to_zero.negate());
            let values = select_negative.choose(negative_values, values);

            values
        }

        //
        // Note: notice that we have lost a symbolic optimization (and precision).
        //
        // As shown on impl c1::w3::pa_01_planar_data_classification::_4::_3::backward, if we needed to calculate
        // m * ∂J/∂z = ∂a/∂z * ∂J/∂a, we could simplify the formula from a(1-a) * (a-y)/(a(1-a)) to just (a-y).
        // In that case we would be merging two upward (inner) functions, for mda and mdz (skipping mda).
        // But that is only for logistical regression, and for a sigmoid activation as the lowest activation.
        impl<const NODELEN: usize, const SETLEN: usize> UpwardJA<NODELEN, SETLEN> for MLogistical {
            /// mda = m * ∂J/∂a.
            type Output = Mda<NODELEN, SETLEN>;

            /// mda = m * ∂J/∂a = (1-y)/(1-a) - y/a = (a-y)/(a(1-a)).
            fn upward_mda(
                &self,
                expect: A<NODELEN, SETLEN>,
                predict: A<NODELEN, SETLEN>,
            ) -> Self::Output {
                let _a_neg = predict.clone().negate();
                let _a = predict.clone();
                (predict - expect) / non_zero(_a * (_a_neg + 1.))
            }
        }

        #[test]
        fn test_upward_j_fixed() {
            // note: this just tests 2-fixed layers, not arbitrary amount of layers.
            //
            let dev = &device();
            let x: X<4, 2> = dev.sample_normal();
            let y: Y<1, 2> = dev.sample_normal();
            let layers = layerc1!(dev, 1., [4, 3, 1]);
            let mut cost_setup = MLogistical::default();
            let (cache_up, (cache_down, ())) = layers.clone().downward(x.clone(), &mut cost_setup);
            let (layer_up, layer_down) = layers;
            let yhat = cache_down.a.clone();

            // layer_down sigmoid (from loss)
            let mda_down = cost_setup.upward_mda(y, yhat);
            assert!(mda_down
                .array()
                .approx([[-2.504804, -5.1476555,],], (1e-5, 0)));

            // sigmoid layer (layer_down)
            #[allow(clippy::let_and_return)]
            let up_mda = {
                let mdz = layer_down
                    .clone()
                    .upward_mdz(cache_down.clone(), mda_down.clone());
                let (_dw2, _db2) = layer_down
                    .clone()
                    .upward_dw_db(cache_up.clone(), mdz.clone());
                let dzda = layer_down.clone().upward_dzdx(cache_up.clone());
                let up_mda = upward_up_mda(dzda, mdz.clone());
                up_mda
            };

            // relu layer (layer_up)
            let up_mdz = layer_up
                .clone()
                .upward_mdz(cache_up.clone(), up_mda.clone());
            let cache0 = Cache { a: x };
            let (dw1, db1) = layer_up.clone().upward_dw_db(cache0, up_mdz.clone());

            assert!(dw1.array().approx(
                [
                    [0.285591, -0.1627307, 0.23043698, 0.03585002,],
                    [-0.0011592763, -0.00022061542, -0.001738474, -0.0008273805,],
                    [0.0, 0.0, 0.0, 0.0,],
                ],
                (1e-7, 0)
            ));
            assert!(db1
                .array()
                .approx([0.3470378, -0.0013506162, 0.0,], (1e-7, 0)));
        }

        pub type Wb<const NODELEN: usize, const FEATLEN: usize> =
            (W<NODELEN, FEATLEN>, TensorF32<Rank1<NODELEN>>);

        /// Helper to reason about splitting a structure between it's Head and Tail, most notably
        /// a pair in a tuple.
        pub trait SplitHead {
            type Head;
            type Headless;
            fn split_head(self) -> (Self::Head, Self::Headless);
        }

        impl<A, B> SplitHead for (A, B) {
            type Head = A;
            type Headless = B;
            /// Splits a tuple in a pair.
            fn split_head(self) -> Self {
                self
            }
        }

        pub trait LastA {
            type Last;
            fn last_a(&self) -> &Self::Last;
        }

        impl<A, B> LastA for (A, B)
        where
            B: LastA,
        {
            type Last = <B as LastA>::Last;
            fn last_a(&self) -> &Self::Last {
                self.1.last_a()
            }
        }

        impl<const NODELEN: usize, const SETLEN: usize> LastA for (Cache<NODELEN, SETLEN>, ()) {
            type Last = A<NODELEN, SETLEN>;
            fn last_a(&self) -> &Self::Last {
                self.0.ref_a()
            }
        }

        pub struct Grads<const NODELEN: usize, const FEATLEN: usize> {
            pub dw: W<NODELEN, FEATLEN>,
            pub db: TensorF32<Rank1<NODELEN>>,
        }

        /// Multiple down-upward calls between many adjacent layers.
        pub trait DownUpGrads<const LOWEST_NODELEN: usize, const SETLEN: usize>: Sized {
            /// The Caches for lower layers.
            type LowerCaches = ();
            /// The input for this layer.  
            /// Is "x" for the top-most layer, and is the output of the layer above for each lower layer
            /// (ie. the NODELEN of each layer is the input "x" for the layer below it).
            type X;
            /// The cache for the current layer.
            type Cache;
            /// The generated gradient (dw, db) plus the mda (of the upper layer), follolwed by the output of layers from down below.
            type Output;
            /// NODELEN for the current layer.
            const NODELEN: usize;

            /// Makes the initial downward call (for access) and then pulls back up, getting the gradients.
            fn gradients<CostType>(
                &mut self,
                expect: A<LOWEST_NODELEN, SETLEN>,
                cost_setup: &mut CostType,
                caches: (Self::X, (Self::Cache, Self::LowerCaches)),
            ) -> Self::Output
            where
                CostType: CostSetup
                    + UpwardJA<LOWEST_NODELEN, SETLEN, Output = Mda<LOWEST_NODELEN, SETLEN>>
                    + Clone;
        }

        /// Down-Up call for the lowest single layer.
        impl<const NODELEN: usize, const FEATLEN: usize, const SETLEN: usize, Z, Act>
            DownUpGrads<NODELEN, SETLEN> for Layer<FEATLEN, NODELEN, Z, Act>
        where
            Layer<FEATLEN, NODELEN, Z, Act>: DownwardZA<FEATLEN, NODELEN, SETLEN>,
            Self: UpwardAZ<NODELEN, Output<SETLEN> = Mdz<NODELEN, SETLEN>>
                + UpwardZwb<FEATLEN, NODELEN, Output = Wb<NODELEN, FEATLEN>>
                + UpwardDownZUpA<FEATLEN, Output<SETLEN> = Mda<NODELEN, FEATLEN>>,
            Act: Clone,
            Z: Clone,
        {
            /// NODELEN for the lowest layer. That is the length of ŷ.
            const NODELEN: usize = NODELEN;
            /// Input for the lowest layer.
            type X = Cache<FEATLEN, SETLEN>;
            /// Cache for the lowest layer.
            type Cache = Cache<NODELEN, SETLEN>;
            /// There are no caches "below" because this already is the lowest layer.
            type LowerCaches = ();
            /// Returns the gradient (dw, db), plus the mda of the upper layer, followed
            /// by an "inexistent" cache.
            type Output = ((Grads<NODELEN, FEATLEN>, Mda<FEATLEN, SETLEN>), ());

            /// Compares the prediction with the expected result, then calculates the gradients (dw, db) for the lowest layer,
            /// and also the mda for the layer above it.
            fn gradients<CostType>(
                &mut self,
                expect: A<NODELEN, SETLEN>,
                cost_setup: &mut CostType,
                caches: (Self::X, (Self::Cache, Self::LowerCaches)),
            ) -> Self::Output
            where
                CostType:
                    CostSetup + UpwardJA<NODELEN, SETLEN, Output = Mda<NODELEN, SETLEN>> + Clone,
            {
                let (x, (cache, ())) = caches;

                // last cost downward intermediate calculation.
                cost_setup.downward(self);

                // calculates m * ∂J/∂a for the current (lowest) layer
                // this calculation is done directly from the derivative of the cost function
                let mda = cost_setup.upward_mda(expect.clone(), cache.clone().a);

                // get m * ∂J/∂z for the current (lowest) layer
                let mdz = self.clone().upward_mdz(cache.clone(), mda);

                // calculates the gradients (∂J/∂w, ∂J/∂b) for the current (lowest) layer
                let (mut dw, mut db) = self.clone().upward_dw_db(x.clone(), mdz.clone());

                // optional additive dw, db term
                if let Some((dw2, db2)) =
                    cost_setup.direct_upward_dwdb::<NODELEN, FEATLEN, SETLEN, _, _>(self)
                {
                    dw = dw + dw2;
                    db = db + db2;
                }

                // calculates the m * ∂J/∂a for the layer above, so the layer above can also have it's m * ∂J/∂a
                let dzda = self.clone().upward_dzdx(x.clone());
                let up_mda = upward_up_mda(dzda, mdz);
                ((Grads { dw, db }, up_mda), ())
            }
        }

        /// Down-Up call for one (or recursively more) pair of layers.
        impl<
                Lower,
                const LOWEST_NODELEN: usize,
                const FEATLEN: usize,
                const NODELEN: usize,
                const SETLEN: usize,
                Z,
                Act,
            > DownUpGrads<LOWEST_NODELEN, SETLEN> for (Layer<FEATLEN, NODELEN, Z, Act>, Lower)
        where
            Layer<FEATLEN, NODELEN, Z, Act>: Clone
                + UpwardAZ<NODELEN, Output<SETLEN> = Mdz<NODELEN, SETLEN>>
                + UpwardZwb<FEATLEN, NODELEN, Output = Wb<NODELEN, FEATLEN>>
                + UpwardDownZUpA<FEATLEN, Output<SETLEN> = Mda<NODELEN, FEATLEN>>,
            Lower: DownUpGrads<LOWEST_NODELEN, SETLEN, X = Cache<NODELEN, SETLEN>>,
            <Lower as DownUpGrads<LOWEST_NODELEN, SETLEN>>::Output:
                SplitHead<Head = (Grads<{ Lower::NODELEN }, NODELEN>, Mda<NODELEN, SETLEN>)>,
        {
            /// NODELEN for the current top layer.
            const NODELEN: usize = NODELEN;
            /// The input for the current top layer.
            type X = Cache<FEATLEN, SETLEN>;
            /// The cache for the current top layer.
            type Cache = Cache<NODELEN, SETLEN>;
            /// The representation of the caches for the lower layers.
            type LowerCaches = (Lower::Cache, Lower::LowerCaches);
            /// The output, which contains the gradients (dw, db) for the current top layer,
            /// plus the mda for the layer above it (which would be for the inputs in the case of the first layer),
            /// plus the outputs of the lower layers.
            type Output = (
                (Grads<NODELEN, FEATLEN>, Mda<FEATLEN, SETLEN>),
                (
                    (Grads<{ Lower::NODELEN }, NODELEN>, Mda<NODELEN, SETLEN>),
                    <<Lower as DownUpGrads<LOWEST_NODELEN, SETLEN>>::Output as SplitHead>::Headless,
                ),
            );

            /// Calls layers below so that the initial m * ∂J/∂a gets calculated, then starts producing gradients (dw, db)
            /// while propagating back up.
            fn gradients<CostType>(
                &mut self,
                expect: A<LOWEST_NODELEN, SETLEN>,
                cost_setup: &mut CostType,
                caches: (Self::X, (Self::Cache, Self::LowerCaches)),
            ) -> Self::Output
            where
                CostType: CostSetup
                    + UpwardJA<LOWEST_NODELEN, SETLEN, Output = Mda<LOWEST_NODELEN, SETLEN>>
                    + Clone,
            {
                // current layer
                let layer = &mut self.0;

                let (x, (cache, lower_caches)) = caches;
                let (lower_cache_head, lower_cache_tail) = lower_caches.split_head();

                // cost downward intermediate calculation.
                cost_setup.downward(layer);

                // recursive call to lower lanes
                let lower_outputs = self.1.gradients(
                    expect,
                    cost_setup,
                    (cache.clone(), (lower_cache_head, lower_cache_tail)),
                );

                // access information from the layer from below
                let (lower_head, lower_tail) = lower_outputs.split_head();

                // get m * ∂J/∂a for the current layer
                let (lower_grads, mda) = lower_head.split_head();

                // get m * ∂J/∂z for the current layer
                let mdz = layer.clone().upward_mdz(cache.clone(), mda.clone());

                // calculates the gradients (∂J/∂w, ∂J/∂b) for the current layer
                let (mut dw, mut db) = layer.clone().upward_dw_db(x.clone(), mdz.clone());

                // optional additive dw, db term
                if let Some((dw2, db2)) =
                    cost_setup.direct_upward_dwdb::<NODELEN, FEATLEN, SETLEN, _, _>(layer)
                {
                    dw = dw + dw2;
                    db = db + db2;
                }

                // calculates the m * ∂J/∂a for the layer above, so the layer above can also have it's m * ∂J/∂a.
                let dzda = layer.clone().upward_dzdx(x.clone());
                let up_mda = upward_up_mda(dzda, mdz);
                ((Grads { dw, db }, up_mda), ((lower_grads, mda), lower_tail))
            }
        }

        pub trait CleanupGrads {
            type Output;
            /// Removes the mda information.
            fn remove_mdas(self) -> Self::Output;
        }

        impl CleanupGrads for () {
            type Output = ();
            /// Removes the mda information.
            fn remove_mdas(self) -> Self::Output {}
        }

        impl<const NODELEN: usize, const FEATLEN: usize, const SETLEN: usize, B> CleanupGrads
            for ((Grads<NODELEN, FEATLEN>, Mda<FEATLEN, SETLEN>), B)
        where
            B: CleanupGrads,
        {
            type Output = (Grads<NODELEN, FEATLEN>, B::Output);
            /// Removes the mda information.
            fn remove_mdas(self) -> Self::Output {
                let (grads, _mda) = self.0;
                let tail = self.1;
                (grads, tail.remove_mdas())
            }
        }

        // same values as the previous test
        #[test]
        fn test_upward_j_variable() {
            let dev = &device();
            let x: X<4, 2> = dev.sample_normal();
            let y: Y<1, 2> = dev.sample_normal();
            let mut layers = layerc1!(dev, 1., [4, 3, 1]);
            let mut cost_setup = MLogistical::default();
            let caches = layers.clone().downward(x.clone(), &mut cost_setup);
            let grads = layers
                .gradients(y, &mut cost_setup, (Cache::from_a(x), caches))
                .remove_mdas()
                .flat3();

            // same values as the previous test
            assert!(grads.0.dw.array().approx(
                [
                    [0.285591, -0.1627307, 0.23043698, 0.03585002,],
                    [-0.0011592763, -0.00022061542, -0.001738474, -0.0008273805,],
                    [0.0, 0.0, 0.0, 0.0,],
                ],
                (1e-7, 0)
            ));
            assert!(grads
                .0
                .db
                .array()
                .approx([0.3470378, -0.0013506162, 0.0,], (1e-7, 0)));
        }
    }
    pub use _3::{non_zero, CleanupGrads, DownUpGrads, Grads, LastA, SplitHead, UpwardJA, Wb};

    /// C01W04PA01 Section 4 - Update Parameters.
    pub mod _4 {
        use super::*;

        /// Recursively update the parameters.
        pub trait UpdateParameters: Sized {
            type Grads;
            fn update_params<CostType>(self, grads: Self::Grads, cost_setup: &CostType) -> Self
            where
                CostType: CostSetup;
        }

        impl<const NODELEN: usize, const FEATLEN: usize, Z, A> UpdateParameters
            for Layer<FEATLEN, NODELEN, Z, A>
        {
            type Grads = (Grads<NODELEN, FEATLEN>, ());
            fn update_params<CostType>(self, grads: Self::Grads, cost_setup: &CostType) -> Self
            where
                CostType: CostSetup,
            {
                cost_setup.update_params(self, grads.0)
            }
        }

        impl<const NODELEN: usize, const FEATLEN: usize, Z, A, Lower> UpdateParameters
            for (Layer<FEATLEN, NODELEN, Z, A>, Lower)
        where
            Lower: UpdateParameters,
        {
            type Grads = (Grads<NODELEN, FEATLEN>, Lower::Grads);
            fn update_params<CostType>(
                mut self,
                (grads, lower_grads): Self::Grads,
                cost_setup: &CostType,
            ) -> Self
            where
                CostType: CostSetup,
            {
                self.0 = cost_setup.update_params(self.0, grads);
                self.1 = self.1.update_params(lower_grads, cost_setup);
                self
            }
        }

        #[test]
        fn test_upward_update_params() {
            let dev = &device();
            let x: X<4, 2> = dev.sample_normal();
            let y: Y<1, 2> = dev.sample_normal();
            let layers = layerc1!(dev, 1., [4, 3, 1]);
            let mut cost_setup = MLogistical::new(0.1);
            let caches = layers.clone().downward(x.clone(), &mut cost_setup);
            let grads = layers
                .clone()
                .gradients(y, &mut cost_setup, (Cache::from_a(x), caches));
            let layers = layers.update_params(grads.remove_mdas(), &cost_setup);

            assert!(layers.0.w.array().approx(
                [
                    [0.7401993, -1.3940384, -0.46163252, 0.8545286,],
                    [-0.28469577, -0.09709492, -0.053475242, 0.9694384,],
                    [-0.31121865, 0.019920077, -0.24145567, 0.5810155,],
                ],
                (1e-7, 0)
            ));
            assert!(layers
                .1
                .w
                .array()
                .approx([[-0.31857067, 0.019019742, 0.23030037,],], (1e-7, 0)));
        }
    }
    pub use _4::UpdateParameters;
}
pub use _6::{
    non_zero, CleanupGrads, DownUpGrads, Dzdx, Grads, LastA, Mda, Mdz, SplitHead, UpdateParameters,
    UpwardAZ, UpwardDownZUpA, UpwardJA, UpwardZwb, Wb,
};

/// C01W04PA01 Part 7 - Conclusion.
mod _7 {
    // no content
}

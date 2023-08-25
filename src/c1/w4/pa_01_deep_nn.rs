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
        use std::marker::PhantomData;

        #[derive(Clone, Debug)]
        pub struct Layer<const FEATLEN: usize, const NODELEN: usize, Z = Linear, A = ReLU> {
            /// Rows for nodes, columns for features.
            pub w: TensorF32<Rank2<NODELEN, FEATLEN>>,

            /// Rows for nodes.
            // for the bias, the implied feature is a single constant 1.0 value.
            pub b: TensorF32<Rank2<NODELEN, 1>>,

            /// Calculation function z(features, w, b).
            pub _z: PhantomData<Z>,

            /// Activation function a(z).
            pub _a: PhantomData<A>,
        }

        impl<const FEATLEN: usize, const NODELEN: usize, Z, A> Layer<FEATLEN, NODELEN, Z, A> {
            pub fn normal(device: &Device) -> Self {
                let w: TensorF32<Rank2<NODELEN, FEATLEN>> = device.sample_normal();
                let b: TensorF32<Rank2<NODELEN, 1>> = device.zeros();
                Self::with(w, b)
            }
            pub fn with(
                w: TensorF32<Rank2<NODELEN, FEATLEN>>,
                b: TensorF32<Rank2<NODELEN, 1>>,
            ) -> Self {
                Self {
                    w,
                    b,
                    _z: PhantomData,
                    _a: PhantomData,
                }
            }
            pub fn from_values(
                device: &Device,
                w: [[f32; FEATLEN]; NODELEN],
                b: [[f32; 1]; NODELEN],
            ) -> Self {
                let w: TensorF32<Rank2<NODELEN, FEATLEN>> = device.tensor(w);
                let b: TensorF32<Rank2<NODELEN, 1>> = device.tensor(b);
                Self::with(w, b)
            }
        }

        /// Weight and Bias calculation.
        ///
        /// `z(w, features, b) = w * features + b`.
        #[derive(Clone, Debug, PartialEq)]
        pub struct Linear;

        /// Activation calculation.
        ///
        /// `a(z) = 0 if z < 0`.
        /// `a(z) = 1 if z >= 0`.
        #[derive(Clone, Debug, PartialEq)]
        pub struct ReLU;

        /// Activation calculation.
        ///
        /// `a(z) = σ(z) = 1 / (1 + e^-z)`
        #[derive(Clone, Debug, PartialEq)]
        pub struct Sigmoid;

        /// Activation calculation.
        ///
        /// `a(z) = tanh(z) = (e^z - e^-z) / (e^z + e^-z)`
        #[derive(Clone, Debug, PartialEq)]
        pub struct Tanh;

        #[test]
        fn example() {
            let dev = device();
            let _model = (
                Layer::<3, 2, Linear, ReLU>::normal(&dev),
                Layer::<2, 1, Linear, Sigmoid>::normal(&dev),
            );

            // (nothing to assert)
        }
    }
    pub use _1::{Layer, Linear, ReLU, Sigmoid, Tanh};

    /// C01W04PA01 Section 2 - L-Layer Neural Network.
    pub mod _2 {
        use super::*;

        /// Creates one or more layers from a normal distribution.
        ///
        /// Example with implicit `Linear`>`Relu` hidden layers and with a final `Linear`>`Sigmoid` final layer
        /// ```rust
        /// # use coursera_exercises::layer;
        /// # use coursera_exercises::c1::w4::pa_01_deep_nn::{Linear, ReLU, Sigmoid, Layer};
        /// # use coursera_exercises::helpers::_dfdx::*;
        /// # let dev = device();
        /// let s = layer!(&dev, [1, 1]);
        /// let rs = layer!(&dev, [1, 1, 1]);
        /// let rrs = layer!(&dev, [1, 1, 1, 1]).flat3();
        /// ```
        ///
        /// Example where all functions are explicit:
        /// ```rust
        /// # use coursera_exercises::layer;
        /// # use coursera_exercises::c1::w4::pa_01_deep_nn::{Linear, ReLU, Sigmoid, Layer};
        /// # use coursera_exercises::helpers::_dfdx::*;
        /// # let dev = device();
        /// let s = layer!(&dev, 1, Linear > Sigmoid => [1]);
        /// let rs = layer!(&dev, 1, Linear > ReLU => [1], Linear > Sigmoid => [1]);
        /// let rrs = layer!(&dev, 1, Linear > ReLU => [1, 1], Linear > Sigmoid => [1]).flat3();
        /// ```
        #[allow(unused_macros)]
        #[allow(unused_attributes)]
        #[macro_export]
        macro_rules! layer {
            // implicit layers creation, hidden layers are Linear>Relu, the last is Linear>Sigmoid
            ($dev:expr, [$head_layer_node:literal, $($tail_layers_nodes:literal),*]) => {
                // separates the feature and forward to another macro call
                layer!($dev, auto, $head_layer_node, [$($tail_layers_nodes),*])
            };

            // explicit single layer creation
            ($dev:expr, $features:literal, $z:ty>$a:ty => $layer_nodes:literal) => {
                // returns the layer
                Layer::<$features, $layer_nodes, $z, $a>::normal($dev)
            };

            // implicit hidden layer creation, all Linear>Relu
            ($dev:expr, auto, $node_features:literal, [$head_layer_node:literal, $($tail_layers_nodes:literal),*]) => {
                (
                    {
                        // creates a single implicit layer
                        type _Linear = $crate::c1::w4::pa_01_deep_nn::Linear;
                        type _Relu = $crate::c1::w4::pa_01_deep_nn::ReLU;
                        layer!($dev, $node_features, _Linear>_Relu => $head_layer_node)
                    },
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    layer!($dev, auto, $head_layer_node, [$($tail_layers_nodes),*])
                )
            };

            // explicit hidden layer creation
            ($dev:expr, $node_features:literal, $z:ty>$a:ty => [$head_layer_node:literal, $($tail_layers_nodes:literal),*] $($other:tt)*) => {
                (
                    // creates a single layer
                    layer!($dev, $node_features, $z>$a => $head_layer_node),
                    // recursive macro call on the remaining layers, forwarding the last "feature" information
                    layer!($dev, $head_layer_node, $z>$a => [$($tail_layers_nodes),*] $($other)*)
                )
            };

            // implicit last layer creation, Linear>Sigmoid
            ($dev:expr, auto, $node_features:literal, [$last_layer_node:literal]) => {
                {
                    // creates a single implicit layer (which is the last)
                    type _Linear = $crate::c1::w4::pa_01_deep_nn::Linear;
                    type _Sigmoid = $crate::c1::w4::pa_01_deep_nn::Sigmoid;
                    layer!($dev, $node_features, _Linear>_Sigmoid => $last_layer_node)
                }
            };

            // explicit last layer creation (with no continuation)
            ($dev:expr, $node_features:literal, $z:ty>$a:ty => [$last_layer_node:literal]) => {
                // returns the layer
                layer!($dev, $node_features, $z>$a => $last_layer_node)
            };

            // explicit "last" layer creation (with a continuation)
            ($dev:expr, $node_features:literal, $z:ty>$a:ty => [$last_layer_node:literal] $($other:tt)*) => {
                (
                    // returns the layer
                    layer!($dev, $node_features, $z>$a => $last_layer_node),
                    // makes a brand new macro call on whatever arguments remains,
                    // forwarding the last "feature" information
                    layer!($dev, $last_layer_node $($other)*)
                )
            };
        }
        pub(crate) use layer;

        #[test]
        fn test_layers() {
            let dev = device();
            let rs = layer!(&dev, [5, 4, 3]);
            let _r1: Layer<5, 4, Linear, ReLU> = rs.0;
            let _s2: Layer<4, 3, Linear, Sigmoid> = rs.1;

            // (nothing to assert)

            // note: I also created some helpers for tuple flattening, such as in:
            let (_, _, _) = layer!(&dev, [1, 1, 1, 1]).flat3();
            // without it:
            let (_, (_, _)) = layer!(&dev, [1, 1, 1, 1]);
        }
    }
    pub(crate) use _2::layer;
}
#[allow(unused_imports)]
pub(crate) use _3::layer;
pub use _3::{Layer, Linear, ReLU, Sigmoid, Tanh};

/// C01W04PA01 Part 4 - Forward Propagation Module.
pub mod _4 {
    use super::*;

    /// C01W04PA01 Section 1 - Linear Forward.
    pub mod _1 {
        use super::*;

        pub trait ForwardZ<const FEATLEN: usize, const NODELEN: usize>: Sized {
            fn forward_z<const SETLEN: usize>(
                self,
                x: TensorF32<Rank2<FEATLEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>>;
        }

        /// Any `Linear` layer can calculate z.
        impl<const FEATLEN: usize, const NODELEN: usize, A> ForwardZ<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            fn forward_z<const SETLEN: usize>(
                self,
                x: TensorF32<Rank2<FEATLEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>> {
                self.w.dot(x)
                    + self
                        .b
                        .reshape::<Rank1<NODELEN>>()
                        .broadcast::<Rank2<NODELEN, SETLEN>, _>()
            }
        }

        #[test]
        fn test_forward_z() {
            let dev = &device();
            let linear = layer!(dev, [3, 1]);
            let x: TensorF32<Rank2<3, 2>> = dev.sample_normal();
            let z = linear.forward_z(x);
            assert!(z.array().approx([[-0.27117705, -3.6855178]], (1e-6, 0)));
        }
    }
    pub use _1::ForwardZ;

    /// C01W04PA01 Section 2 - Linear-Activation Forward.
    pub mod _2 {
        use super::*;

        pub trait ForwardA<const FEATLEN: usize, const NODELEN: usize>: Sized {
            fn forward_a<const SETLEN: usize>(
                self,
                z: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>>;
        }

        /// Any layer can activate with sigmoid.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> ForwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Sigmoid>
        {
            fn forward_a<const SETLEN: usize>(
                self,
                z: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>> {
                z.sigmoid()
            }
        }

        /// Any layer can activate with ReLU.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> ForwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Z, ReLU>
        {
            fn forward_a<const SETLEN: usize>(
                self,
                z: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>> {
                z.relu()
            }
        }

        /// Any layer can activate with Tanh.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> ForwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Tanh>
        {
            fn forward_a<const SETLEN: usize>(
                self,
                z: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> TensorF32<Rank2<NODELEN, SETLEN>> {
                z.tanh()
            }
        }

        pub trait ForwardZA<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize>:
            Sized
        {
            type Output: HasA<NODELEN, SETLEN>;
            fn forward_za(self, x: TensorF32<Rank2<FEATLEN, SETLEN>>) -> Self::Output;
        }

        /// Allows `Linear` layers to make the z->a forward calculation.
        impl<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, A>
            ForwardZA<FEATLEN, NODELEN, SETLEN> for Layer<FEATLEN, NODELEN, Linear, A>
        where
            Self: Clone + ForwardA<FEATLEN, NODELEN>,
        {
            type Output = Cache<NODELEN, SETLEN>;
            fn forward_za(self, x: TensorF32<Rank2<FEATLEN, SETLEN>>) -> Cache<NODELEN, SETLEN> {
                let z = self.clone().forward_z(x);
                let a = self.forward_a(z.clone());
                Cache { a }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Cache<const NODELEN: usize, const SETLEN: usize> {
            // /// Rows for nodes, columns for train/test set size.
            // pub z: TensorF32<Rank2<NODELEN, SETLEN>>,
            /// Activation function. Rows for nodes, columns for train/test set size.
            pub a: TensorF32<Rank2<NODELEN, SETLEN>>,
        }

        // (note: this is useful later on)
        /// Helper trait to access cache info.
        pub trait HasA<const NODELEN: usize, const SETLEN: usize> {
            fn ref_a(&self) -> &TensorF32<Rank2<NODELEN, SETLEN>>;
        }

        impl<const NODELEN: usize, const SETLEN: usize> HasA<NODELEN, SETLEN> for Cache<NODELEN, SETLEN> {
            fn ref_a(&self) -> &TensorF32<Rank2<NODELEN, SETLEN>> {
                &self.a
            }
        }

        #[test]
        fn test_forward() {
            let dev = &device();
            let x: TensorF32<Rank2<3, 2>> = dev.sample_normal();

            let sigmoid = layer!(dev, 3, Linear > Sigmoid => [1]);
            let cache = sigmoid.forward_za(x.clone());
            assert!(cache
                .a
                .array()
                .approx([[0.034352947, 0.40703216]], (1e-7, 0)));

            let relu = layer!(dev, 3, Linear > ReLU => [1]);
            let cache = relu.forward_za(x);
            assert!(cache.a.array().approx([[1.09002, 0.0]], (1e-6, 0)));
        }
    }
    pub use _2::{Cache, ForwardA, ForwardZA, HasA};

    /// C01W04PA01 Section 3 - L-Layer Model.
    pub mod _3 {
        use super::*;

        /// Multiple forward_za calls between many adjacent layers.
        pub trait Forward<const SETLEN: usize>: Sized {
            /// The input set for a single layer, or the first input of a stack of layers.
            type Input;
            /// The Cache for a single layer, or stack of caches of multiple layers.
            type Output;

            /// Makes the forward call for a single layer, or make many calls for a stack of layers.
            fn forward(self, x: Self::Input) -> Self::Output;
        }

        /// Forward call for a single layer.
        impl<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A> Forward<SETLEN>
            for Layer<FEATLEN, NODELEN, Z, A>
        where
            Layer<FEATLEN, NODELEN, Z, A>: ForwardZA<FEATLEN, NODELEN, SETLEN>,
        {
            type Input = TensorF32<Rank2<FEATLEN, SETLEN>>;
            type Output = <Self as ForwardZA<FEATLEN, NODELEN, SETLEN>>::Output;

            fn forward(self, x: Self::Input) -> Self::Output {
                self.forward_za(x)
            }
        }

        /// Forward call for one (or recursively more) pair of layers.
        impl<Lower, const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A>
            Forward<SETLEN> for (Layer<FEATLEN, NODELEN, Z, A>, Lower)
        where
            Layer<FEATLEN, NODELEN, Z, A>: ForwardZA<FEATLEN, NODELEN, SETLEN>,
            Lower: Forward<SETLEN, Input = TensorF32<Rank2<NODELEN, SETLEN>>>,
        {
            type Input = TensorF32<Rank2<FEATLEN, SETLEN>>;
            type Output = (
                <Layer<FEATLEN, NODELEN, Z, A> as ForwardZA<FEATLEN, NODELEN, SETLEN>>::Output,
                Lower::Output,
            );

            fn forward(self, x: Self::Input) -> Self::Output {
                let current_cache = self.0.forward_za(x);
                let x_lower: TensorF32<Rank2<NODELEN, SETLEN>> = current_cache.ref_a().clone();
                (current_cache, self.1.forward(x_lower))
            }
        }

        #[test]
        fn test_l_layers_forward() {
            let dev = &device();
            let x: TensorF32<Rank2<5, 4>> = dev.sample_normal();
            let layers = layer!(dev, [5, 4, 3, 1]);
            let caches = layers.forward(x);
            assert!(caches
                .flat3()
                .2
                .a
                .array()
                .approx([[0.50810283, 0.9082994, 0.7377677, 0.6885722]], (1e-6, 0)));
        }
    }
    pub use _3::Forward;
}
pub use _4::{Cache, Forward, ForwardA, ForwardZ, ForwardZA, HasA};

/// C01W04PA01 Part 5 - Cost Function.
pub mod _5 {
    use super::*;

    pub trait Cost<const NODELEN: usize, const SETLEN: usize> {
        /// Cost between the generated prediction and the given expected values.
        fn cost(
            predict: TensorF32<Rank2<NODELEN, SETLEN>>,
            expected: TensorF32<Rank2<NODELEN, SETLEN>>,
        ) -> TensorF32<Rank1<NODELEN>>;
    }

    /// Logistical cost function.
    ///
    /// cost function m*J(L) = sum (L)
    /// loss function L(ŷ,y) = -(y*log(ŷ) + (1-y)log(1-ŷ))
    ///
    /// Note that the cost function is multiplied by `m` (SETLEN).
    #[derive(Clone, Debug)]
    pub struct MLogistical;

    /// Logistical cost.
    impl<const NODELEN: usize, const SETLEN: usize> Cost<NODELEN, SETLEN> for MLogistical {
        fn cost(
            predict: TensorF32<Rank2<NODELEN, SETLEN>>,
            expect: TensorF32<Rank2<NODELEN, SETLEN>>,
        ) -> TensorF32<Rank1<NODELEN>> {
            // loss function L = -(y*log(ŷ) + (1-y)log(1-ŷ))
            let l1 = expect.clone() * predict.clone().ln();
            let l2 = expect.clone().negate() + 1.;
            let l3 = (predict.clone().negate() + 1.).ln();
            let l = (l1 + l2 * l3).negate();

            // cost function m * J = sum (L)
            l.sum::<Rank1<NODELEN>, _>()
        }
    }

    /// Squared cost function.
    ///
    /// cost function m*J(L) = sum (L)
    /// loss function L(ŷ,y) = (ŷ-y)²
    ///
    /// Note that the cost function is multiplied by `m` (SETLEN).
    #[derive(Clone, Debug)]
    pub struct MSquared;

    /// Squared cost.
    impl<const NODELEN: usize, const SETLEN: usize> Cost<NODELEN, SETLEN> for MSquared {
        fn cost(
            predict: TensorF32<Rank2<NODELEN, SETLEN>>,
            expect: TensorF32<Rank2<NODELEN, SETLEN>>,
        ) -> TensorF32<Rank1<NODELEN>> {
            // loss function L = (ŷ-y)²
            let l = (predict - expect).square();

            // cost function m * J = sum (L)
            l.sum::<Rank1<NODELEN>, _>()
        }
    }

    #[test]
    fn test_cost() {
        let dev = &device();
        let y = dev.tensor([[1., 1., 0.]]);
        let cache = Cache {
            a: dev.tensor([[0.8, 0.9, 0.4]]),
        };
        let setlen = 3.;
        assert_eq!(
            (MLogistical::cost(cache.a, y) / setlen).array(),
            [0.27977654]
        );
    }
}
pub use _5::{Cost, MLogistical};

/// C01W04PA01 Part 6 - Backward Propagation Module.
pub mod _6 {
    use super::*;

    /// C01W04PA01 Section 1 - Linear Backward.
    pub mod _1 {
        use super::*;

        /// ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
        pub trait BackwardDownZUpA: Sized {
            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            type Output<const SETLEN: usize>;

            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            fn backward_dzdx<const UP_NODELEN: usize, const SETLEN: usize>(
                self,
                up_cache: Cache<UP_NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN>;
        }

        /// Any lower layer with a Linear z function can calculate dzda[up] = ∂z[down]/∂a[up].
        impl<const FEATLEN: usize, const NODELEN: usize, A> BackwardDownZUpA
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up].
            type Output<const SETLEN: usize> = TensorF32<Rank2<NODELEN, FEATLEN>>;
            // note: SETLEN is not used but it could be for other formulas (other derivates).

            /// dzdx = ∂z[down]/∂x[down] = ∂z[down]/∂a[up] = w[down].
            fn backward_dzdx<const UP_NODELEN: usize, const SETLEN: usize>(
                self,
                _up_cache: Cache<UP_NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                self.w
            }
        }

        /// mda[up] = m * ∂J/∂a[up] = ∂z[down]/∂a[up] * ∂J/∂z[down] = dzdx * mdz[down].
        pub fn backward_up_mda<
            const DOWN_NODELEN: usize,
            const FEATLEN: usize,
            const SETLEN: usize,
        >(
            // cache: Cache<NODELEN, SETLEN>,
            dzda: TensorF32<Rank2<DOWN_NODELEN, FEATLEN>>,
            down_mdz: TensorF32<Rank2<DOWN_NODELEN, SETLEN>>,
        ) -> TensorF32<Rank2<FEATLEN, SETLEN>> {
            dzda.permute::<_, Axes2<1, 0>>().dot(down_mdz)
        }

        /// m * ∂z/∂w, m * ∂z/∂b.
        pub trait BackwardZwb<const FEATLEN: usize, const NODELEN: usize>: Sized {
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            type Output<const UPPER_NODELEN: usize>;
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            fn backward_dw_db<const SETLEN: usize, const UPPER_NODELEN: usize>(
                self,
                upper_cache: Cache<UPPER_NODELEN, SETLEN>,
                mdz: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<UPPER_NODELEN>;
        }

        /// Any `Linear` layer can calculate dw = ∂z/∂w, db = ∂z/∂b.
        impl<const FEATLEN: usize, const NODELEN: usize, A> BackwardZwb<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, Linear, A>
        {
            /// (dw, db) = (∂J/∂w, ∂J/∂b).
            type Output<const UPPER_NODELEN: usize> = (
                TensorF32<Rank2<NODELEN, UPPER_NODELEN>>,
                TensorF32<Rank1<NODELEN>>,
            );

            /// ∂z/∂w = upper_cache.a  
            /// dw = ∂J/∂w = ∂z/∂w * ∂J/∂z = upper_cache.a * mdz / m.
            ///
            /// ∂z/∂b = 1  
            /// db = ∂J/∂b = ∂z/∂b * ∂J/∂z = mdz / m.
            fn backward_dw_db<const SETLEN: usize, const UPPER_NODELEN: usize>(
                self,
                upper_cache: Cache<UPPER_NODELEN, SETLEN>,
                mdz: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<UPPER_NODELEN> {
                let dw =
                    mdz.clone().dot(upper_cache.a.permute::<_, Axes2<1, 0>>()) / (SETLEN as f32);
                let db = mdz.sum::<Rank1<NODELEN>, _>() / (SETLEN as f32);
                (dw, db)
            }
        }

        #[test]
        fn test_backward_dw_db_and_mdaup() {
            let dev = &device();

            let mdz: TensorF32<Rank2<3, 4>> = dev.sample_normal();
            let up_a: TensorF32<Rank2<5, 4>> = dev.sample_normal();
            let w: TensorF32<Rank2<3, 5>> = dev.sample_normal();
            let b: TensorF32<Rank2<3, 1>> = dev.sample_normal();

            let up_cache = Cache { a: up_a };

            // note: only the 'Linear' part of the layer matters here
            let l: Layer<5, 3, Linear, Sigmoid> = Layer::with(w, b);
            let (dw, db) = l.clone().backward_dw_db(up_cache.clone(), mdz.clone());
            let dzda = l.backward_dzdx(up_cache);
            let up_mda = backward_up_mda(dzda, mdz);

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
    pub use _1::{backward_up_mda, BackwardDownZUpA, BackwardZwb};

    /// C01W04PA01 Section 2 - Linear-Activation Backward.
    pub mod _2 {
        use super::*;

        /// m * ∂a/∂z.
        pub trait BackwardAZ<const NODELEN: usize>: Sized {
            /// mdz = m * ∂J/∂z.
            type Output<const SETLEN: usize>;
            /// mdz = m * ∂J/∂z.
            fn backward_mdz<const SETLEN: usize>(
                self,
                cache: Cache<NODELEN, SETLEN>,
                mda: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN>;
        }

        /// Any layer with a Sigmoid activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> BackwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Sigmoid>
        {
            /// mdz = cache.a * (1 - cache.a) * mda.
            type Output<const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

            // TODO: I thought the formula was cache.a * (1 - cache.a), instead of using sigmoid
            // (because the cache.a already *is* the sigmoid application)
            // TODO: compare with previous lessons

            /// ∂a/∂z = σ'(z) = e^-z / (1 + e^-z)² = σ(z) * (1 - σ(z)) = cache.a * (1 - cache.a).
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = cache.a * (1 - cache.a) * mda
            fn backward_mdz<const SETLEN: usize>(
                self,
                cache: Cache<NODELEN, SETLEN>,
                mda: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN> {
                let _a_neg = cache.a.clone().negate();
                cache.a * (_a_neg + 1.) * mda
            }
        }

        /// Any layer with a Tanh activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> BackwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Tanh>
        {
            /// mdz = (1 - cache.a²) * mda.
            type Output<const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

            /// ∂a/∂z = tanh'(z) = 1 - z² = 1 - cache.a².
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = (1 - cache.a²) * mda.
            fn backward_mdz<const SETLEN: usize>(
                self,
                cache: Cache<NODELEN, SETLEN>,
                mda: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN> {
                (cache.a.square().negate() + 1.) * mda
            }
        }

        /// Any layer with a ReLU activation can calculate mdz = m * ∂a/∂z.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> BackwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, ReLU>
        {
            /// mdz.
            type Output<const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

            /// ∂a/∂z = relu'(z) = {1 if z > 0; 0 if z <= 0} = {1 if cache.a > 0; 0 if cache.a <= 0}
            ///
            /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = {1 if cache.a > 0; 0 if cache.a <= 0} * mda
            fn backward_mdz<const SETLEN: usize>(
                self,
                cache: Cache<NODELEN, SETLEN>,
                mda: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN> {
                let dev = cache.a.device();
                let mask = cache.a.gt(0.);
                let zeros: TensorF32<Rank2<NODELEN, SETLEN>> = dev.zeros();
                mask.choose(mda, zeros)
            }
        }

        #[test]
        fn test_backward_mdz() {
            let dev = &device();

            let mda: TensorF32<Rank2<1, 2>> = dev.sample_normal();
            let w: TensorF32<Rank2<1, 3>> = dev.sample_normal();
            let b: TensorF32<Rank2<1, 1>> = dev.sample_normal();
            let up_cache = Cache {
                a: dev.sample_normal(),
            };
            let z: TensorF32<Rank2<1, 2>> = dev.sample_normal();

            // sigmoid layer test
            {
                // note that we should already have calculated σ(z) and stored it in cache.a during the forward pass:
                let cache = Cache {
                    a: z.clone().sigmoid(),
                };
                let l: Layer<3, 1, Linear, Sigmoid> = Layer::with(w.clone(), b.clone());
                let mdz = l.clone().backward_mdz(cache.clone(), mda.clone());
                let (dw, db) = l.clone().backward_dw_db(up_cache.clone(), mdz.clone());
                let dzda = l.backward_dzdx(up_cache.clone());
                let up_mda = backward_up_mda(dzda, mdz);

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

                // note that we should already have calculated relu(z) and stored it in cache.a during the forward pass:
                let cache = Cache {
                    a: z.clone().relu(),
                };
                let l: Layer<3, 1, Linear, ReLU> = Layer::with(w.clone(), b.clone());
                let mdz = l.clone().backward_mdz(cache.clone(), mda.clone());
                let (dw, db) = l.clone().backward_dw_db(up_cache.clone(), mdz.clone());
                let dzda = l.backward_dzdx(up_cache.clone());
                let up_mda = backward_up_mda(dzda, mdz);

                assert_eq!(
                    up_mda.array(),
                    [[-0.0, -2.0911047,], [0.0, 0.14020352,], [-0.0, -1.0943813,],]
                );
                assert_eq!(dw.array(), [[0.2629047, 0.7462477, -1.2105147,],]);
                assert_eq!(db.array(), [0.42916572]);
            }
        }
    }
    pub use _2::BackwardAZ;

    /// C01W04PA01 Section 3 - L-Model Backward.
    pub mod _3 {
        use super::*;

        /// m * ∂J/∂a.
        pub trait BackwardJA<Cost, const NODELEN: usize>: Sized {
            /// mda = m * ∂J/∂a.
            type Output<const SETLEN: usize>;

            /// mda = m * ∂J/∂a.
            fn backward_mda<const SETLEN: usize>(
                self,
                predicted: TensorF32<Rank2<NODELEN, SETLEN>>,
                expected: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN>;
        }

        /// Any layer with a Sigmoid activation can calculate mda = m * ∂J/∂a for the logistical cost function.
        impl<const FEATLEN: usize, const NODELEN: usize, Z> BackwardJA<MLogistical, NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Sigmoid>
        {
            /// mda = m * ∂J/∂a.
            type Output<const SETLEN: usize> = TensorF32<Rank2<NODELEN, SETLEN>>;

            /// mda = m * ∂J/∂a = (1-y)/(1-a) - y/a = (a-y)/(a(1-a)).
            fn backward_mda<const SETLEN: usize>(
                self,
                predicted: TensorF32<Rank2<NODELEN, SETLEN>>,
                expected: TensorF32<Rank2<NODELEN, SETLEN>>,
            ) -> Self::Output<SETLEN> {
                let _a_neg = predicted.clone().negate();
                let _a = predicted.clone();
                (predicted - expected) / (_a * (_a_neg + 1.))
            }
        }

        #[test]
        fn test_backward_j() {
            let dev = &device();
            let x: TensorF32<Rank2<4, 2>> = dev.sample_normal();
            let y: TensorF32<Rank2<1, 2>> = dev.sample_normal();
            let ls = layer!(dev, [4, 3, 1]);
            let (cache1, cache2) = ls.clone().forward(x.clone());
            let (l1, l2) = ls;

            // l2 sigmoid (from loss)
            let mda_lower = l2.clone().backward_mda(cache2.a.clone(), y);
            assert!(mda_lower
                .array()
                .approx([[-65.21482, -5.9279017]], (1e-5, 0)));

            // sigmoid layer test
            #[allow(clippy::let_and_return)]
            let up_mda = {
                let mdz = l2.clone().backward_mdz(cache2.clone(), mda_lower.clone());
                let (_dw2, _db2) = l2.clone().backward_dw_db(cache1.clone(), mdz.clone());
                let dzda = l2.clone().backward_dzdx(cache1.clone());
                let up_mda = backward_up_mda(dzda, mdz.clone());
                up_mda
            };

            // relu layer test
            let up_mdz = l1.clone().backward_mdz(cache1.clone(), up_mda.clone());
            let cache0 = Cache { a: x };
            let (dw1, db1) = l1.clone().backward_dw_db(cache0, up_mdz.clone());

            assert!(dw1.array().approx(
                [
                    [0.57377756, -0.44744247, 0.35314572, -0.021219313],
                    [-0.0021573375, -0.00041055086, -0.0032351865, -0.001539701],
                    [0.0, 0.0, 0.0, 0.0]
                ],
                (1e-7, 0)
            ));
            assert!(db1
                .array()
                .approx([0.70517296, -0.0025134084, 0.0], (1e-7, 0)));
        }
    }

    /// C01W04PA01 Section 4 - Update Parameters.
    pub mod _4 {
        // no content
    }
}

/// C01W04PA01 Part 7 - Conclusion.
pub mod _7 {
    // no content
}

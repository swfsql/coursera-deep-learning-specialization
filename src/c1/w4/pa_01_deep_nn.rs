//! Building your Deep Neural Network: Step by Step
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Neural%20Networks%20and%20Deep%20Learning/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step.ipynb
//!
//! I recommend first watching all of C1W4.

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
        /// # use crate::c1::w4::pa_01_deep_nn::{Linear, Relu, Sigmoid};
        /// # use crate::helpers::_dfdx::*;
        /// let s = layer!(&dev, [1, 1]);
        /// let rs = layer!(&dev, [1, 1, 1]);
        /// let rrs = layer!(&dev, [1, 1, 1, 1]).flat3();
        /// ```
        ///
        /// Example where all functions are explicit:
        /// ```rust
        /// # use crate::c1::w4::pa_01_deep_nn::{Linear, Relu, Sigmoid};
        /// # use crate::helpers::_dfdx::*;
        /// let s = layer!(&dev, 1, Linear > Sigmoid => [1]);
        /// let rs = layer!(&dev, 1, Linear > Relu => [1], Linear > Sigmoid => [1]);
        /// let rrs = layer!(&dev, 1, Linear > Relu => [1, 1], Linear > Sigmoid => [1]).flat3();
        /// ```
        #[allow(unused_macros)]
        #[allow(unused_attributes)]
        #[macro_use]
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
                        type _Linear = crate::c1::w4::pa_01_deep_nn::Linear;
                        type _Relu = crate::c1::w4::pa_01_deep_nn::ReLU;
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
                    type _Linear = crate::c1::w4::pa_01_deep_nn::Linear;
                    type _Sigmoid = crate::c1::w4::pa_01_deep_nn::Sigmoid;
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
            assert!(z.array().approx([[-0.27117705, -3.6855178]], (1e-6, 0)),);
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
            type Output: AsCache<NODELEN, SETLEN>;
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
                Cache { z, a }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Cache<const NODELEN: usize, const SETLEN: usize> {
            /// Rows for nodes, columns for train/test set size.
            pub z: TensorF32<Rank2<NODELEN, SETLEN>>,
            /// Activation function. Rows for nodes, columns for train/test set size.
            pub a: TensorF32<Rank2<NODELEN, SETLEN>>,
        }

        // (note: this is useful later on)
        /// Helper trait to access cache info.
        pub trait AsCache<const NODELEN: usize, const SETLEN: usize> {
            fn ref_z(&self) -> &TensorF32<Rank2<NODELEN, SETLEN>>;
            fn ref_a(&self) -> &TensorF32<Rank2<NODELEN, SETLEN>>;
        }

        impl<const NODELEN: usize, const SETLEN: usize> AsCache<NODELEN, SETLEN>
            for Cache<NODELEN, SETLEN>
        {
            fn ref_z(&self) -> &TensorF32<Rank2<NODELEN, SETLEN>> {
                &self.z
            }
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
            assert!(cache.a.array().approx([[1.09002, 0.0]], (1e-6, 0)),);
        }
    }
    // pub use _2::{Cache, Forward, ForwardA};
    pub use _2::*;

    /// C01W04PA01 Section 3 - L-Layer Model.
    pub mod _3 {
        use super::*;

        /// Multiple forward_za calls between many adjacent layers.
        pub trait LForward<const SETLEN: usize>: Sized {
            /// The input set for a single layer, or the first input of a stack of layers.
            type Input;
            /// The Cache for a single layer, or stack of caches of multiple layers.
            type Output;

            /// Makes the forward call for a single layer, or make many calls for a stack of layers.
            fn forward(self, x: Self::Input) -> Self::Output;
        }

        /// Forward call for a single layer.
        impl<const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A> LForward<SETLEN>
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
        impl<Next, const FEATLEN: usize, const NODELEN: usize, const SETLEN: usize, Z, A>
            LForward<SETLEN> for (Layer<FEATLEN, NODELEN, Z, A>, Next)
        where
            Layer<FEATLEN, NODELEN, Z, A>: ForwardZA<FEATLEN, NODELEN, SETLEN>,
            Next: LForward<SETLEN, Input = TensorF32<Rank2<NODELEN, SETLEN>>>,
        {
            type Input = TensorF32<Rank2<FEATLEN, SETLEN>>;
            type Output = (
                <Layer<FEATLEN, NODELEN, Z, A> as ForwardZA<FEATLEN, NODELEN, SETLEN>>::Output,
                Next::Output,
            );

            fn forward(self, x: Self::Input) -> Self::Output {
                let current_cache = self.0.forward_za(x);
                let x_next: TensorF32<Rank2<NODELEN, SETLEN>> = current_cache.ref_a().clone();
                (current_cache, self.1.forward(x_next))
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
                .approx([[0.50810283, 0.9082994, 0.7377677, 0.6885722]], (1e-6, 0)),);
        }
    }
}

/// C01W04PA01 Part 5 - Cost Function.
pub mod _5 {
    // no content
}

/// C01W04PA01 Part 6 - Backward Propagation Module.
pub mod _6 {

    /// C01W04PA01 Section 1 - Linear Backward.
    pub mod _1 {
        // no content
    }

    /// C01W04PA01 Section 2 - Linear-Activation Backward.
    pub mod _2 {
        // no content
    }

    /// C01W04PA01 Section 3 - L-Model Backward.
    pub mod _3 {
        // no content
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

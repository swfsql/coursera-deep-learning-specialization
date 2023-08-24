//! Planar Data Classification with One Hidden Layer
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar_data_classification_with_onehidden_layer_v6c.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb
//!
//! I recommend first watching all of C1W3, and then also the chapters 1~3 of 3b1b's "Neural Networks".
//!
//! 3b1b "Neural Networks": https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

/// C01W03PA01 Part 1 - Packages.
mod _1 {
    // no content
}

/// C01W03PA01 Part 2 - Dataset.
pub mod _2 {
    use crate::c1::w2::pa_02_logistic_regression::PreparedData;
    pub use crate::c1::w3::util::{M_TRAIN, X, XLEN, Y};
    use crate::helpers::_dfdx::*;

    pub fn prepare_data(device: &Device) -> PreparedData<XLEN, M_TRAIN> {
        let xtrain: TensorF32<Rank2<XLEN, M_TRAIN>> = device.tensor(X);
        let ytrain: TensorU8<Rank2<1, M_TRAIN>> = device.tensor(Y);
        PreparedData {
            x: xtrain,
            y: ytrain,
        }
    }

    #[test]
    fn test_shapes() {
        // number of features
        assert_eq!(XLEN, 2);
        // number of tests
        assert_eq!(M_TRAIN, 400);

        // X shape
        assert_eq!(X.len(), XLEN);
        assert_eq!(X[0].len(), M_TRAIN);

        // Y shape
        assert_eq!(Y.len(), 1);
        assert_eq!(Y[0].len(), M_TRAIN);
    }
}

/// C01W03PA01 Part 3 - Simple Logistic Regression.
pub mod _3 {

    #[test]
    fn a() -> anyhow::Result<()> {
        use super::_2::prepare_data;
        use crate::c1::w2::pa_02_logistic_regression::{accuracy, Model, PreparedData};
        pub use crate::c1::w3::util::{M_TRAIN, X, XLEN, Y};
        use crate::helpers::_dfdx::*;

        let dev = device();

        // train data
        let train = prepare_data(&dev);

        // test (dummy value)
        // note: the model-training function from last week always require some test-data alongside the train-data,
        // but for now we are only interested in the train-data accuracy.
        // So we just repeat the first entry of the training data (using an empty tensor fails).
        let xtest: TensorF32<Rank2<XLEN, 1>> = dev.tensor([[X[0][0]], [X[1][0]]]);
        let ytest: TensorU8<Rank2<1, 1>> = dev.tensor([[Y[0][0]]]);
        let test = PreparedData { x: xtest, y: ytest };

        let model = Model::<XLEN>::new(&dev);
        let model = model.train(train.clone(), test, 1500, 1e-4, usize::MAX)?;
        let train_accuracy = accuracy(model.train_predict, train.y.clone());

        // the python reference code used sklearn to train the model,
        // but here we used the same training method from the previous week.
        // but we still got a very similar number result
        assert_eq!(train_accuracy.array()[0], 47.750004);

        Ok(())
    }
}

/// C01W03PA01 Part 4 - Neural Network Model.
pub mod _4 {
    #[allow(unused_imports)]
    use crate::helpers::{Approx, _dfdx::*};

    // Note: the python code uses different parameter values for each example,
    // and I originally used the same values as the python code and verified
    // that their outputs matched, but later I changed the rust values to a
    // fixed set of values instead. So the outputs are different from the
    // python's, but they are still (presumably) correct.

    /// - w1: 4 nodes (rows), 2 features (cols);
    /// - b1: 4 nodes (rows);
    /// - w2: 1 node (rows), 4 "features" (cols);
    /// - b2: 1 node (rows);
    fn example_model(device: &Device) -> Model<2, 4, 1> {
        Model::from_values(
            device,
            [
                [-0.00615039, 0.0169021],
                [-0.02311792, 0.03137121],
                [-0.0169217, -0.01752545],
                [0.00935436, -0.05018221],
            ],
            [[-8.97523e-07], [8.15562e-06], [6.04810e-07], [-2.54560e-06]],
            [[-0.0104319, -0.04019007, 0.01607211, 0.04440255]],
            [[9.14954e-05]],
        )
    }

    /// - 2 features (rows), 3 examples (cols);
    fn example_x(device: &Device) -> TensorF32<Rank2<2, 3>> {
        device.tensor([
            [1.6243453, -0.6117564, -0.5281717],
            [-1.0729686, 0.86540763, -2.3015387],
        ])
    }

    /// - 1 output value (rows), 3 examples (cols);
    fn example_y(device: &Device) -> TensorF32<Rank2<1, 3>> {
        let y = device.tensor([[true, false, true]]);
        y.clone()
            .choose(device.ones_like(&y), device.zeros_like(&y))
    }

    /// C01W03PA01 Section 1 - Defining the Neural Network Structure.
    mod _1 {
        // no content
    }

    /// C01W03PA01 Section 2 - Initialize the Model's Parameters.
    pub mod _2 {
        use crate::helpers::_dfdx::*;
        use dfdx::tensor::SampleTensor;

        // Note: c1::w2::pa_02 had a model weight definition that was different.
        // In there the features were in rows, but here they are in columns.
        #[derive(Clone, Debug)]
        pub struct Model<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> {
            /// Layer 1/2. Rows for nodes, columns for features.
            pub w1: TensorF32<Rank2<L1LEN, XLEN>>,
            /// Layer 1/2. Rows for nodes.
            pub b1: TensorF32<Rank2<L1LEN, 1>>,
            /// Layer 2/2. Rows for nodes, columns for features.
            pub w2: TensorF32<Rank2<L2LEN, L1LEN>>,
            /// Layer 2/2. Rows for nodes.
            pub b2: TensorF32<Rank2<L2LEN, 1>>,
        }

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn normal(device: &Device) -> Self {
                let w1: TensorF32<Rank2<L1LEN, XLEN>> = device.sample_normal();
                let b1: TensorF32<Rank2<L1LEN, 1>> = device.sample_normal();
                let w2: TensorF32<Rank2<L2LEN, L1LEN>> = device.sample_normal();
                let b2: TensorF32<Rank2<L2LEN, 1>> = device.sample_normal();
                Self::with(w1, b1, w2, b2)
            }
            pub fn with(
                w1: TensorF32<Rank2<L1LEN, XLEN>>,
                b1: TensorF32<Rank2<L1LEN, 1>>,
                w2: TensorF32<Rank2<L2LEN, L1LEN>>,
                b2: TensorF32<Rank2<L2LEN, 1>>,
            ) -> Self {
                Self { w1, b1, w2, b2 }
            }
            pub fn from_values(
                device: &Device,
                w1: [[f32; XLEN]; L1LEN],
                b1: [[f32; 1]; L1LEN],
                w2: [[f32; L1LEN]; L2LEN],
                b2: [[f32; 1]; L2LEN],
            ) -> Self {
                let w1: TensorF32<Rank2<L1LEN, XLEN>> = device.tensor(w1);
                let b1: TensorF32<Rank2<L1LEN, 1>> = device.tensor(b1);
                let w2: TensorF32<Rank2<L2LEN, L1LEN>> = device.tensor(w2);
                let b2: TensorF32<Rank2<L2LEN, 1>> = device.tensor(b2);
                Self::with(w1, b1, w2, b2)
            }
        }
    }
    pub use _2::Model;

    /// C01W03PA01 Section 3 - The Loop.
    pub mod _3 {
        use super::*;

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn forward<const SETLEN: usize>(
                self,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
            ) -> Cache<L1LEN, L2LEN, SETLEN> {
                // z1(x,w1,b1) = b1 + w1 * x
                let z1 = self.w1.dot(x)
                    + self
                        .b1
                        .reshape::<Rank1<L1LEN>>()
                        .broadcast::<Rank2<L1LEN, SETLEN>, _>();

                // a1(z1) = tanh(z1) = (e^z1-e^-z1)/(e^z+e^-z)
                let a1 = z1.clone().tanh();

                // z2(a1,w2,b2) = b2 + w2 * a1
                let z2 = self.w2.dot(a1.clone())
                    + self
                        .b2
                        .reshape::<Rank1<L2LEN>>()
                        .broadcast::<Rank2<L2LEN, SETLEN>, _>();

                // a2(z2) = σ(z2) = 1/(1 + e^-z2)
                let a2 = z2.clone().sigmoid();

                Cache { z1, a1, z2, a2 }
            }
        }

        /// Forward propagation cache.
        #[derive(Clone, Debug)]
        pub struct Cache<const L1LEN: usize, const L2LEN: usize, const SETLEN: usize> {
            /// Layer 1/2. Rows for nodes, columns for train/test set size.
            pub z1: TensorF32<Rank2<L1LEN, SETLEN>>,
            /// Layer 1/2 activation function. Rows for nodes, columns for train/test set size.
            pub a1: TensorF32<Rank2<L1LEN, SETLEN>>,
            /// Layer 2/2. Rows for nodes, columns for train/test set size.
            pub z2: TensorF32<Rank2<L2LEN, SETLEN>>,
            /// Layer 2/2 activation function. Rows for nodes, columns for train/test set size.
            pub a2: TensorF32<Rank2<L2LEN, SETLEN>>,
        }

        #[test]
        fn test_cache() {
            let dev = device();
            let x = example_x(&dev);
            let model = example_model(&dev);
            let cache = model.forward(x);

            assert!(cache
                .z1
                .mean::<Rank0, _>()
                .array()
                .approx(0.0025779027, (1e-8, 0)));
            assert!(cache
                .a1
                .mean::<Rank0, _>()
                .array()
                .approx(0.0025471698, (1e-8, 0)));
            assert!(cache
                .z2
                .mean::<Rank0, _>()
                .array()
                .approx(0.0035655606, (1e-8, 0)));
            assert!(cache
                .a2
                .mean::<Rank0, _>()
                .array()
                .approx(0.50089145, (1e-7, 0)));
        }

        impl<const L1LEN: usize, const L2LEN: usize, const SETLEN: usize> Cache<L1LEN, L2LEN, SETLEN> {
            pub fn cost(self, y: TensorF32<Rank2<L2LEN, SETLEN>>) -> f32 {
                // loss function L = -(y*log(ŷ) + (1-y)log(1-ŷ))
                let l1 = y.clone() * self.a2.clone().ln();
                let l2 = y.clone().negate() + 1.;
                let l3 = (self.a2.clone().negate() + 1.).ln();
                let l = (l1 + l2 * l3).negate();

                // cost function J = 1/m sum (L)
                let j = l.sum::<Rank0, _>() / (SETLEN as f32);
                j.array()
            }

            pub fn with(
                z1: TensorF32<Rank2<L1LEN, SETLEN>>,
                a1: TensorF32<Rank2<L1LEN, SETLEN>>,
                z2: TensorF32<Rank2<L2LEN, SETLEN>>,
                a2: TensorF32<Rank2<L2LEN, SETLEN>>,
            ) -> Self {
                Self { z1, a1, z2, a2 }
            }
            pub fn from_values(
                device: &Device,
                z1: [[f32; SETLEN]; L1LEN],
                a1: [[f32; SETLEN]; L1LEN],
                z2: [[f32; SETLEN]; L2LEN],
                a2: [[f32; SETLEN]; L2LEN],
            ) -> Self {
                let z1: TensorF32<Rank2<L1LEN, SETLEN>> = device.tensor(z1);
                let a1: TensorF32<Rank2<L1LEN, SETLEN>> = device.tensor(a1);
                let z2: TensorF32<Rank2<L2LEN, SETLEN>> = device.tensor(z2);
                let a2: TensorF32<Rank2<L2LEN, SETLEN>> = device.tensor(a2);
                Self::with(z1, a1, z2, a2)
            }
        }

        #[test]
        fn test_cost() {
            let dev = device();
            let x = example_x(&dev);
            let y = example_y(&dev);
            let model = example_model(&dev);
            let cache = model.forward(x);
            // note: for the cost, only cache.a2 is used
            let cost = cache.cost(y);
            assert_eq!(cost, 0.6900306);
        }

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn backward<const SETLEN: usize>(
                self,
                cache: Cache<L1LEN, L2LEN, SETLEN>,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
                y: TensorF32<Rank2<L2LEN, SETLEN>>,
            ) -> Gradient<XLEN, L1LEN, L2LEN> {
                // note: (re)watching both the C1W3 and the entire series on nn from 3b1b helps.

                // cost function m*J(L) = sum (L)
                // loss function L(a2,y) = -(y*log(a2) + (1-y)log(1-a2)) <- logistic regression

                // J -> L -> a2
                // activation function a2(z2) = σ(z2) = 1/(1 + e^-z2) <- from sigmoid definition
                // ∂a2/∂z2 = ∂a2/∂z2 = σ'(z2) = e^-z2 / (1 + e^-z2)² = σ(z2) * (1-σ(z2)) = a2(1-a2) <- from calculus
                // m * ∂J/∂a2 = y/a2 + (1-y)/(1-a2) = (a2-y)/(a2(1-a2))
                //
                // note: for better precision, 1/m is deferred, so we multiply by m on the partial derivative side.
                // (instead of dividing by m for just dz2, we instead do it for each of dw2,db2,dw1,db1)

                // J -> L -> a2 -> z2
                // z2(a1,w2,b2) = b2 + w2 * a1
                // mdz2 = m * ∂J/∂z2 = ∂a2/∂z2 * ∂J/∂a2 = a2(1-a2) * (a2-y)/(a2(1-a2)) = (a2-y)
                let mdz2 = cache.a2 - y;

                // J -> L -> a2 -> z2 -> w2
                // ∂z2/∂w2 = a1
                // dw2 = ∂J/∂w2 = ∂z2/∂w2 * ∂J/∂z2 / m = a1 * mdz2 / m
                let dw2 = mdz2
                    .clone()
                    .dot(cache.a1.clone().permute::<_, Axes2<1, 0>>())
                    / (SETLEN as f32);

                // J -> L -> a2 -> z2 -> b2
                // ∂z2/∂b2 = 1
                // db2 = ∂J/∂b2 = ∂z2/∂b2 * ∂J/∂z2 / m = mdz2 / m
                let db2 = mdz2.clone().sum::<Rank1<L2LEN>, _>() / (SETLEN as f32);

                // J -> L -> a2 -> z2 -> a1
                // activation function a1(z1) = tanh(z1) = (e^z1-e^-z1)/(e^z+e^-z) <- from tanh definition
                // ∂z2/∂a1 = w2
                // m * ∂J/∂a1 = ∂z2/∂a1 * ∂J/∂z2 = w2 * mdz2

                // J -> L -> a2 -> z2 -> a1 -> z1
                // z1(x,w1,b1) = b1 + w1 * x
                // ∂a1/∂z1 = 1 - a1² <- from calculus
                // mdz1 = m * ∂J/∂z1 = ∂a1/∂z1 * ∂J/∂a1 = (1 - a1²) * w2 * mdz2
                let mdz1 = self.w2.permute::<_, Axes2<1, 0>>().dot(mdz2)
                    * (cache.a1.square().negate() + 1.0);

                // J -> L -> a2 -> z2 -> a1 -> z1 -> w1
                // ∂z1/∂w1 = x
                // dw1 = ∂J/∂w1 = ∂z1/∂w1 * ∂J/∂z1 / m = x * mdz1 / m
                let dw1 = mdz1.clone().dot(x.permute::<_, Axes2<1, 0>>()) / (SETLEN as f32);

                // J -> L -> a2 -> z2 -> a1 -> z1 -> b1
                // ∂z1/∂b1 = 1
                // db1 = ∂J/∂b1 = ∂z1/∂b1 * ∂J/∂z1 / m = mdz1 / m
                let db1 = mdz1.sum::<Rank1<L1LEN>, _>() / (SETLEN as f32);

                Gradient {
                    dw1,
                    db1: db1.reshape::<Rank2<L1LEN, 1>>(),
                    dw2,
                    db2: db2.reshape::<Rank2<L2LEN, 1>>(),
                }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Gradient<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> {
            /// Layer 1/2. Rows for nodes, columns for features.
            pub dw1: TensorF32<Rank2<L1LEN, XLEN>>,
            /// Layer 1/2. Rows for nodes.
            pub db1: TensorF32<Rank2<L1LEN, 1>>,
            /// Layer 2/2. Rows for nodes, columns for features.
            pub dw2: TensorF32<Rank2<L2LEN, L1LEN>>,
            /// Layer 2/2. Rows for nodes.
            pub db2: TensorF32<Rank2<L2LEN, 1>>,
        }

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Gradient<XLEN, L1LEN, L2LEN> {
            pub fn with(
                dw1: TensorF32<Rank2<L1LEN, XLEN>>,
                db1: TensorF32<Rank2<L1LEN, 1>>,
                dw2: TensorF32<Rank2<L2LEN, L1LEN>>,
                db2: TensorF32<Rank2<L2LEN, 1>>,
            ) -> Self {
                Self { dw1, db1, dw2, db2 }
            }
            pub fn from_values(
                device: &Device,
                dw1: [[f32; XLEN]; L1LEN],
                db1: [[f32; 1]; L1LEN],
                dw2: [[f32; L1LEN]; L2LEN],
                db2: [[f32; 1]; L2LEN],
            ) -> Self {
                let w1: TensorF32<Rank2<L1LEN, XLEN>> = device.tensor(dw1);
                let b1: TensorF32<Rank2<L1LEN, 1>> = device.tensor(db1);
                let w2: TensorF32<Rank2<L2LEN, L1LEN>> = device.tensor(dw2);
                let b2: TensorF32<Rank2<L2LEN, 1>> = device.tensor(db2);
                Self::with(w1, b1, w2, b2)
            }
        }

        #[test]
        fn test_backward() {
            let dev = device();
            let x = example_x(&dev);
            let y = example_y(&dev);
            let model = example_model(&dev);
            let cache = model.clone().forward(x.clone());
            let grads = model.backward::<3>(cache, x, y);

            assert!(grads.dw1.array().approx(
                [
                    [0.0029611567, -0.0073388093],
                    [0.011364542, -0.028199548],
                    [-0.0045674066, 0.011302374],
                    [-0.012589335, 0.031008393]
                ],
                (1e-7, 0)
            ),);
            assert!(grads.db1.array().approx(
                [
                    [0.0017263684],
                    [0.006616226],
                    [-0.0026577536],
                    [-0.007254278]
                ],
                (1e-7, 0)
            ));
            assert!(grads.dw2.array().approx(
                [[0.01364471, 0.0286189, -0.0075288084, -0.037893888]],
                (1e-7, 0)
            ));
            assert!(grads.db2.array().approx([[-0.16577528]], (1e-8, 0)));
        }

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn update(
                mut self,
                gradient: Gradient<XLEN, L1LEN, L2LEN>,
                learning_rate: f32,
            ) -> Self {
                self.w1 = self.w1 - gradient.dw1 * learning_rate;
                self.b1 = self.b1 - gradient.db1 * learning_rate;
                self.w2 = self.w2 - gradient.dw2 * learning_rate;
                self.b2 = self.b2 - gradient.db2 * learning_rate;
                self
            }
        }

        #[test]
        fn test_update() {
            let dev = device();
            let x = example_x(&dev);
            let y = example_y(&dev);
            let mut model = example_model(&dev);
            let cache = model.clone().forward(x.clone());
            let grads = model.clone().backward(cache, x, y);
            model = model.update(grads, 1.2);

            assert!(model.w1.array().approx(
                [
                    [-0.009703778, 0.025708672],
                    [-0.03675537, 0.06521067],
                    [-0.011440813, -0.0310883],
                    [0.024461564, -0.087392285]
                ],
                (1e-8, 0)
            ),);
            assert!(model.b1.array().approx(
                [
                    [-0.0020725399],
                    [-0.007931316],
                    [0.0031899095],
                    [0.008702588]
                ],
                (1e-8, 0)
            ),);
            assert!(model.w2.array().approx(
                [[-0.026805554, -0.07453275, 0.02510668, 0.08987522]],
                (1e-7, 0)
            ),);
            assert!(model.b2.array().approx([[0.19902185]], (1e-8, 0)),);
        }
    }
    pub use _3::{Cache, Gradient};

    /// C01W03PA01 Section 4 - Integrate Previous Parts in nn_model().
    pub mod _4 {
        use super::*;

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn optimize<const SETLEN: usize>(
                mut self,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
                y: TensorF32<Rank2<L2LEN, SETLEN>>,
                num_iterations: usize,
                learning_rate: f32,
                imod: usize,
            ) -> Self {
                for i in 0..num_iterations {
                    let cache = self.clone().forward(x.clone());
                    let grads = self.clone().backward(cache.clone(), x.clone(), y.clone());
                    self = self.update(grads, learning_rate);
                    if i % imod == 0 {
                        let cost = cache.cost(y.clone());
                        println!("cost after iteration {}: {}", i, cost);
                    }
                }
                self
            }
        }

        #[test]
        fn test_nn_model() {
            let dev = device();
            let x = example_x(&dev);
            let y = example_y(&dev);
            let mut model = example_model(&dev);
            model = model.optimize(x, y, 10_000, 1.02, 1000);

            assert!(model.w1.array().approx(
                [
                    [-0.63768107, 1.1865728],
                    [-0.74364954, 1.3640525],
                    [0.61709785, -1.1710562],
                    [0.7531452, -1.3976556]
                ],
                (1e-6, 0)
            ),);
            assert!(model.b1.array().approx(
                [[0.26729938], [0.3290354], [-0.26096892], [-0.34003872]],
                (1e-6, 0)
            ),);
            assert!(model
                .w2
                .array()
                .approx([[-2.323175, -3.111068, 2.2466047, 3.262285]], (1e-6, 0)),);
            assert!(model.b2.array().approx([[0.18929027]], (1e-6, 0)),);
        }
    }

    /// C01W03PA01 Section 5 - Predictions.
    pub mod _5 {
        use super::*;

        impl<const XLEN: usize, const L1LEN: usize, const L2LEN: usize> Model<XLEN, L1LEN, L2LEN> {
            pub fn predict<const SETLEN: usize>(
                self,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
            ) -> TensorF32<Rank2<L2LEN, SETLEN>> {
                let cache = self.forward(x.clone());
                let yhat = cache.a2;
                let mask = yhat.ge(0.5);
                let zeros: TensorF32<Rank2<L2LEN, SETLEN>> = mask.device().zeros();
                let ones: TensorF32<Rank2<L2LEN, SETLEN>> = mask.device().ones();
                mask.choose(ones, zeros)
            }
        }

        #[test]
        fn test_prediction() {
            let dev = device();
            let x = example_x(&dev);
            let model = example_model(&dev);
            let prediction = model.predict(x);

            assert_eq!(prediction.mean::<Rank0, _>().array(), 0.6666667);
        }

        #[cfg(feature = "cuda")]
        pub fn accuracy<YD, const L2LEN: usize, const SETLEN: usize>(
            prediction: TensorF32<Rank2<L2LEN, SETLEN>>,
            y: Tensor<Rank2<L2LEN, SETLEN>, YD, Device>,
        ) -> TensorF32<Rank1<L2LEN>>
        where
            YD: dfdx::dtypes::Unit,
            YD: candle_core::cuda_backend::cudarc::types::CudaTypeName,
        {
            (prediction - y.to_dtype::<f32>())
                .abs()
                .mean::<Rank1<L2LEN>, _>()
                .negate()
                * 100.
                + 100.
        }

        #[cfg(not(feature = "cuda"))]
        pub fn accuracy<YD, const L2LEN: usize, const SETLEN: usize>(
            prediction: TensorF32<Rank2<L2LEN, SETLEN>>,
            y: Tensor<Rank2<L2LEN, SETLEN>, YD, Device>,
        ) -> TensorF32<Rank1<L2LEN>>
        where
            YD: dfdx::dtypes::Unit,
            YD: num_traits::cast::AsPrimitive<f32>,
        {
            (prediction - y.to_dtype::<f32>())
                .abs()
                .mean::<Rank1<L2LEN>, _>()
                .negate()
                * 100.
                + 100.
        }

        #[test]
        fn test_planar() {
            use crate::helpers::Approx;

            let dev = device();
            let x = dev.tensor(crate::c1::w3::util::X);
            let y = dev.tensor(crate::c1::w3::util::Y).to_dtype::<f32>();
            let mut model = example_model(&dev);
            model = model.optimize(x.clone(), y.clone(), 10_000, 1.2, 1000);
            let cache = model.clone().forward(x.clone());
            let cost = cache.cost(y.clone());

            // test cost
            // (non-deterministic)
            assert!(cost.approx(0.21835, (0.001, 0)));

            // test accuracy
            let prediction = model.predict(x);
            let accuracy = accuracy(prediction, y);
            // (non-deterministic)
            assert!(accuracy.array().approx([90.5], (5.0, 0)));
        }
    }
    pub use _5::accuracy;

    /// C01W03PA01 Section 6 - Tuning Hidden Layer Size.
    pub mod _6 {
        use super::*;

        /// Creates one or more models with weights from a normal distribution.
        #[allow(unused_macros)]
        macro_rules! models {
            // creates a tuple of models (2, l1size, 1), given a list of l1sizes
            ([$($l1size:literal),*], $dev:expr) => {
                ($(models!($l1size, $dev)),*)
            };
            // creates a single model (2, l1size, 1) given a l1size
            ($l1size:literal, $dev:expr) => {
                crate::c1::w3::pa_01_planar_data_classification::Model::<2, $l1size, 1>::normal(
                    $dev,
                )
            };
        }

        /// Optimizes a model and then check it's accuracy.
        fn optimize_then_accuracy<
            const XLEN: usize,
            const L1LEN: usize,
            const L2LEN: usize,
            const SETLEN: usize,
        >(
            model: Model<XLEN, L1LEN, L2LEN>,
            x: &TensorF32<Rank2<XLEN, SETLEN>>,
            y: &TensorF32<Rank2<L2LEN, SETLEN>>,
        ) -> [f32; L2LEN] {
            let prediction = model
                .optimize(x.clone(), y.clone(), 5000, 1.2, 5000)
                .predict(x.clone());
            accuracy(prediction, y.clone()).array()
        }

        #[test]
        fn test_planar_various() {
            use crate::helpers::Approx;

            let dev = device();
            let x = dev.tensor(crate::c1::w3::util::X);
            let y = dev.tensor(crate::c1::w3::util::Y).to_dtype::<f32>();
            let models = models!([1, 2, 3, 4, 5, 20, 50], &dev);

            assert!(optimize_then_accuracy(models.0, &x, &y)[0].approx(65.0, (10., 0)));
            assert!(optimize_then_accuracy(models.1, &x, &y)[0].approx(65.0, (10., 0)));
            assert!(optimize_then_accuracy(models.2, &x, &y)[0].approx(90.0, (10., 0)));
            assert!(optimize_then_accuracy(models.3, &x, &y)[0].approx(90.0, (10., 0)));
            assert!(optimize_then_accuracy(models.4, &x, &y)[0].approx(91.0, (10., 0)));
            assert!(optimize_then_accuracy(models.5, &x, &y)[0].approx(91.75, (10., 0)));
            assert!(optimize_then_accuracy(models.6, &x, &y)[0].approx(91.75, (10., 0)));
        }
    }
}
pub use _4::{accuracy, Cache, Gradient, Model};

/// C01W03PA01 Part 5 - Performance on Other Datasets.
mod _5 {
    // no content
}

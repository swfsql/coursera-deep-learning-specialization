//! Logistic Regression with a Neural Network Mindset
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic_Regression_with_a_Neural_Network_mindset_v6a.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Neural%20Networks%20and%20Deep%20Learning/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset.ipynb
//! The alternative reference contains less scripting errors.
//!
//! I recommend first watching all of C1W2.

use crate::helpers::_dfdx::*;

/// C01W02PA02 Part 1 - Packages.
mod _1 {
    // no content
}

/// C01W02PA02 Part 2 - Overview of the Problem Set.
pub mod _2 {
    use super::*;

    const TRAIN_FILE: &str =
    "/workspaces/coursera-deep-learning-specialization/r/src/c1/w2/datasets/train_catvnoncat.h5";
    const TRAIN_X: &str = "train_set_x";
    const TRAIN_Y: &str = "train_set_y";
    const TRAIN_CLASSES: &str = "list_classes";
    const TEST_FILE: &str =
        "/workspaces/coursera-deep-learning-specialization/r/src/c1/w2/datasets/test_catvnoncat.h5";
    const TEST_X: &str = "test_set_x";
    const TEST_Y: &str = "test_set_y";

    /// Number of images used for training.
    pub const M_TRAIN: usize = 209;

    /// Number of images used for testing.
    pub const M_TEST: usize = 50;

    /// Squared image dimension (for each width and height).
    pub const NUM_PX: usize = 64;

    /// Number of colorpoints for each image.
    pub const NUM_COLORPOINTS: usize = NUM_PX * NUM_PX * 3;

    fn get_classes(
        file: &hdf5::File,
    ) -> anyhow::Result<ndarray::Array1<hdf5::types::FixedAscii<7>>> {
        assert_eq!(
            file.dataset(TRAIN_CLASSES)?.dtype()?.to_descriptor()?,
            hdf5::types::TypeDescriptor::FixedAscii(7)
        );
        let classes = file
            .dataset(TRAIN_CLASSES)?
            .read::<hdf5::types::FixedAscii<7>, ndarray::Ix1>()?;
        assert_eq!(classes.shape(), [2]);

        Ok(classes)
    }

    pub fn prepare_data() -> anyhow::Result<(
        PreparedData<NUM_COLORPOINTS, M_TRAIN>,
        PreparedData<NUM_COLORPOINTS, M_TEST>,
        ndarray::Array1<hdf5::types::FixedAscii<7>>,
    )> {
        let dev = device();

        let train_file = hdf5::File::open(TRAIN_FILE)?;
        let train_x_orig =
            dev.tensor_from_hdf5::<u8, Rank4<M_TRAIN, NUM_PX, NUM_PX, 3>>(&train_file, TRAIN_X)?;
        let train_y_orig = dev
            .tensor_from_hdf5::<u8, Rank1<M_TRAIN>>(&train_file, TRAIN_Y)?
            .reshape::<Rank2<1, M_TRAIN>>();
        let classes = get_classes(&train_file)?;
        drop(train_file);

        let test_file = hdf5::File::open(TEST_FILE)?;
        let test_x_orig =
            dev.tensor_from_hdf5::<u8, Rank4<M_TEST, NUM_PX, NUM_PX, 3>>(&test_file, TEST_X)?;
        let test_y_orig = dev
            .tensor_from_hdf5::<u8, Rank1<M_TEST>>(&test_file, TEST_Y)?
            .reshape::<Rank2<1, M_TEST>>();
        drop(test_file);

        // check class for 10th image
        assert_eq!(
            train_y_orig
                .clone()
                .slice((0..1, 9..10))
                .realize::<Rank2<1, 1>>()
                .array()[0][0],
            0
        );
        assert_eq!(classes[0], "non-cat");

        // reshapes
        // note: the parent python code forgot to apply the transmute, but the correct expected values are on the "expected output" table.
        // so it's recommended to fix the python code and rerun that notebook.
        // https://github.com/amanchadha/coursera-deep-learning-specialization/issues/23
        let train_x_flatten = train_x_orig
            .reshape::<Rank2<M_TRAIN, NUM_COLORPOINTS>>()
            .permute::<_, Axes2<1, 0>>();
        let test_x_flatten = test_x_orig
            .reshape::<Rank2<M_TEST, NUM_COLORPOINTS>>()
            .permute::<_, Axes2<1, 0>>();
        //
        // sanity check (first 5 color points (R,G,B,R,G) from the first row of the first image)
        assert_eq!(
            train_x_flatten
                .clone()
                .slice((0..5, 0..1))
                .realize::<Rank2<5, 1>>()
                .reshape::<Rank1<5>>()
                .array(),
            [17, 31, 56, 22, 33]
        );

        // standardize the colorpoint values
        let train_set_x = train_x_flatten.to_dtype::<f32>() / 255f32;
        let test_set_x = test_x_flatten.to_dtype::<f32>() / 255f32;

        let train = PreparedData {
            x: train_set_x,
            y: train_y_orig,
        };
        let test = PreparedData {
            x: test_set_x,
            y: test_y_orig,
        };

        Ok((train, test, classes))
    }

    #[derive(Clone)]
    pub struct PreparedData<const XLEN: usize, const SETLEN: usize> {
        pub x: TensorF32<Rank2<XLEN, SETLEN>>,
        pub y: TensorU8<Rank2<1, SETLEN>>,
    }

    #[test]
    fn test_prepared_data() -> anyhow::Result<()> {
        prepare_data()?;
        Ok(())
    }
}
pub use _2::PreparedData;

/// C01W02PA02 Part 3 - General Archtecture of the Learning Algorithm.
mod _3 {
    // no content
}

/// C01W02PA02 Part 4 - Building the Parts of Our Algorithm.
pub mod _4 {
    use super::*;
    use crate::c1::w2::pa_01_basics::sigmoid;

    /// C01W02PA02 Part 4 Section 1 - Helper Functions.
    pub mod _1 {
        #[test]
        fn test_sigmoid() {
            use super::*;
            let dev = device();
            let x: TensorF32<Rank1<2>> = dev.tensor([0., 2.]);
            assert_eq!(sigmoid(x).array(), [0.5, 0.880797]);
        }
    }

    /// C01W02PA02 Part 4 Section 2 - Initializing Parameters.
    pub mod _2 {
        use super::*;

        #[derive(Clone, Debug)]
        pub struct Model<const XLEN: usize> {
            /// Rows for features.
            pub w: TensorF32<Rank2<XLEN, 1>>,
            pub b: TensorF32<Rank1<1>>,
        }

        impl<const XLEN: usize> Model<XLEN> {
            pub fn new(device: &Device) -> Self {
                Self::with(device.tensor([[0.; 1]; XLEN]), device.tensor([0.]))
            }

            pub fn with(w: TensorF32<Rank2<XLEN, 1>>, b: TensorF32<Rank1<1>>) -> Self {
                Self { w, b }
            }
        }
    }
    pub use _2::Model;

    /// C01W02PA02 Part 4 Section 3 - Foward and Backward Propagation.
    pub mod _3 {
        use super::*;

        impl<const XLEN: usize> Model<XLEN> {
            /// Logistic regression, returns ŷ.
            #[allow(clippy::let_and_return)]
            pub fn f<const SETLEN: usize>(
                &self,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
            ) -> TensorF32<Rank2<1, SETLEN>> {
                // forward propagation

                // (inner) linear regression = wt*x + b
                let z =
                        // wt
                        self.w.clone().permute::<_, Axes2<1, 0>>()
                        //
                        .dot(x.clone())
                        + self.b.clone().broadcast::<Rank2<1, SETLEN>, _>();

                // logistic regression f, sigmoid activation over a linear regression -> f = g(z) = sigmoid(z) -> ŷ = a
                let yhat = sigmoid(z.reshape::<Rank1<SETLEN>>()).reshape::<Rank2<1, SETLEN>>();

                yhat
            }

            #[allow(clippy::let_and_return)]
            pub fn cost<const SETLEN: usize>(
                y: TensorF32<Rank2<1, SETLEN>>,
                yhat: TensorF32<Rank2<1, SETLEN>>,
            ) -> TensorF32<Rank1<1>> {
                // loss function L = -(y*log(ŷ) + (1-y)log(1-ŷ))
                let l1 = y.clone() * yhat.clone().ln();
                let l2 = y.clone().negate() + 1.;
                let l3 = (yhat.clone().negate() + 1.).ln();
                let l = (l1 + l2 * l3).negate();

                // cost function J = 1/m sum (L) -> ŷ
                let j = l.sum::<Rank1<1>, _>() / (SETLEN as f32);
                j
            }

            /// Calculates the cost function and its gradient.
            pub fn propagate<const SETLEN: usize>(
                &self,
                data: PreparedData<XLEN, SETLEN>,
            ) -> Propagation<XLEN> {
                // forward propagation

                let y = data.y.to_dtype::<f32>();
                let yhat = self.f(data.x.clone());
                let cost = Self::cost(y.clone(), yhat.clone());

                // backward propagation

                // loss gradient dj/dw = 1/m x(ŷ-y)t
                // note: when writing in terms of ŷ, this is the same loss gradient function
                // as to linear regression
                let dw = {
                    let diff = yhat.clone() - y.clone();
                    let difft = diff.permute::<_, Axes2<1, 0>>();
                    data.x.dot(difft) / (SETLEN as f32)
                };
                // loss gradient dj/db = 1/m sum (ŷ-y)t
                // note: when writing in terms of ŷ, this is the same loss gradient function
                // as to linear regression
                let db = {
                    let diff = yhat.clone() - y.clone();
                    let difft = diff.permute::<_, Axes2<1, 0>>();
                    difft.sum::<Rank1<1>, _>() / (SETLEN as f32)
                };
                Propagation { cost, dw, db }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Propagation<const XLEN: usize> {
            pub cost: TensorF32<Rank1<1>>,
            pub dw: TensorF32<Rank2<XLEN, 1>>,
            pub db: TensorF32<Rank1<1>>,
        }

        pub const W: [[f32; 1]; 2] = [[1.], [2.]];
        pub const B: [f32; 1] = [2.];
        pub const X: [[f32; 3]; 2] = [[1., 2., -1.], [3., 4., -3.2]];
        pub const Y: [[u8; 3]; 1] = [[1, 0, 1]];

        #[test]
        fn test_propagation() {
            let dev = device();

            let w: TensorF32<Rank2<2, 1>> = dev.tensor(W);
            let b: TensorF32<Rank1<1>> = dev.tensor(B);
            let model = Model::with(w, b);
            let x: TensorF32<Rank2<2, 3>> = dev.tensor(X);
            let y: TensorU8<Rank2<1, 3>> = dev.tensor(Y);

            let train = PreparedData { x, y };
            let propagation = model.propagate(train);

            assert_eq!(propagation.cost.array()[0], 5.79859);
            assert_eq!(
                propagation.dw.reshape::<Rank1<2>>().array(),
                [0.998456, 2.3950722]
            );
            assert_eq!(propagation.db.array(), [0.0014555653]);
        }
    }
    pub use _3::Propagation;

    /// C01W02PA02 Part 4 Section 4 - Optimization.
    pub mod _4 {
        use super::*;

        use super::{Model, PreparedData, Propagation};
        #[allow(unused_imports)]
        use crate::helpers::{Approx, _dfdx::*};

        impl<const XLEN: usize> Model<XLEN> {
            pub fn optimize<const SETLEN: usize>(
                mut self,
                data: PreparedData<XLEN, SETLEN>,
                num_iterations: usize,
                learning_rate: f32,
                imod: usize,
            ) -> Optimization<XLEN> {
                let mut last_propagation = None;
                let mut costs = vec![];
                for i in 0..num_iterations {
                    let propagation = self.propagate(data.clone());
                    self.w = self.w - propagation.dw.clone() * learning_rate;
                    self.b = self.b - propagation.db.clone() * learning_rate;
                    if i % imod == 0 {
                        costs.push(propagation.cost.array()[0]);
                        if i != 0 {
                            println!(
                                "cost after iteration {}: {}",
                                i,
                                propagation.cost.array()[0]
                            );
                        }
                    }
                    last_propagation = Some(propagation);
                }

                Optimization {
                    model: self,
                    last_propagation,
                    costs,
                }
            }
        }

        #[derive(Clone, Debug)]
        pub struct Optimization<const XLEN: usize> {
            pub model: Model<XLEN>,
            pub last_propagation: Option<Propagation<XLEN>>,
            pub costs: Vec<f32>,
        }

        #[test]
        fn test_optimization() {
            use super::_3::*;
            let dev = device();

            let w: TensorF32<Rank2<2, 1>> = dev.tensor(W);
            let b: TensorF32<Rank1<1>> = dev.tensor(B);
            let model = Model::with(w, b);
            let x: TensorF32<Rank2<2, 3>> = dev.tensor(X);
            let y: TensorU8<Rank2<1, 3>> = dev.tensor(Y);
            let train = PreparedData { x, y };

            let opt = model.optimize(train, 100, 9e-3, 100);

            assert!(opt
                .model
                .w
                .reshape::<Rank1<2>>()
                .array()
                .approx([0.19033602, 0.12259145], (1e-7, 0)));
            assert!(opt.model.b.array().approx([1.9253595], (1e-7, 0)));
            assert!(opt
                .last_propagation
                .clone()
                .unwrap()
                .dw
                .reshape::<Rank1<2>>()
                .array()
                .approx([0.67752033, 1.4162549], (1e-7, 0)));
            assert!(opt
                .last_propagation
                .as_ref()
                .unwrap()
                .db
                .array()
                .approx([0.21919446], (1e-7, 0)));
        }

        impl<const XLEN: usize> Model<XLEN> {
            pub fn decide_on_prediction<const SETLEN: usize>(
                &self,
                x: TensorF32<Rank2<XLEN, SETLEN>>,
            ) -> TensorF32<Rank2<1, SETLEN>> {
                let yhat = self.f(x.clone());
                let mask = yhat.ge(0.5);
                let zeros: TensorF32<Rank2<1, SETLEN>> = mask.device().zeros();
                let ones: TensorF32<Rank2<1, SETLEN>> = mask.device().ones();
                mask.choose(ones, zeros)
            }
        }

        #[test]
        fn test_decide_on_prediction() {
            use super::*;
            let dev = device();

            let w: TensorF32<Rank2<2, 1>> = dev.tensor([[0.1124579], [0.23106775]]);
            let b: TensorF32<Rank1<1>> = dev.tensor([-0.3]);
            let model = Model::with(w, b);
            let x: TensorF32<Rank2<2, 3>> = dev.tensor([[1., -1.1, -3.2], [1.2, 2., 0.1]]);

            let prediction = model.decide_on_prediction(x);

            assert_eq!(prediction.reshape::<Rank1<3>>().array(), [1., 1., 0.]);
        }
    }
    pub use _4::Optimization;
}
pub use _4::{Model, Optimization, Propagation};

/// C01W02PA02 Part 5 - Merge All Functions Into a Model.
pub mod _5 {
    use super::_2::*;
    use super::*;

    #[allow(unused_imports)]
    use crate::helpers::{Approx, _dfdx::*};

    impl<const XLEN: usize> Model<XLEN> {
        pub fn train<const TRAINLEN: usize, const TESTLEN: usize>(
            self,
            train: PreparedData<XLEN, TRAINLEN>,
            test: PreparedData<XLEN, TESTLEN>,
            num_iterations: usize,
            learning_rate: f32,
            imod: usize,
        ) -> anyhow::Result<TrainedModel<XLEN, TRAINLEN, TESTLEN>> {
            let opt = self.optimize(train.clone(), num_iterations, learning_rate, imod);
            let train_predict = opt.model.decide_on_prediction(train.x);
            let test_predict = opt.model.decide_on_prediction(test.x);
            Ok(TrainedModel {
                model: opt.model,
                train_predict,
                test_predict,
                costs: opt.costs,
                num_iterations,
                learning_rate,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct TrainedModel<const XLEN: usize, const TRAINLEN: usize, const TESTLEN: usize> {
        pub model: Model<XLEN>,
        pub train_predict: TensorF32<Rank2<1, TRAINLEN>>,
        pub test_predict: TensorF32<Rank2<1, TESTLEN>>,
        pub costs: Vec<f32>,
        pub num_iterations: usize,
        pub learning_rate: f32,
    }

    #[cfg(feature = "cuda")]
    pub fn accuracy<YD, const YLEN: usize>(
        prediction: TensorF32<Rank2<1, YLEN>>,
        y: Tensor<Rank2<1, YLEN>, YD, Device>,
    ) -> TensorF32<Rank1<1>>
    where
        YD: dfdx::dtypes::Unit,
        YD: candle_core::cuda_backend::cudarc::types::CudaTypeName,
    {
        (prediction - y.to_dtype::<f32>())
            .abs()
            .mean::<Rank1<1>, _>()
            .negate()
            * 100.
            + 100.
    }

    #[cfg(not(feature = "cuda"))]
    pub fn accuracy<YD, const YLEN: usize>(
        prediction: TensorF32<Rank2<1, YLEN>>,
        y: Tensor<Rank2<1, YLEN>, YD, Device>,
    ) -> TensorF32<Rank1<1>>
    where
        YD: dfdx::dtypes::Unit,
        YD: num_traits::cast::AsPrimitive<f32>,
    {
        (prediction - y.to_dtype::<f32>())
            .abs()
            .mean::<Rank1<1>, _>()
            .negate()
            * 100.
            + 100.
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_train_model() -> anyhow::Result<()> {
        let dev = device();
        let (train, test, _classes) = prepare_data()?;
        let model = Model::<NUM_COLORPOINTS>::new(&dev);
        let model = model.train(train.clone(), test.clone(), 2000, 5e-3, 100)?;
        let train_accuracy = accuracy(model.train_predict, train.y);
        let test_accuracy = accuracy(model.test_predict, test.y);

        assert!(model.costs[0].approx(0.6931472, (1e-5, 0)));
        assert!(train_accuracy.array()[0].approx(99.04306, (1., 0)));
        assert!(test_accuracy.array()[0].approx(70., (2., 0)));

        Ok(())
    }
}
pub use _5::{accuracy, TrainedModel};

/// C01W02PA02 Part 6 - Further Analysis.
pub mod _6 {

    #[test]
    fn test_train_model_varying_lr() -> anyhow::Result<()> {
        use super::_2::*;
        use super::*;

        let dev = device();

        let (train, test, _classes) = prepare_data()?;
        let model = Model::<NUM_COLORPOINTS>::new(&dev);

        let mut res = vec![];
        for lr in [1e-2, 1e-3, 1e-4] {
            let model = model
                .clone()
                .train(train.clone(), test.clone(), 1500, lr, usize::MAX)?;
            let train_accuracy = accuracy(model.train_predict, train.y.clone());
            let test_accuracy = accuracy(model.test_predict, test.y.clone());
            res.push((train_accuracy.array()[0], test_accuracy.array()[0]));
        }

        assert_eq!(
            res.as_slice(),
            &[(99.52153, 68.0), (88.99522, 64.0), (68.42105, 36.0)]
        );

        Ok(())
    }
}

/// C01W02PA02 Part 7 - Test With Your Own Image.
mod _7 {
    // no content
}

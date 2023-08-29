//! Deep Neural Network for Image CLassification: Application
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v8.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Neural%20Networks%20and%20Deep%20Learning/Deep%20Neural%20Network%20-%20Application.ipynb
//!
//! I recommend first watching all of C1W4.

#![allow(unused_imports)]

use crate::c1::w4::pa_01_deep_nn::*;
use crate::helpers::{Approx, _dfdx::*};

/// C01W04PA02 Part 1 - Packages.
mod _1 {
    // no content.
}

/// C01W04PA02 Part 2 - Dataset.
mod _2 {
    pub use crate::c1::w2::pa_02_logistic_regression::_2::{
        prepare_data, PreparedData, M_TEST, M_TRAIN, NUM_COLORPOINTS, NUM_PX,
    };

    // note: already demonstrated on c1::w2::pa_02_logistic_regression::_2
}
use _2::{prepare_data, PreparedData, M_TEST, M_TRAIN, NUM_COLORPOINTS, NUM_PX};

/// C01W04PA02 Part 3 - Architecture.
mod _3 {

    /// C01W04PA02 Section 1 - 2-Layer nn.
    mod _1 {
        // no content.
    }

    /// C01W04PA02 Section 2 - L-Layer nn.
    mod _2 {
        // no content.
    }

    /// C01W04PA02 Section 3 - General Mathodology.
    mod _3 {
        // no content.
    }
}

/// C01W04PA02 Part 4 - 2-Layer nn.
pub mod _4 {
    use super::*;

    pub trait Layers<const X_FEATURES: usize, const Y_FEATURES: usize, const SETLEN: usize>
    where
        Self: Clone
            + Downward<
                SETLEN,
                Input = X<X_FEATURES, SETLEN>,
                Output = (
                    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Cache,
                    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::LowerCaches,
                ),
            > + DownUpGrads<Y_FEATURES, SETLEN, X = Cache<X_FEATURES, SETLEN>>,
        (
            <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Cache,
            <Self as DownUpGrads<Y_FEATURES, SETLEN>>::LowerCaches,
        ): crate::c1::w4::pa_01_deep_nn::_6::_3::LastA<Last = A<Y_FEATURES, SETLEN>>,
        // allows the weights and biases to be updated
        Self: UpdateParameters<
            Grads = <<Self as DownUpGrads<Y_FEATURES, SETLEN>>::Output as CleanupGrads>::Output,
        >,
        // removes the up_mda info
        <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Output: CleanupGrads,
    {
        fn train<CostType>(
            mut self,
            train_x: X<X_FEATURES, SETLEN>,
            train_y: A<Y_FEATURES, SETLEN>,
            cost_setup: &mut CostType,
            num_iterations: usize,
            print_cost_mod: usize,
        ) -> Self
        where
            CostType: Clone
                + CostSetup
                + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>,
        {
            for i in 0..num_iterations {
                // reset cross-training step information from the cost_setup
                cost_setup.new_train_step();

                let caches = self.clone().downward(train_x.clone(), cost_setup);
                let cost = cost_setup
                    .cost(train_y.clone(), caches.last_a().clone())
                    .array();
                if i % print_cost_mod == 0 {
                    println!("Cost after iteration {}: {}", i, cost);
                }

                // reset cost_setup because a downward pass had already happened for the caches,
                // but a new downward pass will happen for the gradients
                cost_setup.new_train_step();

                let wrap_caches = (Cache::from_a(train_x.clone()), caches);
                let grads = self
                    .clone()
                    .gradients(train_y.clone(), cost_setup, wrap_caches);
                let grads = grads.remove_mdas();

                self = self.update_params(grads, cost_setup);
            }

            self
        }

        fn cost<CostType>(
            self,
            train_x: X<X_FEATURES, SETLEN>,
            train_y: Y<Y_FEATURES, SETLEN>,
            cost_setup: &mut CostType,
        ) -> f32
        where
            CostType: Clone
                + CostSetup
                + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>,
        {
            let caches = self.downward(train_x, cost_setup);
            cost_setup.cost(train_y, caches.last_a().clone()).array()
        }

        fn predict<CostType>(
            self,
            train_x: X<X_FEATURES, SETLEN>,
            cost_setup: &mut CostType,
        ) -> A<Y_FEATURES, SETLEN>
        where
            CostType: Clone
                + CostSetup
                + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>,
        {
            let caches = self.downward(train_x, cost_setup);
            caches.last_a().clone()
        }
    }

    impl<L, const X_FEATURES: usize, const Y_FEATURES: usize, const SETLEN: usize>
        Layers<X_FEATURES, Y_FEATURES, SETLEN> for L
    where
        L: Clone
            + Downward<
                SETLEN,
                Input = X<X_FEATURES, SETLEN>,
                Output = (
                    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Cache,
                    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::LowerCaches,
                ),
            > + DownUpGrads<Y_FEATURES, SETLEN, X = Cache<X_FEATURES, SETLEN>>,
        (
            <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Cache,
            <Self as DownUpGrads<Y_FEATURES, SETLEN>>::LowerCaches,
        ): crate::c1::w4::pa_01_deep_nn::_6::_3::LastA<Last = A<Y_FEATURES, SETLEN>>,
        L: UpdateParameters<
            Grads = <<L as DownUpGrads<Y_FEATURES, SETLEN>>::Output as CleanupGrads>::Output,
        >,
        <L as DownUpGrads<Y_FEATURES, SETLEN>>::Output: CleanupGrads,
    {
    }

    #[test]
    fn two_layer_model() -> anyhow::Result<()> {
        use super::*;

        let dev = &device();
        let (train, test, _classes) = prepare_data()?;
        let train_x = train.x;
        let train_y = train.y.to_dtype::<f32>();

        // note: the result values are different from the python ones because
        // the initial weights are different.
        // I have manually checked that they match if the initial values are the same.
        let layers = crate::layerc1!(dev, 1e-1, [12288, 7, 1]);
        let mut cost_setup = MLogistical::new(0.0075);
        let layers = layers.train(train_x.clone(), train_y.clone(), &mut cost_setup, 2500, 100);
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            .approx(0.041310042, (5e-4, 0)));

        // train accuracy
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert_eq!(accuracy.array(), [100.]);
        }

        // test accuracy
        {
            let test_x = test.x;
            let test_y = test.y.to_dtype::<f32>();
            let yhat = layers.clone().predict(test_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert_eq!(accuracy.array(), [72.]);
        }

        Ok(())
    }
}
pub use _4::Layers;

/// C01W04PA02 Part 5 - L-Layer nn.
pub mod _5 {
    #[test]
    fn four_layer_model() -> anyhow::Result<()> {
        use super::*;

        let dev = &device();
        let (train, test, _classes) = prepare_data()?;
        let train_x = train.x;
        let train_y = train.y.to_dtype::<f32>();

        // note: the result values are different from the python ones because
        // the initial weights are different.
        // I have manually checked that they match if the initial values are the same.
        let layers = crate::layerc1!(dev, 1., [12288, 20, 7, 5, 1]);
        let mut cost_setup = MLogistical::new(0.0075);
        let layers = layers.train(train_x.clone(), train_y.clone(), &mut cost_setup, 2500, 100);
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            .approx(0.003624782, (4e-3, 0)));

        // train accuracy
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert_eq!(accuracy.array(), [100.]);
        }

        // test accuracy
        {
            let test_x = test.x;
            let test_y = test.y.to_dtype::<f32>();
            let yhat = layers.clone().predict(test_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert!(accuracy.array().approx([78.], (2., 0)));
        }

        Ok(())
    }
}

/// C01W04PA02 Part 6 - Result Analysis.
pub mod _6 {
    // no content.
}

/// C01W04PA02 Part 7 - Optional.
pub mod _7 {
    // no content.
}

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

    #[test]
    fn two_layer_model() -> anyhow::Result<()> {
        use super::*;

        let dev = &device();
        let (train, test, _classes) = prepare_data()?;
        let train_x = train.x;
        let train_y = MLogistical::from_a(train.y.to_dtype::<f32>());

        // note: the result values are different from the python ones because
        // the initial weights are different.
        // I have manually checked that they match if the initial values are the same.
        let mut layers = crate::layer!(dev, 1e-1, [12288, 7, 1]);
        let mut last_cost = None;
        for i in 0..2500 {
            let caches = layers.clone().downward(train_x.clone());
            let cost = train_y
                .clone()
                .cost(MLogistical::from_a(caches.1 .0.a.clone()))
                .array()
                / (M_TRAIN as f32);
            last_cost = Some(cost);
            if i % 100 == 0 {
                println!("Cost after iteration {}: {}", i, cost);
            }
            let grads = layers.clone().gradients(
                MLogistical::from_a(train_y.a.clone()),
                (Cache::from_a(train_x.clone()), caches),
            );
            let grads = grads.remove_mdas();
            layers = layers.update_params(grads, 0.0075);
        }
        assert!(last_cost.unwrap().approx(0.041310042, (5e-4, 0)));

        // train accuracy
        {
            let yhat = layers.clone().downward(train_x.clone()).flat3().1.a;

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy =
                crate::c1::w2::pa_02_logistic_regression::accuracy(prediction, train_y.a);
            assert_eq!(accuracy.array(), [100.]);
        }

        // test accuracy
        {
            let test_x = test.x;
            let test_y = MLogistical::from_a(test.y.to_dtype::<f32>());
            let yhat = layers.clone().downward(test_x.clone()).flat3().1.a;

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::pa_02_logistic_regression::accuracy(prediction, test_y.a);
            assert_eq!(accuracy.array(), [72.]);
        }

        Ok(())
    }
}

/// C01W04PA02 Part 5 - L-Layer nn.
pub mod _5 {
    #[test]
    fn four_layer_model() -> anyhow::Result<()> {
        use super::*;

        let dev = &device();
        let (train, test, _classes) = prepare_data()?;
        let train_x = train.x;
        let train_y = MLogistical::from_a(train.y.to_dtype::<f32>());

        // note: the result values are different from the python ones because
        // the initial weights are different.
        // I have manually checked that they match if the initial values are the same.
        let mut layers = crate::layer!(dev, 1., [12288, 20, 7, 5, 1]);
        let mut last_cost = None;
        for i in 0..2500 {
            let caches = layers.clone().downward(train_x.clone());
            let cost = train_y
                .clone()
                .cost(MLogistical::from_a(caches.1 .1 .1 .0.a.clone()))
                .array()
                / (M_TRAIN as f32);
            last_cost = Some(cost);
            if i % 100 == 0 {
                println!("Cost after iteration {}: {}", i, cost);
            }
            let grads = layers.clone().gradients(
                MLogistical::from_a(train_y.a.clone()),
                (Cache::from_a(train_x.clone()), caches),
            );
            let grads = grads.remove_mdas();
            layers = layers.update_params(grads, 0.0075);
        }
        assert!(last_cost.unwrap().approx(0.003624782, (4e-3, 0)));

        // train accuracy
        {
            let yhat = layers.clone().downward(train_x.clone()).flat5().3.a;

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy =
                crate::c1::w2::pa_02_logistic_regression::accuracy(prediction, train_y.a);
            assert_eq!(accuracy.array(), [100.]);
        }

        // test accuracy
        {
            let test_x = test.x;
            let test_y = MLogistical::from_a(test.y.to_dtype::<f32>());
            let yhat = layers.clone().downward(test_x.clone()).flat5().3.a;

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::pa_02_logistic_regression::accuracy(prediction, test_y.a);
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

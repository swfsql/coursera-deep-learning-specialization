//! Optimization Methods
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%202/Optimization_methods_v1b.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Optimization%20methods.ipynb
//!
//! I recommend first watching all of C2W2.

#![allow(unused_imports)]

use crate::c1::w4::prelude::*;
use crate::c2::w1::util::pa_01::*;
use crate::helpers::{Approx, _dfdx::*};

/// C02W02PA01 Part 1 - Neural Network Model.
mod _1 {
    #[derive(Clone, Debug)]
    pub struct GradientDescent {
        pub learning_rate: f32,
    }
}

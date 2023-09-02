//! Optimization Methods
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%202/Optimization_methods_v1b.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Optimization%20methods.ipynb
//!
//! I recommend first watching all of C2W2.

#![allow(unused_imports)]

use crate::c1::w4::prelude::*;
use crate::c2::w1::prelude::*;
use crate::helpers::{Approx, _dfdx::*};

/// C02W02PA01 Part 1 - Gradient Descent.
mod _1 {
    use super::*;

    #[test]
    fn test_gradient_descent() {
        let dev = &device();
        let he = HeInit(1.);
        let hes = HeInit::sigmoid();

        // having some layers,
        let mut l = layerc2!(dev, [1, 1 he, 1 he, 1 hes]);

        // some gradients,
        let mut cost_setup = MLogistical;
        let x: X<1, 1> = dev.sample_normal();
        let y = dev.sample_normal();
        let caches = l.downward(x.clone(), &mut cost_setup);
        let grads = l
            .gradients(y, &mut cost_setup, (Cache_::from_a(x), caches))
            .remove_mdas();

        // and an optimizer,
        let mut opt = GradientDescend::new(1.);

        // then can run the update and get a layer
        let l = l.update_params(grads, &mut opt).flat3();
        assert_eq!(l.0.w.array(), [[1.807823,],]);
        assert_eq!(l.0.b.array(), [[4.8961196,],]);
        assert_eq!(l.1.w.array(), [[1.8780298,],]);
        assert_eq!(l.1.b.array(), [[4.0334992,],]);
        assert_eq!(l.2.w.array(), [[-2.767166,],]);
        assert_eq!(l.2.b.array(), [[-1.6556222,],]);
    }
}

// note: Due to complexity, this is the end for this experiment!
// (the current abstractions don't work so well!)
//
// It was a good experience for learning the basics and putting it all together.
//
// From now on it's more productive to more broadly use actual frameworks (other aspects of dfdx, etc).
// Good learning!

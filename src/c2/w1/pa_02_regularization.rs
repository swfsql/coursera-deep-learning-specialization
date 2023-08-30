//! Regularization
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Regularization/Regularization_v2a.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Regularization.ipynb
//!
//! I recommend first watching all of C2W1.

#![allow(unused_imports)]

use crate::c1::w4::prelude::*;
use crate::c2::w1::prelude::*;
use crate::c2::w1::util::pa_02::*;
use crate::helpers::{Approx, _dfdx::*};

/// C02W01PA02 Part 1 - Non-Regularized Model.
mod _1 {
    use super::*;

    #[test]
    fn test_zero_train() {
        let he = HeInit(1.);
        let hes = HeInit::sigmoid();

        let dev = &device();
        let train_x = dev.tensor(XTRAIN);
        let train_y = dev.tensor(YTRAIN);

        let layers = layerc2!(dev, [2, 20 he, 3 he, 1 hes]);
        let mut cost_setup = MLogistical::new(3e-1);
        let layers = layers.train(
            train_x.clone(),
            train_y.clone(),
            &mut cost_setup,
            30_000,
            3000,
        );
        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            .approx(0.06832559, (1e-1, 0)));

        // train accuracy
        {
            let yhat = layers.clone().predict(train_x.clone(), &mut cost_setup);

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert!(accuracy.array().approx([97.156395], (2., 0)));
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
            assert!(accuracy.array().approx([94.0], (2., 0)));
        }
    }
}

// Note: From the youtube lessons, I think this should be called Frobenius regularization?
/// C02W01PA01 Part 2 - L2-Regularization.
pub mod _2 {
    use super::*;

    /// Frobenius Regularization Wrapper for a cost function.
    ///
    /// cost function m*J(L) = sum (L)
    /// loss function L(ŷ,y,w,b) = Inner L(ŷ,y) + λ/2 sum (w² + b²)
    #[derive(Clone, Debug)]
    pub struct FrobeniusReg<Inner> {
        /// Wrapped base cost function.
        pub inner: Inner,

        /// Regularization Parameter.
        pub lambda: f32,

        /// Cost accumulation for element-wise w² + b² for all current and previous w and b.
        ///
        /// Note that this also considers the b parameter for each node, differently from the Python code.
        pub acc: f32,
    }

    impl<Inner> FrobeniusReg<Inner> {
        fn new(inner: Inner, lambda: f32) -> Self {
            Self {
                inner,
                lambda,
                acc: 0.,
            }
        }
    }

    /// Logistical cost.
    impl<Inner> CostSetup for FrobeniusReg<Inner>
    where
        Inner: CostSetup,
    {
        /// M * Cost function m * J = sum (L).
        ///
        /// loss function L(ŷ,y,w,b) = Inner L(ŷ,y) + λ/2m sum (w² + b²)
        fn mcost<const NODELEN: usize, const SETLEN: usize>(
            &self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank1<NODELEN>> {
            let cross_entropy = self.inner.mcost(expect, predict);
            let regularization = self.acc * self.lambda / 2.;

            // M * cost function m * J = sum (L)
            cross_entropy + regularization
        }

        /// Cost function J = sum (L) / m.
        fn cost<const NODELEN: usize, const SETLEN: usize>(
            &self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank0> {
            let l = self.mcost(expect, predict);

            // cost function J = sum (L) / m
            l.sum::<Rank0, _>() / (SETLEN as f32)
        }

        /// Intermediate cost calculation during the downward pass.
        ///
        /// The Loss function L(ŷ,y,w,b) = Inner L(ŷ,y) + λ/2m sum (w² + b²)
        /// has the additive term λ/2m sum (w² + b²).
        ///
        /// acc is an intermediate value for this calculation.  
        /// acc = sum (over each layer) sum (over each node) sum w² + b²
        fn downward<const SOME_NODELEN: usize, const SOME_FEATLEN: usize, SomeZ, SomeA>(
            &mut self,
            layer: &Layer<SOME_FEATLEN, SOME_NODELEN, SomeZ, SomeA>,
        ) {
            // downward recursive call into what's wrapped
            self.inner.downward(layer);

            // sum the layer's nodes' w² and b²
            let layer_sum = layer.w.clone().square().sum::<Rank0, _>().array()
                + layer.b.clone().square().sum::<Rank0, _>().array();

            // acc into the layers' sum
            self.acc += layer_sum;
        }

        /// Required direct additive term for ∂J/∂w and ∂J/∂b.
        ///
        /// The Loss function L(ŷ,y,w,b) = Inner L(ŷ,y) + λ/2m sum (w² + b²)
        /// has the additive term λ/2m sum (w² + b²).
        ///
        /// Since each (w, b) additively and directly affects J, their gradients also have some additive calculations.
        ///
        /// additiveDw = λ/m * w
        /// additiveDb = λ/m * b
        fn direct_upward_dwdb<
            const SOME_NODELEN: usize,
            const SOME_FEATLEN: usize,
            const SETLEN: usize,
            Z,
            A,
        >(
            &self,
            layer: &Layer<SOME_FEATLEN, SOME_NODELEN, Z, A>,
        ) -> Option<Wb<SOME_NODELEN, SOME_FEATLEN>> {
            let mut dw = layer.w.clone() * self.lambda / (SETLEN as f32);
            let db = layer.b.clone() * self.lambda / (SETLEN as f32);
            let mut db = db.reshape::<Rank1<SOME_NODELEN>>();

            // upward recursive call into what's wrapped
            if let Some((inner_dw, inner_db)) = self
                .inner
                .direct_upward_dwdb::<SOME_NODELEN, SOME_FEATLEN, SETLEN, _, _>(layer)
            {
                dw = dw + inner_dw;
                db = db + inner_db;
            }

            Some((dw, db))
        }

        fn update_params<const NODELEN: usize, const FEATLEN: usize, Z, A>(
            &self,
            layer: Layer<FEATLEN, NODELEN, Z, A>,
            gradient: Grads<NODELEN, FEATLEN>,
        ) -> Layer<FEATLEN, NODELEN, Z, A> {
            // the wrapper has no specifics on how the parameters are updated
            self.inner.update_params(layer, gradient)
        }

        fn new_train_step(&mut self) {
            // reset accumulator
            self.acc = 0.;
        }
    }

    /// Any Inner layer should be able to calculate mda = m * ∂J/∂a for the Inner + frobeniusReg cost function.
    impl<const NODELEN: usize, const SETLEN: usize, Inner> UpwardJA<NODELEN, SETLEN>
        for FrobeniusReg<Inner>
    where
        Inner: CostSetup + UpwardJA<NODELEN, SETLEN>,
    {
        /// mda = m * ∂J/∂a.
        type Output = <Inner as UpwardJA<NODELEN, SETLEN>>::Output;

        /// mda = m * ∂J/∂a = (1-y)/(1-a) - y/a = (a-y)/(a(1-a)).
        fn upward_mda(
            &self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> Self::Output {
            // this regularization has no effect on the J->mda->mdz->(..) path
            self.inner.upward_mda(expect, predict)
        }
    }

    #[test]
    fn test_reg_cost() {
        let dev = &device();
        let x: X<3, 2> = dev.sample_normal();
        let y = dev.sample_normal();
        let he = HeInit(1.);
        let hes = HeInit::sigmoid();
        let layers = layerc2!(dev, [3, 2 he, 3 he, 1 hes]);
        let yhat = layers
            .clone()
            // using MLogical for getting the prediction just to simulate that the prediction was a given
            .predict(x.clone(), &mut MLogistical::default());

        // expected cost
        const COST: f32 = 1.1142612;

        // get the cost by calling on "layers"
        {
            let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 1e-1);
            let cost = layers.clone().cost(x.clone(), y.clone(), &mut cost_setup);
            assert_eq!(cost, COST);
        }

        // get the cost by calling on "cost_setup" after manually downwarding it on the layers
        {
            let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 1e-1);
            cost_setup.downward(&layers.0);
            cost_setup.downward(&layers.1 .0);
            cost_setup.downward(&layers.1 .1);
            let cost = cost_setup.cost(y.clone(), yhat.clone()).array();
            assert_eq!(cost, COST);
        }

        // get the cost by calling on "cost_setup" after automatically downwarding it on the layers
        {
            let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 1e-1);
            let caches = layers.downward(x, &mut cost_setup);
            assert_eq!(yhat.array(), caches.last_a().array());
            let cost = cost_setup.cost(y, yhat).array();
            assert_eq!(cost, COST);
        }
    }

    #[test]
    fn test_reg_update() {
        let dev = &device();
        let x: X<3, 2> = dev.sample_normal();
        let y = dev.sample_normal();
        let he = HeInit(1.);
        let hes = HeInit::sigmoid();
        let mut layers = layerc2!(dev, [3, 2 he, 3 he, 1 hes]);
        // make b's also random, otherwise they start set to zero
        layers.0.b = dev.sample_normal() * (2f32 / 3.).sqrt();
        layers.1 .0.b = dev.sample_normal() * (2f32 / 2.).sqrt();
        layers.1 .1.b = dev.sample_normal() * (2f32 / 3.).sqrt();
        let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 7e-1);
        let caches = layers.clone().downward(x.clone(), &mut cost_setup);

        // get the grads values, given the cost_setup used during the caches calculation
        let grads = layers
            .gradients(y, &mut cost_setup, (Cache::from_a(x), caches))
            .remove_mdas()
            .flat4();
        assert_eq!(
            grads.0.dw.array(),
            [
                [0.15945219, 0.49365854, 0.41373608,],
                [-0.92562735, -0.27342856, 0.31114733,],
            ]
        );
        assert_eq!(grads.0.db.array(), [-0.01887869, -0.025309093,]);
        assert_eq!(
            grads.1.dw.array(),
            [
                [-0.1993682, -0.06798189,],
                [-0.20373982, 0.67595494,],
                [-0.116101995, 0.0155322775,],
            ]
        );
        assert_eq!(grads.1.db.array(), [-0.013339308, 1.1011245, 0.26674402,]);
        assert_eq!(grads.2.dw.array(), [[-0.09758315, 1.5412843, 1.313351,],]);
        assert_eq!(grads.2.db.array(), [0.31904832,]);
    }

    #[test]
    fn test_reg_train() {
        let he = HeInit(1.);
        let he2 = HeInit::sigmoid();

        let dev = &device();
        let train_x = dev.tensor(XTRAIN);
        let train_y = dev.tensor(YTRAIN);

        let layers = layerc2!(dev, [2, 20 he, 3 he, 1 he2]);
        let mut cost_setup = FrobeniusReg::new(MLogistical::new(3e-1), 7e-1);
        let layers = layers.train(
            train_x.clone(),
            train_y.clone(),
            &mut cost_setup,
            30_000,
            3000,
        );

        // reset cost_setup accumulators because a new downward pass will happen
        cost_setup.new_train_step();

        assert!(layers
            .clone()
            .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
            .approx(0.26728138, (1e-3, 0)));

        // train accuracy
        {
            let yhat = layers
                .clone()
                .predict(train_x.clone(), &mut MLogistical::default());

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
            assert!(accuracy.array().approx([92.89], (2., 0)));
        }

        // test accuracy
        {
            let test_x = dev.tensor(XTEST);
            let test_y = dev.tensor(YTEST);
            let yhat = layers
                .clone()
                .predict(test_x.clone(), &mut MLogistical::default());

            // rounds ŷ to either 0 or 1
            let mask = yhat.ge(0.5);
            let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
            let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
            let prediction = mask.choose(ones, zeros);

            let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
            assert!(accuracy.array().approx([92.5], (2., 0)));
        }
    }
}

/// C02W01PA01 Part 3 - Dropout.
pub mod _3 {
    use super::*;

    /// C02W01PA01 Section 1 - Forward (Downward) Propagation With Dropout.
    pub mod _1 {}

    /// C02W01PA01 Section 2 - Backward (Upward) Propagation With Dropout.
    pub mod _2 {}
}

/// C02W01PA01 Part 4 - Conclusions.
pub mod _4 {
    use super::*;
}

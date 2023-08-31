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

            // note: it's funny that this is more accurate than with any of the regularizations learned on this week!
            // at least for our initial conditions! (the initial weights, the amount of training and so on)
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
    ///
    /// Notes:
    /// - Accumulator starts unset and is only used for the cost calculation, only relevant
    /// to the downward direction.
    /// - If manually going downward, at the start needs to be set to a new_train_step().
    /// - When the cost is observed (presumably at the lowest layer), the accumulator is unset.
    #[derive(Clone, Debug)]
    pub struct FrobeniusReg<Inner> {
        /// Wrapped base cost function.
        pub inner: Inner,

        /// Regularization Parameter.
        pub lambda: f32,

        /// Cost accumulation for element-wise w² + b² for all current and previous w and b.
        ///
        /// Notes:
        /// - This also considers the b parameter for each node, differently from the Python code.
        /// - This is only relevant to the downward pass, but is irrelevant to the upward pass.
        /// - This should be reset if a new downward pass is getting start (eg. once for each training step).
        pub acc: Option<f32>,
    }

    impl<Inner> FrobeniusReg<Inner> {
        pub fn new(inner: Inner, lambda: f32) -> Self {
            Self {
                inner,
                lambda,
                acc: None,
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
            &mut self,
            expect: A<NODELEN, SETLEN>,
            predict: A<NODELEN, SETLEN>,
        ) -> TensorF32<Rank1<NODELEN>> {
            let cross_entropy = self.inner.mcost(expect, predict);
            let regularization = self.acc.unwrap() * self.lambda / 2.;

            // prevent this accumulator of being used again before it being reset to zero
            self.acc = None;

            // M * cost function m * J = sum (L)
            cross_entropy + regularization
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
            let acc = self.acc.as_mut().unwrap();
            *acc += layer_sum;
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

        /// Resets the accumulator.
        fn refresh_cost(&mut self) {
            // reset accumulator
            self.acc = Some(0.);
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
        let mut layers = layerc2!(dev, [3, 2 he, 3 he, 1 hes]);
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
            let layers = layers.clone().flat3();
            let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 1e-1);
            cost_setup.refresh_cost();
            cost_setup.downward(&layers.0);
            cost_setup.downward(&layers.1);
            cost_setup.downward(&layers.2);
            let cost = cost_setup.cost(y.clone(), yhat.clone()).array();
            assert_eq!(cost, COST);
        }

        // get the cost by calling on "cost_setup" after automatically downwarding it on the layers
        {
            let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 1e-1);
            cost_setup.refresh_cost();
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

        // get caches
        let mut cost_setup = FrobeniusReg::new(MLogistical::new(1.), 7e-1);
        cost_setup.refresh_cost();
        let caches = layers.clone().downward(x.clone(), &mut cost_setup);

        // get the grads values
        cost_setup.refresh_cost();
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
pub use _2::FrobeniusReg;

/// C02W01PA01 Part 3 - Dropout.
pub mod _3 {
    use super::*;

    /// C02W01PA01 Section 1 - Forward (Downward) Propagation With Dropout.
    pub mod _1 {
        use super::*;

        /// Seed used to generate Dropout masks.
        #[derive(Clone, Debug, PartialEq)]
        pub struct DropoutSeed {
            pub is_downward: bool,
            pub seed: u64,
            pub max_seed: u64,
        }

        impl DropoutSeed {
            pub fn new(seed: u64) -> Self {
                Self {
                    is_downward: true,
                    seed,
                    max_seed: 0,
                }
            }
            pub fn set(&mut self, seed: u64, is_downward: bool, max_seed: u64) {
                *self = Self {
                    is_downward,
                    seed,
                    max_seed,
                };
            }
            pub fn is_downward(&self) -> bool {
                self.is_downward
            }
            pub fn is_upward(&self) -> bool {
                !self.is_downward
            }
            /// Increments the seed and returns it.
            /// If was previously going upwards, then first offsets to avoid repetition.
            pub fn tick_downward(&mut self) -> u64 {
                if self.is_upward() {
                    self.is_downward = true;
                    self.seed = self.max_seed;
                }
                self.seed = self.seed.wrapping_add(1);
                self.seed
            }
            /// Decrements the seed and returns it.
            /// If was previously going downwards, then first annotates the max_seed for later offset, and also returns it.
            pub fn tick_upward(&mut self) -> u64 {
                if self.is_downward() {
                    self.max_seed = self.seed;
                    self.is_downward = false;
                    self.max_seed
                } else {
                    self.seed = self.seed.wrapping_sub(1);
                    self.seed
                }
            }
        }

        /// Activation calculation wrapper.
        ///
        /// a(a_inner, for each node) = {
        ///     0 if node was dropped-out,
        ///     a_inner / keep_prob if node was kept,
        /// }
        #[derive(Clone, Debug)]
        pub struct Dropout<Inner> {
            /// The wrapped inner activation.
            pub inner: Inner,

            /// Probability of keeping the nodes on the layer.
            pub keep_prob: f32,

            pub seed: DropoutSeed,
        }

        impl<Inner> Dropout<Inner> {
            pub fn with(inner: Inner, keep_prob: f32, seed: DropoutSeed) -> Self {
                Self {
                    inner,
                    keep_prob,
                    seed,
                }
            }

            pub fn new(device: &Device, inner: Inner, keep_prob: f32) -> Self {
                let seed: TensorF32<Rank0> = device.sample_uniform();
                let seed = seed.array() * (u64::MAX as f32);
                Self {
                    inner,
                    keep_prob,
                    seed: DropoutSeed::new(seed as u64),
                }
            }
        }

        /// Any layer with the Dropout activation wrapper can activate, as long as the inner also can.
        impl<const FEATLEN: usize, const NODELEN: usize, ZF, Inner> DownwardA<FEATLEN, NODELEN>
            for Layer<FEATLEN, NODELEN, ZF, Dropout<Inner>>
        where
            Layer<FEATLEN, NODELEN, ZF, Inner>: DownwardA<FEATLEN, NODELEN>,
            ZF: Clone,
            Inner: Clone,
        {
            /// The activation wrapper has a chance to dropout (zero-out) the inner activation result.
            /// It also upscales the remaining nodes accordingly to the probability.
            ///
            /// Notes:
            /// - The keep_mask must have been unset before this downward pass.
            fn downward_a<const SETLEN: usize>(
                &mut self,
                z: Z<NODELEN, SETLEN>,
            ) -> A<NODELEN, SETLEN> {
                // get the inner activation
                // temporary layer that has inner activation
                //
                // TODO: avoid cloning?
                let mut layer: Layer<FEATLEN, NODELEN, ZF, Inner> = Layer::with(
                    self.w.clone(),
                    self.b.clone(),
                    self.z.clone(),
                    self.a.inner.clone(),
                );
                let inner_a = layer.downward_a(z);

                // re-structure back to self
                self.w = layer.w;
                self.b = layer.b;
                self.z = layer.z;
                self.a.inner = layer.a;

                // get the inner activation
                // let inner_a = self.a.inner.downward_a(z);
                let dev = inner_a.device();

                // generates the mask using a specific seed
                let downward_seed = self.a.seed.tick_downward();
                let mask_dev = &device_seed(downward_seed);
                let keep_prob: TensorF32<Rank2<NODELEN, SETLEN>> = mask_dev.sample_uniform();
                let keep_mask = keep_prob.le(self.a.keep_prob);
                // note: the same seed must be used during the upward pass
                //
                // note: creating a whole device seems quite slow.
                // Could work better if we had direct access to the Cpu's rng data,
                // and if we could set and restore it to what we would like.
                // in this way, creating a whole Device would not be necessary

                // dropped-out nodes get zeroed-out
                let zeros: TensorF32<Rank2<NODELEN, SETLEN>> = dev.zeros();

                // note: wouldn't it make sense to count how many nodes per set were actually dropped-out?
                // (instead of using the probability?)

                // kept nodes are scaled up
                let rescaled = dev.ones() / self.a.keep_prob;
                let scaling = keep_mask.clone().choose(rescaled, zeros);

                // either zero-out or rescale each nodes
                inner_a * scaling
            }
        }

        #[test]
        fn test_dropout_downward() {
            let dev = &device();
            let he = HeInit(1.);
            let hes = HeInit::sigmoid();

            let x = dev.sample_normal();
            let mut layers = layerc2!(
                dev,
                3,
                Linear => Dropout::new(dev, ReLU, 0.7) => [2 he],
                Linear => Dropout::new(dev, ReLU, 0.7) => [3 he],
                Linear => Sigmoid => [1 hes]
            );

            let caches = layers.downward(x, &mut MLogistical::default());
            assert_eq!(
                caches.flat4().2.a.array(),
                [[0.0021308477, 0.5, 0.5, 0.04413341, 0.39451474]]
            );
        }
    }
    pub use _1::{Dropout, DropoutSeed};

    /// C02W01PA01 Section 2 - Backward (Upward) Propagation With Dropout.
    pub mod _2 {
        use super::*;

        /// Any layer with the Dropout activation wrapper can calculate mdz = m * ∂a/∂z, as long as the inner also can.
        impl<const FEATLEN: usize, const NODELEN: usize, Z, Inner> UpwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Dropout<Inner>>
        where
            Layer<FEATLEN, NODELEN, Z, Inner>: UpwardAZ<NODELEN>,
            Z: Clone,
            Inner: Clone,
        {
            /// mdz per node = {
            ///     0 if keep_mask for node was false,
            ///     inner_mdz.a / keep_chance if keep_mask for node was true
            /// }
            type Output<const SETLEN: usize> =
                <Layer<FEATLEN, NODELEN, Z, Inner> as UpwardAZ<NODELEN>>::Output<SETLEN>;

            /// ∂a_dropout/∂z = {
            ///     0 if node was dropped-out,
            ///     (∂a_inner/∂z) / keep_chance if node was kept,
            /// }
            ///
            /// Notes:
            /// - The keep_mask must have been set before this upward pass.
            fn upward_mdz<const SETLEN: usize>(
                &mut self,
                dropout_cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                let dev = self.w.device();

                // re-generates the mask using a specific seed
                let upward_seed = self.a.seed.tick_upward();
                let mask_dev = &device_seed(upward_seed);
                let keep_prob: TensorF32<Rank2<NODELEN, SETLEN>> = mask_dev.sample_uniform();
                let keep_mask = keep_prob.le(self.a.keep_prob);
                // note: the same seed must have been used during the downward pass
                //
                // note: creating a whole device seems quite slow.
                // Could work better if we had direct access to the Cpu's rng data,
                // and if we could set and restore it to what we would like.
                // in this way, creating a whole Device would not be necessary

                // dropped-out nodes get zeroed-out
                let zeros: TensorF32<Rank2<NODELEN, SETLEN>> = dev.zeros();

                // note: wouldn't it make sense to count how many nodes per set were actually dropped-out?
                // (instead of using the probability?)

                // kept nodes are scaled up
                let upscaled = dev.ones() / self.a.keep_prob;
                let scaling = keep_mask.clone().choose(upscaled, zeros);

                let mda = mda * scaling;

                // temporary layer that has inner activation
                // TODO: remove clones?
                let mut layer: Layer<FEATLEN, NODELEN, Z, Inner> = Layer::with(
                    self.w.clone(),
                    self.b.clone(),
                    self.z.clone(),
                    self.a.inner.clone(),
                );

                let res = layer.upward_mdz(dropout_cache, mda);
                self.w = layer.w;
                self.b = layer.b;
                self.z = layer.z;
                self.a.inner = layer.a;
                res
            }
        }

        #[test]
        fn test_dropout_upward() {
            let dev = &device();
            let he = HeInit(1.);
            let hes = HeInit::sigmoid();

            let x: X<3, 5> = dev.sample_normal();
            let y = dev.sample_normal();
            let mut layers = layerc2!(
                dev,
                3,
                Linear => Dropout::new(dev, ReLU, 0.8) => [2 he],
                Linear => Dropout::new(dev, ReLU, 0.8) => [3 he],
                Linear => Sigmoid => [1 hes]
            );
            let mut cost_setup = MLogistical::new(1.0);
            cost_setup.refresh_cost();

            // verify the dropout seeds
            let seeds_initial = [layers.0.a.seed.seed, layers.1 .0.a.seed.seed];

            // get caches (downward)
            let caches = layers.downward(x.clone(), &mut cost_setup);

            // verify final seeds
            let seeds_after_downward = [layers.0.a.seed.seed, layers.1 .0.a.seed.seed];
            assert_eq!(seeds_initial[0] + 1, seeds_after_downward[0]);
            assert_eq!(seeds_initial[1] + 1, seeds_after_downward[1]);

            // get the grads values (upward)
            cost_setup.refresh_cost();
            let grads = layers
                .clone()
                .gradients(
                    y.clone(),
                    &mut cost_setup,
                    (Cache::from_a(x.clone()), caches),
                )
                .remove_mdas()
                .flat4();

            // verify seeds again
            let seeds_after_upward = [layers.0.a.seed.seed, layers.1 .0.a.seed.seed];
            assert_eq!(seeds_initial[0] + 1, seeds_after_upward[0]);
            assert_eq!(seeds_initial[1] + 1, seeds_after_upward[1]);

            // asserts
            assert_eq!(grads.2.dw.array(), [[0.08327425, 0.23616907, 0.018630508]]);
            assert_eq!(
                grads.1.dw.array(),
                [
                    [0.03177176, 0.019490158],
                    [-0.094677165, -0.058079023],
                    [0.0, -0.009395287]
                ]
            );
            assert_eq!(
                grads.0.dw.array(),
                [
                    [-0.04446063, 0.07673529, 0.14610517],
                    [0.16612747, -0.22969477, 0.041919343]
                ]
            );

            // makes another downward-upward round

            // get caches (downward)
            cost_setup.refresh_cost();
            let caches = layers.downward(x.clone(), &mut cost_setup);

            // verify final seeds
            let seeds_after_downward2 = [layers.0.a.seed.seed, layers.1 .0.a.seed.seed];
            assert_eq!(seeds_initial[0] + 2, seeds_after_downward2[0]);
            assert_eq!(seeds_initial[1] + 2, seeds_after_downward2[1]);

            // get the grads values (upward)
            cost_setup.refresh_cost();
            let _grads = layers
                .clone()
                .gradients(y, &mut cost_setup, (Cache::from_a(x), caches))
                .remove_mdas()
                .flat4();

            // verify seeds again
            let seeds_after_upward2 = [layers.0.a.seed.seed, layers.1 .0.a.seed.seed];
            assert_eq!(seeds_initial[0] + 2, seeds_after_upward2[0]);
            assert_eq!(seeds_initial[1] + 2, seeds_after_upward2[1]);
        }

        #[test]
        fn test_dropout_train() {
            let he = HeInit(1.);
            let he2 = HeInit::sigmoid();

            let dev = &device();
            let train_x = dev.tensor(XTRAIN);
            let train_y = dev.tensor(YTRAIN);

            let layers = layerc2!(
                dev,
                2,
                Linear => Dropout::new(dev, ReLU, 0.86) => [20 he],
                Linear => Dropout::new(dev, ReLU, 0.86) => [3 he],
                Linear => Sigmoid => [1 he2]
            );

            let mut cost_setup = MLogistical::new(3e-1);
            let layers = layers.train(
                train_x.clone(),
                train_y.clone(),
                &mut cost_setup,
                // I've reduce the training time because the dropout impl is too slow
                1_000,
                100,
            );

            assert!(layers
                .clone()
                .cost(train_x.clone(), train_y.clone(), &mut cost_setup)
                .approx(0.25611246, (1e-1, 0)));

            // for accuracy testing, remove the dropouts
            let mut layers_clean = layerc2!(
                dev,
                2,
                Linear => ReLU => [20 he],
                Linear => ReLU => [3 he],
                Linear => Sigmoid => [1 he2]
            );

            layers_clean.0.w = layers.0.w;
            layers_clean.0.b = layers.0.b;
            layers_clean.1 .0.w = layers.1 .0.w;
            layers_clean.1 .0.b = layers.1 .0.b;
            layers_clean.1 .1.w = layers.1 .1.w;
            layers_clean.1 .1.b = layers.1 .1.b;

            // train accuracy
            {
                let yhat = layers_clean
                    .clone()
                    .predict(train_x.clone(), &mut MLogistical::default());

                // rounds ŷ to either 0 or 1
                let mask = yhat.ge(0.5);
                let zeros: TensorF32<Rank2<1, M_TRAIN>> = mask.device().zeros();
                let ones: TensorF32<Rank2<1, M_TRAIN>> = mask.device().ones();
                let prediction = mask.choose(ones, zeros);

                let accuracy = crate::c1::w2::prelude::accuracy(prediction, train_y);
                assert!(accuracy.array().approx([92.3128], (3., 0)));
            }

            // test accuracy
            {
                let test_x = dev.tensor(XTEST);
                let test_y = dev.tensor(YTEST);
                let yhat = layers_clean
                    .clone()
                    .predict(test_x.clone(), &mut MLogistical::default());

                // rounds ŷ to either 0 or 1
                let mask = yhat.ge(0.5);
                let zeros: TensorF32<Rank2<1, M_TEST>> = mask.device().zeros();
                let ones: TensorF32<Rank2<1, M_TEST>> = mask.device().ones();
                let prediction = mask.choose(ones, zeros);

                let accuracy = crate::c1::w2::prelude::accuracy(prediction, test_y);
                assert!(accuracy.array().approx([92.0], (2., 0)));
            }
        }
    }
}
pub use _3::{Dropout, DropoutSeed};

/// C02W01PA01 Part 4 - Conclusions.
pub mod _4 {
    use super::*;
}

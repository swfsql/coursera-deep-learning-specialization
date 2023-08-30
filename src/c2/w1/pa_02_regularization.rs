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

        pub trait ActivationSetup {
            fn refresh_activation(&mut self) {}
        }

        impl ActivationSetup for Sigmoid {}
        impl ActivationSetup for ReLU {}
        impl ActivationSetup for Tanh {}

        impl<const FEATLEN: usize, const NODELEN: usize, Z, A> ActivationSetup
            for Layer<FEATLEN, NODELEN, Z, A>
        where
            A: ActivationSetup,
        {
            fn refresh_activation(&mut self) {
                self.a.refresh_activation()
            }
        }

        impl<const FEATLEN: usize, const NODELEN: usize, Z, A, Tail> ActivationSetup
            for (Layer<FEATLEN, NODELEN, Z, A>, Tail)
        where
            Layer<FEATLEN, NODELEN, Z, A>: ActivationSetup,
            Tail: ActivationSetup,
        {
            fn refresh_activation(&mut self) {
                self.0.refresh_activation();
                self.1.refresh_activation()
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

            /// Mask cache created during the downward pass.
            ///
            /// Notes:
            /// - If the keep_mask was unset before a downward pass, it's generated and set.
            /// - If the keep_mask was set before a downward pass, it's re-utilized.
            /// - The keep_mask must have been set before the upward pass.
            //
            // Note: Could not keep this as constant-sized tensor because it would need to
            // Rank2<NODELEN, SETLEN>, but downward_a() fn call requires SETLEN to be defined
            // at call-site.
            // Ie. Otherwise we would have to build a new layer definition for each of the
            // training vs testing, since it would need to be constant towards each of the setlen.
            // In this case, it's better left dynamic-sized.
            pub keep_mask: Option<Vec<bool>>,
        }

        impl<Inner> Dropout<Inner> {
            pub fn with(inner: Inner, keep_prob: f32, keep_mask: Vec<bool>) -> Self {
                Self {
                    inner,
                    keep_prob,
                    keep_mask: Some(keep_mask),
                }
            }

            pub fn new(inner: Inner, keep_prob: f32) -> Self {
                Self {
                    inner,
                    keep_prob,
                    keep_mask: None,
                }
            }

            /// Discards the keep_mask generated during a downward pass.
            pub fn discard_keep_mask(&mut self) {
                self.keep_mask = None;
            }

            pub fn mask_vec_to_tensor<const NODELEN: usize, const SETLEN: usize>(
                &self,
                dev: &Device,
            ) -> Option<Tensor<Rank2<NODELEN, SETLEN>, bool, Device>> {
                match &self.keep_mask {
                    None => None,
                    Some(mask) => {
                        let mut keep_mask_arr = [[false; SETLEN]; NODELEN];
                        keep_mask_arr
                            .iter_mut()
                            .enumerate()
                            .for_each(|(node_index, node_mask)| {
                                let offset = node_index * SETLEN;
                                node_mask.copy_from_slice(&mask[offset..offset + SETLEN]);
                            });
                        Some(dev.tensor(keep_mask_arr))
                    }
                }
            }

            pub fn mask_arr_to_vec<const NODELEN: usize, const SETLEN: usize>(
                &self,
                mask: &[[bool; SETLEN]; NODELEN],
            ) -> Vec<bool> {
                let mut keep_mask_vec = vec![false; NODELEN * SETLEN];
                mask.iter().enumerate().for_each(|(node_index, node_mask)| {
                    let offset = node_index * SETLEN;
                    keep_mask_vec[offset..offset + SETLEN].copy_from_slice(node_mask);
                });
                keep_mask_vec
            }
        }

        impl<Inner> ActivationSetup for Dropout<Inner> {
            /// For each new training step, discards the layer's mask.
            fn refresh_activation(&mut self) {
                self.discard_keep_mask()
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

                // if already set, this re-utilizes the keep_mask
                let keep_mask = if self.a.keep_mask.is_some() {
                    self.a.mask_vec_to_tensor(dev).unwrap()
                } else {
                    // creates a new tensor and saves it as a vector inside the activation
                    let keep_prob: TensorF32<Rank2<NODELEN, SETLEN>> = dev.sample_uniform();
                    let keep_mask = keep_prob.le(self.a.keep_prob);
                    let keep_mask_vec = self.a.mask_arr_to_vec(&keep_mask.array());
                    self.a.keep_mask = Some(keep_mask_vec);
                    keep_mask
                };

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
        fn test_dropout_forward() {
            let dev = &device();
            let he = HeInit(1.);
            let hes = HeInit::sigmoid();

            let x = dev.sample_normal();
            let mut layers = layerc2!(
                dev,
                3,
                Linear => Dropout::new(ReLU, 0.7) => [2 he],
                Linear => Dropout::new(ReLU, 0.7) => [3 he],
                Linear => Sigmoid => [1 hes]
            );

            let caches = layers.downward(x, &mut MLogistical::default());
            assert_eq!(
                caches.flat4().2.a.array(),
                [[0.49617895, 0.9316923, 0.5, 0.5, 0.3315451]]
            );
        }
    }
    pub use _1::{ActivationSetup, Dropout};

    /// C02W01PA01 Section 2 - Backward (Upward) Propagation With Dropout.
    pub mod _2 {
        use super::*;

        /// Any layer with the Dropout activation wrapper can calculate mdz = m * ∂a/∂z, as long as the inner also can.
        impl<const FEATLEN: usize, const NODELEN: usize, Z, Inner> UpwardAZ<NODELEN>
            for Layer<FEATLEN, NODELEN, Z, Dropout<Inner>>
        where
            Layer<FEATLEN, NODELEN, Z, Inner>: UpwardAZ<NODELEN>,
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
                self,
                dropout_cache: Cache<NODELEN, SETLEN>,
                mda: Mda<NODELEN, SETLEN>,
            ) -> Self::Output<SETLEN> {
                let dev = self.w.device();

                // get keep_mask generated during the downward pass for this layer
                let keep_mask = self.a.mask_vec_to_tensor(dev).unwrap();

                // dropped-out nodes get zeroed-out
                let zeros: TensorF32<Rank2<NODELEN, SETLEN>> = dev.zeros();

                // note: wouldn't it make sense to count how many nodes per set were actually dropped-out?
                // (instead of using the probability?)

                // kept nodes are scaled up
                let upscaled = dev.ones() / self.a.keep_prob;
                let scaling = keep_mask.clone().choose(upscaled, zeros);

                let mda = mda * scaling;

                // temporary layer that has inner activation
                let layer: Layer<FEATLEN, NODELEN, Z, Inner> =
                    Layer::with(self.w, self.b, self.z, self.a.inner);

                layer.upward_mdz(dropout_cache, mda)
            }
        }

        #[test]
        fn test_dropout_backward() {
            let dev = &device();
            let he = HeInit(1.);
            let hes = HeInit::sigmoid();

            let x: X<3, 5> = dev.sample_normal();
            let y = dev.sample_normal();
            let mut layers = layerc2!(
                dev,
                3,
                Linear => Dropout::new(ReLU, 0.8) => [2 he],
                Linear => Dropout::new(ReLU, 0.8) => [3 he],
                Linear => Sigmoid => [1 hes]
            );
            let mut cost_setup = MLogistical::new(1.0);
            cost_setup.refresh_cost();
            let caches = layers.downward(x.clone(), &mut cost_setup);

            // get the grads values
            cost_setup.refresh_cost();
            let grads = layers
                .clone()
                .gradients(y, &mut cost_setup, (Cache::from_a(x), caches))
                .remove_mdas()
                .flat4();

            // asserts
            assert_eq!(
                grads.2.dw.array(),
                [[0.08585273, -0.0027280687, -0.0031253458]]
            );
            assert_eq!(
                grads.1.dw.array(),
                [
                    [0.0, 0.040307995],
                    [0.0, -0.00028485357],
                    [0.0, 0.0008488357]
                ]
            );
            assert_eq!(
                grads.0.dw.array(),
                [[0.0, 0.0, 0.0], [-2.105148, 0.11473336, -1.2267761]]
            );
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
                Linear => Dropout::new(ReLU, 0.86) => [20 he],
                Linear => Dropout::new(ReLU, 0.86) => [3 he],
                Linear => Sigmoid => [1 he2]
            );

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
                .approx(0.19930607, (2e-1, 0)));

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
pub use _3::{ActivationSetup, Dropout};

/// C02W01PA01 Part 4 - Conclusions.
pub mod _4 {
    use super::*;
}

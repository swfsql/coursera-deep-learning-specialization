//! Regularization
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C2%20-%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Week%201/Gradient%20Checking/Gradient%20Checking%20v1.ipynb
//!
//! Alternative Reference: https://github.com/Kulbear/deep-learning-coursera/blob/997fdb2e2db67acd45d29ae418212463a54be06d/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Gradient%20Checking.ipynb
//!
//! I recommend first watching all of C2W1.
//!
//! Note: It appears that either my forward/backward prop implementations are wrong,
//! or my implementation for gradient checking is wrong.

#![allow(unused_imports)]

use crate::c1::w4::prelude::*;
use crate::c2::w1::prelude::*;
use crate::c2::w1::util::pa_02::*;
use crate::helpers::{Approx, _dfdx::*};

/// Activation calculation.
///
/// `a(z) = z`
#[derive(Clone, Debug)]
pub struct Identity;

/// Any layer can activate with Idendity.
impl<const FEATLEN: usize, const NODELEN: usize, ZF> DownwardA<FEATLEN, NODELEN>
    for Layer<FEATLEN, NODELEN, ZF, Identity>
{
    fn downward_a<const SETLEN: usize>(&mut self, z: Z<NODELEN, SETLEN>) -> A<NODELEN, SETLEN> {
        z
    }
}

/// Any layer with a Idendity activation can calculate mdz = m * ∂a/∂z.
impl<const FEATLEN: usize, const NODELEN: usize, Z> UpwardAZ<NODELEN>
    for Layer<FEATLEN, NODELEN, Z, Identity>
{
    /// mdz = mda.
    type Output<const SETLEN: usize> = Mdz<NODELEN, SETLEN>;

    /// ∂a/∂z = 1.
    ///
    /// mdz = m * ∂J/∂z = m * ∂a/∂z * ∂J/∂a = mda
    fn upward_mdz<const SETLEN: usize>(
        &mut self,
        _cache: Cache<NODELEN, SETLEN>,
        mda: Mda<NODELEN, SETLEN>,
    ) -> Self::Output<SETLEN> {
        mda
    }
}

/// Identity cost function.
///
/// cost function m*J(L) = sum (L)
/// loss function L(ŷ,y) = ŷ
///
/// Note that the cost function is multiplied by `m` (SETLEN).
#[derive(Clone, Debug)]
pub struct MIdentity;

impl CostSetup for MIdentity {
    fn mcost<const NODELEN: usize, const SETLEN: usize>(
        &mut self,
        _expect: A<NODELEN, SETLEN>,
        predict: A<NODELEN, SETLEN>,
    ) -> TensorF32<Rank1<NODELEN>> {
        predict.sum()
    }

    fn cost<const NODELEN: usize, const SETLEN: usize>(
        &mut self,
        expect: A<NODELEN, SETLEN>,
        predict: A<NODELEN, SETLEN>,
    ) -> TensorF32<Rank0> {
        let cost = self.mcost(expect, predict) / (SETLEN as f32);
        cost.sum()
    }

    fn update_params<const NODELEN: usize, const FEATLEN: usize, Z, A>(
        &self,
        mut layer: Layer<FEATLEN, NODELEN, Z, A>,
        gradient: Grads<NODELEN, FEATLEN>,
    ) -> Layer<FEATLEN, NODELEN, Z, A> {
        layer.w = layer.w - gradient.dw;
        layer.b = layer.b - gradient.db.broadcast();
        layer
    }
}

impl<const NODELEN: usize, const SETLEN: usize> UpwardJA<NODELEN, SETLEN> for MIdentity {
    /// mda = m * ∂J/∂a.
    type Output = Mda<NODELEN, SETLEN>;

    /// mda = m * ∂J/∂a = m.
    fn upward_mda(&self, _expect: A<NODELEN, SETLEN>, predict: A<NODELEN, SETLEN>) -> Self::Output {
        let dev = predict.device();
        dev.ones() * (SETLEN as f32)
    }
}

#[test]
fn gradient_checking_1d_1() {
    let dev = &device();

    // first the dataset (one feature, one sample)
    let x = dev.tensor([[2.]]);

    // then a Linear layer with a single neuron e a single feature, and also with an Idendity activation
    let mut layer = Layer::<1, 1, Linear, Identity>::with(
        dev.tensor([[4.]]),
        dev.tensor([[0.]]),
        Linear,
        Identity,
    );
    // then an "identity" cost function, which just returns the prediction
    let mut cost_setup = MIdentity;

    // then a downward call
    cost_setup.refresh_cost();
    let caches = layer.downward(x.clone(), &mut cost_setup);
    assert_eq!(caches.0.a.array(), [[8.]]);

    // then the derivative
    let y = dev.zeros();
    cost_setup.refresh_cost();
    let grads = layer
        .gradients(y, &mut cost_setup, (Cache::from_a(x), caches))
        .remove_mdas();

    // the value below is just x
    assert_eq!(grads.0.dw.array(), [[2.]]);
    // the value below is always 1
    assert_eq!(grads.0.db.array(), [1.]);
}

pub trait GradientCheck<const FEATLEN: usize, const Y_FEATURES: usize, const SETLEN: usize>
where
    Self: LayersSetup<FEATLEN, Y_FEATURES, SETLEN> + LayerBounds<FEATLEN, Y_FEATURES, SETLEN>,
{
    type Diffs;
    type CurrentGrads;
    fn gradient_check<CostType>(
        self,
        x: X<FEATLEN, SETLEN>,
        y: Y<Y_FEATURES, SETLEN>,
        cost_setup: &mut CostType,
    ) -> Self::Diffs
    where
        CostType: CostSetup
            + Clone
            + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>;
}

/// Implements gradient check for a single layer.
impl<
        const FEATLEN: usize,
        const NODELEN: usize,
        const Y_FEATURES: usize,
        const SETLEN: usize,
        Z,
        A,
    > GradientCheck<FEATLEN, Y_FEATURES, SETLEN> for Layer<FEATLEN, NODELEN, Z, A>
where
    Self: LayersSetup<FEATLEN, Y_FEATURES, SETLEN>,
    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Output:
        CleanupGrads<Output = (Grads<NODELEN, FEATLEN>, ())>,
{
    type Diffs = Wb<NODELEN, FEATLEN>;
    type CurrentGrads = (Grads<NODELEN, FEATLEN>, ());
    fn gradient_check<CostType>(
        mut self,
        x: X<FEATLEN, SETLEN>,
        y: Y<Y_FEATURES, SETLEN>,
        cost_setup: &mut CostType,
    ) -> Self::Diffs
    where
        CostType: CostSetup
            + Clone
            + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>,
    {
        let dev = &self.w.device().clone();
        let layer_w_arr = &mut self.w.array();
        let layer_b_arr = &mut self.b.array();
        let mut diffs_w: [[f32; FEATLEN]; NODELEN] = [[0.; FEATLEN]; NODELEN];
        let mut diffs_b: [f32; NODELEN] = [0.; NODELEN];

        // grad
        cost_setup.refresh_cost();
        let caches = self.downward(x.clone(), cost_setup);
        cost_setup.refresh_cost();
        let wrap_caches = (Cache::from_a(x.clone()), caches);
        let grads: Grads<NODELEN, FEATLEN> = self
            .gradients(y.clone(), cost_setup, wrap_caches)
            .remove_mdas()
            .0;
        let dw = grads.dw.array();
        let db = grads.db.array();

        // for each node
        for (i, (((node_ws, node_dws), node_b), node_db)) in self
            .w
            .array()
            .into_iter()
            .zip(dw)
            .zip(self.b.array().into_iter())
            .zip(db)
            .enumerate()
        {
            // for each w (for an input feature)
            for (j, (feat_w, dw)) in node_ws.into_iter().zip(node_dws).enumerate() {
                // θp = θ + ε
                let tethap = feat_w.next_up();
                // θn = θ - ε
                let tethan = feat_w.next_down();
                // 2ε
                let mut eps2 = tethap - tethan;
                if eps2 == 0. {
                    eps2 = f32::MIN_POSITIVE;
                }
                // dbg!(tethap, feat_w, tethan);
                // dbg!(eps2);

                // θp
                layer_w_arr[i][j] = tethap;
                self.w = dev.tensor(*layer_w_arr);
                cost_setup.refresh_cost();
                let costp = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costp);

                // θn
                layer_w_arr[i][j] = tethan;
                self.w = dev.tensor(*layer_w_arr);
                cost_setup.refresh_cost();
                let costn = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costn);

                // resets the weight to the original θ
                layer_w_arr[i][j] = feat_w;

                // dw_approx = (J(θp) - J(θp)) / 2ε
                let mut diff_cost = costp - costn;
                if diff_cost == 0. && eps2 < f32::EPSILON {
                    diff_cost = eps2;
                }
                let dw_approx = (diff_cost) / eps2;
                // dbg!(dw_approx, dw);

                let numerator = (dw_approx - dw).abs();
                let denominator = dw_approx.abs() + dw.abs();
                let diff = numerator / denominator;
                diffs_w[i][j] = diff;
            }
            // reset the weights to the original w
            self.w = dev.tensor(*layer_w_arr);

            // for the node's b
            {
                // θp = θ + ε
                let tethap = node_b[0].next_up();
                // θn = θ - ε
                let tethan = node_b[0].next_down();
                // 2ε
                let mut eps2 = tethap - tethan;
                if eps2 == 0. {
                    eps2 = f32::MIN_POSITIVE;
                }
                // dbg!(tethap, node_b[0], tethan);
                // dbg!(eps2);

                // θp
                layer_b_arr[i] = [tethap];
                self.b = dev.tensor(*layer_b_arr);
                cost_setup.refresh_cost();
                let costp = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costp);

                // θn
                layer_b_arr[i] = [tethan];
                self.b = dev.tensor(*layer_b_arr);
                cost_setup.refresh_cost();
                let costn = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costn);

                // resets the bias to the original θ
                layer_b_arr[i] = node_b;

                // db_approx = (J(θp) - J(θp)) / 2ε
                let mut diff_cost = costp - costn;
                if diff_cost == 0. && eps2 < f32::EPSILON {
                    diff_cost = eps2;
                }
                let db_approx = (diff_cost) / eps2;
                // dbg!(db_approx, db[0]);

                let numerator = (db_approx - node_db).abs();
                let denominator = db_approx.abs() + node_db.abs();
                let diff = numerator / denominator;
                diffs_b[i] = diff;
                // dbg!(diff);
            }
            // reset the bias to the original b
            self.b = dev.tensor(*layer_b_arr);
        }
        (dev.tensor(diffs_w), dev.tensor(diffs_b))
    }
}

#[test]
fn gradient_checking_1d_2() {
    let dev = &device();

    let x = dev.tensor([[2.]]);
    // y value doesn't matter for MIdentity const
    let y = dev.tensor([[0.]]);
    let layer = Layer::<1, 1, Linear, Identity>::with(
        dev.tensor([[4.]]),
        dev.tensor([[0.]]),
        Linear,
        Identity,
    );
    let mut cost_setup = MIdentity;
    let (diffs_w, diffs_b) = layer.gradient_check(x, y, &mut cost_setup);
    assert_eq!(diffs_w.sum::<Rank0, _>().array(), 0.);
    assert_eq!(diffs_b.sum::<Rank0, _>().array(), 0.);
}

/// Implements gradient check for multiple layers.
impl<
        const FEATLEN: usize,
        const NODELEN: usize,
        const Y_FEATURES: usize,
        const SETLEN: usize,
        Z,
        A,
        Lower,
    > GradientCheck<FEATLEN, Y_FEATURES, SETLEN> for (Layer<FEATLEN, NODELEN, Z, A>, Lower)
where
    Self: LayersSetup<FEATLEN, Y_FEATURES, SETLEN> + LayerBounds<FEATLEN, Y_FEATURES, SETLEN>,
    <Self as DownUpGrads<Y_FEATURES, SETLEN>>::Output:
        CleanupGrads<Output = (Grads<NODELEN, FEATLEN>, Lower::CurrentGrads)>,
    Lower: GradientCheck<NODELEN, Y_FEATURES, SETLEN>,
    <(Layer<FEATLEN, NODELEN, Z, A>, Lower) as DownUpGrads<Y_FEATURES, SETLEN>>::Cache:
        Clone + WrapA<NODELEN, SETLEN>,
{
    type Diffs = (Wb<NODELEN, FEATLEN>, Lower::Diffs);
    type CurrentGrads = (Grads<NODELEN, FEATLEN>, Lower::CurrentGrads);
    fn gradient_check<CostType>(
        mut self,
        x: X<FEATLEN, SETLEN>,
        y: Y<Y_FEATURES, SETLEN>,
        cost_setup: &mut CostType,
    ) -> Self::Diffs
    where
        CostType: CostSetup
            + Clone
            + UpwardJA<Y_FEATURES, SETLEN, Output = TensorF32<Rank2<Y_FEATURES, SETLEN>>>,
    {
        // current gradient check

        let dev = &self.0.w.device().clone();
        let layer_w_arr = &mut self.0.w.array();
        let layer_b_arr = &mut self.0.b.array();
        let mut diffs_w: [[f32; FEATLEN]; NODELEN] = [[0.; FEATLEN]; NODELEN];
        let mut diffs_b: [f32; NODELEN] = [0.; NODELEN];

        // grad
        cost_setup.refresh_cost();
        let caches = self.downward(x.clone(), cost_setup);
        let first_cache = caches.0.ref_a().clone();
        cost_setup.refresh_cost();
        let wrap_caches = (Cache::from_a(x.clone()), caches);
        // let grads: Grads<NODELEN, FEATLEN> = self
        let grads = self
            .gradients(y.clone(), cost_setup, wrap_caches)
            .remove_mdas()
            .0;
        let dw = grads.dw.array();
        let db = grads.db.array();

        // for each node
        for (i, (((node_ws, node_dws), node_b), node_db)) in self
            .0
            .w
            .array()
            .into_iter()
            .zip(dw)
            .zip(self.0.b.array().into_iter())
            .zip(db)
            .enumerate()
        {
            // for each w (for an input feature)
            for (j, (feat_w, dw)) in node_ws.into_iter().zip(node_dws).enumerate() {
                // θp = θ + ε
                let tethap = feat_w.next_up();
                // θn = θ - ε
                let tethan = feat_w.next_down();
                // 2ε
                let mut eps2 = tethap - tethan;
                if eps2 == 0. {
                    eps2 = f32::MIN_POSITIVE;
                }
                // dbg!(tethap, feat_w, tethan);
                // dbg!(eps2);

                // θp
                layer_w_arr[i][j] = tethap;
                self.0.w = dev.tensor(*layer_w_arr);
                cost_setup.refresh_cost();
                let costp = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costp);

                // θn
                layer_w_arr[i][j] = tethan;
                self.0.w = dev.tensor(*layer_w_arr);
                cost_setup.refresh_cost();
                let costn = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costn);

                // resets the weight to the original θ
                layer_w_arr[i][j] = feat_w;

                // dw_approx = (J(θp) - J(θp)) / 2ε
                let mut diff_cost = costp - costn;
                if diff_cost == 0. && eps2 < f32::EPSILON {
                    diff_cost = eps2;
                }
                let dw_approx = (diff_cost) / eps2;
                // dbg!(dw_approx, dw);

                let numerator = (dw_approx - dw).abs();
                let denominator = dw_approx.abs() + dw.abs();
                let diff = numerator / denominator;
                diffs_w[i][j] = diff;
            }
            // reset the weights to the original w
            self.0.w = dev.tensor(*layer_w_arr);

            // for the node's b
            {
                // θp = θ + ε
                let tethap = node_b[0].next_up();
                // θn = θ - ε
                let tethan = node_b[0].next_down();
                // 2ε
                let mut eps2 = tethap - tethan;
                if eps2 == 0. {
                    eps2 = f32::MIN_POSITIVE;
                }
                // dbg!(tethap, node_b[0], tethan);
                // dbg!(eps2);

                // θp
                layer_b_arr[i] = [tethap];
                self.0.b = dev.tensor(*layer_b_arr);
                cost_setup.refresh_cost();
                let costp = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costp);

                // θn
                layer_b_arr[i] = [tethan];
                self.0.b = dev.tensor(*layer_b_arr);
                cost_setup.refresh_cost();
                let costn = self.cost(x.clone(), y.clone(), cost_setup);
                // dbg!(costn);

                // resets the bias to the original θ
                layer_b_arr[i] = node_b;

                // db_approx = (J(θp) - J(θp)) / 2ε
                let mut diff_cost = costp - costn;
                if diff_cost == 0. && eps2 < f32::EPSILON {
                    diff_cost = eps2;
                }
                let db_approx = (diff_cost) / eps2;
                // dbg!(db_approx, db[0]);

                let numerator = (db_approx - node_db).abs();
                let denominator = db_approx.abs() + node_db.abs();
                let diff = numerator / denominator;
                diffs_b[i] = diff;
                // dbg!(diff);
            }
            // reset the bias to the original b
            self.0.b = dev.tensor(*layer_b_arr);
        }

        // current gradient check
        let current_diffs = (dev.tensor(diffs_w), dev.tensor(diffs_b));

        // lower gradient checks
        cost_setup.refresh_cost();
        let lower_diffs = self.1.gradient_check(first_cache, y, cost_setup);

        (current_diffs, lower_diffs)
    }
}

#[test]
fn gradient_checking_l() {
    let dev = &device();
    let he = HeInit(1.);
    let hes = HeInit::sigmoid();

    let x: X<2, 3> = dev.sample_normal();
    let y = dev.sample_normal();
    let layers = layerc2!(dev, [2, 3 he, 1 hes]);
    let mut cost_setup = MLogistical::new(1e-3);
    let diffs = layers.gradient_check(x, y, &mut cost_setup);
    let (diffw1, diffb1) = diffs.0;
    let (diffw2, diffb2) = diffs.1;
    let diffw1 = diffw1.sum::<Rank0, _>().array();
    let diffb1 = diffb1.sum::<Rank0, _>().array();
    let diffw2 = diffw2.sum::<Rank0, _>().array();
    let diffb2 = diffb2.sum::<Rank0, _>().array();

    assert_eq!(diffw1, 5.478448);
    assert_eq!(diffb1, 2.0183713);
    assert_eq!(diffw2, 1.3557003);
    assert_eq!(diffb2, 0.12479658);
    // TODO: so this means that the forward/backward prop implementation is wrong?
    // this is weird because I have been checking the results against the python reference
}

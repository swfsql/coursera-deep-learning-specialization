//! Rust Basics with ML
//!
//! Reference: https://github.com/amanchadha/coursera-deep-learning-specialization/blob/d968708a5318457acdea8f61d6acd4d1db86833f/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%202/Python%20Basics%20with%20Numpy/Python_Basics_With_Numpy_v3a.ipynb
//!
//! I recommend first watching all of C1W2.

/// C01W02PA01 Part 1 - Building Basic Functions.
pub mod _1 {

    /// C01W02PA01 Part 1 Section 1 - Sigmoid Function.
    pub mod _1 {
        #[allow(unused_imports)]
        use crate::helpers::{Approx, _dfdx::*};

        pub mod sigmoid_basic_test {
            /// Returns sigmoid(x).
            pub fn sigmoid_scalar(x: f32) -> f32 {
                1. / (1. + f32::exp(-x))
            }

            /// Basic test for the [`sigmoid`] function.
            #[test]
            fn test_sigmoid_basic() {
                assert_eq!(sigmoid_scalar(3.), 0.95257413);
            }
        }

        pub const X: [f32; 3] = [1., 2., 3.];
        const XEXP: [f32; 3] = [2.7182817, 7.389056, 20.085537];
        const XP3: [f32; 3] = [4., 5., 6.];
        const XSIG: [f32; 3] = [0.73105854, 0.880797, 0.95257413];

        #[test]
        fn _1() {
            let dev = device();

            // creates a 1D tensor
            let x: TensorF32<Rank1<3>> = dev.tensor(X);

            // test exp(x)
            assert_eq!(x.clone().exp().array(), XEXP);

            // test x+3
            assert_eq!((x.clone() + 3.).array(), XP3);
        }

        /// Sigmoid for 1D Tensors.
        pub fn sigmoid<const M: usize>(x: TensorF32<Rank1<M>>) -> TensorF32<Rank1<M>> {
            (x.negate().exp() + 1.).recip()
        }

        #[test]
        fn _2() {
            let dev = device();

            // creates a 1D tensor
            let x: TensorF32<Rank1<3>> = dev.tensor(X);

            // test sigmoid(x)
            sigmoid(x.clone()).array().approx(XSIG, (1e-5, 0));
            assert_eq!(sigmoid(x.clone()).array(), (x.clone()).sigmoid().array());
        }
    }
    pub use _1::sigmoid;

    /// C01W02PA01 Part 1 Section 2 - Sigmoid Gradient.
    pub mod _2 {
        use super::sigmoid;
        pub use super::_1::X;
        use crate::helpers::_dfdx::*;

        const XDSIG: [f32; 3] = [0.19661196, 0.10499363, 0.045176655];

        /// Sigmoid Derivative for 1D Tensors.
        pub fn sigmoid_derivative<const M: usize>(x: TensorF32<Rank1<M>>) -> TensorF32<Rank1<M>> {
            let s = sigmoid(x);
            let s_ = s.clone();
            s * (s_.negate() + 1.)
        }

        #[test]
        fn _1() {
            use crate::helpers::Approx;

            let dev = device();

            // creates a 1D tensor
            let x: TensorF32<Rank1<3>> = dev.tensor(X);

            // test dsig(x)
            // note: CPU vs GPU imprecision
            assert!(sigmoid_derivative(x.clone())
                .array()
                .approx(XDSIG, (0.001, 0)));
        }

        // TODO: try to use the autodiff thing?
    }
    pub use _2::sigmoid_derivative;

    /// C01W02PA01 Part 1 Section 3 - Reshaping Arrays.
    #[allow(clippy::excessive_precision)]
    pub mod _3 {
        use crate::helpers::_dfdx::*;

        #[rustfmt::skip]
        pub const IMAGE: [[[f32; 2]; 3]; 3] = [
            [[0.67826139, 0.29380381],[0.90714982, 0.52835647],[0.4215251, 0.45017551],],
            [[0.92814219, 0.96677647],[0.85304703, 0.52351845],[0.19981397, 0.27417313],],
            [[0.60659855, 0.00533165],[0.10820313, 0.49978937],[0.34144279, 0.94630077],],
        ];
        #[rustfmt::skip]
        pub const IMAGE2: [[f32; 1]; 18] = [
            [0.67826139],[0.29380381],[0.90714982],[0.52835647],[0.4215251],[0.45017551],
            [0.92814219],[0.96677647],[0.85304703],[0.52351845],[0.19981397],[0.27417313],
            [0.60659855],[0.00533165],[0.10820313],[0.49978937],[0.34144279],[0.94630077],
        ];

        pub fn image2vector<const M: usize, const N: usize, const O: usize>(
            x: TensorF32<Rank3<M, N, O>>,
        ) -> TensorF32<Rank2<{ M * N * O }, 1>> {
            x.reshape()
        }

        #[test]
        fn _1() {
            let dev = device();
            let x = dev.tensor(IMAGE);
            assert_eq!(image2vector(x).array(), IMAGE2);
        }
    }

    /// C01W02PA01 Part 1 Section 4 - Normalizing Rows.
    pub mod _4 {
        use crate::helpers::_dfdx::*;
        use dfdx::tensor_ops::{BroadcastTo, SumTo};

        const X: [[f32; 3]; 2] = [[0., 3., 4.], [1., 6., 4.]];
        #[allow(clippy::excessive_precision)]
        const XNORM: [[f32; 3]; 2] = [[0., 0.6, 0.8], [0.13736056, 0.82416338, 0.54944226]];

        pub fn normalize_rows<const M: usize, const N: usize>(
            x: TensorF32<Rank2<M, N>>,
        ) -> TensorF32<Rank2<M, N>> {
            let x2 = x.clone().square();
            let row_sum = x2.sum::<Rank1<M>, Axis<1>>();
            let row_norms = row_sum.sqrt();
            let norms = row_norms.broadcast::<Rank2<M, N>, Axis<1>>();
            x / norms
        }

        #[test]
        fn _1() {
            let dev = device();
            let x = dev.tensor(X);
            assert_eq!(normalize_rows(x).array(), XNORM);
        }
    }
    pub use _4::normalize_rows;

    /// C01W02PA01 Part 1 Section 5 - Broadcasting and the Softmax Function.
    pub mod _5 {
        #[allow(unused_imports)]
        use crate::helpers::{Approx, _dfdx::*};
        use dfdx::tensor_ops::{BroadcastTo, SumTo};

        #[rustfmt::skip]
        const X: [[f32; 5]; 2] = [
            [9., 2., 5., 0., 0.],
            [7., 5., 0., 0., 0.]
        ];
        #[rustfmt::skip]
        const XSOFTMAX: [[f32; 5]; 2] = [
            [0.9808976, 0.00089446286, 0.017965766, 0.00012105238, 0.00012105238],
            [0.8786798, 0.118916385, 0.00080125226, 0.00080125226, 0.00080125226]
        ];
        #[rustfmt::skip]
        const XSOFTMAX2: [[f32; 5]; 2] = [
            [0.9808977, 0.0008944629, 0.01796577, 0.00012105239, 0.00012105239],
            [0.8786798, 0.118916385, 0.00080125226, 0.00080125226, 0.00080125226]
        ];

        pub fn softmax<const M: usize, const N: usize>(
            x: TensorF32<Rank2<M, N>>,
        ) -> TensorF32<Rank2<M, N>> {
            let x = x.exp();
            let row_sum = x.clone().sum::<Rank1<M>, Axis<1>>();
            let norms = row_sum.broadcast::<Rank2<M, N>, Axis<1>>();
            x / norms
        }

        #[test]
        fn _1() {
            let dev = device();
            let x = dev.tensor(X);
            assert!(softmax(x).array().approx(XSOFTMAX2, (1e-5, 0)),);
        }
    }
    pub use _5::softmax;
}
pub use _1::{normalize_rows, sigmoid, sigmoid_derivative, softmax};

/// C01W02PA01 Part 2 - Vectorization.
pub mod _2 {

    pub mod _0 {

        pub const X1: [f32; 15] = [9., 2., 5., 0., 0., 7., 5., 0., 0., 0., 9., 2., 5., 0., 0.];
        pub const X2: [f32; 15] = [9., 2., 2., 9., 0., 9., 2., 5., 0., 0., 9., 2., 5., 0., 0.];
        pub const INNER: f32 = 278.0;
        #[rustfmt::skip]
        pub const OUTER: [[f32; 15]; 15] = [
            [81.0, 18.0, 18.0, 81.0, 0.0, 81.0, 18.0, 45.0, 0.0, 0.0, 81.0, 18.0, 45.0, 0.0, 0.0],
            [18.0, 4.0, 4.0, 18.0, 0.0, 18.0, 4.0, 10.0, 0.0, 0.0, 18.0, 4.0, 10.0, 0.0, 0.0],
            [45.0, 10.0, 10.0, 45.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [63.0, 14.0, 14.0, 63.0, 0.0, 63.0, 14.0, 35.0, 0.0, 0.0, 63.0, 14.0, 35.0, 0.0, 0.0],
            [45.0, 10.0, 10.0, 45.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [81.0, 18.0, 18.0, 81.0, 0.0, 81.0, 18.0, 45.0, 0.0, 0.0, 81.0, 18.0, 45.0, 0.0, 0.0],
            [18.0, 4.0, 4.0, 18.0, 0.0, 18.0, 4.0, 10.0, 0.0, 0.0, 18.0, 4.0, 10.0, 0.0, 0.0],
            [45.0, 10.0, 10.0, 45.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0, 45.0, 10.0, 25.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ];
        #[rustfmt::skip]
        pub const ELEMENT_MUL: [f32; 15] = [ 81., 4., 10., 0., 0., 63., 10., 0., 0., 0., 81., 4., 25., 0., 0.];
        /// Generated randomly from the python code.
        #[rustfmt::skip]
        #[allow(clippy::excessive_precision)]
        pub const W: [[f32; 15]; 3] = [
            [0.49190569, 0.05771136, 0.91080532, 0.71868143, 0.90359229, 0.02728773, 0.56549385, 0.53695982, 0.23358076, 0.03571467, 0.82496654, 0.36568479, 0.29925488, 0.86398078, 0.0988409],
            [0.18888033, 0.54138987, 0.10623625, 0.75765972, 0.83991157, 0.84450423, 0.29730794, 0.62068761, 0.85898898, 0.65458875, 0.53452889, 0.40599233, 0.19362703, 0.24214533, 0.72533643],
            [0.42099865, 0.08020154, 0.29106363, 0.03501952, 0.71853045, 0.52188975, 0.39911255, 0.36212764, 0.54034859, 0.33574637, 0.26193187, 0.72861583, 0.52448071, 0.390605,   0.11923141]
        ];
        pub const GDOT: [f32; 3] = [21.767426, 17.302833, 17.490522];

        #[test]
        fn _1() {
            use crate::helpers::{Approx, _dfdx::*};
            let dev = device();

            let x1: TensorF32<Rank1<15>> = dev.tensor(X1);
            let x2: TensorF32<Rank1<15>> = dev.tensor(X2);

            // vectorized dot product of vectors
            // (the implementation is in the helpers)
            let dot = x1.clone().dot(x2.clone());
            assert_eq!(dot, INNER);

            // vectorized outer product
            // (the implementation is in the helpers)
            let outer = x1.clone().outer(x2.clone());
            assert_eq!(outer.array(), OUTER);

            // vectorized elementwise multiplication
            let mul = x1.clone() * x2.clone();
            assert_eq!(mul.array(), ELEMENT_MUL);

            // vectorized general dot product
            // (the implementation is in the helpers)
            let w: TensorF32<Rank2<3, 15>> = dev.tensor(W);
            let gdot = w.clone().dot(x1.clone());

            // note: cpu vs gpu imprecision
            assert!(gdot.array().approx(GDOT, (0.001, 0)));
        }
    }

    /// C01W02PA01 Part 2 Section 1 - Implement the L1 and L2 Loss Functions.
    pub mod _1 {
        use crate::helpers::_dfdx::*;

        const YHAT: [f32; 5] = [0.9, 0.2, 0.1, 0.4, 0.9];
        const Y: [f32; 5] = [1., 0., 0., 1., 1.];
        const L1: f32 = 1.1;
        const L2: f32 = 0.43;

        pub fn l1<const M: usize>(yhat: TensorF32<Rank1<M>>, y: TensorF32<Rank1<M>>) -> f32 {
            (y - yhat).abs().sum().array()
        }

        #[test]
        fn _test_l1() {
            let dev = device();

            let yhat: TensorF32<Rank1<5>> = dev.tensor(YHAT);
            let y: TensorF32<Rank1<5>> = dev.tensor(Y);

            let l1 = l1(yhat, y);
            assert_eq!(l1, L1);
        }

        pub fn l2<const M: usize>(yhat: TensorF32<Rank1<M>>, y: TensorF32<Rank1<M>>) -> f32 {
            (y - yhat).square().sum().array()
        }

        #[test]
        fn _test_l2() {
            let dev = device();

            let yhat: TensorF32<Rank1<5>> = dev.tensor(YHAT);
            let y: TensorF32<Rank1<5>> = dev.tensor(Y);

            let l2 = l2(yhat, y);
            assert_eq!(l2, L2);
        }
    }
}

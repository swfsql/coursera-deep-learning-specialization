pub mod pa_01_basics;
pub mod pa_02_logistic_regression;

pub mod prelude {
    pub use super::pa_01_basics::{normalize_rows, sigmoid, sigmoid_derivative, softmax};
    pub use super::pa_02_logistic_regression::{
        accuracy, Model, Optimization, PreparedData, Propagation, TrainedModel,
    };
}

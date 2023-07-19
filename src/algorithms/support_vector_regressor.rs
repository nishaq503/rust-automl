//! Support Vector Regressor

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    model_selection::cross_validate,
    model_selection::CrossValidationResult,
    svm::{
        svr::{SVRParameters as SmartcoreSVRParameters, SVR},
        Kernels, LinearKernel, PolynomialKernel, RBFKernel, SigmoidKernel,
    },
};

use crate::{Algorithm, Kernel, Settings};

/// The Support Vector Regressor.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/svm.html#svm-regression)
/// for a more in-depth description of the algorithm.
pub struct SupportVectorRegressorWrapper {}

impl super::ModelWrapper for SupportVectorRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        let parameters = SmartcoreSVRParameters::default()
            .with_tol(settings.svr_settings.as_ref().unwrap().tol)
            .with_c(settings.svr_settings.as_ref().unwrap().c)
            .with_eps(settings.svr_settings.as_ref().unwrap().c);
        let cv = match settings.svr_settings.as_ref().unwrap().kernel {
            Kernel::Linear => cross_validate(
                SVR::fit,
                x,
                y,
                parameters.with_kernel(Kernels::linear()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::Polynomial(degree, gamma, coef) => cross_validate(
                SVR::fit,
                x,
                y,
                parameters.with_kernel(Kernels::polynomial(degree, gamma, coef)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::RBF(gamma) => cross_validate(
                SVR::fit,
                x,
                y,
                parameters.with_kernel(Kernels::rbf(gamma)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Kernel::Sigmoid(gamma, coef) => cross_validate(
                SVR::fit,
                x,
                y,
                parameters.with_kernel(Kernels::sigmoid(gamma, coef)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
        };
        (cv, Algorithm::SVR)
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        let parameters = SmartcoreSVRParameters::default()
            .with_tol(settings.svr_settings.as_ref().unwrap().tol)
            .with_c(settings.svr_settings.as_ref().unwrap().c)
            .with_eps(settings.svr_settings.as_ref().unwrap().c);
        match settings.svr_settings.as_ref().unwrap().kernel {
            Kernel::Linear => bincode::serialize(
                &SVR::fit(x, y, parameters.with_kernel(Kernels::linear())).unwrap(),
            )
            .unwrap(),
            Kernel::Polynomial(degree, gamma, coef) => bincode::serialize(
                &SVR::fit(
                    x,
                    y,
                    parameters.with_kernel(Kernels::polynomial(degree, gamma, coef)),
                )
                .unwrap(),
            )
            .unwrap(),
            Kernel::RBF(gamma) => bincode::serialize(
                &SVR::fit(x, y, parameters.with_kernel(Kernels::rbf(gamma))).unwrap(),
            )
            .unwrap(),
            Kernel::Sigmoid(gamma, coef) => bincode::serialize(
                &SVR::fit(x, y, parameters.with_kernel(Kernels::sigmoid(gamma, coef))).unwrap(),
            )
            .unwrap(),
        }
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        match settings.svr_settings.as_ref().unwrap().kernel {
            Kernel::Linear => {
                let model: SVR<f32, DenseMatrix<f32>, LinearKernel> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::Polynomial(_, _, _) => {
                let model: SVR<f32, DenseMatrix<f32>, PolynomialKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::RBF(_) => {
                let model: SVR<f32, DenseMatrix<f32>, RBFKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Kernel::Sigmoid(_, _) => {
                let model: SVR<f32, DenseMatrix<f32>, SigmoidKernel<f32>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}

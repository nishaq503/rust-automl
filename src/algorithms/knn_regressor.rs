//! KNN Regressor

use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    math::distance::{
        euclidian::Euclidian, hamming::Hamming, mahalanobis::Mahalanobis, manhattan::Manhattan,
        minkowski::Minkowski, Distances,
    },
    model_selection::cross_validate,
    model_selection::CrossValidationResult,
    neighbors::knn_regressor::{
        KNNRegressor, KNNRegressorParameters as SmartcoreKNNRegressorParameters,
    },
};

use crate::{Algorithm, Distance, Settings};

/// The KNN Regressor.
///
/// See [scikit-learn's user guide](https://scikit-learn.org/stable/modules/neighbors.html#regression)
/// for a more in-depth description of the algorithm.
pub struct KNNRegressorWrapper {}

impl super::ModelWrapper for KNNRegressorWrapper {
    fn cv(
        x: &DenseMatrix<f32>,
        y: &Vec<f32>,
        settings: &Settings,
    ) -> (CrossValidationResult<f32>, Algorithm) {
        let parameters = SmartcoreKNNRegressorParameters::default()
            .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
            .with_algorithm(
                settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .algorithm
                    .clone(),
            )
            .with_weight(
                settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .weight
                    .clone(),
            );

        let cv = match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                parameters.with_distance(Distances::euclidian()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Manhattan => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                parameters.with_distance(Distances::manhattan()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Minkowski(p) => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                parameters.with_distance(Distances::minkowski(p)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Mahalanobis => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                parameters.with_distance(Distances::mahalanobis(x)),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
            Distance::Hamming => cross_validate(
                KNNRegressor::fit,
                x,
                y,
                parameters.with_distance(Distances::hamming()),
                settings.get_kfolds(),
                settings.get_metric(),
            )
            .unwrap(),
        };

        (cv, Algorithm::KNNRegressor)
    }

    fn train(x: &DenseMatrix<f32>, y: &Vec<f32>, settings: &Settings) -> Vec<u8> {
        let parameters = SmartcoreKNNRegressorParameters::default()
            .with_k(settings.knn_regressor_settings.as_ref().unwrap().k)
            .with_algorithm(
                settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .algorithm
                    .clone(),
            )
            .with_weight(
                settings
                    .knn_regressor_settings
                    .as_ref()
                    .unwrap()
                    .weight
                    .clone(),
            );
        match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => bincode::serialize(
                &KNNRegressor::fit(x, y, parameters.with_distance(Distances::euclidian())).unwrap(),
            )
            .unwrap(),
            Distance::Manhattan => bincode::serialize(
                &KNNRegressor::fit(x, y, parameters.with_distance(Distances::manhattan())).unwrap(),
            )
            .unwrap(),
            Distance::Minkowski(p) => bincode::serialize(
                &KNNRegressor::fit(x, y, parameters.with_distance(Distances::minkowski(p)))
                    .unwrap(),
            )
            .unwrap(),
            Distance::Mahalanobis => bincode::serialize(
                &KNNRegressor::fit(x, y, parameters.with_distance(Distances::mahalanobis(x)))
                    .unwrap(),
            )
            .unwrap(),
            Distance::Hamming => bincode::serialize(
                &KNNRegressor::fit(x, y, parameters.with_distance(Distances::hamming())).unwrap(),
            )
            .unwrap(),
        }
    }

    fn predict(x: &DenseMatrix<f32>, final_model: &Vec<u8>, settings: &Settings) -> Vec<f32> {
        match settings.knn_regressor_settings.as_ref().unwrap().distance {
            Distance::Euclidean => {
                let model: KNNRegressor<f32, Euclidian> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Manhattan => {
                let model: KNNRegressor<f32, Manhattan> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Minkowski(_) => {
                let model: KNNRegressor<f32, Minkowski> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Mahalanobis => {
                let model: KNNRegressor<f32, Mahalanobis<f32, DenseMatrix<f32>>> =
                    bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
            Distance::Hamming => {
                let model: KNNRegressor<f32, Hamming> = bincode::deserialize(final_model).unwrap();
                model.predict(x).unwrap()
            }
        }
    }
}

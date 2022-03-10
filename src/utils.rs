use smartcore::{algorithm::neighbour::KNNAlgorithmName, neighbors::KNNWeightFunction};
use std::fmt::{Debug, Display, Formatter};
use std::ops::BitAnd;

pub(crate) fn print_option<T: Display>(x: Option<T>) -> String {
    match x {
        None => "None".to_string(),
        Some(y) => format!("{}", y),
    }
}
pub(crate) fn debug_option<T: Debug>(x: Option<T>) -> String {
    match x {
        None => "None".to_string(),
        Some(y) => format!("{:#?}", y),
    }
}

pub(crate) fn print_knn_weight_function(f: &KNNWeightFunction) -> String {
    match f {
        KNNWeightFunction::Uniform => "Uniform".to_string(),
        KNNWeightFunction::Distance => "Distance".to_string(),
    }
}

pub(crate) fn print_knn_search_algorithm(a: &KNNAlgorithmName) -> String {
    match a {
        KNNAlgorithmName::LinearSearch => "Linear Search".to_string(),
        KNNAlgorithmName::CoverTree => "Cover Tree".to_string(),
    }
}

/// Kernel options for use with support vector machines
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Kernel {
    /// Linear Kernel
    Linear,

    /// Polynomial kernel
    Polynomial(f32, f32, f32),

    /// Radial basis function kernel
    RBF(f32),

    /// Sigmoid kernel
    Sigmoid(f32, f32),
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernel::Linear => write!(f, "Linear"),
            Kernel::Polynomial(degree, gamma, coef) => write!(
                f,
                "Polynomial\n    degree = {}\n    gamma = {}\n    coef = {}",
                degree, gamma, coef
            ),
            Kernel::RBF(gamma) => write!(f, "RBF\n    gamma = {}", gamma),
            Kernel::Sigmoid(gamma, coef) => {
                write!(f, "Sigmoid\n    gamma = {}\n    coef = {}", gamma, coef)
            }
        }
    }
}

/// Distance metrics
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Distance {
    /// Euclidean distance
    Euclidean,

    /// Manhattan distance
    Manhattan,

    /// Minkowski distance, parameterized by p
    Minkowski(u16),

    /// Mahalanobis distance
    Mahalanobis,

    /// Hamming distance
    Hamming,
}

impl Display for Distance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Distance::Euclidean => write!(f, "Euclidean"),
            Distance::Manhattan => write!(f, "Manhattan"),
            Distance::Minkowski(n) => write!(f, "Minkowski\n    p = {}", n),
            Distance::Mahalanobis => write!(f, "Mahalanobis"),
            Distance::Hamming => write!(f, "Hamming"),
        }
    }
}

/// Function to do element-wise multiplication fo two vectors
pub fn elementwise_multiply(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    v1.iter().zip(v2).map(|(&i1, &i2)| i1 * i2).collect()
}

#[cfg(any(feature = "csv"))]
use polars::prelude::{
    BooleanChunked, BooleanChunkedBuilder, CsvReader, DataFrame, DataType, PolarsError, SerReader,
};

#[cfg(any(feature = "csv"))]
pub(crate) fn validate_and_read(file_path: &str, header: bool) -> DataFrame {
    match CsvReader::from_path(file_path) {
        Ok(csv) => csv
            .infer_schema(Some(10))
            .has_header(header)
            .finish()
            .expect("Cannot read file as CSV")
            .remove_nulls()
            .expect("Cannot remove null values")
            .convert_to_float()
            .expect("Cannot convert types"),
        Err(_) => {
            if let Ok(_) = url::Url::parse(file_path) {
                let file_contents = minreq::get(file_path).send().unwrap();
                let temp = temp_file::with_contents(file_contents.as_bytes());

                CsvReader::from_path(temp.path())
                    .expect("Cannot find file")
                    .infer_schema(Some(10))
                    .has_header(header)
                    .finish()
                    .expect("Cannot read file as CSV")
                    .remove_nulls()
                    .expect("Cannot remove null values")
                    .convert_to_float()
                    .expect("Cannot convert types")
            } else {
                panic!("The string {} is not a valid URL or file path.", file_path)
            }
        }
    }
}
#[cfg(any(feature = "csv"))]
trait Cleanup {
    fn convert_to_float(self) -> Result<DataFrame, PolarsError>;
    fn remove_nulls(self) -> Result<DataFrame, PolarsError>;
}

#[cfg(any(feature = "csv"))]
impl Cleanup for DataFrame {
    fn convert_to_float(self) -> Result<DataFrame, PolarsError> {
        for field in self.schema().fields() {
            let name = field.name();
            println!("{}", field.data_type().to_string());
            match field.data_type() {
                DataType::Boolean => {}
                DataType::Utf8 => {}
                DataType::Date => {}
                DataType::Datetime => {}
                DataType::Time => {}
                DataType::List(_) => {}
                DataType::Null => {}
                DataType::Categorical => {}
                _ => {}
            }
        }

        Ok(self)
    }

    fn remove_nulls(self) -> Result<DataFrame, PolarsError> {
        let mut mask = self.get_columns()[0].is_not_null();
        for column in self.get_columns() {
            mask = column.is_not_null().bitand(mask);
        }
        Ok(self.filter(&mask).expect("Cannot mask"))
    }
}

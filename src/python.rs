use std::usize;

use numpy::ndarray::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;

use crate::{
    Batch, BatchEncoding, Encoding, Node, Span, StaticBatcher, Text, TokenizedText, Tokenizer,
    TxtLoader,
};

impl<T: ToPyObjectConsume> Node for Box<dyn Node<Output = T>> {
    type Output = T;
    fn get(&self, index: usize) -> Option<Self::Output> {
        self.get(index)
    }
    fn len(&self) -> Option<usize> {
        self.len()
    }
    fn next(&mut self) -> Option<Self::Output> {
        self.next()
    }
}

pub trait ToPyObjectConsume {
    fn to_object_consume(self, py: Python<'_>) -> PyObject;
}

impl ToPyObjectConsume for Text {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.text.to_object(py)
    }
}

impl ToPyObjectConsume for TokenizedText {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.encoding.to_object_consume(py)
    }
}

#[pyclass(name = "Encoding")]
pub struct EncodingPy {
    input_ids: Array1<u32>,
    #[pyo3(get)]
    pad_token: u32,
}
impl ToPyObjectConsume for Encoding {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let encoding = EncodingPy {
            input_ids: self.input_ids,
            pad_token: self.pad_token,
        };
        encoding.into_py(py)
    }
}

#[pymethods]
impl EncodingPy {
    #[getter]
    fn get_input_ids(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.input_ids.to_pyarray(py).to_owned()
    }
}

#[pyclass]
struct BatchEncodingPy {
    inner: BatchEncoding,
}

#[pymethods]
impl BatchEncodingPy {
    #[getter]
    fn input_ids(&self, py: Python<'_>) -> Py<PyArray2<u32>> {
        self.inner.input_ids.to_pyarray(py).to_owned()
    }
}

impl ToPyObjectConsume for BatchEncoding {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let encoding = BatchEncodingPy { inner: self };
        encoding.into_py(py)
    }
}

impl ToPyObjectConsume for Batch {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.encoding.to_object_consume(py)
    }
}

impl ToPyObjectConsume for Span {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

macro_rules! add_py_node {
    ($($native_type:ty: $py_type:ident: $py_name:expr,)*) => {
        $(
            #[pyclass(name = $py_name)]
            struct $py_type {
                inner: Option<Box<dyn Node<Output = $native_type>>>,
            }

            #[pymethods]
            impl $py_type {
                fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                    slf
                }
                fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
                    match self {
                        $py_type { inner: None } => Err(PyErr::new::<PyBaseException, _>(
                            "This node is already in use by another node.",
                        )),
                        $py_type { inner: Some(node) } => match node.next() {
                            None => Ok(None),
                            Some(result) => Ok(Some(result.to_object_consume(py))),
                        },
                    }
                }
            }
        )*
    };
}

type SameType<T> = T;
macro_rules! add_node_constructor {
    ($rust_constructor_name:ident: $py_constructor_name:expr => (node: &mut $input_node:ty, $($arg_name:ident: $arg_type:ty,)*) => $node_type_py:ty: $node_type_rust:ty) => {
        #[pyfunction(name = $py_constructor_name)]
        fn $rust_constructor_name(node: &mut $input_node, $($arg_name: $arg_type,)*) -> PyResult<$node_type_py> {
            match node.inner.take() {
                None => Err(PyErr::new::<PyBaseException, _>(
                    "This node is already in use by another node.",
                )),
                Some(node) => match <$node_type_rust>::new(node, $($arg_name,)*) {
                    Err(err) => Err(PyErr::new::<PyBaseException, _>(format!("{}", err))),
                    Ok(tokenizer) => Ok(SameType::<$node_type_py> {
                        inner: Some(Box::new(tokenizer)),
                    }),
                },
            }
        }
    };
    ($rust_constructor_name:ident: $py_constructor_name:expr => ($($arg_name:ident: $arg_type:ty,)*) => $node_type_py:ty: $node_type_rust:ty) => {
        #[pyfunction(name = $py_constructor_name)]
        fn $rust_constructor_name($($arg_name: $arg_type,)*) -> PyResult<$node_type_py> {
            match <$node_type_rust>::new($($arg_name,)*) {
                Err(err) => Err(PyErr::new::<PyBaseException, _>(format!("{}", err))),
                Ok(tokenizer) => Ok(SameType::<$node_type_py> {
                    inner: Some(Box::new(tokenizer)),
                }),
            }
        }
    };
}

add_py_node!(
    Text: TextNodePy: "TextNode",
    TokenizedText: TokenizedTextNodePy: "TokenizedTextNode",
    Batch: BatchNodePy: "BatchNode",
);

add_node_constructor!(create_txt_loader: "TxtLoader" => (filename: String,) => TextNodePy: TxtLoader);
add_node_constructor!(create_tokenizer: "Tokenizer" => (node: &mut TextNodePy, tokenizer: String,) => TokenizedTextNodePy: Tokenizer<Box<dyn Node<Output = Text>>>);
add_node_constructor!(create_static_batcher: "StaticBatcher" => (node: &mut TokenizedTextNodePy, batch_size: usize, seq_length: usize,) => BatchNodePy: StaticBatcher<Box<dyn Node<Output = TokenizedText>>>);

#[pymodule]
fn pyo3_test(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_txt_loader, m)?)?;
    m.add_function(wrap_pyfunction!(create_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(create_static_batcher, m)?)?;
    Ok(())
}

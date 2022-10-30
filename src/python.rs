use std::any::Any;
use std::usize;

use numpy::ndarray::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;

use crate::{
    datasets::{SQuADLoader, TxtLoader},
    Batch, BatchEncoding, BatchLabel, Encoding, Label, NoLabel, NoTokenizedLabel, Node, Span,
    StaticBatcher, Text, TextPair, TokenizedLabel, TokenizedSpan, TokenizedText, Tokenizer,
};
use crate::{BatchSpan, NoBatchLabel};

pub trait ToPyObjectConsume: Send {
    fn to_object_consume(self, py: Python<'_>) -> PyObject;
}

impl<T: Label> ToPyObjectConsume for Text<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.text.to_object(py)
    }
}

impl<T: Label> ToPyObjectConsume for TextPair<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.text.to_object(py)
    }
}

impl<T: TokenizedLabel> ToPyObjectConsume for TokenizedText<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let TokenizedText { encoding, label } = self;
        (encoding.to_object_consume(py), label.to_object_consume(py)).into_py(py)
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

#[pyclass(name = "BatchEncoding")]
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

impl<T: BatchLabel> ToPyObjectConsume for Batch<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        (
            self.encoding.to_object_consume(py),
            self.labels.to_object_consume(py),
        )
            .to_object(py)
    }
}

impl ToPyObjectConsume for NoLabel {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let none: Option<()> = None;
        none.to_object(py)
    }
}
impl ToPyObjectConsume for Span {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl ToPyObjectConsume for NoTokenizedLabel {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let none: Option<()> = None;
        none.to_object(py)
    }
}

impl ToPyObjectConsume for TokenizedSpan {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl ToPyObjectConsume for NoBatchLabel {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let none: Option<()> = None;
        none.to_object(py)
    }
}

#[pyclass(name = "BatchSpan")]
pub struct BatchSpanPy {
    start: Array1<usize>,
    end: Array1<usize>,
}
impl ToPyObjectConsume for BatchSpan {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        let BatchSpan { start, end } = self;
        BatchSpanPy {
            start: start,
            end: end,
        }
        .into_py(py)
    }
}
#[pymethods]
impl BatchSpanPy {
    #[getter]
    fn get_start(&self, py: Python<'_>) -> Py<PyArray1<usize>> {
        self.start.to_pyarray(py).to_owned()
    }
    #[getter]
    fn get_end(&self, py: Python<'_>) -> Py<PyArray1<usize>> {
        self.end.to_pyarray(py).to_owned()
    }
}

struct NodeWrapper<T: ToPyObjectConsume>(Box<dyn Node<Output = T>>);

impl<T: ToPyObjectConsume> Node for NodeWrapper<T> {
    type Output = T;
    fn get(&self, index: usize) -> Option<Self::Output> {
        self.0.get(index)
    }
    fn len(&self) -> Option<usize> {
        self.0.len()
    }
    fn next(&mut self) -> Option<Self::Output> {
        self.0.next()
    }
}

trait NodePyOutput {
    fn get(&self, index: usize, py: Python<'_>) -> Option<PyObject>;
    fn len(&self) -> Option<usize>;
    fn next(&mut self, py: Python<'_>) -> Option<PyObject>;
    fn get_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<S: ToPyObjectConsume + 'static, T: Node<Output = S> + 'static> NodePyOutput for T {
    fn next(&mut self, py: Python<'_>) -> Option<PyObject> {
        match self.next() {
            Some(output) => Some(output.to_object_consume(py)),
            None => None,
        }
    }
    fn len(&self) -> Option<usize> {
        self.len()
    }
    fn get(&self, index: usize, py: Python<'_>) -> Option<PyObject> {
        match self.get(index) {
            Some(output) => Some(output.to_object_consume(py)),
            None => None,
        }
    }
    fn get_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(NodeWrapper(Box::new(*self)))
    }
}

#[pyclass(name = "Node")]
struct NodePy {
    inner: Option<Box<dyn NodePyOutput + Send>>,
}

#[pymethods]
impl NodePy {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self {
            NodePy { inner: None } => Err(PyErr::new::<PyBaseException, _>(
                "This node is already in use by another node.",
            )),
            NodePy { inner: Some(node) } => Ok(node.next(py)),
        }
    }
}

macro_rules! add_node_constructor {
    ($rust_constructor_name:ident: $py_constructor_name:expr => (node: &mut $input_node:ty, $($arg_name:ident: $arg_type:ty,)*) => $node_type_rust:ty { $($input_type:ty),+ }) => {
        #[pyfunction(name = $py_constructor_name)]
        fn $rust_constructor_name(node: &mut $input_node, $($arg_name: $arg_type,)*) -> PyResult<NodePy> {
            #[allow(unused_assignments)] {
                match node.inner.take() {
                    None => {
                        return Err(PyErr::new::<PyBaseException, _>(
                            "This node is already in use by another node.",
                        ))
                    },
                    Some(node) => {
                        let mut node = node.get_any();
                        add_node_constructor!(call node, $node_type_rust, ($($input_type,)+), ($($arg_name),*));
                    }
                }
            }
            return Err(PyErr::new::<PyBaseException, _>(
                "The provided input node is not compatible :(",
            ))
        }
    };
    ($rust_constructor_name:ident: $py_constructor_name:expr => ($($arg_name:ident: $arg_type:ty,)*) => $node_type_rust:tt) => {
        #[pyfunction(name = $py_constructor_name)]
        fn $rust_constructor_name($($arg_name: $arg_type,)*) -> PyResult<NodePy> {
            match <$node_type_rust>::new($($arg_name,)*) {
                Err(err) => Err(PyErr::new::<PyBaseException, _>(format!("{}", err))),
                Ok(node) => Ok(NodePy {
                    inner: Some(Box::new(node)),
                }),
            }
        }
    };
    (call $node:ident, $node_type_rust:ty, ($($input_type:ty,)+), $args:tt) => {
        $(
            match $node.downcast::<NodeWrapper<$input_type>>() {
                Ok(node) => {
                    let node = *node;
                    return match add_node_constructor!(hi node, $node_type_rust, $args) {
                        Err(err) => Err(PyErr::new::<PyBaseException, _>(format!("{}", err))),
                        Ok(tokenizer) => {
                            Ok(NodePy {
                                inner: Some(Box::new(tokenizer)),
                            })
                        }
                    }
                }
                Err(node) => {
                    $node = node;
                }
            }
        )+
    };
    (hi $node:ident, $node_type_rust:ty, ($($arg_name:ident),*)) => {
        <$node_type_rust>::new($node, $($arg_name,)*)
    };
}

add_node_constructor!(create_txt_loader: "TxtLoader" => (filename: String,) => TxtLoader);
add_node_constructor!(create_squad_loader: "SQuADLoader" => (filename: String,) => SQuADLoader);
add_node_constructor!(create_tokenizer: "Tokenizer" => (node: &mut NodePy, tokenizer: String,) => Tokenizer<_> {Text<NoLabel>, Text<Span>, TextPair<Span>});
add_node_constructor!(create_static_batcher: "StaticBatcher" => (node: &mut NodePy, batch_size: usize, seq_length: usize,) => StaticBatcher<_, _> {TokenizedText<NoTokenizedLabel>, TokenizedText<TokenizedSpan>});

#[pymodule]
#[pyo3(name = "ayp")]
fn pyo3_test(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_txt_loader, m)?)?;
    m.add_function(wrap_pyfunction!(create_squad_loader, m)?)?;
    m.add_function(wrap_pyfunction!(create_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(create_static_batcher, m)?)?;
    m.add_class::<EncodingPy>()?;
    m.add_class::<BatchEncodingPy>()?;
    m.add_class::<BatchSpanPy>()?;
    m.add_class::<NodePy>()?;
    Ok(())
}

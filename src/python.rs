use std::usize;

use numpy::ndarray::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;

use crate::{
    Batch, BatchEncoding, BatchLabel, Encoding, Label, Node, Span, StaticBatcher, Text,
    TokenizedLabel, TokenizedText, Tokenizer, TxtLoader,
};

pub trait ToPyObjectConsume {
    fn to_object_consume(self, py: Python<'_>) -> PyObject;
}

impl<T: Label> ToPyObjectConsume for Text<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.text.to_object(py)
    }
}

impl<T: TokenizedLabel> ToPyObjectConsume for TokenizedText<T> {
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

impl<T: BatchLabel> ToPyObjectConsume for Batch<T> {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.encoding.to_object_consume(py)
    }
}

impl ToPyObjectConsume for Span {
    fn to_object_consume(self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

struct NodePy<T: Node> {
    inner: Option<T>,
}

unsafe impl<T: Node> pyo3::type_object::PyTypeInfo for NodePy<T> {
    type AsRefTarget = pyo3::PyCell<Self>;
    const NAME: &'static str = "TxtLoader";
    const MODULE: ::std::option::Option<&'static str> = ::core::option::Option::None;
    #[inline]
    fn type_object_raw(py: pyo3::Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        use pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}
impl<T: Node> pyo3::PyClass for NodePy<T> {
    type Dict = pyo3::impl_::pyclass::PyClassDummySlot;
    type WeakRef = pyo3::impl_::pyclass::PyClassDummySlot;
    type BaseNativeType = pyo3::PyAny;
}
impl<'a, T: Node> pyo3::derive_utils::ExtractExt<'a> for &'a NodePy<T> {
    type Target = pyo3::PyRef<'a, NodePy<T>>;
}
impl<'a, T: Node> pyo3::derive_utils::ExtractExt<'a> for &'a mut NodePy<T> {
    type Target = pyo3::PyRefMut<'a, NodePy<T>>;
}
impl<T: Node> pyo3::IntoPy<pyo3::PyObject> for NodePy<T> {
    fn into_py(self, py: pyo3::Python) -> pyo3::PyObject {
        pyo3::IntoPy::into_py(pyo3::Py::new(py, self).unwrap(), py)
    }
}

impl<T: Node> pyo3::impl_::pyclass::PyClassImpl for NodePy<T> {
    const DOC: &'static str = "\u{0}";
    const IS_BASETYPE: bool = false;
    const IS_SUBCLASS: bool = false;
    const IS_MAPPING: bool = false;
    type Layout = pyo3::PyCell<Self>;
    type BaseType = pyo3::PyAny;
    type ThreadChecker = pyo3::impl_::pyclass::ThreadCheckerStub<NodePy<T>>;
    fn for_all_items(visitor: &mut dyn ::std::ops::FnMut(&pyo3::impl_::pyclass::PyClassItems)) {
        use pyo3::impl_::pyclass::*;
        let collector = PyClassImplCollector::<Self>::new();
        static INTRINSIC_ITEMS: PyClassItems = PyClassItems {
            methods: &[],
            slots: &[],
        };
        visitor(&INTRINSIC_ITEMS);
        visitor(collector.py_methods());
        visitor(collector.object_protocol_items());
        visitor(collector.number_protocol_items());
        visitor(collector.iter_protocol_items());
        visitor(collector.gc_protocol_items());
        visitor(collector.descr_protocol_items());
        visitor(collector.mapping_protocol_items());
        visitor(collector.sequence_protocol_items());
        visitor(collector.async_protocol_items());
        visitor(collector.buffer_protocol_items());
    }
}
impl<T: Node> NodePy<T> {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self {
            NodePy { inner: None } => Err(PyErr::new::<PyBaseException, _>(
                "This node is already in use by another node.",
            )),
            NodePy { inner: Some(node) } => match node.next() {
                None => Ok(None),
                Some(result) => Ok(Some(result.to_object_consume(py))),
            },
        }
    }
}

impl<T: Node> pyo3::impl_::pyclass::PyMethods<NodePy<T>>
    for pyo3::impl_::pyclass::PyClassImplCollector<NodePy<T>>
{
    fn py_methods(self) -> &'static pyo3::impl_::pyclass::PyClassItems {
        static ITEMS: pyo3::impl_::pyclass::PyClassItems = pyo3::impl_::pyclass::PyClassItems {
            methods: &[],
            slots: &[
                {
                    unsafe extern "C" fn __wrap(
                        _raw_slf: *mut pyo3::ffi::PyObject,
                    ) -> *mut pyo3::ffi::PyObject {
                        let _slf = _raw_slf;
                        let gil = pyo3::GILPool::new();
                        let _py = gil.python();
                        pyo3::callback::panic_result_into_callback_output(
                            _py,
                            ::std::panic::catch_unwind(move || -> pyo3::PyResult<_> {
                                let _cell = _py
                                    .from_borrowed_ptr::<pyo3::PyAny>(_slf)
                                    .downcast::<pyo3::PyCell<NodePy<TxtLoader>>>()?;
                                #[allow(clippy::useless_conversion)]
                                let _slf = ::std::convert::TryFrom::try_from(_cell)?;
                                pyo3::callback::convert(_py, NodePy::<TxtLoader>::__iter__(_slf))
                            }),
                        )
                    }
                    pyo3::ffi::PyType_Slot {
                        slot: pyo3::ffi::Py_tp_iter,
                        pfunc: __wrap as pyo3::ffi::getiterfunc as _,
                    }
                },
                {
                    unsafe extern "C" fn __wrap(
                        _raw_slf: *mut pyo3::ffi::PyObject,
                    ) -> *mut pyo3::ffi::PyObject {
                        let _slf = _raw_slf;
                        let gil = pyo3::GILPool::new();
                        let _py = gil.python();
                        pyo3::callback::panic_result_into_callback_output(
                            _py,
                            ::std::panic::catch_unwind(move || -> pyo3::PyResult<_> {
                                let _cell = _py
                                    .from_borrowed_ptr::<pyo3::PyAny>(_slf)
                                    .downcast::<pyo3::PyCell<NodePy<TxtLoader>>>()?;
                                let mut _ref = _cell.try_borrow_mut()?;
                                let _slf: &mut NodePy<TxtLoader> = &mut *_ref;
                                let _result: pyo3::PyResult<
                                    pyo3::class::iter::IterNextOutput<_, _>,
                                > = pyo3::callback::convert(
                                    _py,
                                    NodePy::<TxtLoader>::__next__(_slf, _py),
                                );
                                pyo3::callback::convert(_py, _result)
                            }),
                        )
                    }
                    pyo3::ffi::PyType_Slot {
                        slot: pyo3::ffi::Py_tp_iternext,
                        pfunc: __wrap as pyo3::ffi::iternextfunc as _,
                    }
                },
            ],
        };
        &ITEMS
    }
}
/*macro_rules! add_py_node {
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
add_node_constructor!(create_static_batcher: "StaticBatcher" => (node: &mut TokenizedTextNodePy, batch_size: usize, seq_length: usize,) => BatchNodePy: StaticBatcher<Box<dyn Node<Output = TokenizedText>>>);*/

#[pyfunction]
fn create_txt_loader(filename: String) -> NodePy<TxtLoader> {
    NodePy {
        inner: Some(TxtLoader::new(filename).unwrap()),
    }
}

#[pymodule]
#[pyo3(name = "preprocessing")]
fn pyo3_test(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /*    m.add_function(wrap_pyfunction!(create_txt_loader, m)?)?;
    m.add_function(wrap_pyfunction!(create_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(create_static_batcher, m)?)?;*/
    m.add_function(wrap_pyfunction!(create_txt_loader, m)?)?;
    Ok(())
}

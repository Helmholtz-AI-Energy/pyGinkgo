// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "python.hpp"
#include "utils.hpp"

namespace py = pybind11;

// Helper to extract a device pointer and element count from an object
// that exposes __cuda_array_interface__ (e.g. a CuPy ndarray).
#ifdef GINKGO_BUILD_CUDA
template <typename T>
static std::pair<T *, size_t> cai_ptr_and_size(py::object obj)
{
    auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto data = cai["data"].cast<py::tuple>();
    auto shape = cai["shape"].cast<py::tuple>();
    return {reinterpret_cast<T *>(data[0].cast<uintptr_t>()),
            shape[0].cast<size_t>()};
}
#endif

template <template <typename ValueType, typename IndexType> class MatrixType,
          typename ValueType, typename IndexType>
void init_matrix(py::module_ &module, const std::string matrix_type,
                 const std::string value_type, const std::string index_type)
{
    std::string matrix_type_repr = "pygko.matrix." + matrix_type + " object";
    std::string value_index_str = value_type + "_" + index_type;
    std::string pyclass_name = matrix_type + "_" + value_index_str;

    auto cls =
        py::class_<MatrixType<ValueType, IndexType>,
                   std::shared_ptr<MatrixType<ValueType, IndexType>>,
                   gko::LinOp>(module, pyclass_name.c_str(),
                               py::buffer_protocol())
            .def(py::init([](std::shared_ptr<gko::Executor> exec) {
                return MatrixType<ValueType, IndexType>::create(exec);
            }))
            .def(py::init([](std::shared_ptr<gko::Executor> exec,
                             py::tuple dim, gko::array<ValueType> &vals,
                             gko::array<IndexType> &cols,
                             gko::array<IndexType> &rows) {
                return gko::share(MatrixType<ValueType, IndexType>::create(
                    exec,
                    gko::dim<2>{dim[0].cast<size_t>(),
                                dim[1].cast<size_t>()},
                    vals, cols, rows));
            }))
            .def(py::init([](std::shared_ptr<gko::Executor> exec,
                             py::tuple dim, py::buffer data, py::buffer cols,
                             py::buffer rows) {
                /* Request a buffer descriptor from Python */
                py::buffer_info data_info = data.request();
                check_buffer_dtype<ValueType>(data_info);

                py::buffer_info cols_info = cols.request();
                check_buffer_dtype<IndexType>(cols_info);

                py::buffer_info rows_info = rows.request();
                check_buffer_dtype<IndexType>(rows_info);

                auto ref = gko::ReferenceExecutor::create();

                /* create a view into numpy data */
                auto nnz = data_info.shape[0];

                auto data_view = gko::array<ValueType>::view(
                    ref, nnz, (ValueType *)data_info.ptr);
                auto cols_view = gko::array<IndexType>::view(
                    ref, nnz, (IndexType *)cols_info.ptr);

                auto rows_view = gko::array<IndexType>::view(
                    ref, rows_info.shape[0], (IndexType *)rows_info.ptr);

                return gko::share(MatrixType<ValueType, IndexType>::create(
                    exec,
                    gko::dim<2>{dim[0].cast<size_t>(),
                                dim[1].cast<size_t>()},
                    data_view, cols_view, rows_view));
            }))
            .def("__repr__",
                 [=](const MatrixType<ValueType, IndexType> &o) {
                     return matrix_type_repr;
                 })
            .def("get_num_stored_elements",
                 &MatrixType<ValueType, IndexType>::get_num_stored_elements,
                 "Get the number of non zero elements")
            .def(
                "T",
                [](MatrixType<ValueType, IndexType> &m) {
                    return gko::share(m.transpose());
                },
                "Computes the transpose of a matrix")
            .def(
                "get_size",
                [](const MatrixType<ValueType, IndexType> &m) {
                    // Deprecation in favor of .shape
                    // https://stackoverflow.com/a/62559865
                    PyErr_WarnEx(PyExc_DeprecationWarning,
                                 ".get_size() is deprecated, use .shape "
                                 "instead",
                                 1);
                    return m.get_size();
                },
                "Get the size of the matrix")
            .def_property_readonly(
                "size",
                [](const MatrixType<ValueType, IndexType> &m) {
                    // Deprecation in favor of .shape
                    // https://stackoverflow.com/a/62559865
                    PyErr_WarnEx(PyExc_DeprecationWarning,
                                 ".size is deprecated, use .shape instead", 1);
                    return m.get_size();
                },
                "Get the size of the matrix")
            .def_property_readonly(
                "shape",
                [](const MatrixType<ValueType, IndexType> &m) {
                    auto size = m.get_size();
                    return py::make_tuple(size[0], size[1]);
                },
                "Get the size of the matrix")
            .def(
                "convert_to_dense",
                [](MatrixType<ValueType, IndexType> &m) {
                    auto exec = m.get_executor();
                    auto dense = gko::share(
                        gko::matrix::Dense<ValueType>::create(exec));
                    m.convert_to(dense);
                    return dense;
                },
                "Returns dense representation of the matrix");

#ifdef GINKGO_BUILD_CUDA
    // -----------------------------------------------------------------
    // CUDA / CuPy interop  (zero-copy via gko::array::view)
    // -----------------------------------------------------------------

    // Factory: create a sparse matrix from CuPy device arrays.
    // The three arrays (values, col_idxs, row_ptrs/row_idxs) are
    // wrapped as non-owning views; py::keep_alive ensures the source
    // Python objects stay alive while the Ginkgo matrix exists.
    cls.def_static(
        "from_device_ptrs",
        [](std::shared_ptr<gko::Executor> exec, py::tuple dim,
           py::object values_obj, py::object col_idxs_obj,
           py::object row_component_obj) {
            auto [vals_ptr, nnz] = cai_ptr_and_size<ValueType>(values_obj);
            auto [cols_ptr, cols_n] =
                cai_ptr_and_size<IndexType>(col_idxs_obj);
            auto [rows_ptr, rows_n] =
                cai_ptr_and_size<IndexType>(row_component_obj);

            auto vals_view =
                gko::array<ValueType>::view(exec, nnz, vals_ptr);
            auto cols_view =
                gko::array<IndexType>::view(exec, cols_n, cols_ptr);
            auto rows_view =
                gko::array<IndexType>::view(exec, rows_n, rows_ptr);

            return gko::share(MatrixType<ValueType, IndexType>::create(
                exec,
                gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
                std::move(vals_view), std::move(cols_view),
                std::move(rows_view)));
        },
        py::keep_alive<0, 3>(),  // prevent GC of values array
        py::keep_alive<0, 4>(),  // prevent GC of col_idxs array
        py::keep_alive<0, 5>(),  // prevent GC of row_ptrs/row_idxs array
        py::arg("exec"), py::arg("dim"), py::arg("values"),
        py::arg("col_idxs"), py::arg("row_component"),
        "Create a sparse matrix by wrapping CuPy device arrays as "
        "zero-copy views (via gko::array::view). "
        "The source arrays must stay alive while the matrix is in use; "
        "this is handled automatically by py::keep_alive.");

    // Expose raw device pointers for the internal component arrays so
    // that the Python layer can construct CuPy sparse matrices from a
    // Ginkgo sparse matrix without a device→host round-trip.

    cls.def(
        "get_values_device_ptr",
        [](const MatrixType<ValueType, IndexType> &m) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(m.get_const_values());
        },
        "Return the device pointer to the values array as an integer.");

    cls.def(
        "get_col_idxs_device_ptr",
        [](const MatrixType<ValueType, IndexType> &m) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(m.get_const_col_idxs());
        },
        "Return the device pointer to the column indices array.");

    // CSR: row_ptrs  (n_rows + 1 elements)
    // COO: row_idxs  (nnz elements)
    if constexpr (std::is_same_v<MatrixType<ValueType, IndexType>,
                                 gko::matrix::Csr<ValueType, IndexType>>) {
        cls.def(
            "get_row_ptrs_device_ptr",
            [](const gko::matrix::Csr<ValueType, IndexType> &m) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(m.get_const_row_ptrs());
            },
            "Return the device pointer to the row-pointers array "
            "(CSR format, n_rows + 1 elements).");
    } else {
        cls.def(
            "get_row_idxs_device_ptr",
            [](const gko::matrix::Coo<ValueType, IndexType> &m) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(m.get_const_row_idxs());
            },
            "Return the device pointer to the row-indices array "
            "(COO format, nnz elements).");
    }
#endif

    std::string read_fn = "read_" + matrix_type + "_" + value_index_str;
    module.def(read_fn.c_str(), [](const std::string &fn,
                                   std::shared_ptr<gko::Executor> exec) {
        return gko::share(gko::read<MatrixType<ValueType, IndexType>>(
            std::ifstream(fn), exec));
    });
}

void init_sparse_all_types(py::module_ &module)
{
#define DECLARE_COO_MATRIX(ValueType, IndexType)         \
    init_matrix<gko::matrix::Coo, ValueType, IndexType>( \
        module, "Coo", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_COO_MATRIX);

#define DECLARE_CSR_MATRIX(ValueType, IndexType)         \
    init_matrix<gko::matrix::Csr, ValueType, IndexType>( \
        module, "Csr", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_CSR_MATRIX);
}

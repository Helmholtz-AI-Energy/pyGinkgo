// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

#define GKO_MATRIX_BINDING(Name)                                               \
  py::class_<gko::matrix::Name<ValueType>,                                     \
             std::shared_ptr<gko::matrix::Name<ValueType>>, gko::LinOp>(       \
      module_matrix, #Name, py::buffer_protocol())                             \
      .def(py::init([](std::shared_ptr<gko::Executor> exec) {                  \
        return gko::matrix::Name<ValueType>::create(exec);                     \
      }))                                                                      \
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,     \
                       gko::array<ValueType> &vals,                            \
                       gko::array<IndexType> &cols,                            \
                       gko::array<IndexType> &rows) {                          \
        return gko::share(gko::matrix::Name<ValueType>::create(                \
            exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},   \
            vals, cols, rows));                                                \
      }))                                                                      \
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,     \
                       py::buffer data, py::buffer rows, py::buffer cols) {    \
        /* Request a buffer descriptor from Python */                          \
        py::buffer_info data_info = data.request();                            \
        py::buffer_info rows_info = rows.request();                            \
        py::buffer_info cols_info = cols.request();                            \
                                                                               \
        if (data_info.format != py::format_descriptor<ValueType>::format())    \
          throw std::runtime_error(                                            \
              "Provided values have an incompatible dtype");                   \
                                                                               \
        if (rows_info.format != py::format_descriptor<IndexType>::format())    \
          throw std::runtime_error(                                            \
              "Provided rows have an incompatible dtype");                     \
                                                                               \
        if (cols_info.format != py::format_descriptor<IndexType>::format())    \
          throw std::runtime_error(                                            \
              "Provided cols have an incompatible dtype");                     \
                                                                               \
        auto ref = gko::ReferenceExecutor::create();                           \
                                                                               \
        /* create a view into numpy data */                                    \
        auto nnz = data_info.shape[0];                                         \
                                                                               \
        auto data_view =                                                       \
            gko::array<ValueType>::view(ref, nnz, (ValueType *)data_info.ptr); \
        auto rows_view =                                                       \
            gko::array<IndexType>::view(ref, nnz, (IndexType *)rows_info.ptr); \
        auto cols_view =                                                       \
            gko::array<IndexType>::view(ref, nnz, (IndexType *)cols_info.ptr); \
                                                                               \
        return gko::share(gko::matrix::Name<ValueType>::create(                \
            exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},   \
            data_view, cols_view, rows_view));                                 \
      }))                                                                      \
      .def(                                                                    \
          "apply",                                                             \
          [](const gko::matrix::Name<ValueType, IndexType> &m,                 \
             std::shared_ptr<const gko::LinOp> b,                              \
             std::shared_ptr<gko::LinOp> x) { m.apply(b, x); },                \
          "")                                                                  \
      .def("__repr__",                                                         \
           [](const gko::matrix::Name<ValueType> &o) {                         \
             auto str = std::string("pygko.matrix.Name object");               \
             return str;                                                       \
           })                                                                  \
      .def("get_num_stored_elements",                                          \
           &gko::matrix::Name<ValueType>::get_num_stored_elements,             \
           "Applies Name matrix axpy to a vector (or a sequence of vectors)."  \
           "Performs the operation x = Name * b + x")

void init_coo(py::module_ &module_matrix) {
  module_matrix.def("read_Coo", [](const std::string &fn,
                                   std::shared_ptr<gko::Executor> exec) {
    return gko::share(
        gko::read<gko::matrix::Coo<ValueType>>(std::ifstream(fn), exec));
  });

  GKO_MATRIX_BINDING(Coo);
}

void init_csr(py::module_ &module_matrix) {
  module_matrix.def("read_Csr", [](const std::string &fn,
                                   std::shared_ptr<gko::Executor> exec) {
    return gko::share(
        gko::read<gko::matrix::Csr<ValueType>>(std::ifstream(fn), exec));
  });

  GKO_MATRIX_BINDING(Csr);
}

// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

void init_array(py::module_ &);
void init_dense(py::module_ &);
void init_coo(py::module_ &);
void init_csr(py::module_ &);
void add_allocator_classes(py::module_ &);
void add_stream_classes(py::module_ &);
void add_executor_classes(py::module_ &);

PYBIND11_MODULE(pyGinkgo, m) {
  m.doc() = "Python bindings for the Ginkgo framework";

  py::class_<gko::LinOp, std::shared_ptr<gko::LinOp>>(m, "LinOp");

  py::class_<gko::ptr_param<gko::LinOp>>(m, "ptr_param");

  py::class_<gko::dim<2>>(m, "dim2").def(
      py::init<unsigned long, unsigned long>());

  add_allocator_classes(m);
  add_stream_classes(m);
  add_executor_classes(m);

  py::module_ module_base =
      m.def_submodule("base", "Submodule for Ginkgos low level type bindings");
  init_array(module_base);

  py::module_ module_matrix =
      m.def_submodule("matrix", "Submodule for Ginkgos matrix type bindings");

  init_dense(module_matrix);
  init_coo(module_matrix);
  init_csr(module_matrix);
}

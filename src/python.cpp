// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

void init_array(py::module_ &);
void init_dense(py::module_ &);
void init_coo(py::module_ &);
void init_csr(py::module_ &);

PYBIND11_MODULE(pyGinkgo, m) {
  m.doc() = "Python bindings for the Ginkgo framework";

  py::class_<gko::LinOp, std::shared_ptr<gko::LinOp>>(m, "LinOp");

  py::class_<gko::ptr_param<gko::LinOp>>(m, "ptr_param");

  py::class_<gko::dim<2>>(m, "dim2").def(
      py::init<unsigned long, unsigned long>());

  py::class_<gko::Executor, std::shared_ptr<gko::Executor>>(m, "Executor");

  py::class_<gko::detail::ExecutorBase<gko::OmpExecutor>, gko::Executor,
             std::shared_ptr<gko::detail::ExecutorBase<gko::OmpExecutor>>>(
      m, "OmpExecutorBase");

  py::class_<gko::OmpExecutor, gko::detail::ExecutorBase<gko::OmpExecutor>,
             std::shared_ptr<gko::OmpExecutor>>(m, "OmpExecutor")
      .def(py::init([]() { return gko::OmpExecutor::create(); }));

  py::class_<gko::ReferenceExecutor, gko::OmpExecutor,
             std::shared_ptr<gko::ReferenceExecutor>>(m, "ReferenceExecutor")
      .def(py::init([]() { return gko::ReferenceExecutor::create(); }));

  py::module_ module_base =
      m.def_submodule("base", "Submodule for Ginkgos low level type bindings");
  init_array(module_base);

  py::module_ module_matrix =
      m.def_submodule("matrix", "Submodule for Ginkgos matrix type bindings");

  init_dense(module_matrix);
  init_coo(module_matrix);
  init_csr(module_matrix);
}

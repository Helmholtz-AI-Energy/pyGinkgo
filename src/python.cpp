// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

template<typename ValueType>
void declare_array(py::module_&);

PYBIND11_MODULE(pyGinkgo, m) {
  m.doc() = "Python bindings for the Ginkgo framework";

  py::class_<gko::ptr_param<gko::LinOp>>(m, "ptr_param");

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

    declare_array<float>(module_base);
    declare_array<double>(module_base);
    declare_array<int>(module_base);
    declare_array<long>(module_base);
}

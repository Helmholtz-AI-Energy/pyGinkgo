// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

namespace py = pybind11;

void init_array_all_types(py::module_ &);
void init_dense(py::module_ &);
void init_coo(py::module_ &);
void init_csr(py::module_ &);
void init_logger(py::module_ &);
void init_gmres(py::module_ &);
void init_direct(py::module_ &);
void init_config_solver(py::module_ &);
void init_ilu(py::module_ &);
void add_allocator_classes(py::module_ &);
void add_stream_classes(py::module_ &);
void add_executor_classes(py::module_ &);

PYBIND11_MODULE(pyGinkgoBindings, m)
{
    m.doc() = "Python bindings for the Ginkgo framework";

    add_allocator_classes(m);
    add_stream_classes(m);
    add_executor_classes(m);

    py::class_<gko::LinOp, std::shared_ptr<gko::LinOp>>(m, "LinOp")
        .def("get_executor", &gko::PolymorphicObject::get_executor,
             "Get the executor")
        .def(
            "apply",
            [](const gko::LinOp &m, std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { m.apply(b, x); },
            "");

    py::class_<gko::dim<2>>(m, "dim2")
        .def(py::init<unsigned long, unsigned long>())
        .def("__getitem__",
             [](const gko::dim<2> &o, size_t i) { return o[i]; });

    py::module_ module_base = m.def_submodule(
        "base", "Submodule for Ginkgos low level type bindings");
    init_array_all_types(module_base);

    py::module_ module_matrix =
        m.def_submodule("matrix", "Submodule for Ginkgos matrix type bindings");

    init_dense(module_matrix);
    init_coo(module_matrix);
    init_csr(module_matrix);

    py::module_ module_logger =
        m.def_submodule("logger", "Submodule for Ginkgos logger type bindings");
    init_logger(module_logger);

    py::module_ module_preconditioner = m.def_submodule(
        "preconditioner", "Submodule for Ginkgos preconditioner type bindings");
    init_ilu(module_preconditioner);

    py::module_ module_solver =
        m.def_submodule("solver", "Submodule for Ginkgos solver type bindings");
    init_gmres(module_solver);
    init_direct(module_solver);
    init_config_solver(module_solver);
}

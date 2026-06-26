// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
// SPDX-FileCopyrightText: 2026 Imad Kissami
//
// SPDX-License-Identifier: MIT

#include "python.hpp"

namespace py = pybind11;

void init_array_all_types(py::module_ &);
void init_dense_all_types(py::module_ &);
void init_sparse_all_types(py::module_ &);
void init_logger_all_types(py::module_ &);
void init_gmres_all_types(py::module_ &);
void init_direct_all_types(py::module_ &);
void init_trs_all_types(py::module_ &);
void init_cg_all_types(py::module_ &);
void init_factorization(py::module_ &);
void init_config_solver_all_types(py::module_ &);
void init_ilu_all_types(py::module_ &);
void add_allocator_classes(py::module_ &);
void add_stream_classes(py::module_ &);
void add_dpcpp_queue_property_enum(py::module_ &);
void add_executor_classes(py::module_ &);
void init_distributed(py::module_ &);

PYBIND11_MODULE(pyGinkgoBindings, m)
{
    m.doc() = "Python bindings for the Ginkgo framework";

    add_allocator_classes(m);
    add_stream_classes(m);
    add_dpcpp_queue_property_enum(m);
    add_executor_classes(m);

    // Declaring common types
    py::class_<gko::PolymorphicObject, std::shared_ptr<gko::PolymorphicObject>>(
        m, "PolymorphicObject")
        .def("get_executor", &gko::PolymorphicObject::get_executor,
             "Get the executor");

    py::class_<gko::LinOp, gko::PolymorphicObject, std::shared_ptr<gko::LinOp>>(
        m, "LinOp")
        .def(
            "apply",
            [](const gko::LinOp &m, std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { m.apply(b, x); },
            py::arg("b"), py::arg("x"),
            "Applies a linear operator to a vector (or a sequence of "
            "vectors).\n\n"
            "Performs the operation x = op(b), where op is this linear "
            "operator.\n\n"
            "@param b  the input vector(s) on which the operator is applied\n"
            "@param x  the output vector(s) where the result is stored\n\n"
            "@return this");

    py::class_<gko::dim<2>>(m, "dim2")
        .def(py::init<unsigned long, unsigned long>())
        .def("__getitem__",
             [](const gko::dim<2> &o, size_t i) { return o[i]; });


    // Declaring submodules
    py::module_ module_base = m.def_submodule(
        "base", "Submodule for Ginkgos low level type bindings");
    init_array_all_types(module_base);

    py::module_ module_matrix =
        m.def_submodule("matrix", "Submodule for Ginkgos matrix type bindings");

    init_dense_all_types(module_matrix);
    init_sparse_all_types(module_matrix);

    py::module_ module_logger =
        m.def_submodule("logger", "Submodule for Ginkgos logger type bindings");
    init_logger_all_types(module_logger);

    py::module_ module_solver =
        m.def_submodule("solver", "Submodule for Ginkgos solver type bindings");
    init_gmres_all_types(module_solver);
    init_direct_all_types(module_solver);
    init_config_solver_all_types(module_solver);
    init_trs_all_types(module_solver);
    init_cg_all_types(module_solver);

    py::module_ module_factorization = m.def_submodule(
        "factorization", "Submodule for Ginkgos factorization type bindings");
    init_factorization(module_factorization);

    py::module_ module_preconditioner = m.def_submodule(
        "preconditioner", "Submodule for Ginkgos preconditioner type bindings");
    init_ilu_all_types(module_preconditioner);

    py::module_ module_distributed = m.def_submodule(
        "distributed", "Submodule for Ginkgos MPI-distributed type bindings");
    init_distributed(module_distributed);
}

// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType, typename IndexType>
void init_jacobi(py::module_& module_preconditioner,
                 const std::string value_type, const std::string index_type)
{
    using Jacobi = gko::preconditioner::Jacobi<ValueType, IndexType>;
    std::string pyclass_name = "Jacobi_" + value_type + "_" + index_type;
    std::string repr_str = "pygko.preconditioner." + pyclass_name + " object";

    py::class_<Jacobi, std::shared_ptr<Jacobi>, gko::LinOp>(
        module_preconditioner, pyclass_name.c_str(),
        "Block-Jacobi preconditioner. Works as a local preconditioner for "
        "distributed solvers via the Schwarz wrapper.")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         gko::uint32 max_block_size) {
                 auto fact = gko::share(Jacobi::build()
                                            .with_max_block_size(max_block_size)
                                            .on(exec));
                 return gko::share(fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"),
             py::arg("max_block_size") = 1u,
             "max_block_size=1 = scalar Jacobi (diagonal scaling).")
        .def("__repr__", [=](const Jacobi&) { return repr_str; });
}

void init_jacobi_all_types(py::module_& module_preconditioner)
{
#define DECLARE_JACOBI(ValueType, IndexType)                                 \
    init_jacobi<ValueType, IndexType>(module_preconditioner, #ValueType,     \
                                      #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(
        DECLARE_JACOBI);
#undef DECLARE_JACOBI
}

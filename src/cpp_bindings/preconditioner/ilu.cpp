// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType, typename IndexType>
#if GKO_VERSION_MAJOR == 1
using Ilu =
    gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                             gko::solver::UpperTrs<ValueType, IndexType>, false,
                             IndexType>;
#else
using Ilu = gko::preconditioner::Ilu<ValueType, false, IndexType>;
#endif

template <typename ValueType, typename IndexType>
void init_ilu(py::module_ &module_preconditioner, const std::string value_type,
              const std::string index_type)
{
    std::string pyclass_name = "Ilu_" + value_type + "_" + index_type;
    std::string repr_str = "pygko.preconditioner." + pyclass_name + " object";
    py::class_<Ilu<ValueType, IndexType>,
               std::shared_ptr<Ilu<ValueType, IndexType>>, gko::LinOp>(
        module_preconditioner, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            auto par_ilu_fact =
                gko::factorization::ParIlu<ValueType, IndexType>::build().on(
                    exec);
            // Generate concrete factorization for input matrix
            auto par_ilu = gko::share(par_ilu_fact->generate(system_matrix));

            // Generate an ILU preconditioner factory by setting lower and upper
            // triangular solver - in this case the exact triangular solves
            auto ilu_pre_factory = Ilu<ValueType, IndexType>::build().on(exec);

            // Use incomplete factors to generate ILU preconditioner
            return gko::share(ilu_pre_factory->generate(par_ilu));
        }))
        .def("__repr__",
             [=](const Ilu<ValueType, IndexType> &o) { return repr_str; });
}

void init_ilu_all_types(py::module_ &module_preconditioner)
{
#define DECLARE_ILU_PRECONDITIONER(ValueType, IndexType)              \
    init_ilu<ValueType, IndexType>(module_preconditioner, #ValueType, \
                                   #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_ILU_PRECONDITIONER);
}

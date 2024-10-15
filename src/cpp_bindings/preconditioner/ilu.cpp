// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

void init_ilu(py::module_ &module_solver)
{
    py::class_<gko::solver::Ilu, std::shared_ptr<gko::preconditioner::Ilu>,
               gko::LinOp>(module_solver, "Ilu")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            auto par_ilu_fact =
                gko::factorization::ParIlu<ValueType, IndexType>::build().on(
                    exec);
            // Generate concrete factorization for input matrix
            auto par_ilu = gko::share(par_ilu_fact->generate(system_matrix));

            // Generate an ILU preconditioner factory by setting lower and upper
            // triangular solver - in this case the exact triangular solves
            auto ilu_pre_factory =
                gko::preconditioner::Ilu<
                    gko::solver::LowerTrs<ValueType, IndexType>,
                    gko::solver::UpperTrs<ValueType, IndexType>, false>::build()
                    .on(exec);

            // Use incomplete factors to generate ILU preconditioner
            return gko::share(ilu_pre_factory->generate(par_ilu));
        }))
        .def("__repr__", [](const gko::solver::Gmres<ValueType> &o) {
            auto str = std::string("pygko.preconditioner.Ilu object");
            return str;
        });
}

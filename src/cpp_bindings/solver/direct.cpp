// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

void init_direct(py::module_ &module_solver)
{
    py::class_<gko::experimental::solver::Direct<ValueType, IndexType>,
               std::shared_ptr<
                   gko::experimental::solver::Direct<ValueType, IndexType>>,
               gko::LinOp>(module_solver, "direct")
        .def(
            py::init([](std::shared_ptr<gko::Executor> exec,
                        std::shared_ptr<const gko::LinOp> system_matrix,
                        std::string factorization) {
                std::shared_ptr<gko::LinOpFactory> factorization_fact;

                if (factorization == "Cholesky") {
                    factorization_fact =
                        gko::share(gko::experimental::factorization::Cholesky<
                                       ValueType, IndexType>::build()
                                       .with_skip_sorting(true)
                                       .on(exec));
                }
                if (factorization == "LU") {
                    factorization_fact = gko::share(
                        gko::experimental::factorization::Lu<ValueType,
                                                             IndexType>::build()
                            .with_skip_sorting(true)
                            .on(exec));
                }

                auto solver_fact = gko::share(
                    gko::experimental::solver::Direct<ValueType,
                                                      IndexType>::build()
                        .with_factorization(factorization_fact)
                        .on(exec));
                return gko::share(solver_fact->generate(system_matrix));
            }),
            py::arg("exec"), py::arg("system_matrix"), py::arg("factorization"))
        .def("__repr__",
             [](const gko::experimental::solver::Direct<ValueType, IndexType>
                    &o) {
                 auto str = std::string("pygko.solver.Direct object");
                 return str;
             })
        .def(
            "apply",
            [](const gko::experimental::solver::Direct<ValueType, IndexType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}

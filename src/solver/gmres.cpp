// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"

void init_gmres(py::module_ &module_solver) {
  py::class_<gko::solver::Gmres<ValueType>,
             std::shared_ptr<gko::solver::Gmres<ValueType>>, gko::LinOp>(
      module_solver, "gmres")
      .def(py::init([](std::shared_ptr<gko::Executor> exec,
                       std::shared_ptr<const gko::LinOp> system_matrix,
                       size_t max_iters, size_t krylov_dim,
                       ValueType reduction_factor) {
        auto fact = gko::share(
            gko::solver::Gmres<ValueType>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(max_iters),
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_baseline(gko::stop::mode::rhs_norm)
                        .with_reduction_factor(reduction_factor))
                .with_krylov_dim(krylov_dim)
                .on(exec));
        return gko::share(fact->generate(system_matrix));
      }))
      .def("__repr__",
           [](const gko::solver::Gmres<ValueType> &o) {
             auto str = std::string("pygko.solver.Gmres object");
             return str;
           })
      .def(
          "apply",
          [](const gko::solver::Gmres<ValueType> &d,
             std::shared_ptr<const gko::LinOp> b,
             std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
          "");
}

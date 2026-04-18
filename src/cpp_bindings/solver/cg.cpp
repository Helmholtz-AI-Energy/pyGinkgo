// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <tuple>

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType>
void init_cg(py::module_& module_solver, const std::string value_type)
{
    using Cg = gko::solver::Cg<ValueType>;
    std::string pyclass_name = "cg_" + value_type;
    std::string repr_str = "pygko.solver." + pyclass_name + " object";

    auto initialize_logger = [](Cg& o) {
        std::shared_ptr<gko::log::Convergence<ValueType>> logger =
            gko::log::Convergence<ValueType>::create();
        o.add_logger(logger);
        return logger;
    };

    py::class_<Cg, std::shared_ptr<Cg>, gko::LinOp>(module_solver,
                                                    pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         size_t max_iters, ValueType reduction_factor,
                         bool relative_stop_mode) {
                 auto stop_mode = (relative_stop_mode)
                                      ? gko::stop::mode::rhs_norm
                                      : gko::stop::mode::absolute;
                 auto fact = gko::share(
                     Cg::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(
                                 max_iters),
                             gko::stop::ResidualNorm<ValueType>::build()
                                 .with_baseline(stop_mode)
                                 .with_reduction_factor(reduction_factor))
                         .on(exec));
                 return gko::share(fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"), py::arg("max_iters"),
             py::arg("reduction_factor"), py::arg("relative_stop_mode") = true,
             "Build a single-process or distributed CG solver.\n\n"
             "system_matrix may be a single-process LinOp (e.g. CSR) or a "
             "gko.distributed.Matrix; CG dispatches polymorphically.")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         std::shared_ptr<const gko::LinOp> preconditioner,
                         size_t max_iters, ValueType reduction_factor,
                         bool relative_stop_mode) {
                 auto stop_mode = (relative_stop_mode)
                                      ? gko::stop::mode::rhs_norm
                                      : gko::stop::mode::absolute;
                 auto fact = gko::share(
                     Cg::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(
                                 max_iters),
                             gko::stop::ResidualNorm<ValueType>::build()
                                 .with_baseline(stop_mode)
                                 .with_reduction_factor(reduction_factor))
                         .with_generated_preconditioner(preconditioner)
                         .on(exec));
                 return gko::share(fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"),
             py::arg("preconditioner"), py::arg("max_iters"),
             py::arg("reduction_factor"), py::arg("relative_stop_mode") = true)
        .def("initialize_logger", initialize_logger)
        .def("__repr__", [=](const Cg&) { return repr_str; })
        .def(
            "apply",
            [=](Cg& d, std::shared_ptr<const gko::LinOp> b,
                std::shared_ptr<gko::LinOp> x) {
                auto logger = initialize_logger(d);
                d.apply(b, x);
                return std::make_tuple(logger, x);
            },
            py::arg("b"), py::arg("x"),
            "Solve A x = b in place. Returns (logger, x).");
}

void init_cg_all_types(py::module_& module_solver)
{
#define DECLARE_CG_SOLVER(ValueType) \
    init_cg<ValueType>(module_solver, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(DECLARE_CG_SOLVER);
#undef DECLARE_CG_SOLVER
}

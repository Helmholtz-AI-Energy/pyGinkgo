// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <tuple>

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType>
void init_cg(py::module_ &module_solver, const std::string value_type)
{
    std::string pyclass_name = "cg_" + value_type;
    std::string repr_str = "pygko.solver." + pyclass_name + " object";

    auto initialize_logger = [](gko::solver::Cg<ValueType> &o) {
        std::shared_ptr<gko::log::Convergence<ValueType>> convergence_logger =
            gko::log::Convergence<ValueType>::create();
        o.add_logger(convergence_logger);
        return convergence_logger;
    };

    py::class_<gko::solver::Cg<ValueType>,
               std::shared_ptr<gko::solver::Cg<ValueType>>, gko::LinOp>(
        module_solver, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         size_t max_iters, ValueType reduction_factor,
                         bool relative_stop_mode) {
                 auto stop_mode = (relative_stop_mode)
                                      ? gko::stop::mode::rhs_norm
                                      : gko::stop::mode::absolute;
                 auto factory = gko::share(
                     gko::solver::Cg<ValueType>::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(
                                 max_iters),
                             gko::stop::ResidualNorm<ValueType>::build()
                                 .with_baseline(stop_mode)
                                 .with_reduction_factor(reduction_factor))
                         .on(exec));
                 return gko::share(factory->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"), py::arg("max_iters"),
             py::arg("reduction_factor"), py::arg("relative_stop_mode"))
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix,
                         std::shared_ptr<const gko::LinOp> preconditioner,
                         size_t max_iters, ValueType reduction_factor,
                         bool relative_stop_mode) {
                 auto stop_mode = (relative_stop_mode)
                                      ? gko::stop::mode::rhs_norm
                                      : gko::stop::mode::absolute;
                 auto factory = gko::share(
                     gko::solver::Cg<ValueType>::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(
                                 max_iters),
                             gko::stop::ResidualNorm<ValueType>::build()
                                 .with_baseline(stop_mode)
                                 .with_reduction_factor(reduction_factor))
                         .with_generated_preconditioner(preconditioner)
                         .on(exec));
                 return gko::share(factory->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"),
             py::arg("preconditioner"), py::arg("max_iters"),
             py::arg("reduction_factor"), py::arg("relative_stop_mode"))
        // TODO: not sure, whether we actually still need this function
        .def("initialize_logger", initialize_logger)
        .def("__repr__",
             [=](const gko::solver::Cg<ValueType> &o) { return repr_str; })
        .def(
            "apply",
            [=](gko::solver::Cg<ValueType> &d,
                std::shared_ptr<const gko::LinOp> b,
                std::shared_ptr<gko::LinOp> x) {
                auto logger = initialize_logger(d);
                d.apply(b, x);
                return std::make_tuple(logger, x);
            },
            py::arg("b"), py::arg("x"),
            "Applies the solver on b (rhs) and x (initial guess).\n"
            "Changes the initial guess inplace.\n"
            "Returns a tuple: (logger object, result)");
}

void init_cg_all_types(py::module_ &module_solver)
{
#define DECLARE_CG_SOLVER(ValueType) \
    init_cg<ValueType>(module_solver, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_CG_SOLVER);
}

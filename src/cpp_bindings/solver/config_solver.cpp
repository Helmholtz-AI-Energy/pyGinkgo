// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
// SPDX-FileCopyrightText: 2026 Imad Kissami
//
// SPDX-License-Identifier: MIT

#include <ginkgo/extensions/config/json_config.hpp>
#include <ginkgo/ginkgo.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../python.hpp"
#include "../utils.hpp"

namespace py = pybind11;

// TODO: This does not need the template. Either binding type_descriptor or
// accept the dtype= like numpy array.
template <typename ValueType>
std::shared_ptr<gko::LinOp> config_solver(std::shared_ptr<gko::Executor> exec,
                                          std::shared_ptr<gko::LinOp> A,
                                          std::string json)
{
    auto config = gko::ext::config::parse_json(nlohmann::json::parse(json));
    // Create the registry, which allows passing the existing data into config
    // This example does not use existing data.
    auto reg = gko::config::registry();
    // generate the linopfactory on the given executors
    // Global index type int32 (matches the int32-global distributed binding;
    // unused for non-distributed types, which only need ValueType + int32 local).
    auto solver_gen =
        gko::config::parse(
            config, reg,
            gko::config::make_type_descriptor<ValueType, gko::int32, gko::int32>())
            .on(exec);

    // Create solver
    auto solver = solver_gen->generate(A);

    return solver;
}

template <typename ValueType>
std::shared_ptr<const gko::log::Convergence<ValueType>> config_solve(
    std::shared_ptr<gko::Executor> exec, std::shared_ptr<gko::LinOp> A,
    std::shared_ptr<gko::LinOp> b, std::shared_ptr<gko::LinOp> x,
    std::string json)
{
    // Create solver
    auto solver = config_solver<ValueType>(exec, A, json);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    solver->apply(b, x);

    return logger;
}

template <typename ValueType>
void init_config_solver(py::module_ &m, const std::string value_type)
{
    std::string pyfunc_name = "config_solve_" + value_type;
    m.def(pyfunc_name.c_str(), &config_solve<ValueType>,
          "wrapper for config solve");
    pyfunc_name = "config_solver_" + value_type;
    m.def(pyfunc_name.c_str(), &config_solver<ValueType>,
          "wrapper for generating config solver");
}

void init_config_solver_all_types(py::module_ &m)
{
#define DECLARE_CONFIG_SOLVER(ValueType) \
    init_config_solver<ValueType>(m, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_CONFIG_SOLVER);
#undef DECLARE_CONFIG_SOLVER
}

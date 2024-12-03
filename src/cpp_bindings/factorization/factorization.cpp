// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"


void init_factorization(py::module_ &module_factorization)
{
    py::class_<
        gko::experimental::factorization::Factorization<ValueType, IndexType>,
        std::shared_ptr<gko::experimental::factorization::Factorization<
            ValueType, IndexType>>,
        gko::LinOp>(module_factorization, "Factorization")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            auto mtx =
                gko::as<gko::matrix::Csr<ValueType, IndexType>>(system_matrix);
            return gko::share(
                gko::experimental::factorization::Factorization<
                    ValueType,
                    IndexType>::create_from_combined_ldu(mtx->clone()));
        }))
        .def("get_lower_factor",
             &gko::experimental::factorization::Factorization<
                 ValueType, IndexType>::get_lower_factor,
             "Returns lower factors");
}

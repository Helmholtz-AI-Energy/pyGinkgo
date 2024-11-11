
// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "../python.hpp"


void init_cholesky(py::module_ &module_factorization)
{
    //     py::class_<
    //         gko::experimental::factorization::Cholesky<ValueType, IndexType>,
    //         std::shared_ptr<
    //             gko::experimental::factorization::Cholesky<ValueType,
    //             IndexType>>
    //         // ,gko::experimental::factorization::Factorization<ValueType,
    //         IndexType>> (module_factorization, "Cholesky")
    //         .def(py::init([](std::shared_ptr<gko::Executor> exec,
    //                          std::shared_ptr<const gko::LinOp> system_matrix)
    //                          {
    //             auto fact = gko::share(
    //                 gko::experimental::factorization::Cholesky<ValueType,
    //                                                            IndexType>::build()
    //                     // .on(exec)
    // );
    //             return gko::share(fact->generate(system_matrix));
    //         }));
}

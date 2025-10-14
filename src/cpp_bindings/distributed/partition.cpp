// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include "../python.hpp"
#include "../utils.hpp"

void init_partition(py::module_ &module)
{
    using Partition = gko::experimental::distributed::Partition<int, int>;
    py::class_<Partition, std::shared_ptr<Partition>>(module, "Partition")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         gko::experimental::mpi::communicator comm) {
            return gko::share(
                Partition::build_from_global_size_uniform(exec, 1, 1));
        }));
}

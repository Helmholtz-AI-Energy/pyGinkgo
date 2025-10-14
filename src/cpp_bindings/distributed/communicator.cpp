// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType>
void init_communicator(py::module_ &module)
{
    py::class_<gko::experimental::mpi::communicator>(module, "Communicator")
        .def(py::init([](const MPI_Comm &comm, bool force_host_buffer) {
            return gko::experimental::mpi::communicator(comm,
                                                        force_host_buffer);
        }));
}

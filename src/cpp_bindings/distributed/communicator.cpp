// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include "../python.hpp"
#include "../utils.hpp"

void init_communicator(py::module_ &module)
{
    using Communicator = gko::experimental::mpi::communicator;
    py::class_<Communicator>(module, "Communicator")
        // TODO not working check
        // https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
        // if helpful
        //.def(py::init([](const MPI_Comm &comm, bool force_host_buffer) {
        .def(py::init([](bool force_host_buffer) {
            return Communicator(MPI_COMM_WORLD, force_host_buffer);
        }))
        .def_property_readonly("size", &Communicator::size,
                               "get the size of the communicator");
}

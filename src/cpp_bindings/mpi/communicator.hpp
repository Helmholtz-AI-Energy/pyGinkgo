// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef PYGINKGO_BUILD_MPI

#include <mpi.h>
#include <ginkgo/core/base/mpi.hpp>

#include "../python.hpp"

namespace pyginkgo_mpi {

/// Idempotently runs mpi4py's runtime initialization. Throws std::runtime_error
/// on failure. Must be called before any PyMPIComm_Get usage in this module.
void ensure_mpi4py_imported();

/// Convert an mpi4py.MPI.Comm Python object into a raw MPI_Comm. Raises a
/// Python TypeError via std::runtime_error if the object is not an mpi4py
/// communicator, or MPI_COMM_NULL.
MPI_Comm pycomm_to_mpi_comm(py::handle py_comm);

/// Convert an mpi4py.MPI.Comm into a Ginkgo communicator wrapper.
/// `force_host_buffer` mirrors gko::experimental::mpi::communicator's flag.
gko::experimental::mpi::communicator make_gko_communicator(
    py::handle py_comm, bool force_host_buffer = false);

void init_communicator(py::module_& module);
void init_abi_check(py::module_& module);

}  // namespace pyginkgo_mpi

#endif  // PYGINKGO_BUILD_MPI

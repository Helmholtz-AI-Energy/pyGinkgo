// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include "communicator.hpp"

#include <pyGinkgo_mpi_abi.hpp>

#include <mpi.h>

#include <cstring>
#include <string>

namespace pyginkgo_mpi {

namespace {

std::string current_runtime_mpi_version()
{
    char buf[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0;
    if (MPI_Get_library_version(buf, &len) != MPI_SUCCESS) {
        return "<unavailable>";
    }
    return std::string(buf, static_cast<size_t>(len));
}

bool string_starts_with(const std::string& s, const char* prefix)
{
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

std::string detected_runtime_impl(const std::string& v)
{
    if (v.find("Open MPI") != std::string::npos ||
        v.find("OpenRTE") != std::string::npos)
        return "OpenMPI";
    if (v.find("MPICH") != std::string::npos) return "MPICH";
    if (v.find("Intel(R) MPI") != std::string::npos) return "IntelMPI";
    return "Unknown";
}

}  // namespace


void init_abi_check(py::module_& module)
{
    module.attr("BUILD_MPI_IMPL") = py::str(PYGINKGO_MPI_IMPL);
    module.attr("BUILD_MPI_LIBRARY_VERSION") =
        py::str(PYGINKGO_MPI_LIBRARY_VERSION);

    module.def(
        "runtime_mpi_library_version",
        []() {
            ensure_mpi4py_imported();
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized) {
                throw std::runtime_error(
                    "MPI is not initialized; import mpi4py.MPI first.");
            }
            return current_runtime_mpi_version();
        },
        "Return MPI_Get_library_version() of the currently loaded MPI.");

    module.def(
        "verify_abi",
        [](py::handle py_comm) {
            // Round-trip MPI_Comm_size — catches catastrophic ABI mismatches
            // (e.g. mpich-built pyGinkgo loaded against an openmpi mpi4py)
            // even when the version strings happen to match.
            MPI_Comm c = pycomm_to_mpi_comm(py_comm);
            int sz = -1;
            if (MPI_Comm_size(c, &sz) != MPI_SUCCESS || sz < 1) {
                throw std::runtime_error(
                    "pyGinkgo: MPI_Comm_size failed on the provided "
                    "communicator. This typically indicates an MPI ABI "
                    "mismatch between mpi4py and pyGinkgo (e.g. one was "
                    "built against MPICH, the other against OpenMPI).");
            }

            std::string runtime_v = current_runtime_mpi_version();
            std::string runtime_impl = detected_runtime_impl(runtime_v);
            std::string build_impl = PYGINKGO_MPI_IMPL;

            if (build_impl != "Unknown" && runtime_impl != "Unknown" &&
                build_impl != runtime_impl) {
                throw std::runtime_error(
                    "pyGinkgo: MPI implementation mismatch. Built against '" +
                    build_impl + "' but loaded MPI is '" + runtime_impl +
                    "'. Install the matching pyGinkgo conda variant "
                    "(e.g. `conda install 'pyginkgo-mpi=*=*" +
                    std::string(runtime_impl == "MPICH" ? "mpich"
                                                        : "openmpi") +
                    "*'`).");
            }
            return true;
        },
        py::arg("comm"),
        "Verify that the MPI library mpi4py is bound to matches the one "
        "pyGinkgo was built against. Raises a clear error if not.");
}

}  // namespace pyginkgo_mpi

#endif  // PYGINKGO_BUILD_MPI

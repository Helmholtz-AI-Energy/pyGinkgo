// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include "communicator.hpp"

#include <mpi4py/mpi4py.h>

#include <stdexcept>
#include <string>


namespace pyginkgo_mpi {

namespace {

bool& mpi4py_imported_flag()
{
    static bool flag = false;
    return flag;
}

}  // namespace


void ensure_mpi4py_imported()
{
    if (mpi4py_imported_flag()) {
        return;
    }
    if (import_mpi4py() < 0) {
        throw std::runtime_error(
            "pyGinkgo: failed to import mpi4py at the C API level. Make sure "
            "mpi4py is installed and was built against the same MPI library "
            "that pyGinkgo was linked with.");
    }
    mpi4py_imported_flag() = true;
}


MPI_Comm pycomm_to_mpi_comm(py::handle py_comm)
{
    ensure_mpi4py_imported();

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error(
            "pyGinkgo: MPI is not initialized. Import mpi4py.MPI before "
            "creating distributed objects (this triggers MPI_Init).");
    }

    MPI_Comm* p = PyMPIComm_Get(py_comm.ptr());
    if (!p) {
        throw py::type_error(
            "pyGinkgo: expected an mpi4py.MPI.Comm; got " +
            std::string(py::str(py_comm.get_type().attr("__name__"))));
    }
    if (*p == MPI_COMM_NULL) {
        throw std::runtime_error("pyGinkgo: communicator is MPI_COMM_NULL");
    }
    return *p;
}


gko::experimental::mpi::communicator make_gko_communicator(
    py::handle py_comm, bool force_host_buffer)
{
    return gko::experimental::mpi::communicator(pycomm_to_mpi_comm(py_comm),
                                                force_host_buffer);
}


void init_communicator(py::module_& module)
{
    py::class_<gko::experimental::mpi::communicator>(module, "Communicator",
        "Thin wrapper around gko::experimental::mpi::communicator that takes "
        "an mpi4py.MPI.Comm. The underlying MPI_Comm is NOT duplicated; the "
        "lifetime of the mpi4py communicator must outlive this object.")
        .def(py::init([](py::handle py_comm, bool force_host_buffer) {
                 return std::make_unique<
                     gko::experimental::mpi::communicator>(
                     make_gko_communicator(py_comm, force_host_buffer));
             }),
             py::arg("comm"), py::arg("force_host_buffer") = false)
        .def("rank", &gko::experimental::mpi::communicator::rank)
        .def("size", &gko::experimental::mpi::communicator::size)
        .def("node_local_rank",
             &gko::experimental::mpi::communicator::node_local_rank)
        .def_static("world",
                    [](bool force_host_buffer) {
                        return std::make_unique<
                            gko::experimental::mpi::communicator>(
                            MPI_COMM_WORLD, force_host_buffer);
                    },
                    py::arg("force_host_buffer") = false,
                    "Build a Communicator from MPI_COMM_WORLD. Useful when "
                    "mpi4py is not in use.");

    module.def(
        "map_rank_to_device_id",
        [](py::handle py_comm, int num_devices) {
            return gko::experimental::mpi::map_rank_to_device_id(
                pycomm_to_mpi_comm(py_comm), num_devices);
        },
        py::arg("comm"), py::arg("num_devices"),
        "Return the device id this rank should bind to (round-robin across "
        "the node-local ranks).");

    module.def(
        "is_gpu_aware",
        []() { return gko::experimental::mpi::is_gpu_aware(); },
        "Returns whether the linked MPI library is built with CUDA-aware "
        "support.");
}

}  // namespace pyginkgo_mpi

#endif  // PYGINKGO_BUILD_MPI

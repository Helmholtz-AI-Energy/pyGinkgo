// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "../mpi/communicator.hpp"
#include "../python.hpp"
#include "../utils.hpp"

namespace gd = gko::experimental::distributed;
using dim_type = gko::dim<2>::dimension_type;

namespace {

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void init_distributed_matrix(py::module_& module, const std::string& vt,
                             const std::string& lit, const std::string& git)
{
    using M = gd::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using P = gd::Partition<LocalIndexType, GlobalIndexType>;
    using IndexMap = gd::index_map<LocalIndexType, GlobalIndexType>;
    std::string name = "Matrix_" + vt + "_" + lit + "_" + git;

    py::class_<M, std::shared_ptr<M>, gko::LinOp>(
        module, name.c_str(),
        "Distributed sparse matrix split into local-diagonal and "
        "non-local-offdiagonal blocks.")
        .def_static(
            "create_empty",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                return std::shared_ptr<M>(M::create(exec, comm));
            },
            py::arg("exec"), py::arg("comm"),
            "Create an empty distributed matrix; populate via "
            "read_distributed.")
        .def_static(
            "create_from_local_linop",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               py::tuple global_size,
               std::shared_ptr<gko::LinOp> local_linop) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                return std::shared_ptr<M>(M::create(
                    exec, comm,
                    gko::dim<2>{global_size[0].cast<dim_type>(),
                                global_size[1].cast<dim_type>()},
                    local_linop));
            },
            py::arg("exec"), py::arg("comm"), py::arg("global_size"),
            py::arg("local_linop"),
            "Create a local-only distributed matrix from an existing local "
            "LinOp (e.g. a CSR). No off-process communication is set up.")
        .def_static(
            "create_from_local_and_non_local",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               std::shared_ptr<const P> partition, py::object recv_connections,
               std::shared_ptr<gko::LinOp> local_linop,
               std::shared_ptr<gko::LinOp> non_local_linop) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                auto recv = gko_array_from_pyobject<GlobalIndexType>(
                    exec, recv_connections);
                IndexMap imap{exec, partition,
                              static_cast<gko::experimental::distributed::
                                              comm_index_type>(comm.rank()),
                              recv};
                return std::shared_ptr<M>(M::create(
                    exec, comm, std::move(imap), local_linop, non_local_linop));
            },
            py::arg("exec"), py::arg("comm"), py::arg("partition"),
            py::arg("recv_connections"), py::arg("local_linop"),
            py::arg("non_local_linop"),
            "Create a distributed matrix from local-diagonal and non-local "
            "(off-diagonal) LinOps. `recv_connections` is a 1D array of "
            "global column indices that the non-local block accesses; the "
            "non-local LinOp's column indices must be consistent with the "
            "induced index_map ordering.")
        .def(
            "get_local_matrix",
            [](const M& self) { return self.get_local_matrix(); },
            "Returns the locally-owned (diagonal) block as a LinOp.")
        .def(
            "get_non_local_matrix",
            [](const M& self) { return self.get_non_local_matrix(); },
            "Returns the off-process (non-local) block as a LinOp.")
        .def_property_readonly(
            "shape",
            [](const M& m) {
                auto s = m.get_size();
                return py::make_tuple(s[0], s[1]);
            },
            "Global shape of the distributed matrix.");
}

}  // namespace

void init_distributed_matrix_all_types(py::module_& module)
{
    init_distributed_matrix<double, gko::int32, gko::int64>(
        module, "double", "int32", "int64");
    init_distributed_matrix<float, gko::int32, gko::int64>(
        module, "float", "int32", "int64");
    init_distributed_matrix<double, gko::int64, gko::int64>(
        module, "double", "int64", "int64");
    init_distributed_matrix<float, gko::int64, gko::int64>(
        module, "float", "int64", "int64");
}

#endif  // PYGINKGO_BUILD_MPI

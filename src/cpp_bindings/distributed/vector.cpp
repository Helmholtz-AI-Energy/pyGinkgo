// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/distributed/vector.hpp>

#include "../mpi/communicator.hpp"
#include "../python.hpp"
#include "../utils.hpp"

namespace gd = gko::experimental::distributed;
using dim_type = gko::dim<2>::dimension_type;

namespace {

template <typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>> dense_view_from_pyobject(
    std::shared_ptr<gko::Executor> exec, py::object obj)
{
    auto arr = gko_array_from_pyobject<ValueType>(exec, obj);
    auto rows = arr.get_size();
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{static_cast<dim_type>(rows), 1}, std::move(arr), 1));
}

template <typename ValueType>
void init_distributed_vector(py::module_& module, const std::string& typestr)
{
    using V = gd::Vector<ValueType>;
    using Dense = gko::matrix::Dense<ValueType>;
    std::string pyclass_name = std::string("Vector_") + typestr;

    auto cls = py::class_<V, std::shared_ptr<V>, gko::LinOp>(
                   module, pyclass_name.c_str(),
                   "Distributed (multi-)vector with an owned local slice.")
        .def_static(
            "create",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               py::tuple global_size, py::tuple local_size) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                return std::shared_ptr<V>(V::create(
                    exec, comm,
                    gko::dim<2>{global_size[0].cast<dim_type>(),
                                global_size[1].cast<dim_type>()},
                    gko::dim<2>{local_size[0].cast<dim_type>(),
                                local_size[1].cast<dim_type>()}));
            },
            py::arg("exec"), py::arg("comm"), py::arg("global_size"),
            py::arg("local_size"),
            "Create an empty distributed vector with given global and local "
            "sizes.")
        .def_static(
            "from_local_array",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               py::tuple global_size, py::object local_array) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                auto local = dense_view_from_pyobject<ValueType>(
                    exec, local_array);
                return std::shared_ptr<V>(V::create(
                    exec, comm,
                    gko::dim<2>{global_size[0].cast<dim_type>(),
                                global_size[1].cast<dim_type>()},
                    gko::clone(exec, local)));
            },
            py::arg("exec"), py::arg("comm"), py::arg("global_size"),
            py::arg("local_array"),
            "Create a distributed vector from a local 1D array (numpy or "
            "cupy). The data is copied into a new local Dense vector.")
        .def_static(
            "from_local_array_view",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               py::tuple global_size, py::object local_array) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                // gko_array_from_pyobject is zero-copy on a CudaExecutor with
                // a __cuda_array_interface__-backed input. The resulting
                // Dense view is moved into the distributed.Vector.
                auto arr = gko_array_from_pyobject<ValueType>(
                    exec, local_array);
                auto rows = arr.get_size();
                auto local = Dense::create(
                    exec, gko::dim<2>{static_cast<dim_type>(rows), 1},
                    std::move(arr), 1);
                return std::shared_ptr<V>(V::create(
                    exec, comm,
                    gko::dim<2>{global_size[0].cast<dim_type>(),
                                global_size[1].cast<dim_type>()},
                    std::move(local)));
            },
            py::arg("exec"), py::arg("comm"), py::arg("global_size"),
            py::arg("local_array"),
            py::keep_alive<0, 4>(),
            "Zero-copy variant of from_local_array. On a CudaExecutor with "
            "a __cuda_array_interface__-backed array (e.g. cupy ndarray), "
            "the distributed.Vector aliases the caller's buffer instead of "
            "copying. The local_array is kept alive for the lifetime of "
            "the returned vector via py::keep_alive.")
        .def_static(
            "from_local_array_deduce_size",
            [](std::shared_ptr<gko::Executor> exec, py::handle py_comm,
               py::object local_array) {
                auto comm = pyginkgo_mpi::make_gko_communicator(py_comm);
                auto local = dense_view_from_pyobject<ValueType>(
                    exec, local_array);
                return std::shared_ptr<V>(
                    V::create(exec, comm, gko::clone(exec, local)));
            },
            py::arg("exec"), py::arg("comm"), py::arg("local_array"),
            "Like from_local_array but deduces the global size via a "
            "collective sum.")
        .def("fill", &V::fill, py::arg("value"))
        .def(
            "scale",
            [](V& self, std::shared_ptr<const gko::LinOp> alpha) {
                self.scale(alpha);
            },
            py::arg("alpha"))
        .def(
            "inv_scale",
            [](V& self, std::shared_ptr<const gko::LinOp> alpha) {
                self.inv_scale(alpha);
            },
            py::arg("alpha"))
        .def(
            "add_scaled",
            [](V& self, std::shared_ptr<const gko::LinOp> alpha,
               std::shared_ptr<const gko::LinOp> b) {
                self.add_scaled(alpha, b);
            },
            py::arg("alpha"), py::arg("b"))
        .def(
            "sub_scaled",
            [](V& self, std::shared_ptr<const gko::LinOp> alpha,
               std::shared_ptr<const gko::LinOp> b) {
                self.sub_scaled(alpha, b);
            },
            py::arg("alpha"), py::arg("b"))
        .def(
            "compute_dot",
            [](const V& self, std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> result) {
                self.compute_dot(b, result);
            },
            py::arg("b"), py::arg("result"))
        .def(
            "compute_norm2",
            [](const V& self, std::shared_ptr<gko::LinOp> result) {
                self.compute_norm2(result);
            },
            py::arg("result"))
        .def(
            "compute_norm1",
            [](const V& self, std::shared_ptr<gko::LinOp> result) {
                self.compute_norm1(result);
            },
            py::arg("result"))
        .def(
            "get_local_vector",
            [](const V& self) {
                // Return an owning clone of the local Dense slice. The clone
                // shares the executor with self; lifetime issues avoided.
                auto local = self.get_local_vector();
                return std::shared_ptr<const Dense>(gko::clone(local).release());
            },
            "Returns an owning clone of the local Dense vector slice.")
        .def_property_readonly(
            "shape",
            [](const V& v) {
                auto s = v.get_size();
                return py::make_tuple(s[0], s[1]);
            },
            "Global shape of the distributed vector.")
        .def_property_readonly(
            "local_shape",
            [](const V& v) {
                auto s = v.get_local_vector()->get_size();
                return py::make_tuple(s[0], s[1]);
            },
            "Local shape of this rank's slice.")
        .def(
            "gather_on_root",
            [](const V& self, int root) -> py::object {
                // Gathers all local slices into a single host numpy array on
                // `root`; returns None on every other rank.
                auto comm = self.get_communicator();
                auto rank = comm.rank();
                auto size = comm.size();
                auto host = self.get_executor()->get_master();
                auto local = gko::clone(host, self.get_local_vector());

                auto local_rows =
                    static_cast<int>(local->get_size()[0]);
                auto cols =
                    static_cast<int>(local->get_size()[1]);

                std::vector<int> recvcounts(size, 0);
                comm.gather(host, &local_rows, 1, recvcounts.data(), 1, root);

                std::vector<int> displs(size, 0);
                int total_rows = 0;
                if (rank == root) {
                    for (int r = 0; r < size; ++r) {
                        displs[r] = total_rows;
                        recvcounts[r] *= cols;  // gatherv counts in elements
                        total_rows += recvcounts[r] / cols;
                    }
                }

                py::object result = py::none();
                ValueType* recvbuf = nullptr;
                py::array_t<ValueType> arr;
                if (rank == root) {
                    arr = py::array_t<ValueType>(
                        std::vector<ssize_t>{total_rows, cols});
                    recvbuf = static_cast<ValueType*>(arr.request().ptr);
                    result = std::move(arr);
                }
                int sendcount = local_rows * cols;
                comm.gather_v(host, local->get_const_values(), sendcount,
                              recvbuf, recvcounts.data(), displs.data(), root);
                return result;
            },
            py::arg("root") = 0,
            "Collective: gather the distributed vector to a host numpy array "
            "on rank `root`. Returns the assembled numpy array on `root` and "
            "None on every other rank. Slow path intended for diagnostics, "
            "snapshots, and validators -- not for the hot loop.");
}

}  // namespace

void init_distributed_vector_all_types(py::module_& module)
{
#define DECLARE_DIST_VECTOR(ValueType) \
    init_distributed_vector<ValueType>(module, #ValueType)
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
        DECLARE_DIST_VECTOR);
#undef DECLARE_DIST_VECTOR
}

#endif  // PYGINKGO_BUILD_MPI

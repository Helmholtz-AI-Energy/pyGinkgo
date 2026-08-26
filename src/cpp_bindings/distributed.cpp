// SPDX-FileCopyrightText: 2026 Imad Kissami
//
// SPDX-License-Identifier: MIT

#include "python.hpp"
#include "utils.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

#if GINKGO_BUILD_MPI

#include <mpi.h>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace dist = gko::experimental::distributed;
namespace mpi = gko::experimental::mpi;

using LIT = gko::int32;  // local index type
using GIT = gko::int32;  // global index type (int32: matches manapy ls_compute
                         // triplets; caps global DOFs at ~2.1e9, fine here)
using PartT = dist::Partition<LIT, GIT>;

static mpi::communicator comm_from_fortran(long fortran_handle)
{
    MPI_Comm c = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_handle));
    return mpi::communicator(c);
}

static py::buffer_info checked_1d(py::array array, const char* name)
{
    auto info = array.request();
    if (info.ndim != 1) {
        std::ostringstream os;
        os << name << " must be a 1-D array, got " << info.ndim << " dimensions";
        throw py::value_error(os.str());
    }
    return info;
}

static void check_same_length(py::ssize_t lhs, const char* lhs_name,
                              py::ssize_t rhs, const char* rhs_name)
{
    if (lhs != rhs) {
        std::ostringstream os;
        os << lhs_name << " and " << rhs_name << " must have matching lengths ("
           << lhs << " != " << rhs << ")";
        throw py::value_error(os.str());
    }
}

template <typename T>
static gko::array<T> host_array_from_numpy(
    std::shared_ptr<const gko::Executor> host, py::array_t<T> array,
    const char* name)
{
    auto info = checked_1d(array, name);
    auto n = static_cast<gko::size_type>(info.shape[0]);
    gko::array<T> out(host, n);
    std::copy_n(static_cast<const T*>(info.ptr), n, out.get_data());
    return out;
}

template <typename IndexType>
static void validate_global_indices(py::array_t<IndexType> array, const char* name,
                                    gko::size_type global_size)
{
    auto info = checked_1d(array, name);
    const auto* data = static_cast<const IndexType*>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        const auto idx = data[i];
        if (idx < 0 || static_cast<gko::size_type>(idx) >= global_size) {
            std::ostringstream os;
            os << name << " contains out-of-range global index " << idx
               << " at position " << i << " for global size " << global_size;
            throw py::value_error(os.str());
        }
    }
}

static void validate_partition_size(const PartT& partition,
                                    gko::size_type global_size)
{
    if (static_cast<gko::size_type>(partition.get_size()) != global_size) {
        std::ostringstream os;
        os << "partition size (" << partition.get_size()
           << ") must match global_size (" << global_size << ")";
        throw py::value_error(os.str());
    }
}

#ifdef GINKGO_BUILD_CUDA
// Extract a device pointer + element count from a 1-D __cuda_array_interface__
// object (CuPy, numba device array, ...).
template <typename T>
static std::pair<T*, gko::size_type> cai_ptr_and_size(py::object obj)
{
    auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto data = cai["data"].cast<py::tuple>();
    auto shape = cai["shape"].cast<py::tuple>();
    if (shape.size() != 1) {
        throw py::value_error(
            "__cuda_array_interface__ array must be 1-D for the distributed "
            "binding");
    }
    return {reinterpret_cast<T*>(data[0].cast<uintptr_t>()),
            shape[0].cast<gko::size_type>()};
}
#endif

// 1-D length of a numpy host array OR a __cuda_array_interface__ device array.
static gko::size_type pyobj_len(py::object obj, const char* name)
{
#ifdef GINKGO_BUILD_CUDA
    if (py::hasattr(obj, "__cuda_array_interface__")) {
        auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
        auto shape = cai["shape"].cast<py::tuple>();
        if (shape.size() != 1) {
            throw py::value_error(std::string(name) + " must be a 1-D array");
        }
        return shape[0].cast<gko::size_type>();
    }
#endif
    auto info = py::array(obj).request();
    if (info.ndim != 1) {
        throw py::value_error(std::string(name) + " must be a 1-D array");
    }
    return static_cast<gko::size_type>(info.shape[0]);
}

// Build an owning gko::array<T> on `exec` from either a numpy host array
// (copied) or a __cuda_array_interface__ device array (zero-copy view, then an
// owning device->device copy so the array can be moved into device_matrix_data
// independently of the Python source's lifetime).
template <typename T>
static gko::array<T> array_on_exec(std::shared_ptr<const gko::Executor> exec,
                                   py::object obj, const char* name)
{
#ifdef GINKGO_BUILD_CUDA
    if (py::hasattr(obj, "__cuda_array_interface__")) {
        auto pn = cai_ptr_and_size<T>(obj);
        auto view = gko::array<T>::view(exec, pn.second, pn.first);
        return gko::array<T>{exec, view};
    }
#endif
    py::array_t<T, py::array::c_style | py::array::forcecast> arr(obj);
    auto info = checked_1d(arr, name);
    auto n = static_cast<gko::size_type>(info.shape[0]);
    auto host = exec->get_master();
    gko::array<T> host_arr(host, n);
    std::copy_n(static_cast<const T*>(info.ptr), n, host_arr.get_data());
    return gko::array<T>{exec, host_arr};
}

static std::shared_ptr<PartT> build_partition(
    std::shared_ptr<gko::Executor> exec,
    py::array_t<int, py::array::c_style | py::array::forcecast> owners,
    int num_parts)
{
    if (num_parts <= 0) {
        throw py::value_error("num_parts must be positive");
    }

    auto owner_info = checked_1d(owners, "owners");
    if (owner_info.shape[0] == 0) {
        throw py::value_error("owners must not be empty");
    }
    const auto* owner_data = static_cast<const int*>(owner_info.ptr);
    for (py::ssize_t i = 0; i < owner_info.shape[0]; ++i) {
        if (owner_data[i] < 0 || owner_data[i] >= num_parts) {
            std::ostringstream os;
            os << "owners contains invalid part id " << owner_data[i]
               << " at position " << i << " for num_parts=" << num_parts;
            throw py::value_error(os.str());
        }
    }

    auto host = gko::ReferenceExecutor::create();
    auto mapping_host = host_array_from_numpy<int>(host, owners, "owners");
    auto mapping = gko::array<int>(exec, mapping_host);
    return gko::share(PartT::build_from_mapping(exec, mapping, num_parts));
}

template <typename ValueType>
static std::shared_ptr<gko::LinOp> build_matrix(
    std::shared_ptr<gko::Executor> exec, long comm_handle,
    std::shared_ptr<PartT> partition, py::object rows, py::object cols,
    py::object vals, gko::size_type global_size)
{
    auto n = pyobj_len(rows, "rows");
    check_same_length(static_cast<py::ssize_t>(n), "rows",
                      static_cast<py::ssize_t>(pyobj_len(cols, "cols")), "cols");
    check_same_length(static_cast<py::ssize_t>(n), "rows",
                      static_cast<py::ssize_t>(pyobj_len(vals, "vals")), "vals");
    validate_partition_size(*partition, global_size);

    auto comm = comm_from_fortran(comm_handle);

    // Assemble directly on the requested executor (device for CUDA). Inputs may
    // be host numpy or device (__cuda_array_interface__) arrays; the latter are
    // ingested zero-copy. The partition is already on `exec`, so assembly is
    // executor-consistent (the host-staging workaround is no longer needed).
    auto row_arr = array_on_exec<GIT>(exec, rows, "rows");
    auto col_arr = array_on_exec<GIT>(exec, cols, "cols");
    auto val_arr = array_on_exec<ValueType>(exec, vals, "vals");

    gko::device_matrix_data<ValueType, GIT> data(
        exec, gko::dim<2>{global_size, global_size}, std::move(row_arr),
        std::move(col_arr), std::move(val_arr));

    auto mat = dist::Matrix<ValueType, LIT, GIT>::create(exec, comm);
    mat->read_distributed(data, partition, dist::assembly_mode::communicate);
    return gko::share(std::move(mat));
}

template <typename ValueType>
static std::shared_ptr<gko::LinOp> build_vector(
    std::shared_ptr<gko::Executor> exec, long comm_handle,
    std::shared_ptr<PartT> partition, py::object rows, py::object vals,
    gko::size_type global_size)
{
    auto n = pyobj_len(rows, "rows");
    check_same_length(static_cast<py::ssize_t>(n), "rows",
                      static_cast<py::ssize_t>(pyobj_len(vals, "vals")), "vals");
    validate_partition_size(*partition, global_size);

    auto comm = comm_from_fortran(comm_handle);

    auto row_arr = array_on_exec<GIT>(exec, rows, "rows");
    auto val_arr = array_on_exec<ValueType>(exec, vals, "vals");
    auto nnz = row_arr.get_size();

    gko::array<GIT> col_arr(exec, nnz);
    col_arr.fill(GIT{0});

    gko::device_matrix_data<ValueType, GIT> data(
        exec, gko::dim<2>{global_size, 1}, std::move(row_arr),
        std::move(col_arr), std::move(val_arr));

    auto vec = dist::Vector<ValueType>::create(exec, comm);
    vec->read_distributed(data, partition);
    return gko::share(std::move(vec));
}

template <typename ValueType>
static py::array_t<ValueType> vector_local(std::shared_ptr<gko::LinOp> v)
{
    auto vec = std::dynamic_pointer_cast<dist::Vector<ValueType>>(v);
    if (!vec) {
        throw py::type_error(
            "vector_local: object is not a distributed vector of the requested value type");
    }
    auto host = vec->get_executor()->get_master();
    auto local = vec->get_local_vector();
    auto host_local = gko::clone(host, local);
    const auto nrows = static_cast<py::ssize_t>(host_local->get_size()[0]);
    const auto ncols = static_cast<py::ssize_t>(host_local->get_size()[1]);
    const auto stride = static_cast<py::ssize_t>(host_local->get_stride());
    const auto* values = host_local->get_const_values();

    if (ncols == 1) {
        py::array_t<ValueType> out(nrows);
        auto* out_data = static_cast<ValueType*>(out.request().ptr);
        for (py::ssize_t row = 0; row < nrows; ++row) {
            out_data[row] = values[row];
        }
        return out;
    }

    py::array_t<ValueType> out({nrows, ncols});
    auto* out_data = static_cast<ValueType*>(out.request().ptr);
    for (py::ssize_t col = 0; col < ncols; ++col) {
        for (py::ssize_t row = 0; row < nrows; ++row) {
            out_data[row * ncols + col] = values[row + col * stride];
        }
    }
    return out;
}

// Overwrite a distributed vector's processor-local values IN PLACE (no
// reallocation, no read_distributed). `vals` is a 1-D host numpy array or a
// __cuda_array_interface__ device array whose length must equal the local size;
// its entries are already in Ginkgo's processor-local ordering. Lets callers
// reuse a single vector across solves instead of rebuilding it each time.
template <typename ValueType>
static void vector_set_local(std::shared_ptr<gko::LinOp> v, py::object vals)
{
    auto vec = std::dynamic_pointer_cast<dist::Vector<ValueType>>(v);
    if (!vec) {
        throw py::type_error(
            "vector_set_local: object is not a distributed vector of the "
            "requested value type");
    }
    // The local part is owned by the vector; we only overwrite its values
    // (no resize), so const_cast on the buffer is safe here.
    auto* local = const_cast<gko::matrix::Dense<ValueType>*>(
        vec->get_local_vector());
    const auto nrows = local->get_size()[0];
    const auto ncols = local->get_size()[1];
    if (ncols != 1) {
        throw py::value_error(
            "vector_set_local supports column vectors only (ncols == 1)");
    }
    auto n = pyobj_len(vals, "vals");
    if (n != nrows) {
        std::ostringstream os;
        os << "vals length (" << n << ") must match the vector's local size ("
           << nrows << ")";
        throw py::value_error(os.str());
    }
    auto exec = local->get_executor();
    // Bring vals onto the vector's executor (host copy, or zero-copy device view
    // then an owning copy), then overwrite the contiguous local buffer.
    auto src = array_on_exec<ValueType>(exec, vals, "vals");
    exec->copy(n, src.get_const_data(), local->get_values());
}

#ifdef GINKGO_BUILD_CUDA
// Expose the device pointer + layout of a distributed vector's processor-local
// part, so Python can wrap it zero-copy (CuPy / numba) without a device->host
// copy. The buffer stays valid as long as the source vector is alive.
template <typename ValueType>
static py::tuple vector_local_device_info(std::shared_ptr<gko::LinOp> v)
{
    auto vec = std::dynamic_pointer_cast<dist::Vector<ValueType>>(v);
    if (!vec) {
        throw py::type_error(
            "vector_local_device: object is not a distributed vector of the "
            "requested value type");
    }
    auto local = vec->get_local_vector();
    return py::make_tuple(
        reinterpret_cast<uintptr_t>(local->get_const_values()),
        static_cast<py::ssize_t>(local->get_size()[0]),
        static_cast<py::ssize_t>(local->get_size()[1]),
        static_cast<py::ssize_t>(local->get_stride()));
}
#endif

template <typename ValueType>
static void init_distributed_for_type(py::module_& m, const std::string& vt)
{
    m.def(("matrix_" + vt).c_str(), &build_matrix<ValueType>, py::arg("exec"),
          py::arg("comm_handle"), py::arg("partition"), py::arg("rows"),
          py::arg("cols"), py::arg("vals"), py::arg("global_size"),
          "Build a distributed Csr matrix from global COO triplets. The matrix "
          "is square: its shape is global_size x global_size. Row/column indices "
          "must lie in [0, global_size).");

    m.def(("vector_" + vt).c_str(), &build_vector<ValueType>, py::arg("exec"),
          py::arg("comm_handle"), py::arg("partition"), py::arg("rows"),
          py::arg("vals"), py::arg("global_size"),
          "Build a distributed vector from (global_row, value) entries. Global "
          "rows not present in 'rows' default to zero.");

    m.def(("vector_local_" + vt).c_str(), &vector_local<ValueType>,
          py::arg("vector"),
          "Return the processor-local part of a distributed vector as numpy.");

    m.def(("vector_set_local_" + vt).c_str(), &vector_set_local<ValueType>,
          py::arg("vector"), py::arg("vals"),
          "Overwrite a distributed vector's processor-local values in place "
          "(no reallocation). vals length must equal the local size.");

#ifdef GINKGO_BUILD_CUDA
    m.def(("vector_local_device_" + vt).c_str(),
          &vector_local_device_info<ValueType>, py::arg("vector"),
          "Return (device_ptr, nrows, ncols, stride) of a distributed vector's "
          "processor-local part for zero-copy CuPy/numba wrapping.");
#endif
}

void init_distributed(py::module_& m)
{
    py::class_<PartT, std::shared_ptr<PartT>>(m, "Partition")
        .def("get_num_parts", &PartT::get_num_parts)
        .def_property_readonly("size", [](const PartT& p) { return p.get_size(); })
        .def("__repr__", [](const PartT& p) {
            std::ostringstream os;
            os << "<pyGinkgo.distributed.Partition size=" << p.get_size()
               << " num_parts=" << p.get_num_parts() << ">";
            return os.str();
        });

    m.def("build_partition", &build_partition, py::arg("exec"),
          py::arg("owners"), py::arg("num_parts"),
          "Build a distributed Partition from a global owner mapping "
          "(owners[global_row] = rank).");

    // half/bfloat16 distributed types are instantiated by Ginkgo only when it
    // is built with GINKGO_ENABLE_HALF; PYGKO_ADAPT_HF compiles the half
    // bindings out when half support is disabled (e.g. HALF=OFF on GCC 13).
    PYGKO_ADAPT_HF(init_distributed_for_type<half>(m, "half"));
    init_distributed_for_type<float>(m, "float");
    init_distributed_for_type<double>(m, "double");

    m.attr("available") = true;
}

#else  // GINKGO_BUILD_MPI

void init_distributed(py::module_& m) { m.attr("available") = false; }

#endif  // GINKGO_BUILD_MPI

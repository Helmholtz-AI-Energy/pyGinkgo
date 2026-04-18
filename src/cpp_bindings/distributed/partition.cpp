// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "../python.hpp"
#include "../utils.hpp"

namespace {

template <typename LocalIndexType, typename GlobalIndexType>
void init_partition(py::module_& module, const std::string& l, const std::string& g)
{
    using P = gko::experimental::distributed::Partition<LocalIndexType,
                                                        GlobalIndexType>;
    std::string name = "Partition_" + l + "_" + g;

    py::class_<P, std::shared_ptr<P>>(module, name.c_str(),
        "Distributed partition mapping global row IDs to (rank, local) pairs.")
        .def_static(
            "build_from_global_size_uniform",
            [](std::shared_ptr<gko::Executor> exec, gko::int32 num_parts,
               GlobalIndexType global_size) {
                return std::shared_ptr<P>(
                    P::build_from_global_size_uniform(exec, num_parts,
                                                       global_size));
            },
            py::arg("exec"), py::arg("num_parts"), py::arg("global_size"),
            "Evenly split [0, global_size) into num_parts contiguous chunks.")
        .def_static(
            "build_from_contiguous",
            [](std::shared_ptr<gko::Executor> exec, py::object ranges_obj) {
                auto ranges = gko_array_from_pyobject<GlobalIndexType>(
                    exec, ranges_obj);
                return std::shared_ptr<P>(
                    P::build_from_contiguous(exec, ranges));
            },
            py::arg("exec"), py::arg("ranges"),
            "Build from a contiguous-ranges array of length num_parts+1, "
            "where part i owns [ranges[i], ranges[i+1]).")
        .def_static(
            "build_from_mapping",
            [](std::shared_ptr<gko::Executor> exec, py::object mapping_obj,
               gko::int32 num_parts) {
                auto mapping =
                    gko_array_from_pyobject<gko::int32>(exec, mapping_obj);
                return std::shared_ptr<P>(
                    P::build_from_mapping(exec, mapping, num_parts));
            },
            py::arg("exec"), py::arg("mapping"), py::arg("num_parts"),
            "Build from an explicit owner-per-row map.")
        .def("get_size", [](const P& p) { return p.get_size(); })
        .def("get_num_parts", [](const P& p) { return p.get_num_parts(); })
        .def("get_num_ranges", [](const P& p) { return p.get_num_ranges(); })
        .def("get_part_size",
             [](const P& p, gko::int32 part) { return p.get_part_size(part); },
             py::arg("part"));
}

}  // namespace

void init_distributed_partition_all_types(py::module_& module)
{
    init_partition<gko::int32, gko::int64>(module, "int32", "int64");
    init_partition<gko::int64, gko::int64>(module, "int64", "int64");
    init_partition<gko::int32, gko::int32>(module, "int32", "int32");
}

#endif  // PYGINKGO_BUILD_MPI

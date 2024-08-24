// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;

void add_stream_classes(py::module_ &root_module)
{
    py::class_<gko::cuda_stream, std::shared_ptr<gko::cuda_stream>>(root_module, "cuda_stream")
        // The comments are copied from the source of the class at ginkgo-src/include/ginkgo/core/base/stream.hpp
        .def(py::init(), "Creates an empty stream wrapper, representing the default stream.")
        .def(py::init<int>(), "Creates a new custom CUDA stream on the given CUDA device ID");
}
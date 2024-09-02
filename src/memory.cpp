// TODO: might require placement in a separate dedicated folder
//      (in this case, something like 'base' folder)

#include "python.hpp"

namespace py = pybind11;

void add_allocator_classes(py::module_ &root_module)
{
    py::class_<gko::Allocator, std::shared_ptr<gko::Allocator>>
        (root_module, "Allocator");

    // TODO: implement CudaAllocatorBase binding,
    //    to be able to define custom allocator
    py::class_<gko::CudaAllocatorBase, gko::Allocator,
        std::shared_ptr<gko::CudaAllocatorBase>>
            (root_module, "CudaAllocatorBase");
    
    py::class_<gko::CudaAllocator, gko::CudaAllocatorBase,
         std::shared_ptr<gko::CudaAllocator>>
            (root_module, "CudaAllocator")
        .def(py::init());
}
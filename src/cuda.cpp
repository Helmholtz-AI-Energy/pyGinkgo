// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;

void add_cuda_executor_class(py::module_ &root_module)
{
    py::class_<gko::detail::ExecutorBase<gko::CudaExecutor>, gko::Executor,
        std::shared_ptr<gko::detail::ExecutorBase<gko::CudaExecutor>>>
            (root_module, "CudaExecutorBase");

    py::class_<gko::CudaExecutor, gko::detail::ExecutorBase<gko::CudaExecutor>,
        std::shared_ptr<gko::CudaExecutor>>
            (root_module, "CudaExecutor")
        .def(py::init([](
                int dev_id, std::shared_ptr<gko::Executor> master,              
                std::shared_ptr<gko::CudaAllocatorBase> alloc,
                std::shared_ptr<gko::cuda_stream> stream)
            {
                return gko::CudaExecutor::create(
                    dev_id, master, alloc, stream->get());
            }),
            // The first two are the deviation from the original library
            py::arg("device_id") = 0,
            py::arg("master") = gko::ReferenceExecutor::create(),
            py::arg("allocator") = std::make_shared<gko::CudaAllocator>(),
            py::arg("stream") = std::make_shared<gko::cuda_stream>()
        )
        
        .def_property_readonly("master",
            [](std::shared_ptr<gko::CudaExecutor> self)
                -> std::shared_ptr<gko::Executor>
            // Hopefully there would be no problems with converting val to lval
            { return self->get_master(); }
        )

        .def_property_readonly("device_id",
            &gko::CudaExecutor::get_device_id)

        .def_property_readonly("num_devices",
            &gko::CudaExecutor::get_num_devices)
        ;
}
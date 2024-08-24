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
                // Cannot pass the stream argument,
                return gko::CudaExecutor::create(
                    dev_id, master, std::make_shared<gko::CudaAllocator>(), nullptr);
            }),
            py::arg("device_id"),
            py::arg("master"),
            py::arg("allocator") = std::make_shared<gko::CudaAllocator>(),
            py::arg("stream") = std::make_shared<gko::cuda_stream>()
        );
}
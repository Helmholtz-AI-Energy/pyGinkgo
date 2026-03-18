// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <pybind11/numpy.h>

#include "python.hpp"
#include "utils.hpp"

template <typename ValueType>
void init_array(py::module_ &module, const std::string typestr)
{
    std::string pyclass_name = std::string("array_") + typestr;

    auto cls =
        py::class_<gko::array<ValueType>>(module, pyclass_name.c_str(),
                                          py::buffer_protocol())
            .def(py::init<std::shared_ptr<const gko::Executor>, int>())
            .def(py::init<std::shared_ptr<const gko::Executor>,
                          gko::array<ValueType> &>())
            .def(py::init(
                [](std::shared_ptr<gko::Executor> exec,
                   py::object obj) {
#ifdef GINKGO_BUILD_CUDA
                    // Fast path: if the input exposes
                    // __cuda_array_interface__ and the executor is a
                    // CUDA executor, create a zero-copy view instead
                    // of going through host memory.
                    // py::keep_alive<1,3> ensures the source object
                    // stays alive while this array exists.
                    if (py::hasattr(obj, "__cuda_array_interface__") &&
                        std::dynamic_pointer_cast<const gko::CudaExecutor>(
                            exec)) {
                        auto cai =
                            obj.attr("__cuda_array_interface__")
                                .cast<py::dict>();
                        auto shape = cai["shape"].cast<py::tuple>();
                        if (py::len(shape) != 1) {
                            throw std::runtime_error(
                                "Only 1D arrays are supported");
                        }
                        auto typestr =
                            cai["typestr"].cast<std::string>();
                        auto expected =
                            get_cuda_array_typestr<ValueType>();
                        if (typestr != expected) {
                            throw std::runtime_error(
                                "dtype mismatch: "
                                "__cuda_array_interface__ reports '" +
                                typestr +
                                "' but this array type expects '" +
                                expected + "'");
                        }
                        auto data = cai["data"].cast<py::tuple>();
                        auto ptr = data[0].cast<uintptr_t>();
                        auto size = shape[0].cast<size_t>();
                        return gko::array<ValueType>::view(
                            exec, size,
                            reinterpret_cast<ValueType *>(ptr));
                    }
#endif
                    // Fallback: use buffer protocol (host memory)
                    auto b = py::array_t<ValueType,
                                         py::array::c_style |
                                             py::array::forcecast>(obj);
                    py::buffer_info info = b.request();
                    check_buffer_dtype<ValueType>(info);

                    if (info.ndim != 1) {
                        throw std::runtime_error(
                            "Only 1D arrays are supported");
                    }

                    auto elems = info.shape[0];
                    return gko::array<ValueType>(
                        exec, (ValueType *)info.ptr,
                        (ValueType *)info.ptr + elems);
                }),
                py::keep_alive<1, 3>())
            .def_buffer([](gko::array<ValueType> &a) -> py::buffer_info {
                return py::buffer_info(
                    a.get_data(),      /* Pointer to buffer */
                    sizeof(ValueType), /* Size of one scalar */
                    py::format_descriptor<
                        ValueType>::format(), /* Python struct-style
                                                 format descriptor */
                    1,                        /* Number of dimensions */
                    {a.get_size()},           /* Buffer dimensions */
                    {
                        sizeof(ValueType) /* Strides (in bytes) for each
                                             index */
                    });
            })
            .def("fill", &gko::array<ValueType>::fill,
                 "Fill the array with the given value.")
            .def("get_size",
                 [](const gko::array<ValueType> &arr) {
                     // Deprecation in favor of .shape
                     // https://stackoverflow.com/a/62559865
                     PyErr_WarnEx(PyExc_DeprecationWarning,
                                  ".get_size() is deprecated, use .shape[0] "
                                  "instead",
                                  1);
                     return arr.get_size();
                 })
            .def_property_readonly(
                "size",
                [](const gko::array<ValueType> &arr) {
                    // Deprecation in favor of .shape
                    // https://stackoverflow.com/a/62559865
                    PyErr_WarnEx(PyExc_DeprecationWarning,
                                 ".size is deprecated, use .shape instead", 1);
                    return arr.get_size();
                })
            .def_property_readonly(
                "shape",
                [](const gko::array<ValueType> &arr) {
                    return py::make_tuple(arr.get_size());
                })
            .def("at",
                 [](const gko::array<ValueType> &arr, int idx) {
                     return arr.get_const_data()[idx];
                 })
            .def("__repr__", [=](const gko::array<ValueType> &arr) {
                auto str =
                    "pygko.base." + pyclass_name + " object of size ";
                auto elems = arr.get_size();
                str += std::to_string(elems);
                if (arr.get_executor() ==
                    arr.get_executor()->get_master()) {
                    str += " on host";
                    if (elems < 10) {
                        str += " [ ";
                        auto data = arr.get_const_data();
                        for (int i = 0; i < elems; i++) {
                            str += std::to_string(data[i]);
                            str += " ";
                        }
                        str += " ] ";
                    }
                }
                return str;
            });

#ifdef GINKGO_BUILD_CUDA
    // __cuda_array_interface__ (v3) for zero-copy interop with CuPy and
    // other CUDA-aware Python libraries.
    // Only available when the array is on a CUDA executor.
    cls.def_property_readonly(
        "__cuda_array_interface__",
        [](gko::array<ValueType> &a) -> py::dict {
            auto exec = a.get_executor();
            if (!std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
                throw py::attribute_error(
                    "__cuda_array_interface__ is only available for "
                    "arrays on CUDA executors");
            }
            py::dict interface;
            interface["shape"] = py::make_tuple(a.get_size());
            interface["typestr"] = get_cuda_array_typestr<ValueType>();
            interface["data"] =
                py::make_tuple(reinterpret_cast<uintptr_t>(a.get_data()),
                               false);  // (ptr, read_only)
            interface["version"] = 3;
            interface["strides"] = py::none();
            return interface;
        });

    // Factory method to create an array from a raw CUDA device pointer.
    // Creates an owning copy of the data pointed to.
    cls.def_static(
        "from_device_ptr",
        [](std::shared_ptr<gko::Executor> exec, uintptr_t ptr, size_t size) {
            auto view = gko::array<ValueType>::view(
                exec, size, reinterpret_cast<ValueType *>(ptr));
            return gko::array<ValueType>(exec, view);
        },
        py::arg("exec"), py::arg("ptr"), py::arg("size"),
        "Create an owning array by copying data from a device pointer. "
        "The pointer must be valid on the same CUDA device as the "
        "executor.");
#endif

    module.def(
        "reduce_add",
        pybind11::overload_cast<const gko::array<ValueType> &, const ValueType>(
            &gko::reduce_add<ValueType>));
}

void init_array_all_types(py::module_ &module)
{
#define DECLARE_ARRAY_VALUE(ValueType) \
    init_array<ValueType>(module, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_ARRAY_VALUE);
    PYGKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_ARRAY_VALUE);
}

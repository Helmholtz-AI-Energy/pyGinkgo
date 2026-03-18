// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "python.hpp"
#include "utils.hpp"

template <typename ValueType>
void init_dense(py::module_ &module_matrix, const std::string typestr)
{
    std::string pyclass_name = std::string("dense_") + typestr;

    using dim_type = gko::dim<2>::dimension_type;
    /* function to create a dense matrix from py::buffer object
     *
     */
    auto init_func = [](std::shared_ptr<gko::Executor> exec, py::buffer b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();
        check_buffer_dtype<ValueType>(info);

        /* create a view into numpy data */
        auto elems =
            (info.ndim == 1) ? info.shape[0] : info.shape[0] * info.shape[1];

        auto view = gko::array<ValueType>::view(exec->get_master(), elems,
                                                (ValueType *)info.ptr);

        auto rows = info.shape[0];
        auto cols = (info.ndim == 1) ? 1 : info.shape[1];

        return gko::matrix::Dense<ValueType>::create(
            exec,
            gko::dim<2>{static_cast<dim_type>(rows),
                        static_cast<dim_type>(cols)},
            view, cols);
    };

    auto cls = py::class_<gko::matrix::Dense<ValueType>,
                           std::shared_ptr<gko::matrix::Dense<ValueType>>,
                           gko::LinOp>(module_matrix, pyclass_name.c_str(),
                                       py::buffer_protocol())
                   .def(py::init([init_func](py::buffer b) {
                       auto ref = gko::ReferenceExecutor::create();
                       return init_func(ref, b);
                   }))
                   .def(py::init([](std::shared_ptr<gko::Executor> exec) {
                       return gko::share(
                           gko::matrix::Dense<ValueType>::create(exec));
                   }))
                   .def(py::init([](std::shared_ptr<gko::Executor> exec,
                                    py::tuple dim) {
                       return gko::share(
                           gko::matrix::Dense<ValueType>::create(
                               exec,
                               gko::dim<2>{dim[0].cast<size_t>(),
                                           dim[1].cast<size_t>()}));
                   }))
                   .def(py::init([init_func](
                                     std::shared_ptr<gko::Executor> exec,
                                     py::object obj) {
#ifdef GINKGO_BUILD_CUDA
                       // Fast path: if the input exposes
                       // __cuda_array_interface__ and the executor is
                       // a CUDA executor, create a zero-copy view.
                       // py::keep_alive<1,3> ensures the source object
                       // stays alive while this dense matrix exists.
                       if (py::hasattr(obj, "__cuda_array_interface__") &&
                           std::dynamic_pointer_cast<const gko::CudaExecutor>(
                               exec)) {
                           auto cai =
                               obj.attr("__cuda_array_interface__")
                                   .cast<py::dict>();
                           auto shape = cai["shape"].cast<py::tuple>();
                           auto typestr =
                               cai["typestr"].cast<std::string>();
                           auto expected =
                               get_cuda_array_typestr<ValueType>();
                           if (typestr != expected) {
                               throw std::runtime_error(
                                   "dtype mismatch: "
                                   "__cuda_array_interface__ reports '" +
                                   typestr +
                                   "' but this dense type expects '" +
                                   expected + "'");
                           }
                           auto data = cai["data"].cast<py::tuple>();
                           auto ptr = data[0].cast<uintptr_t>();
                           auto rows = shape[0].cast<size_t>();
                           auto cols = (py::len(shape) > 1)
                                           ? shape[1].cast<size_t>()
                                           : 1;
                           auto stride = cols;  // C-contiguous

                           auto view = gko::array<ValueType>::view(
                               exec, rows * stride,
                               reinterpret_cast<ValueType *>(ptr));
                           return gko::share(
                               gko::matrix::Dense<ValueType>::create(
                                   exec,
                                   gko::dim<2>{
                                       static_cast<dim_type>(rows),
                                       static_cast<dim_type>(cols)},
                                   std::move(view), stride));
                       }
#endif
                       // Fallback: use buffer protocol (host memory)
                       auto b = py::array_t<ValueType,
                                            py::array::c_style |
                                                py::array::forcecast>(obj);
                       return gko::share(init_func(exec, b));
                   }),
                   py::keep_alive<1, 3>())
                   .def(py::init([](std::shared_ptr<gko::Executor> exec,
                                    py::tuple dim, int stride) {
                       return gko::share(
                           gko::matrix::Dense<ValueType>::create(
                               exec,
                               gko::dim<2>{dim[0].cast<dim_type>(),
                                           dim[1].cast<dim_type>()},
                               stride));
                   }))
                   .def(py::init([](std::shared_ptr<gko::Executor> exec,
                                    py::tuple dim, py::buffer b,
                                    size_t stride) {
                       py::buffer_info info = b.request();
                       check_buffer_dtype<ValueType>(info);

                       auto ref = gko::ReferenceExecutor::create();

                       /* create a view into numpy data */
                       auto elems = (info.ndim == 1)
                                        ? info.shape[0]
                                        : info.shape[0] * info.shape[1];

                       auto view = gko::array<ValueType>::view(
                           ref, elems, (ValueType *)info.ptr);

                       auto rows = info.shape[0];
                       auto cols = (info.ndim == 1) ? 1 : info.shape[1];

                       return gko::share(
                           gko::matrix::Dense<ValueType>::create(
                               exec,
                               gko::dim<2>{dim[0].cast<dim_type>(),
                                           dim[1].cast<dim_type>()},
                               view, stride));
                   }))
                   .def(py::init([](std::shared_ptr<gko::Executor> exec,
                                    py::tuple dim,
                                    gko::array<ValueType> view,
                                    size_t stride) {
                       return gko::share(
                           gko::matrix::Dense<ValueType>::create(
                               exec,
                               gko::dim<2>{dim[0].cast<dim_type>(),
                                           dim[1].cast<dim_type>()},
                               view, stride));
                   }));

    cls.def("__repr__",
            [=](const gko::matrix::Dense<ValueType> &o) {
                auto str =
                    "pygko.matrix." + pyclass_name + " object of size ";
                auto elems = o.get_num_stored_elements();
                str += std::to_string(elems);
                if (o.get_executor() == o.get_executor()->get_master()) {
                    str += " on host";
                    if (elems < 10) {
                        str += " [ ";
                        for (int i = 0; i < elems; i++) {
                            str += std::to_string(o.at(i));
                            str += " ";
                        }
                        str += " ] ";
                    }
                }
                return str;
            })
        .def("copy_to_host",
             [](gko::matrix::Dense<ValueType> &m) {
                 auto host_exec = m.get_executor()->get_master();
                 std::cout << __FILE__ << "Warning creating a copy of dense\n";
                 if (m.get_executor() != host_exec) {
                     auto host_dense = gko::share(
                         gko::matrix::Dense<ValueType>::create(host_exec));
                     host_dense->operator=(m);
                     return host_dense;
                 } else {
                     return std::make_shared<gko::matrix::Dense<ValueType>>(m);
                 }
             })
        .def(
            "T",
            [](gko::matrix::Dense<ValueType> &m) {
                return gko::share(m.transpose());
            },
            "Computes and returns transpose of the matrix")
        .def(
            "convert_to_csr",
            [](gko::matrix::Dense<ValueType> &m) {
                auto exec = m.get_executor();
                auto csr =
                    gko::share(gko::matrix::Csr<ValueType, int>::create(exec));
                m.convert_to(csr);
                return csr;
            },
            "Computes and returns transpose of the matrix")
        .def_buffer([](gko::matrix::Dense<ValueType> &m) -> py::buffer_info {
            // buffer info needs data on host, thus if data is on device it
            // should be copied to host first
            size_t rows = m.get_num_stored_elements() / m.get_stride();
            size_t cols = m.get_stride();
            size_t dim = (m.get_stride() == 1) ? 1 : 2;

            ValueType *buffer_ptr = nullptr;
            if (m.get_executor() != m.get_executor()->get_master()) {
                GKO_NOT_IMPLEMENTED;
            }
            buffer_ptr = m.get_values();

            if (dim == 1) {
                return py::buffer_info(
                    buffer_ptr,        /* Pointer to buffer */
                    sizeof(ValueType), /* Size of one scalar */
                    py::format_descriptor<ValueType>::format(), /* Python
                                                                struct-style
                                                                format
                                                                descriptor */
                    dim,                /* Number of dimensions */
                    {rows},             /* Buffer dimensions */
                    {sizeof(ValueType)} /* Strides (in bytes) for each index */
                );
            } else {
                return py::buffer_info(
                    buffer_ptr,        /* Pointer to buffer */
                    sizeof(ValueType), /* Size of one scalar */
                    py::format_descriptor<ValueType>::format(), /* Python
                                                                struct-style
                                                                format
                                                                descriptor */
                    dim, /* Number of dimensions */
                    // TODO: potential mistake:
                    {rows, cols}, /* Buffer dimensions */
                    {sizeof(ValueType) * cols, sizeof(ValueType)}
                    /* Strides (in bytes) for each index */
                );
            }
        })
        .def(
            "apply",
            [](const gko::matrix::Dense<ValueType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "")
        .def("scale",
             [](gko::matrix::Dense<ValueType> &m, ValueType s) {
                 auto o = gko::matrix::Dense<ValueType>::create(
                     m.get_executor(), gko::dim<2>(1, 1));
                 o->fill(s);
                 m.scale(o);
             })
        .def("inv_scale",
             [](gko::matrix::Dense<ValueType> &m, ValueType s) {
                 auto o = gko::matrix::Dense<ValueType>::create(
                     m.get_executor(), gko::dim<2>(1, 1));
                 o->fill(s);
                 m.inv_scale(o);
             })
        .def(
            "add_scaled",
            [](gko::matrix::Dense<ValueType> &self,
               std::shared_ptr<gko::LinOp> alpha,
               std::shared_ptr<gko::LinOp> b) { self.add_scaled(alpha, b); },
            py::arg("alpha"), py::arg("b"),
            "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
        .def(
            "sub_scaled",
            [](gko::matrix::Dense<ValueType> &self,
               std::shared_ptr<gko::LinOp> alpha,
               std::shared_ptr<gko::LinOp> b) { self.sub_scaled(alpha, b); },
            py::arg("alpha"), py::arg("b"),
            "Subtracts `b` scaled by `alpha` from the matrix (aka: BLAS axpy).")
        .def("at",
             py::overload_cast<size_t>(&gko::matrix::Dense<ValueType>::at,
                                       py::const_),
             "Returns an element using linearized index.")
        .def("fill", &gko::matrix::Dense<ValueType>::fill,
             "Fills the matrix with a given value")
        .def("at",
             py::overload_cast<size_t, size_t>(
                 &gko::matrix::Dense<ValueType>::at, py::const_),
             "Returns an element at row, column index.")
        .def(
            "get_size",
            [](const gko::matrix::Dense<ValueType> &m) {
                // Deprecation in favor of .shape
                // https://stackoverflow.com/a/62559865
                PyErr_WarnEx(PyExc_DeprecationWarning,
                             ".get_size() is deprecated, use .shape instead",
                             1);
                return m.get_size();
            },
            "Returns the dimension of the dense matrix.")
        .def_property_readonly(
            "size",
            [](const gko::matrix::Dense<ValueType> &m) {
                // Deprecation in favor of .shape
                // https://stackoverflow.com/a/62559865
                PyErr_WarnEx(PyExc_DeprecationWarning,
                             ".size is deprecated, use .shape instead", 1);
                return m.get_size();
            },
            "Returns the dimension of the dense matrix.")
        .def_property_readonly(
            "shape",
            [](const gko::matrix::Dense<ValueType> &m) {
                auto size = m.get_size();
                return py::make_tuple(size[0], size[1]);
            },
            "Returns the dimension of the dense matrix.")
        .def("get_num_stored_elements",
             &gko::matrix::Dense<ValueType>::get_num_stored_elements,
             "Returns the number of elements explicitly stored in the "
             "matrix.");

#ifdef GINKGO_BUILD_CUDA
    // __cuda_array_interface__ (v3) for zero-copy interop with CuPy and
    // other CUDA-aware Python libraries.
    // Only available when the matrix is on a CUDA executor.
    cls.def_property_readonly(
        "__cuda_array_interface__",
        [](gko::matrix::Dense<ValueType> &m) -> py::dict {
            auto exec = m.get_executor();
            if (!std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
                throw py::attribute_error(
                    "__cuda_array_interface__ is only available for "
                    "dense matrices on CUDA executors");
            }
            auto gko_size = m.get_size();
            size_t rows = gko_size[0];
            size_t cols = gko_size[1];

            py::dict interface;
            if (cols == 1) {
                interface["shape"] = py::make_tuple(rows);
                interface["strides"] = py::none();
            } else {
                interface["shape"] = py::make_tuple(rows, cols);
                interface["strides"] =
                    py::make_tuple(static_cast<long>(m.get_stride() *
                                                     sizeof(ValueType)),
                                   static_cast<long>(sizeof(ValueType)));
            }
            interface["typestr"] = get_cuda_array_typestr<ValueType>();
            interface["data"] = py::make_tuple(
                reinterpret_cast<uintptr_t>(m.get_values()),
                false);  // (ptr, read_only)
            interface["version"] = 3;
            return interface;
        });

    // Factory method to create a dense matrix from a raw CUDA device
    // pointer. Creates an owning copy of the data pointed to.
    cls.def_static(
        "from_device_ptr",
        [](std::shared_ptr<gko::Executor> exec, uintptr_t ptr, size_t rows,
           size_t cols, size_t stride) {
            auto view = gko::array<ValueType>::view(
                exec, rows * stride, reinterpret_cast<ValueType *>(ptr));
            auto source = gko::matrix::Dense<ValueType>::create(
                exec,
                gko::dim<2>{static_cast<dim_type>(rows),
                            static_cast<dim_type>(cols)},
                view, stride);
            // Create an owning copy
            auto result = gko::share(gko::matrix::Dense<ValueType>::create(
                exec,
                gko::dim<2>{static_cast<dim_type>(rows),
                            static_cast<dim_type>(cols)}));
            result->operator=(*source);
            return result;
        },
        py::arg("exec"), py::arg("ptr"), py::arg("rows"), py::arg("cols"),
        py::arg("stride"),
        "Create a dense matrix by copying data from a device pointer. "
        "The pointer must be valid on the same CUDA device as the "
        "executor.");
#endif

    std::string read_dense_name = std::string("read_dense_") + typestr;
    module_matrix.def(
        read_dense_name.c_str(),
        [](const std::string &fn, std::shared_ptr<gko::Executor> exec) {
            return gko::share(gko::read<gko::matrix::Dense<ValueType>>(
                std::ifstream(fn), exec));
        });
}

void init_dense_all_types(py::module_ &module)
{
#define DECLARE_DENSE_VALUE(ValueType) init_dense<ValueType>(module, #ValueType)
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_DENSE_VALUE);
}

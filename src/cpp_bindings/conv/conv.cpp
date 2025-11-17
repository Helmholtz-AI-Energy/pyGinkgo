// SPDX-FileCopyrightText: 2025 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <pybind11/stl.h>

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType>
void init_conv2d(py::module_ &module_matrix, const std::string &typestr)
{
    using Conv2d = gko::matrix::Conv2d<ValueType>;
    using Dense = gko::matrix::Dense<ValueType>;

    std::string pyclass_name = std::string("conv2d_") + typestr;

    py::class_<Conv2d, std::shared_ptr<Conv2d>, gko::LinOp>(
        module_matrix, pyclass_name.c_str())
        // --- Constructors ---
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {
            return gko::share(Conv2d::create(exec));
        }))

        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const Dense> kernel) {
            return gko::share(Conv2d::create(exec, kernel));
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::vector<std::shared_ptr<const Dense>> kernels) {
            return gko::share(Conv2d::create(exec, kernels));
        }))

        // --- Repr ---
        .def("__repr__",
             [pyclass_name](const Conv2d &op) {
                 std::ostringstream oss;
                 oss << "pygko.matrix." << pyclass_name << " with "
                     << op.kernels_.size() << " kernel(s)";
                 return oss.str();
             })

        // --- Apply methods ---
        /*        .def(
                    "apply",
                    [](const Conv2d &op,
                       std::shared_ptr<const gko::LinOp> b,
                       std::shared_ptr<gko::LinOp> x) { op.apply(b, x); },
                    py::arg("b"), py::arg("x"),
                    "Applies the 2D convolution to input `b` and stores result
           in `x`.")
        */
        .def(
            "apply_multi",
            [](const Conv2d &op, std::shared_ptr<const gko::LinOp> b,
               std::vector<std::shared_ptr<gko::LinOp>> xs) {
                op.apply(b, xs);
            },
            py::arg("b"), py::arg("xs"),
            "Applies 2D convolution using multiple kernels (multi-output).")

        // --- Accessors ---
        .def_property_readonly(
            "kernels", [](const Conv2d &op) { return op.kernels_; },
            "Returns the list of kernel matrices associated with this Conv2d.");
}

void init_conv2d_all_types(py::module_ &module)
{
#define DECLARE_CONV2D_VALUE(ValueType) \
    init_conv2d<ValueType>(module, #ValueType)
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_CONV2D_VALUE);
}

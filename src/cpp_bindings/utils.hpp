// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/core/base/types.hpp>

#include "python.hpp"

template <typename ValueType>
void check_buffer_dtype(const py::buffer_info &info)
{
    auto expected = py::format_descriptor<ValueType>::format();
    if (info.format == expected) return;

    // On platforms where two distinct C types have the same size (e.g.
    // 'l' vs 'q' for 64-bit integers on x86_64 Linux, or 'l' vs 'i' for
    // 32-bit integers on Windows), pybind11's format string and numpy's
    // format string can disagree even though the layout is identical.
    // Treat them as compatible if the itemsize and integer/float-ness
    // match.
    if (static_cast<size_t>(info.itemsize) == sizeof(ValueType) &&
        info.format.size() == 1 && expected.size() == 1) {
        const char a = info.format[0];
        const char b = expected[0];
        auto is_signed_int = [](char c) {
            return c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' ||
                   c == 'n';
        };
        auto is_unsigned_int = [](char c) {
            return c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' ||
                   c == 'N';
        };
        if ((is_signed_int(a) && is_signed_int(b)) ||
            (is_unsigned_int(a) && is_unsigned_int(b))) {
            return;
        }
    }

    throw std::runtime_error("Incompatible dtypes: " + info.format +
                             " vs " + expected);
}

/**
 * Returns a NumPy-style dtype string for the given ValueType,
 * used by the __cuda_array_interface__ protocol.
 *
 * For example: float -> "<f4", double -> "<f8", half -> "<f2"
 */
template <typename ValueType>
std::string get_cuda_array_typestr()
{
    auto np = py::module_::import("numpy");
    return np.attr("dtype")(py::format_descriptor<ValueType>::format())
        .attr("str")
        .template cast<std::string>();
}

/**
 * Create a gko::array<T> from a Python object.
 *
 * CUDA executor + __cuda_array_interface__: zero-copy device view.
 * Otherwise: copy via buffer protocol (host memory).
 *
 * This is the single source of truth for the "py::object → gko::array"
 * conversion used by the array, CSR, and COO pybind11 constructors.
 */
template <typename T>
gko::array<T> gko_array_from_pyobject(std::shared_ptr<gko::Executor> exec,
                                      py::object obj)
{
#ifdef GINKGO_BUILD_CUDA
    if (py::hasattr(obj, "__cuda_array_interface__") &&
        std::dynamic_pointer_cast<const gko::CudaExecutor>(exec)) {
        auto cai = obj.attr("__cuda_array_interface__").cast<py::dict>();
        auto shape = cai["shape"].cast<py::tuple>();
        if (py::len(shape) != 1) {
            throw std::runtime_error(
                "__cuda_array_interface__ object must be 1D (got " +
                std::to_string(py::len(shape)) + " dimensions)");
        }

        auto typestr = cai["typestr"].cast<std::string>();
        auto expected = get_cuda_array_typestr<T>();
        if (typestr != expected) {
            throw std::runtime_error(
                "dtype mismatch: __cuda_array_interface__ reports '" +
                typestr + "' but expected '" + expected + "'");
        }

        // Validate contiguous storage: strides must be None or sizeof(T).
        if (cai.contains("strides")) {
            py::handle strides_obj = cai["strides"];
            if (!strides_obj.is_none()) {
                auto strides = strides_obj.cast<py::tuple>();
                if (strides.size() != 1 ||
                    strides[0].cast<ssize_t>() !=
                        static_cast<ssize_t>(sizeof(T))) {
                    throw std::runtime_error(
                        "__cuda_array_interface__ object must be 1D and "
                        "contiguous in memory");
                }
            }
        }

        auto data = cai["data"].cast<py::tuple>();
        auto ptr = data[0].cast<uintptr_t>();
        auto size = shape[0].cast<size_t>();
        return gko::array<T>::view(exec, size, reinterpret_cast<T *>(ptr));
    }
#endif
    // Fallback: copy via buffer protocol (host memory)
    auto buf =
        py::array_t<T, py::array::c_style | py::array::forcecast>(obj);
    py::buffer_info info = buf.request();
    check_buffer_dtype<T>(info);
    if (info.ndim != 1) {
        throw std::runtime_error("Only 1D arrays are supported");
    }
    auto elems = info.shape[0];
    return gko::array<T>(exec, (T *)info.ptr, (T *)info.ptr + elems);
}


/**
 * Instantiates a template for each non-complex value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro) \
    _macro(float)
#else
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro) \
    _macro(float);                                                     \
    _macro(double)
#endif

// cuda half operation is supported from arch 5.3
#define PYGKO_HALF_ENABLED \
    GINKGO_ENABLE_HALF && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
#if PYGKO_HALF_ENABLED
#define PYGKO_ADAPT_HF(_macro) _macro
#else
#define PYGKO_ADAPT_HF(_macro)
#endif

#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    PYGKO_ADAPT_HF(_macro(half));                                 \
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(_macro)


/**
 * Instantiates a template for each index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define PYGKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro) \
    _macro(int32);                                    \
    _macro(int64)


/**
 * Instantiates a template for each non-complex value and index type compiled by
 * Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE( \
    _macro)                                                               \
    _macro(float, int32);                                                 \
    _macro(float, int64)
#else
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE( \
    _macro)                                                               \
    _macro(float, int32);                                                 \
    _macro(double, int32);                                                \
    _macro(float, int64);                                                 \
    _macro(double, int64)
#endif
#define PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro) \
    PYGKO_ADAPT_HF(_macro(half, int32));                                    \
    PYGKO_ADAPT_HF(_macro(half, int64));                                    \
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(_macro)

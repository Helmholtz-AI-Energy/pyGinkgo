// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#ifdef PYGINKGO_BUILD_MPI

#include <ginkgo/ginkgo.hpp>

#include "../python.hpp"

namespace {

// Pybind11 trampoline that lets Python subclass a Ginkgo LinOp by overriding
// `apply_impl`. This is the matrix-free path: the Python subclass receives the
// input/output LinOps (typically distributed::Vector) and is responsible for
// performing y = A @ x using whatever Python code (cupy, numba, custom CUDA).
class PyLinOp : public gko::EnableLinOp<PyLinOp> {
    friend class gko::EnablePolymorphicObject<PyLinOp, gko::LinOp>;

public:
    PyLinOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {})
        : gko::EnableLinOp<PyLinOp>(exec, size)
    {}

    static std::unique_ptr<PyLinOp> create(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
    {
        return std::unique_ptr<PyLinOp>(new PyLinOp(exec, size));
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(this, "apply_impl");
        if (!override) {
            throw std::runtime_error(
                "PyLinOp subclass must override apply_impl(b, x)");
        }
        // Wrap the raw pointers in non-owning shared_ptrs (no-op deleter)
        // so pybind11 can hand them to Python. The Python override should
        // not store or outlive these pointers.
        auto b_sp = std::shared_ptr<const gko::LinOp>(b, [](auto*) {});
        auto x_sp = std::shared_ptr<gko::LinOp>(x, [](auto*) {});
        override(b_sp, x_sp);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(this, "apply_impl_scaled");
        if (!override) {
            // No generic implementation: 4-arg apply on an arbitrary LinOp
            // requires a Dense axpby that we cannot dispatch on a bare
            // LinOp. Most matrix-free users only need the 2-arg form (the
            // one Krylov solvers actually call).
            throw gko::NotImplemented(
                __FILE__, __LINE__,
                "PyLinOp: override `apply_impl_scaled(alpha, b, beta, x)` to "
                "use the 4-arg apply on a matrix-free Python LinOp.");
        }
        auto alpha_sp = std::shared_ptr<const gko::LinOp>(alpha, [](auto*) {});
        auto b_sp = std::shared_ptr<const gko::LinOp>(b, [](auto*) {});
        auto beta_sp = std::shared_ptr<const gko::LinOp>(beta, [](auto*) {});
        auto x_sp = std::shared_ptr<gko::LinOp>(x, [](auto*) {});
        override(alpha_sp, b_sp, beta_sp, x_sp);
    }
};

class PyLinOpTrampoline : public PyLinOp {
public:
    using PyLinOp::PyLinOp;
};

}  // namespace

void init_distributed_pylinop(py::module_& module)
{
    py::class_<PyLinOp, std::shared_ptr<PyLinOp>, gko::LinOp,
               PyLinOpTrampoline>(
        module, "PyLinOp",
        "Subclass-this LinOp for matrix-free operators in Python. Override "
        "`apply_impl(b, x)` to compute `x = A @ b` (and optionally "
        "`apply_impl_scaled(alpha, b, beta, x)` for `x = alpha*A*b + "
        "beta*x`).\n\n"
        "Distributed usage: when the enclosing solver is invoked on a "
        "`distributed.Vector`, the callback receives the *distributed* "
        "vectors directly (cast as `gko::LinOp*`); only the local block "
        "is owned by this rank. The callback is responsible for any "
        "halo exchange required by the matrix-free matvec -- Ginkgo does "
        "not perform implicit halo communication for user-defined "
        "operators. A typical CuPy callback should call "
        "`cupy.cuda.Stream.null.synchronize()` before returning to "
        "guarantee Ginkgo sees the writes (Ginkgo dispatches on its own "
        "executor stream which may differ from CuPy's default).")
        .def(py::init([](std::shared_ptr<const gko::Executor> exec,
                         py::tuple size) {
                 return std::shared_ptr<PyLinOpTrampoline>(
                     new PyLinOpTrampoline(
                         exec, gko::dim<2>{size[0].cast<size_t>(),
                                           size[1].cast<size_t>()}));
             }),
             py::arg("exec"), py::arg("size"));
}

#endif  // PYGINKGO_BUILD_MPI

// SPDX-FileCopyrightText: 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <ginkgo/ginkgo.hpp>

#include <pybind11/pybind11.h>


namespace py = pybind11;


namespace gko {


class PyLinOp : public EnableLinOp<PyLinOp> {
public:
    friend class EnableLinOp<PyLinOp>;
    friend class EnablePolymorphicObject<PyLinOp, LinOp>;

public:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    explicit PyLinOp(std::shared_ptr<const gko::Executor> exec,
                     gko::dim<2> dim = gko::dim<2>{})
        : gko::EnableLinOp<PyLinOp>(std::move(exec), dim)
    {}
};


class PyLinOpTrampoline : public PyLinOp {
    using PyLinOp::PyLinOp;

public:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        PYBIND11_OVERRIDE(void, PyLinOp, apply_impl, b, x);
    }
};


std::unique_ptr<PyLinOp> create(std::shared_ptr<const gko::Executor> exec)
{
    return std::unique_ptr<PyLinOp>{new PyLinOpTrampoline{exec}};
}

std::unique_ptr<PyLinOp> create(std::shared_ptr<const gko::Executor> exec,
                                gko::dim<2> dim)
{
    return std::unique_ptr<PyLinOp>{new PyLinOpTrampoline{exec, dim}};
}


}  // namespace gko


class Publicist
    : public gko::LinOp {  // helper type for exposing protected functions
public:
    using gko::LinOp::apply_impl;  // inherited with different access modifier
};

PYBIND11_MODULE(pyGinkgoExtensions, m, py::mod_gil_not_used())
{
    py::class_<gko::PyLinOp, gko::LinOp, gko::PyLinOpTrampoline,
               std::shared_ptr<gko::PyLinOp>>(m, "PyLinOp")
        .def(py::init([](std::shared_ptr<const gko::Executor> exec) {
            auto A = std::shared_ptr(create(exec));
            return A;
        }))
        .def(py::init(
            [](std::shared_ptr<const gko::Executor> exec, py::tuple dim) {
                auto A = std::shared_ptr(create(
                    exec,
                    gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()}));
                return A;
            }))
        .def("apply_impl", py::overload_cast<const gko::LinOp *, gko::LinOp *>(
                               &Publicist::apply_impl, py::const_));

    m.def("call_apply", [](const gko::LinOp *linop, const gko::LinOp *b,
                           gko::LinOp *x) { linop->apply(b, x); });
}

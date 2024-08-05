// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2024 pyGinkgo authors

#include "python.hpp"

void init_dense(py::module_ &module_matrix) {

  module_matrix.def("read_dense", [](const std::string &fn,
                                     std::shared_ptr<gko::Executor> exec) {
    return gko::share(
        gko::read<gko::matrix::Dense<ValueType>>(std::ifstream(fn), exec));
  });

  /* function to create a dense matrix from py::buffer object
   *
   */
  auto init_func = [](std::shared_ptr<gko::Executor> exec, py::buffer b) {
    /* Request a buffer descriptor from Python */
    py::buffer_info info = b.request();

    if (info.format != py::format_descriptor<ValueType>::format())
      throw std::runtime_error("Incompatible dtype");

    /* create a view into numpy data */
    auto elems =
        (info.ndim == 1) ? info.shape[0] : info.shape[0] * info.shape[1];

    auto view = gko::array<ValueType>::view(exec->get_master(), elems,
                                            (ValueType *)info.ptr);

    auto rows = info.shape[0];
    auto cols = (info.ndim == 1) ? 1 : info.shape[1];

    return gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{rows, cols},
                                                 view, cols);
  };

  py::class_<gko::matrix::Dense<ValueType>,
             std::shared_ptr<gko::matrix::Dense<ValueType>>, gko::LinOp>(
      module_matrix, "dense", py::buffer_protocol())
      .def(py::init([init_func](py::buffer b) {
        auto ref = gko::ReferenceExecutor::create();
        return init_func(ref, b);
      }))
      .def(py::init([init_func](std::shared_ptr<gko::Executor> exec,
                                py::buffer b) { return init_func(exec, b); }))
      .def(py::init([](std::shared_ptr<gko::Executor> exec) {
        return gko::share(gko::matrix::Dense<ValueType>::create(exec));
      }))
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim) {
        return gko::share(gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()}));
      }))
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                       int stride) {
        return gko::share(gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{dim[0].cast<int>(), dim[1].cast<int>()}, stride));
      }))
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                       py::buffer b, size_t stride) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<ValueType>::format())
          throw std::runtime_error("Incompatible dtype");

        auto ref = gko::ReferenceExecutor::create();

        /* create a view into numpy data */
        auto elems =
            (info.ndim == 1) ? info.shape[0] : info.shape[0] * info.shape[1];

        auto view =
            gko::array<ValueType>::view(ref, elems, (ValueType *)info.ptr);

        auto rows = info.shape[0];
        auto cols = (info.ndim == 1) ? 1 : info.shape[1];

        return gko::share(gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
            view, stride));
      }))
      .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,
                       gko::array<ValueType> view, size_t stride) {
        return gko::share(gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},
            view, stride));
      }))
      .def("__repr__",
           [](const gko::matrix::Dense<ValueType> &o) {
             auto str = std::string("pygko.matrix.Dense object of size ");
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
               }
               str += " ] ";
             }
             return str;
           })
      .def("copy_to_host",
           [](gko::matrix::Dense<ValueType> &m) {
             auto host_exec = m.get_executor()->get_master();
             std::cout << __FILE__ << "Warning creating a copy of dense\n";
             if (m.get_executor() != host_exec) {
               auto host_dense =
                   gko::share(gko::matrix::Dense<ValueType>::create(host_exec));
               host_dense->operator=(m);
               return host_dense;
             } else {
               return std::make_shared<gko::matrix::Dense<ValueType>>(m);
             }
           })
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
              dim,          /* Number of dimensions */
              {rows, cols}, /* Buffer dimensions */
              {sizeof(ValueType), sizeof(ValueType) * rows}
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
             auto o = gko::matrix::Dense<ValueType>::create(m.get_executor(),
                                                            gko::dim<2>(1, 1));
             o->fill(s);
             m.scale(o);
           })
      .def("inv_scale",
           [](gko::matrix::Dense<ValueType> &m, ValueType s) {
             auto o = gko::matrix::Dense<ValueType>::create(m.get_executor(),
                                                            gko::dim<2>(1, 1));
             o->fill(s);
             m.inv_scale(o);
           })
      .def("add_scaled", &gko::matrix::Dense<ValueType>::add_scaled,
           "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
      .def("sub_scaled", &gko::matrix::Dense<ValueType>::sub_scaled,
           "Subtracts `b` scaled by `alpha` from the matrix (aka: BLAS axpy).")
      .def("at",
           py::overload_cast<size_t>(&gko::matrix::Dense<ValueType>::at,
                                     py::const_),
           "Returns an element using linearized index.")
      .def("at",
           py::overload_cast<size_t, size_t>(&gko::matrix::Dense<ValueType>::at,
                                             py::const_),
           "Returns an element at row, column index.")
      .def("get_num_stored_elements",
           &gko::matrix::Dense<ValueType>::get_num_stored_elements,
           "Returns the number of elements explicitly stored in the "
           "matrix.");
}

// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"


template <typename ValueType>
void init_vector(py::module_ &module, const std::string typestr)
{
    using Vector = gko::experimental::distributed::Vector<ValueType>;
    using Partition = gko::experimental::distributed::Partition<int, int>;
    using Dense = gko::matrix::Dense<ValueType>;
    using Communicator = gko::experimental::mpi::communicator;
    using Matrix_data = gko::matrix_data<ValueType, int64>;
    using dim_type = gko::dim<2>::dimension_type;

    std::string pyclass_name = std::string("vector_") + typestr;

    py::class_<Vector, std::shared_ptr<Vector>, gko::LinOp>(
        module, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<const gko::Executor> exec,
                         Communicator comm, std::shared_ptr<Partition> part,
                         std::shared_ptr<Dense> in) {
            return gko::share(Vector::create(
                exec, comm, gko::dim<2>{part->get_size(), 1}, in->clone()));
        }))
        .def("fill", &Vector::fill, "Fill the vector with the given value.");
}

void init_vector_all_types(py::module_ &module)
{
#define DECLARE_VECTOR_VALUE(ValueType) \
    init_vector<ValueType>(module, #ValueType);

    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_VECTOR_VALUE);
}

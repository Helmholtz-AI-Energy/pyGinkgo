// SPDX-FileCopyrightText: 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"


template <typename ValueType, typename IndexType>
void init_distributed_matrix(py::module_ &module, const std::string value_type,
                             const std::string index_type)
{
    std::string matrix_type_repr = "dist_matrix_object";
    std::string value_index_str = value_type + "_" + index_type;
    std::string pyclass_name = matrix_type_repr + "_" + value_index_str;

    using Matrix =
        gko::experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
    py::class_<Matrix, std::shared_ptr<Matrix>, gko::LinOp>(
        module, pyclass_name.c_str())
        .def(py::init(
            [](std::shared_ptr<const gko::Executor> exec,
               gko::experimental::mpi::communicator comm,
               std::shared_ptr<const gko::experimental::distributed::Partition<
                   IndexType, IndexType>>
                   part,
               std::shared_ptr<const gko::matrix::Coo<ValueType, IndexType>>
                   in) {
                return gko::share(
                    gko::experimental::distributed::create_from_super_rank(
                        exec, comm, in, part));
            }));
}

void init_dist_matrix_all_types(py::module_ &module)
{
#define DECLARE_DISTRIBUTED_MATRIX_VALUE_INDEX(ValueType, IndexType)  \
    init_distributed_matrix<ValueType, IndexType>(module, #ValueType, \
                                                  #IndexType);

    DECLARE_DISTRIBUTED_MATRIX_VALUE_INDEX(float, int);
    // PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    //     DECLARE_DISTRIBUTED_MATRIX_VALUE_INDEX);
}

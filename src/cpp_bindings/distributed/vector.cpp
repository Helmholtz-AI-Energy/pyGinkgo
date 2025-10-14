// SPDX-FileCopyrightText: 2024 - 2025 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"


template <typename ValueType>
void init_vector(py::module_ &module, const std::string typestr)
{
    std::string pyclass_name = std::string("vector_") + typestr;
}

void init_vector_all_types(py::module_ &module)
{
#define DECLARE_VECTOR_VALUE(ValueType) \
    init_vector<ValueType>(module, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_ARRAY_VALUE);
    PYGKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(DECLARE_ARRAY_VALUE);
}

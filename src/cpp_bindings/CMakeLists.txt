# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

set(PYGINKGO_CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/python.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/array.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/dense.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/dpcpp_queue_property.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/matrix.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/stream.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/logger.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/preconditioner/ilu.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/cg.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/config_solver.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/direct.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/gmres.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/triangular.cpp
    ${PROJECT_SOURCE_DIR}/src/cpp_bindings/factorization/factorization.cpp
    PARENT_SCOPE)

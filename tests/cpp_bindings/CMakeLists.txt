# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_path(GET CMAKE_LIBRARY_OUTPUT_DIRECTORY PARENT_PATH
           PYGINKGO_TESTS_WORK_DIR)

add_test(
  NAME pyginkgo_bindings_import_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_import.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
add_test(
  NAME pyginkgo_bindings_array_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_array.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
add_test(
  NAME pyginkgo_bindings_dense_matrix_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_dense.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
add_test(
  NAME pyginkgo_bindings_sparse_matrix_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_sparse.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
add_test(
  NAME pyginkgo_bindings_solver_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_solver.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
add_test(
  NAME pyginkgo_bindings_factorization_test
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/test_factorization.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})

# add CUDA specific tests, if building with CUDA enabled
if(GINKGO_BUILD_CUDA)
  add_test(
    NAME pyginkgo_bindings_cuda_executor_test
    COMMAND ${PYTHON_EXECUTABLE} -m pytest
            ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_executor.py
    WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endif()

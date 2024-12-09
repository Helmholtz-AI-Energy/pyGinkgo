# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_path(GET CMAKE_LIBRARY_OUTPUT_DIRECTORY PARENT_PATH
           PYGINKGO_TESTS_WORK_DIR)

function(pyGinkgo_bindings_test TEST)
add_test(
  NAME pyginkgo_bindings_test_${TEST}
  COMMAND ${PYTHON_EXECUTABLE} -m pytest
  ${CMAKE_CURRENT_SOURCE_DIR}/test_${TEST}.py
  WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endfunction()

pyGinkgo_bindings_test(import)
pyGinkgo_bindings_test(array)
pyGinkgo_bindings_test(dense)
pyGinkgo_bindings_test(sparse)
pyGinkgo_bindings_test(iterative_solver)
pyGinkgo_bindings_test(direct_solver)

# add CUDA specific tests, if building with CUDA enabled
if(GINKGO_BUILD_CUDA)
    pyGinkgo_bindings_test(cuda_executor)
endif()

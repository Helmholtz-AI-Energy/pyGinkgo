# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_path(GET CMAKE_LIBRARY_OUTPUT_DIRECTORY PARENT_PATH
           PYGINKGO_TESTS_WORK_DIR)

# * Function to generate bindings unit test.
function(pyGinkgo_bindings_test TEST)
  add_test(
    NAME pyginkgo_bindings_test_${TEST}
    COMMAND ${PYTHON_EXECUTABLE} -m pytest
            ${CMAKE_CURRENT_SOURCE_DIR}/test_${TEST}.py
    WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endfunction()

pyginkgo_bindings_test(import)
pyginkgo_bindings_test(array)
pyginkgo_bindings_test(dense)
pyginkgo_bindings_test(sparse)
pyginkgo_bindings_test(iterative_solver)
pyginkgo_bindings_test(direct_solver)

# add CUDA specific tests, if building with CUDA enabled
if(GINKGO_BUILD_CUDA)
  pyginkgo_bindings_test(cuda_executor)
endif()

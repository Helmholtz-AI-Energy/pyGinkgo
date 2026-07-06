# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_path(GET CMAKE_LIBRARY_OUTPUT_DIRECTORY PARENT_PATH
           PYGINKGO_TESTS_WORK_DIR)

# * Function to generate Python-side unit test.
function(pyginkgo_python_test TEST)
  # Correct variable according to
  # https://cmake.org/cmake/help/latest/module/FindPython.html Printing on test
  # skip: https://stackoverflow.com/a/64776498
  add_test(
    NAME pyginkgo_${TEST}_test
    COMMAND ${Python_EXECUTABLE} -m pytest -vv -rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test_${TEST}.py
    WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endfunction()

pyginkgo_python_test(solve)
pyginkgo_python_test(RR)
pyginkgo_python_test(core)

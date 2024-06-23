# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_minimum_required(VERSION 3.20)
project(pyGinkgo)

enable_testing()

# Find Python interpreter and library
find_package(Python COMPONENTS Interpreter Development)

# Find OpenMPI find_package(MPI REQUIRED)

# Find pybind11
find_package(pybind11 QUIET)

# Fetch pybind11 if not found
if(NOT pybind11_FOUND)
  # If not found, fetch Pybind11
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0)
  FetchContent_MakeAvailable(pybind11)
endif()

# Find  Ginkgo find_package(Ginkgo 1.7.0 QUIET)
find_package(ginkgo QUIET)

# Fetch Ginkgo if not found
if(NOT ginkgo_FOUND)
  # If not found, fetch Ginkgo
  include(FetchContent)
  if(NOT DEFINED GINKGO_BUILD_TESTS)
    set(GINKGO_BUILD_TESTS OFF)
  endif()
  if(NOT DEFINED GINKGO_BUILD_BENCHMARKS)
    set(GINKGO_BUILD_BENCHMARKS OFF)
  endif()
  if(NOT DEFINED GINKGO_BUILD_EXAMPLES)
    set(GINKGO_BUILD_EXAMPLES OFF)
  endif()
  FetchContent_Declare(
    ginkgo
    GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
    GIT_TAG v1.8.0)
  FetchContent_MakeAvailable(ginkgo)

endif()

pybind11_add_module(
  pyGinkgo ${PROJECT_SOURCE_DIR}/src/python.cpp
  ${PROJECT_SOURCE_DIR}/src/array.cpp ${PROJECT_SOURCE_DIR}/src/dense.cpp)

target_include_directories(pyGinkgo PRIVATE ${Ginkgo_BINARY_DIR}/include)
target_link_libraries(pyGinkgo PRIVATE Ginkgo::ginkgo)
set_property(TARGET pyGinkgo PROPERTY CXX_STANDARD 14)

# adding tests
add_test(NAME pyginkgo_import_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_import.py)
add_test(NAME pyginkgo_array_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_array.py)
add_test(NAME pyginkgo_matrix_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dense.py)

# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_minimum_required(VERSION 3.20)
project(pyGinkgo)

include(cmake/AutoEnableDevice.cmake)
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
    GIT_TAG v1.8.0
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(ginkgo)
endif()

# Pass build type down to c++ macro: https://stackoverflow.com/a/9017635/8302811
if(GINKGO_BUILD_CUDA)
  add_compile_definitions(GINKGO_BUILD_CUDA)
endif()

pybind11_add_module(
  pyGinkgoBindings
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/python.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/array.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/dense.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/memory.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/stream.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/executor.cpp
  ${PROJECT_SOURCE_DIR}/src/cpp_bindings/solver/gmres.cpp)

target_include_directories(pyGinkgoBindings
                           PRIVATE ${Ginkgo_BINARY_DIR}/include)
target_link_libraries(pyGinkgoBindings PRIVATE Ginkgo::ginkgo)
set_property(TARGET pyGinkgoBindings PROPERTY CXX_STANDARD 14)

# adding tests
add_test(NAME pyginkgo_import_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_import.py)
add_test(NAME pyginkgo_array_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_array.py)
add_test(NAME pyginkgo_dense_matrix_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dense.py)
add_test(NAME pyginkgo_sparse_matrix_test
         COMMAND ${PYTHON_EXECUTABLE} -m pytest
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sparse.py)

# add CUDA specific tests, if building with CUDA enabled
if(GINKGO_BUILD_CUDA)
  add_test(NAME pyginkgo_cuda_executor_test
           COMMAND ${PYTHON_EXECUTABLE} -m pytest
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_executor.py)
endif()

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()
set(CMAKE_INSTALL_RPATH ${Python_SITELIB}/${PROJECT_NAME})

message("Install path: ${Python_SITELIB}")
install(TARGETS pyGinkgoBindings DESTINATION ${Python_SITELIB}/${PROJECT_NAME})
install(
  IMPORTED_RUNTIME_ARTIFACTS
  ginkgo
  ginkgo_device
  ginkgo_hip
  ginkgo_cuda
  ginkgo_omp
  ginkgo_dpcpp
  ginkgo_reference
  DESTINATION
  ${Python_SITELIB}/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/pyGinkgo/
        DESTINATION ${Python_SITELIB}/${PROJECT_NAME})

# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_minimum_required(VERSION 3.20)
project(pyGinkgo)

include(cmake/AutoEnableDevice.cmake)
enable_testing()

# Find Python interpreter and library
find_package(Python COMPONENTS Interpreter Development)

# copy all generated files to pyGinkgo that makes testing and installing easier
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

# Find OpenMPI find_package(MPI REQUIRED)

# Find pybind11
find_package(pybind11 QUIET)

# Fetch pybind11 if not found
include(FetchContent)
if(NOT pybind11_FOUND)
  # If not found, fetch Pybind11
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0)
  FetchContent_MakeAvailable(pybind11)
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3)
  FetchContent_MakeAvailable(nlohmann_json)
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

add_subdirectory(src/cpp_bindings)
pybind11_add_module(pyGinkgoBindings ${PYGINKGO_CPP_SOURCES})

target_include_directories(pyGinkgoBindings
                           PRIVATE ${Ginkgo_BINARY_DIR}/include)
target_link_libraries(pyGinkgoBindings PRIVATE Ginkgo::ginkgo)
target_link_libraries(pyGinkgoBindings PUBLIC nlohmann_json::nlohmann_json)
set_property(TARGET pyGinkgoBindings PROPERTY CXX_STANDARD 14)

# Generating stubs for the library (interface annotation) using
# https://github.com/sizmailov/pybind11-stubgen Thus this library should be
# first installed before enabling the generation
option(
  ENABLE_PYGINKGOBINDINGS_STUBS
  "Whether the stub (interface) files are to be generated for pyGinkgoBindings"
  OFF)
message(
  "Stubs generation for pyGinkgoBindings is: ${ENABLE_PYGINKGOBINDINGS_STUBS}")
if(${ENABLE_PYGINKGOBINDINGS_STUBS})
  add_custom_command(
    TARGET pyGinkgoBindings
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    VERBATIM
    COMMAND
      ${PYTHON_EXECUTABLE} -c
      "import pybind11_stubgen; pybind11_stubgen.main();" pyGinkgoBindings
      "-o./"
    COMMENT "Generating stub files at ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

# Generating the virtual installation within the build folder
set(PYGINKGO_PYTHON_FILES ${PROJECT_SOURCE_DIR}/src/pyGinkgo/__init__.py
                          ${PROJECT_SOURCE_DIR}/src/pyGinkgo/core.py)
add_custom_target(
  pyGinkgo_virtual_installation ALL
  DEPENDS pyGinkgoBindings
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYGINKGO_PYTHON_FILES}
          ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  VERBATIM
  COMMENT "Creating self-contained library folder at: \
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# adding tests
add_subdirectory(tests/cpp_bindings)
add_subdirectory(tests/pyGinkgo)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()
set(CMAKE_INSTALL_RPATH ${Python_SITELIB}/${PROJECT_NAME})

message("Install path: ${Python_SITELIB}")
install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        DESTINATION ${Python_SITELIB})
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

# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

cmake_minimum_required(VERSION 3.20)
project(pyGinkgo)

include(cmake/AutoEnableDevice.cmake)
enable_testing()

# Find Python interpreter and library
set(Python_FIND_VIRTUALENV "FIRST")
find_package(Python COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# copy all generated files to pyGinkgo that makes testing and installing easier
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

# Multi-config generators (Visual Studio) append the configuration name to the
# output directory, which would put the extension module in pyGinkgo/Release/
# instead of next to the Python sources. It would then be packaged one level too
# deep and fail to import as pyGinkgo.pyGinkgoBindings. Pin every configuration
# to the same directory. CMAKE_CONFIGURATION_TYPES is empty for single-config
# generators, so this is a no-op there.
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper}
      ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
endforeach()

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
    GIT_TAG develop
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(ginkgo)
endif()

# Pass build type down to c++ macro: https://stackoverflow.com/a/9017635/8302811
if(GINKGO_BUILD_CUDA)
  add_compile_definitions(GINKGO_BUILD_CUDA)
endif()
if(GINKGO_BUILD_HIP)
  add_compile_definitions(GINKGO_BUILD_HIP)
endif()
if(GINKGO_BUILD_SYCL)
  add_compile_definitions(GINKGO_BUILD_SYCL)
endif()

add_subdirectory(src/cpp_bindings)

pybind11_add_module(pyGinkgoBindings ${PYGINKGO_CPP_SOURCES})

target_include_directories(pyGinkgoBindings
                           PRIVATE ${Ginkgo_BINARY_DIR}/include)
target_link_libraries(pyGinkgoBindings PRIVATE Ginkgo::ginkgo)
target_link_libraries(pyGinkgoBindings PUBLIC nlohmann_json::nlohmann_json)
set_property(TARGET pyGinkgoBindings PROPERTY CXX_STANDARD 14)

# MPI / distributed bindings: only built when Ginkgo is compiled with MPI.
# FetchContent builds expose GINKGO_BUILD_MPI as a cache variable, while installed
# Ginkgo packages may only expose it through target compile definitions.
set(PYGINKGO_GINKGO_HAS_MPI OFF)
if(DEFINED GINKGO_BUILD_MPI AND GINKGO_BUILD_MPI)
  set(PYGINKGO_GINKGO_HAS_MPI ON)
endif()
if(TARGET Ginkgo::ginkgo)
  get_target_property(_ginkgo_compile_defs Ginkgo::ginkgo INTERFACE_COMPILE_DEFINITIONS)
  if(_ginkgo_compile_defs)
    foreach(_ginkgo_def IN LISTS _ginkgo_compile_defs)
      if(_ginkgo_def MATCHES "(^|[<:])GINKGO_BUILD_MPI($|[>:])")
        set(PYGINKGO_GINKGO_HAS_MPI ON)
      endif()
    endforeach()
  endif()
endif()

if(PYGINKGO_GINKGO_HAS_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(pyGinkgoBindings PRIVATE GINKGO_BUILD_MPI)
  target_link_libraries(pyGinkgoBindings PRIVATE MPI::MPI_CXX)
  message(STATUS "pyGinkgo: MPI-distributed bindings enabled")
else()
  message(STATUS "pyGinkgo: MPI-distributed bindings disabled")
endif()

# Disable false positive warnings from GCC (>= 12.4 and <
# 13)(https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115824 and
# https://github.com/pybind/pybind11/issues/5224)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.4.0
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(pyGinkgoBindings PRIVATE -Wno-array-bounds
                                                  -Wno-stringop-overread)
  message(
    WARNING
      "GCC >= 12.4.0 < 13 gives false positive warning, so we need to add \
      `-Wno-array-bounds -Wno-stringop-overread`. Please refer to \
      https://github.com/pybind/pybind11/issues/5224 for more details.")
endif()

# Generating stubs for the library (interface annotation) using
# https://github.com/sizmailov/pybind11-stubgen Thus this library should be
# first installed before enabling the generation
option(
  ENABLE_PYGINKGOBINDINGS_STUBS
  "Whether the stub (interface) files are to be generated for pyGinkgoBindings"
  OFF)
message(
  STATUS
    "Stubs generation for pyGinkgoBindings is: ${ENABLE_PYGINKGOBINDINGS_STUBS}"
)
if(${ENABLE_PYGINKGOBINDINGS_STUBS})
  # Correct variable according to
  # https://cmake.org/cmake/help/latest/module/FindPython.html
  add_custom_command(
    TARGET pyGinkgoBindings
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    VERBATIM
    COMMAND
      ${Python_EXECUTABLE} -c
      "import pybind11_stubgen; pybind11_stubgen.main();" pyGinkgoBindings
      "-o./"
    COMMENT "Generating stub files at ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

option(
  ENABLE_PYGINKGOBINDINGS_DEV_STUBS
  "Whether the stub (interface) files are to be generated for pyGinkgoBindings\
   at src/pyGinkgo folder" OFF)
if(${ENABLE_PYGINKGOBINDINGS_DEV_STUBS})
  message(STATUS "Enabled stubs generation for pyGinkgoBindings at\
     '${PROJECT_SOURCE_DIR}/src/pyGinkgo/pyGinkgoBindings'")
  # Correct variable according to
  # https://cmake.org/cmake/help/latest/module/FindPython.html
  add_custom_command(
    TARGET pyGinkgoBindings
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    VERBATIM
    COMMAND
      ${Python_EXECUTABLE} -c
      "import pybind11_stubgen, distutils.dir_util; pybind11_stubgen.main();\
       distutils.dir_util.copy_tree('./pyGinkgoBindings',\
       '${PROJECT_SOURCE_DIR}/src/pyGinkgo/pyGinkgoBindings', update=1)"
      pyGinkgoBindings "-o./"
    COMMENT "Generating stub files at ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

# Generating the virtual installation within the build folder
set(PYGINKGO_PYTHON_FILES
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/__init__.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/core.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/gko_types.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/device.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/solver.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/rayleigh_ritz.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/preconditioner.py
    ${PROJECT_SOURCE_DIR}/src/pyGinkgo/distributed.py)
# TODO: it seems reasonable to be to start using glob recuse above

add_custom_target(
  pyGinkgo_virtual_installation ALL
  DEPENDS pyGinkgoBindings
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYGINKGO_PYTHON_FILES}
          ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  VERBATIM
  COMMENT "Creating self-contained library folder at: \
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# adding tests
include(CTest) # https://cmake.org/pipermail/cmake/2015-January/059742.html -
               # removing the issue with checking with dart availability
add_subdirectory(tests/cpp_bindings)
add_subdirectory(tests/pyGinkgo)

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()
# The extension module and the Ginkgo shared libraries are installed next to
# each other, so a relative RPATH keeps the result relocatable. An absolute
# RPATH into the build environment would not survive wheel packaging.
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Install destinations must stay RELATIVE. CMake ignores CMAKE_INSTALL_PREFIX
# whenever a DESTINATION is absolute, and py-build-cmake relies on that prefix
# to redirect the install into the wheel staging directory -- an absolute
# ${Python_SITELIB} destination silently installs into the isolated build
# environment instead, producing a wheel with no binaries in it. Defaulting the
# prefix to site-packages keeps a plain `cmake --install .` behaving as before.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "${Python_SITELIB}"
      CACHE PATH "Where the pyGinkgo package is installed" FORCE)
endif()

message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} DESTINATION ".")
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
  ${PROJECT_NAME})

# On MSVC/MINGW shared builds Ginkgo splits its core out into a separate
# ginkgo_core library to work around the LNK1189 exported-symbol limit; see
# `set(ginkgo_core ...)` in ginkgo's core/CMakeLists.txt. Everywhere else
# ginkgo_core *is* ginkgo, so the target only exists on Windows -- where
# ginkgo.dll imports it and the package is unimportable without it.
if(TARGET ginkgo_core)
  install(IMPORTED_RUNTIME_ARTIFACTS ginkgo_core DESTINATION ${PROJECT_NAME})
endif()

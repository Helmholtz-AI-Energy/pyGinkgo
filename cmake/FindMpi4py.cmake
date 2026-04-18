# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# Locate the mpi4py headers via the Python interpreter.
#
# Provides:
#   Mpi4py_FOUND           - set when headers were located
#   Mpi4py_INCLUDE_DIR     - directory containing mpi4py/mpi4py.h
#   Mpi4py_VERSION         - mpi4py package version string
#   Mpi4py_MPI_LIBRARY_DIR - directory of the MPI library mpi4py was built
#                            against (used to verify ABI compatibility with
#                            find_package(MPI))
#
# This is intentionally Python-driven: mpi4py installs the headers with the
# package, so the only reliable way to find them is to ask Python.

if(NOT Python_EXECUTABLE)
  message(FATAL_ERROR "FindMpi4py.cmake requires Python_EXECUTABLE to be set "
                      "(call find_package(Python ...) first).")
endif()

execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import mpi4py, sys; sys.stdout.write(mpi4py.get_include())"
  OUTPUT_VARIABLE _mpi4py_include
  RESULT_VARIABLE _mpi4py_rc
  ERROR_QUIET)

if(NOT _mpi4py_rc EQUAL 0)
  set(Mpi4py_FOUND FALSE)
  if(Mpi4py_FIND_REQUIRED)
    message(FATAL_ERROR "mpi4py not found in ${Python_EXECUTABLE}; "
                        "install it with `pip install mpi4py` or via conda.")
  endif()
  return()
endif()

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import mpi4py, sys; sys.stdout.write(mpi4py.__version__)"
  OUTPUT_VARIABLE Mpi4py_VERSION
  ERROR_QUIET)

# mpi4py records the MPI implementation it was built against in its config.
# The dict may be empty (e.g. when mpi4py was built with the default mpicc
# in PATH). When it has 'library_dirs', use that to cross-check against MPI.
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import mpi4py, sys
cfg = mpi4py.get_config() or {}
sys.stdout.write(cfg.get('library_dirs', '') or '')"
  OUTPUT_VARIABLE Mpi4py_MPI_LIBRARY_DIR
  ERROR_QUIET)

set(Mpi4py_INCLUDE_DIR "${_mpi4py_include}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Mpi4py
  REQUIRED_VARS Mpi4py_INCLUDE_DIR
  VERSION_VAR Mpi4py_VERSION)

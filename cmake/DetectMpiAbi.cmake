# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# Detect the MPI implementation flavor (OpenMPI, MPICH, Intel MPI, ...) and
# the library version string and bake them into a generated header so we can
# verify at import time that the mpi4py loaded by the user matches the MPI
# library pyGinkgo was linked against.
#
# Inputs:
#   MPI_C_COMPILER, MPI_C_INCLUDE_DIRS, MPI_C_LIBRARIES (from find_package(MPI))
# Outputs (cache vars):
#   PYGINKGO_MPI_IMPL              - e.g. "OpenMPI", "MPICH", "IntelMPI"
#   PYGINKGO_MPI_LIBRARY_VERSION   - full string from MPI_Get_library_version
#
# A header `pyGinkgo_mpi_abi.hpp` is configured from
# `pyGinkgo_mpi_abi.hpp.in` into ${CMAKE_CURRENT_BINARY_DIR}/generated/.

function(pyginkgo_detect_mpi_abi)
  if(DEFINED PYGINKGO_MPI_IMPL AND DEFINED PYGINKGO_MPI_LIBRARY_VERSION)
    return()
  endif()

  set(_probe_src "${CMAKE_CURRENT_BINARY_DIR}/pyginkgo_mpi_probe.c")
  set(_probe_bin "${CMAKE_CURRENT_BINARY_DIR}/pyginkgo_mpi_probe")
  file(
    WRITE ${_probe_src}
    "
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    char ver[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0;
    MPI_Init(&argc, &argv);
    MPI_Get_library_version(ver, &len);
    fputs(ver, stdout);
    MPI_Finalize();
    return 0;
}
")

  try_run(
    _probe_run _probe_compile
    "${CMAKE_CURRENT_BINARY_DIR}/pyginkgo_mpi_probe_dir"
    SOURCES "${_probe_src}"
    LINK_LIBRARIES MPI::MPI_C
    RUN_OUTPUT_VARIABLE _probe_output
    COMPILE_OUTPUT_VARIABLE _probe_log)

  if(NOT _probe_compile)
    message(
      WARNING
        "Could not compile MPI ABI probe (cross-compiling?). Falling back to "
        "MPI_C_VERSION=${MPI_C_VERSION}.\n${_probe_log}")
    set(_probe_output "Unknown MPI ${MPI_C_VERSION}")
  endif()

  set(_impl "Unknown")
  if(_probe_output MATCHES "[Oo]pen MPI" OR _probe_output MATCHES "OpenRTE")
    set(_impl "OpenMPI")
  elseif(_probe_output MATCHES "MPICH")
    set(_impl "MPICH")
  elseif(_probe_output MATCHES "Intel.* MPI")
    set(_impl "IntelMPI")
  endif()

  # Strip newlines / tabs to keep the C string single-line.
  string(REGEX REPLACE "[\r\n\t]+" " " _probe_output "${_probe_output}")
  string(STRIP "${_probe_output}" _probe_output)

  set(PYGINKGO_MPI_IMPL
      "${_impl}"
      CACHE INTERNAL "Detected MPI implementation")
  set(PYGINKGO_MPI_LIBRARY_VERSION
      "${_probe_output}"
      CACHE INTERNAL "MPI_Get_library_version output")
  message(
    STATUS
      "pyGinkgo MPI: implementation='${PYGINKGO_MPI_IMPL}', version='${PYGINKGO_MPI_LIBRARY_VERSION}'"
  )
endfunction()

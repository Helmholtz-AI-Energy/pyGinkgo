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

# Distributed / MPI bindings. The test file skips itself gracefully when the
# build has no MPI support or when mpi4py is missing, so it is always registered
# (the wrapper-layer tests run even in a CPU-only, MPI-off build).
# PYGINKGO_GINKGO_HAS_MPI is set by the top-level CMakeLists before this
# directory is added.
pyginkgo_python_test(distributed)

# In an MPI-enabled build additionally run the distributed suite under an MPI
# launcher with multiple ranks, so the off-process communication paths are
# actually exercised (a single-rank pytest run keeps everything process-local).
if(PYGINKGO_GINKGO_HAS_MPI AND MPIEXEC_EXECUTABLE)
  # Two ranks keeps the run cheap and avoids OpenMPI slot oversubscription on
  # the small (2-core) CI runners while still crossing at least one rank
  # boundary.
  add_test(
    NAME pyginkgo_distributed_mpi_test
    # FindMPI layout: mpiexec <NUMPROC_FLAG> <N> <PREFLAGS> <target> <POSTFLAGS>
    # <args>. POSTFLAGS must sit right after the target (python), not after the
    # pytest args, or non-empty postflags would be handed to pytest.
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      ${Python_EXECUTABLE} ${MPIEXEC_POSTFLAGS} -m pytest -vv -rs -p
      no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed.py
    WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endif()

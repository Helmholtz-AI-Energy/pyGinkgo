# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""mpi4py.Comm <-> gko::mpi::communicator bridge tests."""

from __future__ import annotations

from pyGinkgo import pyGinkgoBindings as _pgb


def test_mpi_module_exposed():
    assert hasattr(_pgb, "mpi"), "C++ side did not expose pyGinkgoBindings.mpi"
    assert hasattr(_pgb.mpi, "verify_abi")


def test_build_metadata_strings():
    impl = getattr(_pgb.mpi, "BUILD_MPI_IMPL", "")
    ver = getattr(_pgb.mpi, "BUILD_MPI_LIBRARY_VERSION", "")
    assert isinstance(impl, str) and impl
    assert isinstance(ver, str) and ver


def test_runtime_abi_check(comm):
    """Round-trip an MPI_Comm_size on the user-supplied comm."""
    _pgb.mpi.verify_abi(comm)


def test_world_size_matches(comm):
    c = _pgb.mpi.Communicator(comm)
    assert c.size() == comm.size


def test_world_rank_matches(comm):
    c = _pgb.mpi.Communicator(comm)
    assert c.rank() == comm.rank


def test_split_communicator(comm):
    """Splitting and using a sub-communicator must be safe."""
    color = comm.rank % 2
    sub = comm.Split(color=color, key=comm.rank)
    try:
        c = _pgb.mpi.Communicator(sub)
        assert c.size() == sub.size
        assert c.rank() == sub.rank
    finally:
        sub.Free()

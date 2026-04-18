# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Shared fixtures for distributed (MPI) C++-binding tests.

These tests must be run under ``mpirun -n N pytest tests/cpp_bindings/distributed``.
Each test process runs all the tests independently; collective calls are
made by all ranks together. We don't try to be clever about per-rank
test selection — the harness keeps the model simple.
"""

from __future__ import annotations

import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

pyGinkgo = pytest.importorskip("pyGinkgo")
try:
    from pyGinkgo import distributed as gkd  # noqa: F401
except ImportError as e:  # pragma: no cover
    pytest.skip(f"pyGinkgo built without MPI: {e}", allow_module_level=True)

from pyGinkgo.pyGinkgoBindings import OmpExecutor, ReferenceExecutor  # noqa: E402


@pytest.fixture(scope="session")
def comm():
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def rank(comm):
    return comm.rank


@pytest.fixture(scope="session")
def nprocs(comm):
    return comm.size


@pytest.fixture(scope="session")
def exec():
    """A single shared executor used across the tests."""
    return ReferenceExecutor()


@pytest.fixture(scope="session", autouse=True)
def _require_mpi(comm):
    """Skip the whole module when the runtime isn't actually MPI-launched."""
    if comm.size < 2:
        pytest.skip(
            "Distributed tests require mpirun -n >= 2 "
            "(currently running with size=1).",
            allow_module_level=False,
        )


@pytest.fixture(autouse=True)
def _barrier_around_test(comm):
    """Synchronise ranks at start/end of each test so failures don't deadlock."""
    comm.Barrier()
    yield
    comm.Barrier()

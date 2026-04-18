# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Facade-level tests for ``pyGinkgo.distributed``.

These exercise the Python wrappers (``Partition``, ``DistributedVector``,
``DistributedMatrix``) rather than the C++ bindings directly. Run via
``mpirun -n 2 pytest tests/pyGinkgo/distributed/``.
"""

from __future__ import annotations

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

pyGinkgo = pytest.importorskip("pyGinkgo")
try:
    from pyGinkgo import distributed as gkd  # noqa: F401
except ImportError as e:  # pragma: no cover
    pytest.skip(f"pyGinkgo built without MPI: {e}", allow_module_level=True)

from pyGinkgo.pyGinkgoBindings import ReferenceExecutor  # noqa: E402


@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD


@pytest.fixture(scope="module")
def exec():
    return ReferenceExecutor()


@pytest.fixture(autouse=True)
def _require_mpi(comm):
    if comm.size < 2:
        pytest.skip("requires mpirun -n >= 2")


def test_partition_facade(exec, comm):
    p = gkd.Partition.uniform(exec, comm.size, 12)
    assert p.size == 12
    assert p.num_parts == comm.size
    assert isinstance(p.raw, gkd.partition._resolve(np.int32, np.int64))


def test_vector_facade_roundtrip(exec, comm):
    n_global = 4 * comm.size
    local = np.full(4, float(comm.rank + 1), dtype=np.float64)
    v = gkd.DistributedVector.from_local_array(
        exec, comm, (n_global, 1), local
    )
    assert v.shape == (n_global, 1)
    assert v.local_shape == (4, 1)
    out = np.asarray(v.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_array_equal(out, local)


def test_matrix_facade_apply(exec, comm):
    from pyGinkgo import pyGinkgoBindings as _pgb

    n_local = 4
    n_global = n_local * comm.size

    Csr = _pgb.matrix.Csr_double_int32
    row_ptrs = np.arange(n_local + 1, dtype=np.int32)
    col_idxs = np.arange(n_local, dtype=np.int32)
    values = 2.0 * np.ones(n_local, dtype=np.float64)
    diag2 = Csr(exec, (n_local, n_local), values, col_idxs, row_ptrs)

    M = gkd.DistributedMatrix.from_local_linop(
        exec, comm, (n_global, n_global), diag2
    )

    x_local = np.full(n_local, 3.0, dtype=np.float64)
    x = gkd.DistributedVector.from_local_array(
        exec, comm, (n_global, 1), x_local
    )
    y = gkd.DistributedVector.from_local_array(
        exec, comm, (n_global, 1), np.zeros(n_local, dtype=np.float64)
    )
    M.raw.apply(x.raw, y.raw)
    out = np.asarray(y.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_array_equal(out, 6.0 * np.ones(n_local))

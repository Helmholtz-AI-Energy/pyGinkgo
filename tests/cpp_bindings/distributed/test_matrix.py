# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Tests for ``pyGinkgo.distributed.DistributedMatrix``.

We verify the local-only constructor (``from_local_linop``) and the
matrix-vector apply against a hand-rolled identity, using a local CSR
that's just the identity on each rank's slice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyGinkgo.distributed import DistributedMatrix, DistributedVector
from pyGinkgo import pyGinkgoBindings as _pgb


def _local_slice(rank, nprocs, n):
    base = n // nprocs
    rem = n % nprocs
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return start, size


def _identity_csr(exec, n):
    Csr = _pgb.matrix.Csr_double_int32
    row_ptrs = np.arange(n + 1, dtype=np.int32)
    col_idxs = np.arange(n, dtype=np.int32)
    values = np.ones(n, dtype=np.float64)
    return Csr(exec, (n, n), values, col_idxs, row_ptrs)


@pytest.fixture
def n_global(nprocs):
    return 6 * nprocs


def test_local_only_matrix_shape(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    local_csr = _identity_csr(exec, local_n)
    M = DistributedMatrix.from_local_linop(
        exec, comm, (n_global, n_global), local_csr
    )
    assert M.shape == (n_global, n_global)


def test_apply_identity(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    local_csr = _identity_csr(exec, local_n)
    M = DistributedMatrix.from_local_linop(
        exec, comm, (n_global, n_global), local_csr
    )

    x_local = np.full(local_n, float(rank + 1), dtype=np.float64)
    x = DistributedVector.from_local_array(exec, comm, (n_global, 1), x_local)
    y = DistributedVector.from_local_array(
        exec, comm, (n_global, 1),
        np.zeros(local_n, dtype=np.float64),
    )

    M.raw.apply(x.raw, y.raw)

    y_local = np.asarray(y.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_array_equal(y_local, x_local)

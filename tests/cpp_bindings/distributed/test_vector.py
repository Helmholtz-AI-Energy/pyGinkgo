# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Tests for ``pyGinkgo.distributed.DistributedVector``."""

from __future__ import annotations

import numpy as np
import pytest

from pyGinkgo.distributed import DistributedVector
from pyGinkgo import pyGinkgoBindings as _pgb


def _local_slice(rank, nprocs, n):
    base = n // nprocs
    rem = n % nprocs
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return start, size


@pytest.fixture
def n_global(nprocs):
    return 4 * nprocs  # ensures ≥ 4 elements per rank


def _scalar_dense(exec, value=0.0):
    Dense = _pgb.matrix.dense_double
    d = Dense(exec, (1, 1))
    d.fill(value)
    return d


def _dense_to_float(d):
    return float(np.asarray(d.copy_to_host()).item())


def test_from_local_array_shape(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    local = np.full(local_n, float(rank + 1), dtype=np.float64)
    v = DistributedVector.from_local_array(
        exec, comm, (n_global, 1), local
    )
    assert v.shape == (n_global, 1)
    assert v.local_shape == (local_n, 1)


def test_fill_then_norm(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    local = np.zeros(local_n, dtype=np.float64)
    v = DistributedVector.from_local_array(exec, comm, (n_global, 1), local)
    v.fill(2.0)
    result = _scalar_dense(exec)
    v.raw.compute_norm2(result)
    norm = _dense_to_float(result)
    assert norm == pytest.approx(np.sqrt(4.0 * n_global), rel=1e-12)


def test_dot_cross_rank(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    a = np.full(local_n, 1.0, dtype=np.float64)
    b = np.full(local_n, float(rank + 1), dtype=np.float64)
    va = DistributedVector.from_local_array(exec, comm, (n_global, 1), a)
    vb = DistributedVector.from_local_array(exec, comm, (n_global, 1), b)
    result = _scalar_dense(exec)
    va.raw.compute_dot(vb.raw, result)
    got = _dense_to_float(result)

    expected = 0.0
    for r in range(nprocs):
        _, ln = _local_slice(r, nprocs, n_global)
        expected += ln * (r + 1)
    assert got == pytest.approx(expected, rel=1e-12)


def test_get_local_vector(exec, comm, rank, nprocs, n_global):
    _, local_n = _local_slice(rank, nprocs, n_global)
    local = np.arange(local_n, dtype=np.float64) + 100.0 * rank
    v = DistributedVector.from_local_array(exec, comm, (n_global, 1), local)
    lv = v.get_local_vector()
    arr = np.asarray(lv.copy_to_host()).reshape(-1)
    np.testing.assert_array_equal(arr, local)

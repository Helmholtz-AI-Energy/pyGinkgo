# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the re-added CG / BiCGSTAB / Jacobi bindings used in
the distributed code paths and for the new zero-copy / gather helpers
on ``distributed::Vector``.

The block-diagonal construction mirrors :mod:`test_solver` so the
distributed solve is mathematically identical to ``nprocs`` independent
serial solves on the same SPD block.
"""

from __future__ import annotations

import numpy as np

from pyGinkgo.distributed import DistributedMatrix, DistributedVector
from pyGinkgo import pyGinkgoBindings as _pgb


BLOCK = 4


def _spd_block(n):
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def _spd_block_csr(exec, n):
    A = _spd_block(n)
    rows, cols = np.where(A != 0)
    order = np.argsort(rows * n + cols)
    rows, cols = rows[order], cols[order]
    vals = A[rows, cols]
    row_ptrs = np.zeros(n + 1, dtype=np.int32)
    np.add.at(row_ptrs, rows + 1, 1)
    np.cumsum(row_ptrs, out=row_ptrs)
    return _pgb.matrix.Csr_double_int32(
        exec, (n, n),
        vals.astype(np.float64),
        cols.astype(np.int32),
        row_ptrs,
    )


def _make_block_system(exec, comm, nprocs):
    n_global = BLOCK * nprocs
    local_csr = _spd_block_csr(exec, BLOCK)
    A = DistributedMatrix.from_local_linop(
        exec, comm, (n_global, n_global), local_csr
    )
    b_local = (_spd_block(BLOCK) @ np.ones(BLOCK)).astype(np.float64)
    b = DistributedVector.from_local_array(exec, comm, (n_global, 1), b_local)
    x = DistributedVector.from_local_array(
        exec, comm, (n_global, 1), np.zeros(BLOCK, dtype=np.float64)
    )
    return A, b, x


def test_distributed_cg(exec, comm, rank, nprocs):
    A, b, x = _make_block_system(exec, comm, nprocs)
    cg = _pgb.solver.cg_double(
        exec, A.raw,
        max_iters=100,
        reduction_factor=1e-12, relative_stop_mode=True,
    )
    cg.apply(b.raw, x.raw)
    x_local = np.asarray(x.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_allclose(x_local, np.ones(BLOCK), atol=1e-9, rtol=0)


def test_distributed_bicgstab(exec, comm, rank, nprocs):
    A, b, x = _make_block_system(exec, comm, nprocs)
    bicg = _pgb.solver.bicgstab_double(
        exec, A.raw,
        max_iters=200,
        reduction_factor=1e-12, relative_stop_mode=True,
    )
    bicg.apply(b.raw, x.raw)
    x_local = np.asarray(x.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_allclose(x_local, np.ones(BLOCK), atol=1e-9, rtol=0)


def test_vector_from_local_array_view_matches_copy(exec, comm, rank, nprocs):
    """The view variant must yield the same values as the copy variant."""
    n_global = BLOCK * nprocs
    a = np.arange(BLOCK, dtype=np.float64) + 10.0 * rank
    v_copy = DistributedVector.from_local_array(
        exec, comm, (n_global, 1), a
    )
    v_view = _pgb.distributed.Vector_double.from_local_array_view(
        exec, comm, (n_global, 1), a
    )
    a_copy = np.asarray(v_copy.get_local_vector().copy_to_host()).reshape(-1)
    a_view = np.asarray(v_view.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_array_equal(a_copy, a_view)


def test_vector_gather_on_root(exec, comm, rank, nprocs):
    """gather_on_root returns the concatenated global vector on root only."""
    n_global = BLOCK * nprocs
    a = (np.arange(BLOCK, dtype=np.float64) + 100.0 * rank)
    v = DistributedVector.from_local_array(
        exec, comm, (n_global, 1), a
    )
    out = v.raw.gather_on_root(0)
    if rank == 0:
        assert out is not None
        assert out.shape == (n_global,) or out.shape == (n_global, 1)
        flat = np.asarray(out).reshape(-1)
        expected = np.concatenate([
            np.arange(BLOCK, dtype=np.float64) + 100.0 * r
            for r in range(nprocs)
        ])
        np.testing.assert_array_equal(flat, expected)
    else:
        assert out is None

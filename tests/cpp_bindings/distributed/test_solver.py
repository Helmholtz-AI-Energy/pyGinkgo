# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Distributed solve via the existing GMRES binding.

The serial ``pyGinkgo.solver.gmres_<T>`` factory accepts any ``LinOp``
including a ``distributed::Matrix`` (Ginkgo dispatches polymorphically),
so no separate distributed-solver binding is needed. We exercise that
contract here on a 1D Poisson-like SPD block-diagonal matrix split one
block per rank: the off-diagonal block is empty so the solve is
mathematically identical to ``nprocs`` independent serial solves.
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


def test_distributed_gmres_block_diagonal(exec, comm, rank, nprocs):
    n_global = BLOCK * nprocs
    local_csr = _spd_block_csr(exec, BLOCK)
    A = DistributedMatrix.from_local_linop(
        exec, comm, (n_global, n_global), local_csr
    )

    # x_true = ones; b = A_block @ ones (the same on every rank)
    b_local = (_spd_block(BLOCK) @ np.ones(BLOCK)).astype(np.float64)
    b = DistributedVector.from_local_array(exec, comm, (n_global, 1), b_local)
    x = DistributedVector.from_local_array(
        exec, comm, (n_global, 1), np.zeros(BLOCK, dtype=np.float64)
    )

    gmres = _pgb.solver.gmres_double(
        exec, A.raw,
        max_iters=100, krylov_dim=20,
        reduction_factor=1e-12, relative_stop_mode=True,
    )
    gmres.apply(b.raw, x.raw)

    x_local = np.asarray(x.get_local_vector().copy_to_host()).reshape(-1)
    np.testing.assert_allclose(x_local, np.ones(BLOCK), atol=1e-9, rtol=0)

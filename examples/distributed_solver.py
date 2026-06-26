# SPDX-FileCopyrightText: 2026 Imad Kissami
#
# SPDX-License-Identifier: MIT

"""Minimal distributed pyGinkgo example.

Solves the 1D Poisson problem ``A x = b`` where ``A`` is the N x N tridiagonal
Laplacian (diag = 2, off-diag = -1) and ``b = 1``, using Ginkgo's MPI-distributed
matrix/vector and a CG solver.

Each MPI rank owns a contiguous block of rows and assembles ONLY its own rows
(like PETSc): Ginkgo handles the off-process communication during the solve.

Run with:
    mpirun -n 4 python examples/distributed_solver.py

Requires pyGinkgo built with MPI support (GINKGO_BUILD_MPI=ON).
"""

import json

import numpy as np
from mpi4py import MPI

import pyGinkgo as pg
from pyGinkgo import pyGinkgoBindings as pGB

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if not pg.distributed.available:
    raise RuntimeError(
        "pyGinkgo was built without MPI support (GINKGO_BUILD_MPI=ON). "
        "The distributed bindings are unavailable.")

# ---------------------------------------------------------------- parameters
N = 16                       # global system size
executor = pg.device("cpu")  # "cpu", "omp", or "cuda" / "cuda:0"
value = "double"             # matches np.float64

# ----------------------------------------------------- row distribution (1D)
# owners[g] = rank that owns global row g. Split the N rows into contiguous
# blocks, one per rank.
counts = [N // size + (1 if r < N % size else 0) for r in range(size)]
offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
start, end = offsets[rank], offsets[rank + 1]
owned_rows = np.arange(start, end, dtype=np.int32)

owners = np.empty(N, dtype=np.int32)
for r in range(size):
    owners[offsets[r]:offsets[r + 1]] = r

partition = pg.distributed.build_partition(executor, owners, size)

# --------------------------------------------- local assembly (owned rows only)
# 1D Laplacian stencil: row i has 2 on the diagonal and -1 to its neighbours.
# Indices are 0-based GLOBAL indices.
rows, cols, vals = [], [], []
for i in range(start, end):
    rows.append(i); cols.append(i); vals.append(2.0)
    if i > 0:
        rows.append(i); cols.append(i - 1); vals.append(-1.0)
    if i < N - 1:
        rows.append(i); cols.append(i + 1); vals.append(-1.0)

rows = np.asarray(rows, dtype=np.int32)
cols = np.asarray(cols, dtype=np.int32)
vals = np.asarray(vals, dtype=np.float64)

A = pg.distributed.matrix(executor, comm, partition, rows, cols, vals, N,
                          dtype=value)

# ------------------------------------------------------ distributed RHS / sol
b_local = np.ones(owned_rows.size, dtype=np.float64)        # b = 1
x_local = np.zeros(owned_rows.size, dtype=np.float64)       # initial guess 0

b = pg.distributed.vector(executor, comm, partition, owned_rows, b_local, N,
                          dtype=value)
x = pg.distributed.vector(executor, comm, partition, owned_rows, x_local, N,
                          dtype=value)

# ------------------------------------------------------------------- solve
solver_args = json.dumps({
    "type": "solver::Cg",
    "criteria": [
        {"type": "Iteration", "max_iters": 1000},
        {"type": "ResidualNorm", "reduction_factor": 1e-10},
    ],
})

logger = pGB.solver.config_solve_double(executor, A, b, x, solver_args)

# ----------------------------------------------------------------- results
# Each rank reads its own slice of the solution (processor-local part).
sol_local = pg.distributed.vector_local(x, dtype=value)

if rank == 0:
    print(f"[pyGinkgo distributed] N={N} ranks={size} "
          f"converged={logger.has_converged()} "
          f"iters={logger.get_num_iterations()}")

# Gather the full solution on rank 0 just to print it (only for this demo).
full = comm.gather(np.asarray(sol_local).ravel(), root=0)
if rank == 0:
    x_full = np.concatenate(full)
    print("solution:", np.array2string(x_full, precision=4, suppress_small=True))

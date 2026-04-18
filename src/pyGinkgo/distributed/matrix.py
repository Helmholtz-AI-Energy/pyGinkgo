# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Distributed Matrix wrapper."""

from __future__ import annotations

import numpy as np

from .. import pyGinkgoBindings as _pgb
from . import _ensure_mpi_abi
from .partition import Partition


_DTYPE_NAMES = {
    "float64": "double",
    "float32": "float",
    "int32": "int32",
    "int64": "int64",
}


def _gko_name(dtype):
    name = np.dtype(dtype).name
    return _DTYPE_NAMES.get(name, name)


def _resolve(value_dtype, local_index_dtype, global_index_dtype):
    vt = _gko_name(value_dtype)
    li = _gko_name(local_index_dtype)
    gi = _gko_name(global_index_dtype)
    name = f"Matrix_{vt}_{li}_{gi}"
    cls = getattr(_pgb.distributed, name, None)
    if cls is None:
        raise TypeError(
            f"No distributed Matrix bound for ({vt}, {li}, {gi}). "
            "Supported: (double|float, int32|int64, int64)."
        )
    return cls


class DistributedMatrix:
    """Distributed matrix with local-diag and non-local-offdiag blocks."""

    def __init__(self, c_matrix):
        self._m = c_matrix

    @classmethod
    def empty(cls, exec, comm, value_dtype=np.float64,
              local_index_dtype=np.int32, global_index_dtype=np.int64):
        _ensure_mpi_abi(comm)
        c = _resolve(value_dtype, local_index_dtype, global_index_dtype)
        return cls(c.create_empty(exec, comm))

    @classmethod
    def from_local_linop(cls, exec, comm, global_size, local_linop,
                         value_dtype=np.float64,
                         local_index_dtype=np.int32,
                         global_index_dtype=np.int64):
        """Local-only matrix (no off-process columns)."""
        _ensure_mpi_abi(comm)
        c = _resolve(value_dtype, local_index_dtype, global_index_dtype)
        return cls(c.create_from_local_linop(
            exec, comm, tuple(global_size), local_linop))

    @classmethod
    def from_local_and_non_local(cls, exec, comm, partition,
                                 recv_connections,
                                 local_linop, non_local_linop,
                                 value_dtype=np.float64,
                                 local_index_dtype=np.int32,
                                 global_index_dtype=np.int64):
        """Build a distributed matrix from local-diagonal and non-local
        (off-diagonal) LinOps.

        Parameters
        ----------
        partition : Partition
            Row partition; the same partition is used for columns.
        recv_connections : 1D array of global column indices
            Global column ids accessed by the non-local block. Duplicates
            are filtered.
        local_linop, non_local_linop : LinOp
            Diagonal block (local rows × local cols) and off-diagonal block
            (local rows × non-local cols, with column indices already
            ordered to match the induced index_map).
        """
        _ensure_mpi_abi(comm)
        c = _resolve(value_dtype, local_index_dtype, global_index_dtype)
        if isinstance(partition, Partition):
            partition = partition.raw
        return cls(c.create_from_local_and_non_local(
            exec, comm, partition, recv_connections,
            local_linop, non_local_linop))

    @property
    def raw(self):
        return self._m

    @property
    def shape(self):
        return self._m.shape

    def get_local_matrix(self):
        return self._m.get_local_matrix()

    def get_non_local_matrix(self):
        return self._m.get_non_local_matrix()


__all__ = ["DistributedMatrix"]

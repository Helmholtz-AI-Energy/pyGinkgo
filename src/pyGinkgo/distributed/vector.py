# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Distributed Vector wrapper."""

from __future__ import annotations

import numpy as np

from .. import pyGinkgoBindings as _pgb
from . import _ensure_mpi_abi


_DTYPE_NAMES = {
    "float64": "double",
    "float32": "float",
    "int32": "int32",
    "int64": "int64",
}


def _gko_name(dtype):
    name = np.dtype(dtype).name
    return _DTYPE_NAMES.get(name, name)


def _resolve(value_dtype):
    name = f"Vector_{_gko_name(value_dtype)}"
    cls = getattr(_pgb.distributed, name, None)
    if cls is None:
        raise TypeError(
            f"No distributed Vector bound for dtype {value_dtype}. "
            "Supported: float32, float64."
        )
    return cls


class DistributedVector:
    """Owned-local-slice distributed vector with cupy/numpy interop."""

    def __init__(self, c_vector):
        self._v = c_vector

    @classmethod
    def empty(cls, exec, comm, global_size, local_size, value_dtype=np.float64):
        _ensure_mpi_abi(comm)
        c = _resolve(value_dtype)
        return cls(c.create(exec, comm, tuple(global_size), tuple(local_size)))

    @classmethod
    def from_local_array(cls, exec, comm, global_size, local_array,
                         value_dtype=None):
        _ensure_mpi_abi(comm)
        if value_dtype is None:
            value_dtype = local_array.dtype
        c = _resolve(value_dtype)
        return cls(c.from_local_array(
            exec, comm, tuple(global_size), local_array))

    @classmethod
    def from_local_array_deduce_size(cls, exec, comm, local_array,
                                     value_dtype=None):
        _ensure_mpi_abi(comm)
        if value_dtype is None:
            value_dtype = local_array.dtype
        c = _resolve(value_dtype)
        return cls(c.from_local_array_deduce_size(exec, comm, local_array))

    @property
    def raw(self):
        return self._v

    @property
    def shape(self):
        return self._v.shape

    @property
    def local_shape(self):
        return self._v.local_shape

    def fill(self, value):
        self._v.fill(value)

    def get_local_vector(self):
        return self._v.get_local_vector()


__all__ = ["DistributedVector"]

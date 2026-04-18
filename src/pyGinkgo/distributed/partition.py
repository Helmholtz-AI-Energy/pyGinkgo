# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Distributed Partition wrapper."""

from __future__ import annotations

import numpy as np

from .. import pyGinkgoBindings as _pgb


def _resolve(local_index_dtype, global_index_dtype):
    li = np.dtype(local_index_dtype).name
    gi = np.dtype(global_index_dtype).name
    name = f"Partition_{li}_{gi}"
    cls = getattr(_pgb.distributed, name, None)
    if cls is None:
        raise TypeError(
            f"No distributed Partition bound for ({li}, {gi}). "
            "Supported: (int32,int64), (int64,int64), (int32,int32)."
        )
    return cls


class Partition:
    """Thin Python facade over the C++ ``Partition_<L>_<G>`` classes."""

    def __init__(self, c_partition):
        self._p = c_partition

    @classmethod
    def uniform(cls, exec, num_parts, global_size,
                local_index_dtype=np.int32, global_index_dtype=np.int64):
        c = _resolve(local_index_dtype, global_index_dtype)
        return cls(c.build_from_global_size_uniform(exec, num_parts,
                                                    global_size))

    @classmethod
    def from_contiguous(cls, exec, ranges,
                        local_index_dtype=np.int32,
                        global_index_dtype=np.int64):
        c = _resolve(local_index_dtype, global_index_dtype)
        return cls(c.build_from_contiguous(exec, ranges))

    @classmethod
    def from_mapping(cls, exec, mapping, num_parts,
                     local_index_dtype=np.int32,
                     global_index_dtype=np.int64):
        c = _resolve(local_index_dtype, global_index_dtype)
        return cls(c.build_from_mapping(exec, mapping, num_parts))

    @property
    def raw(self):
        return self._p

    @property
    def size(self):
        return self._p.get_size()

    @property
    def num_parts(self):
        return self._p.get_num_parts()

    def part_size(self, part_id):
        return self._p.get_part_size(part_id)


__all__ = ["Partition"]

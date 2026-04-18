# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Tests for ``pyGinkgo.distributed.Partition``."""

from __future__ import annotations

import numpy as np
import pytest

from pyGinkgo.distributed import Partition


GLOBAL = 24


def test_uniform_partition(exec, nprocs):
    p = Partition.uniform(exec, nprocs, GLOBAL)
    assert p.size == GLOBAL
    assert p.num_parts == nprocs
    sizes = [p.part_size(i) for i in range(nprocs)]
    assert sum(sizes) == GLOBAL
    # uniform split: difference between min and max ≤ 1
    assert max(sizes) - min(sizes) <= 1


def test_from_contiguous(exec, nprocs):
    # Build the same uniform partition via explicit contiguous ranges
    base = GLOBAL // nprocs
    rem = GLOBAL % nprocs
    ranges = [0]
    for i in range(nprocs):
        ranges.append(ranges[-1] + base + (1 if i < rem else 0))
    ranges = np.asarray(ranges, dtype=np.int64)
    p = Partition.from_contiguous(exec, ranges)
    assert p.size == GLOBAL
    assert p.num_parts == nprocs


def test_from_mapping(exec, nprocs):
    # Round-robin owner assignment
    mapping = np.arange(GLOBAL, dtype=np.int32) % nprocs
    p = Partition.from_mapping(exec, mapping, nprocs)
    assert p.size == GLOBAL
    assert p.num_parts == nprocs


def test_unsupported_dtype_raises(exec, nprocs):
    with pytest.raises(TypeError):
        Partition.uniform(exec, nprocs, GLOBAL, local_index_dtype=np.float32)

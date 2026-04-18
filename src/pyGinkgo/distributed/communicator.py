# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Communicator helpers for the distributed bindings."""

from __future__ import annotations

from .. import pyGinkgoBindings as _pgb
from . import _ensure_mpi_abi


def map_rank_to_device_id(comm, num_devices: int) -> int:
    """Return a device id for this rank using Ginkgo's intra-node mapping."""
    _ensure_mpi_abi(comm)
    return _pgb.mpi.map_rank_to_device_id(comm, num_devices)


def is_gpu_aware() -> bool:
    """Return True if the linked MPI advertises CUDA-aware support."""
    return _pgb.mpi.is_gpu_aware()


__all__ = ["map_rank_to_device_id", "is_gpu_aware"]

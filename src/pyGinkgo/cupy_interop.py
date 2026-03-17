# SPDX-FileCopyrightText: 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

"""CuPy interoperability for pyGinkgo.

This module provides utilities for zero-copy data exchange between
CuPy arrays and Ginkgo objects on CUDA devices, avoiding unnecessary
device-host-device memory transfers.

The primary mechanism uses ``__cuda_array_interface__`` (CAI v3) for
direct GPU memory sharing between CuPy and Ginkgo's CUDA executor
objects.

When CAI is not available (e.g., CPU arrays or non-CUDA builds),
a fallback path through host memory is used.

Design rationale (``__cuda_array_interface__`` vs DLPack):
    We chose ``__cuda_array_interface__`` because:

    1. It is CuPy's native protocol for zero-copy CUDA interop.
    2. It is simpler to implement and maintain at the C++ binding level.
    3. It directly addresses the CUDA use case that prompted this work.

    DLPack is a candidate for future work to provide a universal protocol
    that also covers HIP/ROCm and SYCL/DPC++ backends.

Output (Ginkgo → CuPy):
    Ginkgo arrays and dense matrices on a CUDA executor expose
    ``__cuda_array_interface__``, so ``cupy.asarray(gko_obj)`` creates
    a zero-copy view directly into Ginkgo-managed GPU memory.

Input (CuPy → Ginkgo):
    The ``from_cupy`` helpers read the CuPy array's
    ``__cuda_array_interface__``, extract the device pointer, and call
    the C++ ``from_device_ptr`` factory to perform a fast
    device-to-device copy into Ginkgo-owned memory.
    When the C++ CUDA path is unavailable the data is copied through
    host memory as a fallback.
"""

import numpy as np
from typing import Optional

try:
    import cupy

    cupy_avail = True
except ImportError:
    cupy_avail = False


def is_cupy_array(obj) -> bool:
    """Return *True* if *obj* is a CuPy ndarray."""
    if not cupy_avail:
        return False
    return isinstance(obj, cupy.ndarray)


# ------------------------------------------------------------------
# CuPy → Ginkgo
# ------------------------------------------------------------------

def _cupy_dtype_to_gko_dtype(cupy_arr) -> str:
    """Map a CuPy array's dtype to a Ginkgo dtype string."""
    dtype_map = {
        np.dtype("float16"): "half",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
        np.dtype("int32"): "int32",
        np.dtype("int64"): "int64",
    }
    gko_dtype = dtype_map.get(cupy_arr.dtype)
    if gko_dtype is None:
        raise TypeError(
            f"Unsupported CuPy dtype for Ginkgo conversion: {cupy_arr.dtype}"
        )
    return gko_dtype


def from_cupy_to_gko_array(cupy_arr, executor, dtype: Optional[str] = None):
    """Create a Ginkgo 1-D array from a CuPy array.

    Parameters
    ----------
    cupy_arr : cupy.ndarray
        Source array (must be 1-D and C-contiguous).
    executor : Ginkgo Executor
        The target executor (should be a CUDA executor for zero-copy).
    dtype : str, optional
        Ginkgo dtype string (e.g. ``"float"``).  Inferred from
        *cupy_arr* when not given.

    Returns
    -------
    gko array object
    """
    from pyGinkgo import pyGinkgoBindings as pGB

    if not is_cupy_array(cupy_arr):
        raise TypeError("Expected a CuPy ndarray")

    if cupy_arr.ndim != 1:
        raise ValueError("Only 1-D CuPy arrays can be converted to gko arrays")

    if not cupy_arr.flags["C_CONTIGUOUS"]:
        cupy_arr = cupy.ascontiguousarray(cupy_arr)

    if dtype is None:
        dtype = _cupy_dtype_to_gko_dtype(cupy_arr)

    array_cls = getattr(pGB.base, "array_" + dtype)

    # Fast path: use from_device_ptr (device-to-device copy, no host round-trip)
    if hasattr(array_cls, "from_device_ptr"):
        cai = cupy_arr.__cuda_array_interface__
        ptr = cai["data"][0]
        size = cai["shape"][0]
        return array_cls.from_device_ptr(executor, ptr, size)

    # Fallback: copy through host memory
    np_array = cupy.asnumpy(cupy_arr)
    return array_cls(executor, np_array)


def from_cupy_to_gko_dense(cupy_arr, executor, dtype: Optional[str] = None):
    """Create a Ginkgo dense matrix from a CuPy array.

    Parameters
    ----------
    cupy_arr : cupy.ndarray
        Source array (1-D or 2-D, C-contiguous).
    executor : Ginkgo Executor
        Target executor (should be a CUDA executor for zero-copy).
    dtype : str, optional
        Ginkgo dtype string.  Inferred from *cupy_arr* when not given.

    Returns
    -------
    gko dense matrix object
    """
    from pyGinkgo import pyGinkgoBindings as pGB

    if not is_cupy_array(cupy_arr):
        raise TypeError("Expected a CuPy ndarray")

    if cupy_arr.ndim not in (1, 2):
        raise ValueError(
            "Only 1-D or 2-D CuPy arrays can be converted to gko dense matrices"
        )

    if not cupy_arr.flags["C_CONTIGUOUS"]:
        cupy_arr = cupy.ascontiguousarray(cupy_arr)

    if dtype is None:
        dtype = _cupy_dtype_to_gko_dtype(cupy_arr)

    dense_cls = getattr(pGB.matrix, "dense_" + dtype)

    # Fast path: use from_device_ptr (device-to-device copy, no host round-trip)
    if hasattr(dense_cls, "from_device_ptr"):
        cai = cupy_arr.__cuda_array_interface__
        ptr = cai["data"][0]
        shape = cai["shape"]
        rows = shape[0]
        cols = shape[1] if len(shape) > 1 else 1
        stride = cols  # C-contiguous
        return dense_cls.from_device_ptr(executor, ptr, rows, cols, stride)

    # Fallback: copy through host memory
    np_array = cupy.asnumpy(cupy_arr)
    return dense_cls(executor, np_array)


# ------------------------------------------------------------------
# Ginkgo → CuPy
# ------------------------------------------------------------------

def gko_to_cupy(gko_obj):
    """Convert a Ginkgo array or dense matrix to a CuPy array.

    If the Ginkgo object is on a CUDA executor and exposes
    ``__cuda_array_interface__``, CuPy creates a **zero-copy view**
    directly into Ginkgo-managed device memory.

    Otherwise the data is copied through host memory.

    Parameters
    ----------
    gko_obj : gko array or dense matrix
        The source Ginkgo object.

    Returns
    -------
    cupy.ndarray
    """
    if not cupy_avail:
        raise ImportError(
            "CuPy is required for gko_to_cupy(). "
            "Install it with: pip install cupy-cuda12x"
        )

    # Zero-copy path via __cuda_array_interface__
    if hasattr(gko_obj, "__cuda_array_interface__"):
        return cupy.asarray(gko_obj)

    # Fallback: copy through host
    if hasattr(gko_obj, "copy_to_host"):
        host_obj = gko_obj.copy_to_host()
        return cupy.asarray(np.array(host_obj))

    return cupy.asarray(np.array(gko_obj))

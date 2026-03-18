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

Dense / 1-D arrays (no imports from this module required):
    CuPy dense arrays and Ginkgo dense / array objects interoperate
    through the standard constructor and ``cupy.asarray`` paths,
    mirroring the pattern used by PyTorch::

        # CuPy → Ginkgo (zero-copy view via __cuda_array_interface__)
        gko_arr = array_cls(cuda_executor, cp_arr)

        # Ginkgo → CuPy (zero-copy view via __cuda_array_interface__)
        cp_arr = cupy.asarray(gko_obj)

    Both directions are zero-copy when the executor is a CUDA executor
    and the array dtype matches.  The source object's lifetime is tied
    to the Ginkgo object via ``py::keep_alive``, so the device memory
    stays valid.  On non-CUDA executors or when dtype conversion is
    needed, data is copied through host memory automatically.

Sparse matrices (import from this module):
    CuPy sparse matrices (CSR / COO) require explicit conversion
    functions because they consist of multiple component arrays::

        from pyGinkgo.cupy_interop import from_cupy_csr_to_gko
        gko_csr = from_cupy_csr_to_gko(cp_csr, executor)

        from pyGinkgo.cupy_interop import gko_csr_to_cupy
        cp_csr = gko_csr_to_cupy(gko_csr)

    The C++ ``from_device_ptrs`` factory wraps the CuPy component
    arrays (values, col indices, row pointers / row indices) as
    non-owning ``gko::array::view`` objects.  This is true zero-copy;
    ``py::keep_alive`` prevents garbage-collection of the source CuPy
    arrays while the Ginkgo matrix is alive.

    When the C++ CUDA path is unavailable the data is copied through
    host memory as a fallback.

Design rationale (``__cuda_array_interface__`` vs DLPack):
    We chose ``__cuda_array_interface__`` because:

    1. It is CuPy's native protocol for zero-copy CUDA interop.
    2. It is simpler to implement and maintain at the C++ binding level.
    3. It directly addresses the CUDA use case that prompted this work.

    DLPack is a candidate for future work to provide a universal protocol
    that also covers HIP/ROCm and SYCL/DPC++ backends.
"""

import numpy as np
from typing import Optional

try:
    import cupy

    cupy_avail = True
except ImportError:
    cupy_avail = False

try:
    import cupyx.scipy.sparse as cupy_sparse

    cupy_sparse_avail = True
except ImportError:
    cupy_sparse_avail = False


def is_cupy_array(obj) -> bool:
    """Return *True* if *obj* is a CuPy ndarray."""
    if not cupy_avail:
        return False
    return isinstance(obj, cupy.ndarray)


def is_cupy_sparse(obj) -> bool:
    """Return *True* if *obj* is a CuPy sparse matrix (CSR or COO)."""
    if not cupy_sparse_avail:
        return False
    return isinstance(obj, (cupy_sparse.csr_matrix, cupy_sparse.coo_matrix))


# ------------------------------------------------------------------
# dtype helpers
# ------------------------------------------------------------------

_NP_TO_GKO_VALUE_DTYPE = {
    np.dtype("float16"): "half",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
}

_NP_TO_GKO_INDEX_DTYPE = {
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
}

_NP_TO_GKO_DTYPE = {**_NP_TO_GKO_VALUE_DTYPE, **_NP_TO_GKO_INDEX_DTYPE}

# Reverse maps: Ginkgo dtype string → numpy dtype
_GKO_VALUE_TO_NP = {v: k for k, v in _NP_TO_GKO_VALUE_DTYPE.items()}
_GKO_INDEX_TO_NP = {v: k for k, v in _NP_TO_GKO_INDEX_DTYPE.items()}


def _gko_class_dtypes(gko_obj):
    """Extract (np_value_dtype, np_index_dtype) from a Ginkgo class name.

    Class names follow the pattern ``Csr_float_int32``, ``Coo_double_int64``,
    etc.  Returns ``(None, None)`` if the name cannot be parsed.
    """
    parts = type(gko_obj).__name__.split("_")
    if len(parts) < 3:
        return None, None
    return _GKO_VALUE_TO_NP.get(parts[1]), _GKO_INDEX_TO_NP.get(parts[2])


# ------------------------------------------------------------------
# CuPy sparse  →  Ginkgo   (zero-copy view via from_device_ptrs)
# ------------------------------------------------------------------

def from_cupy_csr_to_gko(
    cupy_csr, executor,
    dtype: Optional[str] = None,
    itype: Optional[str] = None,
):
    """Create a Ginkgo CSR matrix from a CuPy CSR sparse matrix."""
    from pyGinkgo import pyGinkgoBindings as pGB

    if not cupy_sparse_avail:
        raise ImportError("cupyx.scipy.sparse is required")

    if not isinstance(cupy_csr, cupy_sparse.csr_matrix):
        raise TypeError("Expected a cupyx.scipy.sparse.csr_matrix")

    if dtype is None:
        dtype = _cupy_value_dtype(cupy_csr.data)
    if itype is None:
        itype = _cupy_index_dtype(cupy_csr.indices)

    csr_cls = getattr(pGB.matrix, f"Csr_{dtype}_{itype}")
    shape = cupy_csr.shape

    # Fast path: zero-copy device views
    if hasattr(csr_cls, "from_device_ptrs"):
        return csr_cls.from_device_ptrs(
            executor,
            (shape[0], shape[1]),
            cupy_csr.data,
            cupy_csr.indices,
            cupy_csr.indptr,
        )

    # Fallback: copy through host
    np_data = cupy.asnumpy(cupy_csr.data)
    np_indices = cupy.asnumpy(cupy_csr.indices)
    np_indptr = cupy.asnumpy(cupy_csr.indptr)
    return csr_cls(executor, (shape[0], shape[1]), np_data, np_indices, np_indptr)


def from_cupy_coo_to_gko(
    cupy_coo, executor,
    dtype: Optional[str] = None,
    itype: Optional[str] = None,
):
    """Create a Ginkgo COO matrix from a CuPy COO sparse matrix."""
    from pyGinkgo import pyGinkgoBindings as pGB

    if not cupy_sparse_avail:
        raise ImportError("cupyx.scipy.sparse is required")

    if not isinstance(cupy_coo, cupy_sparse.coo_matrix):
        raise TypeError("Expected a cupyx.scipy.sparse.coo_matrix")

    if dtype is None:
        dtype = _cupy_value_dtype(cupy_coo.data)
    if itype is None:
        itype = _cupy_index_dtype(cupy_coo.col)

    coo_cls = getattr(pGB.matrix, f"Coo_{dtype}_{itype}")
    shape = cupy_coo.shape

    # Fast path: zero-copy device views
    if hasattr(coo_cls, "from_device_ptrs"):
        return coo_cls.from_device_ptrs(
            executor,
            (shape[0], shape[1]),
            cupy_coo.data,
            cupy_coo.col,
            cupy_coo.row,
        )

    # Fallback: copy through host
    np_data = cupy.asnumpy(cupy_coo.data)
    np_col = cupy.asnumpy(cupy_coo.col)
    np_row = cupy.asnumpy(cupy_coo.row)
    return coo_cls(executor, (shape[0], shape[1]), np_data, np_col, np_row)


# ------------------------------------------------------------------
# Ginkgo sparse  →  CuPy sparse
# ------------------------------------------------------------------


def gko_csr_to_cupy(gko_csr):
    """Convert a Ginkgo CSR matrix to a CuPy CSR sparse matrix."""
    if not cupy_avail or not cupy_sparse_avail:
        raise ImportError("CuPy with cupyx.scipy.sparse is required")

    shape = gko_csr.shape
    nnz = gko_csr.get_num_stored_elements()

    # Fast path: wrap device pointers directly
    if hasattr(gko_csr, "get_values_device_ptr"):
        vals_ptr = gko_csr.get_values_device_ptr()
        cols_ptr = gko_csr.get_col_idxs_device_ptr()
        rows_ptr = gko_csr.get_row_ptrs_device_ptr()

        np_vdtype, np_idtype = _gko_class_dtypes(gko_csr)

        if np_vdtype is None or np_idtype is None:
            import warnings
            warnings.warn(
                f"Cannot determine dtypes from class name "
                f"'{type(gko_csr).__name__}'; falling back to "
                f"dense conversion through host memory.",
                stacklevel=2,
            )
        else:
            values = cupy.ndarray(
                nnz,
                dtype=np_vdtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        vals_ptr,
                        nnz * np_vdtype.itemsize,
                        gko_csr,
                    ),
                    0,
                ),
            )
            col_idxs = cupy.ndarray(
                nnz,
                dtype=np_idtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        cols_ptr,
                        nnz * np_idtype.itemsize,
                        gko_csr,
                    ),
                    0,
                ),
            )
            row_ptrs = cupy.ndarray(
                shape[0] + 1,
                dtype=np_idtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        rows_ptr,
                        (shape[0] + 1) * np_idtype.itemsize,
                        gko_csr,
                    ),
                    0,
                ),
            )
            return cupy_sparse.csr_matrix(
                (values, col_idxs, row_ptrs), shape=shape
            )

    # Fallback: dense → host → CuPy (expensive, last resort)
    dense = gko_csr.convert_to_dense()
    if hasattr(dense, "copy_to_host"):
        dense = dense.copy_to_host()
    import scipy.sparse

    np_dense = np.array(dense)
    sp = scipy.sparse.csr_matrix(np_dense)
    return cupy_sparse.csr_matrix(sp)


def gko_coo_to_cupy(gko_coo):
    """Convert a Ginkgo COO matrix to a CuPy COO sparse matrix."""
    if not cupy_avail or not cupy_sparse_avail:
        raise ImportError("CuPy with cupyx.scipy.sparse is required")

    shape = gko_coo.shape
    nnz = gko_coo.get_num_stored_elements()

    # Fast path: wrap device pointers directly
    if hasattr(gko_coo, "get_values_device_ptr"):
        vals_ptr = gko_coo.get_values_device_ptr()
        cols_ptr = gko_coo.get_col_idxs_device_ptr()
        rows_ptr = gko_coo.get_row_idxs_device_ptr()

        np_vdtype, np_idtype = _gko_class_dtypes(gko_coo)

        if np_vdtype is None or np_idtype is None:
            import warnings
            warnings.warn(
                f"Cannot determine dtypes from class name "
                f"'{type(gko_coo).__name__}'; falling back to "
                f"dense conversion through host memory.",
                stacklevel=2,
            )
        else:
            values = cupy.ndarray(
                nnz,
                dtype=np_vdtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        vals_ptr,
                        nnz * np_vdtype.itemsize,
                        gko_coo,
                    ),
                    0,
                ),
            )
            col_idxs = cupy.ndarray(
                nnz,
                dtype=np_idtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        cols_ptr,
                        nnz * np_idtype.itemsize,
                        gko_coo,
                    ),
                    0,
                ),
            )
            row_idxs = cupy.ndarray(
                nnz,
                dtype=np_idtype,
                memptr=cupy.cuda.MemoryPointer(
                    cupy.cuda.UnownedMemory(
                        rows_ptr,
                        nnz * np_idtype.itemsize,
                        gko_coo,
                    ),
                    0,
                ),
            )
            return cupy_sparse.coo_matrix(
                (values, (row_idxs, col_idxs)), shape=shape
            )

    # Fallback: dense → host → CuPy (expensive, last resort)
    dense = gko_coo.convert_to_dense()
    if hasattr(dense, "copy_to_host"):
        dense = dense.copy_to_host()
    import scipy.sparse

    np_dense = np.array(dense)
    sp = scipy.sparse.coo_matrix(np_dense)
    return cupy_sparse.coo_matrix(sp)


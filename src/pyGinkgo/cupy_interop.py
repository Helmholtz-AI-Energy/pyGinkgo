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

API:
    The :class:`CuPyBridge` class provides an executor-centric
    interface that mirrors the pattern used by other backends (HIP,
    DPC++) where the executor drives array construction::

        bridge = CuPyBridge("cuda")          # or CuPyBridge(executor)
        gko_arr  = bridge.array(cp_arr)      # CuPy → Ginkgo array
        gko_csr  = bridge.csr(cp_csr)        # CuPy → Ginkgo CSR
        cp_arr   = bridge.to_cupy(gko_arr)   # Ginkgo → CuPy

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
    **Dense / 1-D arrays** – the helpers read the CuPy array's
    ``__cuda_array_interface__``, extract the device pointer, and call
    the C++ ``from_device_ptr`` factory to perform a fast
    device-to-device copy into Ginkgo-owned memory.

    **Sparse matrices (CSR / COO)** – the C++ ``from_device_ptrs``
    factory wraps the CuPy component arrays (values, col indices,
    row pointers / row indices) as non-owning ``gko::array::view``
    objects.  This is true zero-copy; ``py::keep_alive`` prevents
    garbage-collection of the source CuPy arrays while the Ginkgo
    matrix is alive.

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


def _cupy_dtype_to_gko_dtype(cupy_arr) -> str:
    """Map a CuPy array's dtype to a Ginkgo dtype string."""
    gko_dtype = _NP_TO_GKO_DTYPE.get(cupy_arr.dtype)
    if gko_dtype is None:
        raise TypeError(
            f"Unsupported CuPy dtype for Ginkgo conversion: {cupy_arr.dtype}"
        )
    return gko_dtype


def _cupy_value_dtype(cupy_arr) -> str:
    """Map a CuPy array's value dtype to a Ginkgo value-type string."""
    gko_dtype = _NP_TO_GKO_VALUE_DTYPE.get(cupy_arr.dtype)
    if gko_dtype is None:
        raise TypeError(
            f"Unsupported CuPy value dtype for Ginkgo: {cupy_arr.dtype}"
        )
    return gko_dtype


def _cupy_index_dtype(cupy_arr) -> str:
    """Map a CuPy array's index dtype to a Ginkgo index-type string."""
    gko_dtype = _NP_TO_GKO_INDEX_DTYPE.get(cupy_arr.dtype)
    if gko_dtype is None:
        raise TypeError(
            f"Unsupported CuPy index dtype for Ginkgo: {cupy_arr.dtype}"
        )
    return gko_dtype


# ------------------------------------------------------------------
# CuPy dense / array  →  Ginkgo
# ------------------------------------------------------------------

def _from_cupy_to_gko_array(cupy_arr, executor, dtype: Optional[str] = None):
    """Create a Ginkgo 1-D array from a CuPy array."""
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


def _from_cupy_to_gko_dense(cupy_arr, executor, dtype: Optional[str] = None):
    """Create a Ginkgo dense matrix from a CuPy array."""
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
        dtype = _cupy_value_dtype(cupy_arr)

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
# CuPy sparse  →  Ginkgo   (zero-copy view via from_device_ptrs)
# ------------------------------------------------------------------

def _from_cupy_csr_to_gko(
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


def _from_cupy_coo_to_gko(
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
# Ginkgo  →  CuPy
# ------------------------------------------------------------------

def _gko_to_cupy(gko_obj):
    """Convert a Ginkgo array or dense matrix to a CuPy array."""
    if not cupy_avail:
        raise ImportError(
            "CuPy is required for CuPyBridge.to_cupy(). "
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


def _gko_csr_to_cupy(gko_csr):
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


def _gko_coo_to_cupy(gko_coo):
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


# ------------------------------------------------------------------
# Executor-bound CuPy bridge (aligned with HIP / DPC++ patterns)
# ------------------------------------------------------------------

class CuPyBridge:
    """Executor-bound interface for CuPy ↔ Ginkgo zero-copy conversions.

    In the HIP and DPC++ backends, Ginkgo objects are constructed via
    the executor itself (``array_cls(executor, data)``).  ``CuPyBridge``
    brings CuPy conversions in line with that pattern by binding an
    executor once and exposing factory methods that mirror the standard
    array / dense / sparse constructors.

    Parameters
    ----------
    executor : Ginkgo Executor
        The target executor (typically a ``CudaExecutor`` for zero-copy).
        May also be a device string such as ``"cuda"`` or ``"cuda:0"``.

    Examples
    --------
    >>> import pyGinkgo.pyGinkgoBindings as pGB
    >>> executor = pGB.CudaExecutor()
    >>> bridge = CuPyBridge(executor)
    >>> gko_arr = bridge.array(cp_arr, dtype="float")
    >>> gko_csr = bridge.csr(cp_csr)
    >>> cp_result = bridge.to_cupy(gko_arr)
    """

    def __init__(self, executor):
        from pyGinkgo import pyGinkgoBindings as pGB

        if isinstance(executor, str):
            from .device import device
            executor = device(executor)

        if not isinstance(executor, pGB.Executor):
            raise TypeError(
                f"Expected a Ginkgo Executor or device string, "
                f"got {type(executor).__name__}"
            )
        self._executor = executor

    @property
    def executor(self):
        """The underlying Ginkgo executor."""
        return self._executor

    # -- CuPy → Ginkgo (input) ------------------------------------------

    def array(self, cupy_arr, dtype=None):
        """Create a Ginkgo 1-D array from a CuPy array.

        Parameters
        ----------
        cupy_arr : cupy.ndarray
            Source array (must be 1-D and C-contiguous).
        dtype : str, optional
            Ginkgo dtype string.  Inferred from *cupy_arr* when omitted.

        Returns
        -------
        gko array object
        """
        return _from_cupy_to_gko_array(cupy_arr, self._executor, dtype)

    def dense(self, cupy_arr, dtype=None):
        """Create a Ginkgo dense matrix from a CuPy array.

        Parameters
        ----------
        cupy_arr : cupy.ndarray
            Source array (1-D or 2-D, C-contiguous).
        dtype : str, optional
            Ginkgo dtype string.  Inferred from *cupy_arr* when omitted.

        Returns
        -------
        gko dense matrix object
        """
        return _from_cupy_to_gko_dense(cupy_arr, self._executor, dtype)

    def csr(self, cupy_csr, dtype=None, itype=None):
        """Create a Ginkgo CSR matrix from a CuPy CSR sparse matrix.

        Parameters
        ----------
        cupy_csr : cupyx.scipy.sparse.csr_matrix
            Source sparse matrix (on a CUDA device).
        dtype : str, optional
            Ginkgo value-type string.
        itype : str, optional
            Ginkgo index-type string.

        Returns
        -------
        Ginkgo CSR matrix
        """
        return _from_cupy_csr_to_gko(
            cupy_csr, self._executor, dtype=dtype, itype=itype
        )

    def coo(self, cupy_coo, dtype=None, itype=None):
        """Create a Ginkgo COO matrix from a CuPy COO sparse matrix.

        Parameters
        ----------
        cupy_coo : cupyx.scipy.sparse.coo_matrix
            Source sparse matrix (on a CUDA device).
        dtype : str, optional
            Ginkgo value-type string.
        itype : str, optional
            Ginkgo index-type string.

        Returns
        -------
        Ginkgo COO matrix
        """
        return _from_cupy_coo_to_gko(
            cupy_coo, self._executor, dtype=dtype, itype=itype
        )

    # -- Ginkgo → CuPy (output) -----------------------------------------

    @staticmethod
    def to_cupy(gko_obj):
        """Convert a Ginkgo array or dense matrix to a CuPy array.

        Parameters
        ----------
        gko_obj : gko array or dense matrix

        Returns
        -------
        cupy.ndarray
        """
        return _gko_to_cupy(gko_obj)

    @staticmethod
    def to_cupy_csr(gko_csr):
        """Convert a Ginkgo CSR matrix to a CuPy CSR sparse matrix.

        Parameters
        ----------
        gko_csr : Ginkgo CSR matrix

        Returns
        -------
        cupyx.scipy.sparse.csr_matrix
        """
        return _gko_csr_to_cupy(gko_csr)

    @staticmethod
    def to_cupy_coo(gko_coo):
        """Convert a Ginkgo COO matrix to a CuPy COO sparse matrix.

        Parameters
        ----------
        gko_coo : Ginkgo COO matrix

        Returns
        -------
        cupyx.scipy.sparse.coo_matrix
        """
        return _gko_coo_to_cupy(gko_coo)


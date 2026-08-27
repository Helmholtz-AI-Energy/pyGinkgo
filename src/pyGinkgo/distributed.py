# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from .pyGinkgoBindings import distributed as _distributed

available = bool(_distributed.available)
Partition = getattr(_distributed, "Partition", None)

# Value types the API understands. Which ones are actually compiled depends on
# the Ginkgo build (half requires GINKGO_ENABLE_HALF); _binding() checks at call
# time and raises if a requested dtype is unavailable in this build.
_VALID_DTYPES = ("half", "float", "double")


def _require_available():
    if not available:
        raise RuntimeError(
            "pyGinkgo distributed bindings are unavailable; build Ginkgo with MPI support."
        )


def _binding(prefix, dtype):
    """Resolve a typed binding (e.g. ``matrix_double``), validating availability."""
    _require_available()
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"dtype must be one of {_VALID_DTYPES}, got {dtype!r}")
    fn = getattr(_distributed, f"{prefix}_{dtype}", None)
    if fn is None:
        raise ValueError(
            f"dtype {dtype!r} is not available in this build "
            f"(Ginkgo built without {dtype} support)."
        )
    return fn


def _comm_handle(comm):
    """Accept an mpi4py communicator or an already-converted Fortran MPI handle."""
    if isinstance(comm, int):
        return comm
    py2f = getattr(comm, "py2f", None)
    if py2f is None:
        raise TypeError(
            "comm must be an mpi4py communicator or an integer MPI_Comm handle "
            "(as returned by comm.py2f())."
        )
    return py2f()


def build_partition(exec, owners, num_parts):
    _require_available()
    return _distributed.build_partition(exec, owners, num_parts)


def matrix(exec, comm, partition, rows, cols, vals, global_size, dtype="double"):
    return _binding("matrix", dtype)(
        exec, _comm_handle(comm), partition, rows, cols, vals, global_size
    )


def vector(exec, comm, partition, rows, vals, global_size, dtype="double"):
    return _binding("vector", dtype)(
        exec, _comm_handle(comm), partition, rows, vals, global_size
    )


_TYPESTR = {"double": "<f8", "float": "<f4", "half": "<f2"}
_ITEMSIZE = {"double": 8, "float": 4, "half": 2}


class _DeviceArray:
    """Zero-copy ``__cuda_array_interface__`` view over a distributed vector's
    local device buffer. Holds a reference to the source vector so the GPU
    memory stays alive. Wrap with ``cupy.asarray(obj)`` or
    ``numba.cuda.as_cuda_array(obj)``."""

    def __init__(self, vector, ptr, shape, strides, typestr):
        self._vector = vector  # keep-alive: GPU buffer is owned by the vector
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),  # (device pointer, read-only=False)
            "strides": strides,
            "version": 3,
        }


def vector_set_local(vector, vals, dtype="double"):
    """Overwrite a distributed vector's processor-local values in place (no
    reallocation, no re-distribution). ``vals`` may be a host numpy array or a
    ``__cuda_array_interface__`` device array; its length must equal the
    vector's local size and its entries must already be in Ginkgo's
    processor-local ordering. Use to reuse a vector across solves instead of
    rebuilding it with :func:`vector` every time."""
    return _binding("vector_set_local", dtype)(vector, vals)


def vector_local(vector, dtype="double", on_device=False):
    """Processor-local part of a distributed vector.

    on_device=False -> host numpy array (device->host copy).
    on_device=True  -> a __cuda_array_interface__ wrapper over the device buffer
                       (zero-copy); requires a CUDA-enabled build.
    """
    if not on_device:
        return _binding("vector_local", dtype)(vector)
    ptr, nrows, ncols, stride = _binding("vector_local_device", dtype)(vector)
    itemsize = _ITEMSIZE[dtype]
    if ncols == 1:
        shape, strides = (nrows,), None  # contiguous
    else:
        # Ginkgo Dense is row-major with leading dimension `stride`.
        shape, strides = (nrows, ncols), (stride * itemsize, itemsize)
    return _DeviceArray(vector, ptr, shape, strides, _TYPESTR[dtype])


if available:
    # Re-export the typed bindings that this build actually compiled.
    for _prefix in ("matrix", "vector", "vector_local", "vector_set_local"):
        for _dtype in _VALID_DTYPES:
            _name = f"{_prefix}_{_dtype}"
            _fn = getattr(_distributed, _name, None)
            if _fn is not None:
                globals()[_name] = _fn
    del _prefix, _dtype, _name, _fn


__all__ = [
    "available",
    "Partition",
    "build_partition",
    "matrix",
    "vector",
    "vector_local",
    "vector_set_local",
]

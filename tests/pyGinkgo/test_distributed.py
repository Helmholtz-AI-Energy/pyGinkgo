# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

"""Unit tests for the MPI-distributed pyGinkgo bindings (``pyGinkgo.distributed``).

The functional tests build the 1-D Poisson system ``A x = b`` (tri-diagonal
Laplacian, ``b = 1``) exactly like ``examples/distributed_solver.py`` and drive
it through a distributed CG solve.  They are written to be correct at *any* rank
count:

* run plainly (``pytest test_distributed.py``) they execute on a single rank and
  still exercise the full assemble/solve/read-back path;
* run under an MPI launcher (``mpirun -n 4 python -m pytest test_distributed.py``)
  they additionally exercise the off-process communication paths.

Every test is parametrised over the executors available in this build/host --
CPU (Reference) always, CUDA and ROCm/HIP when a device is present -- so the same
assertions cover the CPU and GPU code paths.  Anything that cannot run in the
current environment (build without MPI, no GPU, no ``mpi4py``/``cupy``) is
skipped rather than failed, so ``-rs`` reports exactly what was and was not
covered.
"""

import json

import numpy as np
import pytest

import pyGinkgo.pyGinkgoBindings as pGB
from pyGinkgo import distributed as dist


# --------------------------------------------------------------------------- #
#  Capability probes
# --------------------------------------------------------------------------- #
DIST_AVAILABLE = bool(dist.available)

requires_dist = pytest.mark.skipif(
    not DIST_AVAILABLE,
    reason="pyGinkgo built without MPI support (GINKGO_BUILD_MPI=ON required)",
)

# numpy dtype for each value-type name understood by the distributed bindings.
_NP_DTYPE = {"half": np.float16, "float": np.float32, "double": np.float64}


def _cuda_available():
    try:
        return hasattr(pGB, "CudaExecutor") and pGB.CudaExecutor.get_num_devices() > 0
    except RuntimeError:
        return False


def _hip_available():
    try:
        return hasattr(pGB, "HipExecutor") and pGB.HipExecutor.get_num_devices() > 0
    except RuntimeError:
        return False


def _dist_dtypes():
    """Value types the distributed bindings actually compiled in this build.

    float/double are always instantiated when MPI is enabled; half only when
    Ginkgo was built with GINKGO_ENABLE_HALF (re-exported as ``vector_half``).
    """
    dtypes = ["float", "double"]
    if DIST_AVAILABLE and hasattr(dist, "vector_half"):
        dtypes.append("half")
    return dtypes


def _has_device_binding(dtype):
    """True if the zero-copy ``vector_local_device_<dtype>`` (CUDA-only) exists."""
    return getattr(dist._distributed, f"vector_local_device_{dtype}", None) is not None


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def comm():
    """The MPI communicator.  Skips the whole module if mpi4py is missing."""
    MPI = pytest.importorskip("mpi4py.MPI")
    return MPI.COMM_WORLD


@pytest.fixture(params=["cpu", "cuda", "hip"])
def executor(request):
    """A fresh executor for each supported backend; GPU backends skip when absent.

    CUDA and ROCm/HIP remain in the parameter list even when no device is present
    so ``-rs`` reports them as skipped (matching test_cuda_executor.py's style).
    """
    backend = request.param
    if backend == "cpu":
        return pGB.ReferenceExecutor()
    if backend == "cuda":
        if not _cuda_available():
            pytest.skip("CUDA is not available")
        return pGB.CudaExecutor(0, pGB.ReferenceExecutor())
    if backend == "hip":
        if not _hip_available():
            pytest.skip("HIP is not available")
        return pGB.HipExecutor(0, pGB.ReferenceExecutor())
    raise AssertionError(f"unknown backend {backend!r}")


# --------------------------------------------------------------------------- #
#  Helpers -- 1-D Laplacian, contiguous block row distribution (PETSc-style)
# --------------------------------------------------------------------------- #
def _block_distribution(N, comm):
    """owners[g] = rank, plus this rank's [start, end) contiguous block."""
    rank, size = comm.Get_rank(), comm.Get_size()
    counts = [N // size + (1 if r < N % size else 0) for r in range(size)]
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    start, end = int(offsets[rank]), int(offsets[rank + 1])
    owners = np.empty(N, dtype=np.int32)
    for r in range(size):
        owners[offsets[r] : offsets[r + 1]] = r
    return owners, start, end


def _laplacian_triplets(start, end, N, np_dtype):
    """Global COO triplets for the owned rows of the [-1, 2, -1] Laplacian."""
    rows, cols, vals = [], [], []
    for i in range(start, end):
        rows.append(i)
        cols.append(i)
        vals.append(2.0)
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            vals.append(-1.0)
        if i < N - 1:
            rows.append(i)
            cols.append(i + 1)
            vals.append(-1.0)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np_dtype),
    )


def _dense_laplacian(N):
    A = 2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    return A


def _gather_global(local, comm):
    """Concatenate the per-rank local slices in rank order into the full vector.

    The block partition assigns rank r the contiguous global rows it owns and
    keeps them in ascending order within the local part, so rank-order
    concatenation reproduces the global ordering (see examples/distributed_solver.py).
    """
    pieces = comm.allgather(np.asarray(local).ravel())
    return np.concatenate(pieces)


# --------------------------------------------------------------------------- #
#  Pure-Python wrapper layer -- runs even in a build WITHOUT MPI
# --------------------------------------------------------------------------- #
class TestWrapperLayer:
    def test_available_is_bool(self):
        assert isinstance(dist.available, bool)

    def test_public_exports(self):
        for name in (
            "available",
            "build_partition",
            "matrix",
            "vector",
            "vector_local",
            "vector_set_local",
        ):
            assert name in dist.__all__

    def test_comm_handle_passes_through_int(self):
        assert dist._comm_handle(42) == 42

    def test_comm_handle_uses_py2f(self):
        class FakeComm:
            def py2f(self):
                return 7

        assert dist._comm_handle(FakeComm()) == 7

    def test_comm_handle_rejects_bad_object(self):
        with pytest.raises(TypeError, match="mpi4py communicator"):
            dist._comm_handle(object())

    def test_require_available_matches_flag(self):
        if DIST_AVAILABLE:
            assert dist._require_available() is None
        else:
            with pytest.raises(
                RuntimeError, match="distributed bindings are unavailable"
            ):
                dist._require_available()

    @requires_dist
    def test_binding_rejects_unknown_dtype(self):
        with pytest.raises(ValueError, match="dtype must be one of"):
            dist._binding("vector", "float128")

    @pytest.mark.skipif(DIST_AVAILABLE, reason="only meaningful without MPI support")
    def test_matrix_raises_when_unavailable(self):
        with pytest.raises(RuntimeError, match="distributed bindings are unavailable"):
            dist.matrix(None, 0, None, None, None, None, 1)


# --------------------------------------------------------------------------- #
#  Partition
# --------------------------------------------------------------------------- #
@requires_dist
class TestPartition:
    def test_build_and_properties(self, executor, comm):
        N, size = 12, comm.Get_size()
        owners, _, _ = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        assert part.size == N
        assert part.get_num_parts() == size
        assert "Partition" in repr(part)

    def test_rejects_non_positive_num_parts(self, executor):
        owners = np.zeros(4, dtype=np.int32)
        with pytest.raises(ValueError, match="num_parts must be positive"):
            dist.build_partition(executor, owners, 0)

    def test_rejects_empty_owners(self, executor):
        with pytest.raises(ValueError, match="owners must not be empty"):
            dist.build_partition(executor, np.empty(0, dtype=np.int32), 1)

    def test_rejects_out_of_range_part_id(self, executor):
        owners = np.array([0, 1, 5, 0], dtype=np.int32)  # 5 >= num_parts
        with pytest.raises(ValueError, match="invalid part id"):
            dist.build_partition(executor, owners, 2)


# --------------------------------------------------------------------------- #
#  Distributed matrix / vector assembly and read-back
# --------------------------------------------------------------------------- #
@requires_dist
class TestAssembly:
    @pytest.mark.parametrize("dtype", _dist_dtypes())
    def test_vector_local_roundtrips_global_values(self, executor, comm, dtype):
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)

        owned = np.arange(start, end, dtype=np.int32)
        # Distinct value per global row so the ordering is checked, not just sums.
        owned_vals = (owned + 1).astype(_NP_DTYPE[dtype])
        vec = dist.vector(executor, comm, part, owned, owned_vals, N, dtype=dtype)

        local = dist.vector_local(vec, dtype=dtype)
        assert local.shape[0] == owned.size
        np.testing.assert_allclose(np.asarray(local).ravel(), owned_vals)

        full = _gather_global(local, comm)
        np.testing.assert_allclose(full, np.arange(1, N + 1))

    @pytest.mark.parametrize("dtype", _dist_dtypes())
    def test_vector_set_local_overwrites_in_place(self, executor, comm, dtype):
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)

        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.zeros(owned.size, dtype=_NP_DTYPE[dtype]),
            N,
            dtype=dtype,
        )

        replacement = (100 + np.arange(owned.size)).astype(_NP_DTYPE[dtype])
        dist.vector_set_local(vec, replacement, dtype=dtype)

        np.testing.assert_allclose(
            np.asarray(dist.vector_local(vec, dtype=dtype)).ravel(), replacement
        )

    def test_matrix_builds(self, executor, comm):
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        rows, cols, vals = _laplacian_triplets(start, end, N, np.float64)
        A = dist.matrix(executor, comm, part, rows, cols, vals, N, dtype="double")
        assert A is not None


# --------------------------------------------------------------------------- #
#  End-to-end distributed CG solve  (CPU + GPU)
# --------------------------------------------------------------------------- #
@requires_dist
class TestDistributedSolve:
    # half is too inaccurate for a meaningful residual check on this system.
    # reduction_factor / comparison tolerance per precision: 1e-10 is below the
    # float32 residual floor, so CG would never satisfy it and never converge.
    @pytest.mark.parametrize(
        "dtype, reduction, tol",
        [
            pytest.param("double", 1e-10, 1e-6, id="double"),
            pytest.param("float", 1e-5, 1e-2, id="float"),
        ],
    )
    def test_cg_solves_poisson(self, executor, comm, dtype, reduction, tol):
        if dtype not in _dist_dtypes():
            pytest.skip(f"build has no distributed {dtype} bindings")

        N, size = 24, comm.Get_size()
        np_dtype = _NP_DTYPE[dtype]
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)

        rows, cols, vals = _laplacian_triplets(start, end, N, np_dtype)
        A = dist.matrix(executor, comm, part, rows, cols, vals, N, dtype=dtype)

        owned = np.arange(start, end, dtype=np.int32)
        b = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.ones(owned.size, dtype=np_dtype),
            N,
            dtype=dtype,
        )
        x = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.zeros(owned.size, dtype=np_dtype),
            N,
            dtype=dtype,
        )

        solver_args = json.dumps(
            {
                "type": "solver::Cg",
                "criteria": [
                    {"type": "Iteration", "max_iters": 1000},
                    {"type": "ResidualNorm", "reduction_factor": reduction},
                ],
            }
        )
        config_solve = getattr(pGB.solver, f"config_solve_{dtype}")
        logger = config_solve(executor, A, b, x, solver_args)

        assert logger.has_converged()
        assert 0 < logger.get_num_iterations() <= 1000

        # Compare the reassembled global solution against a dense reference.
        x_full = _gather_global(dist.vector_local(x, dtype=dtype), comm)
        x_ref = np.linalg.solve(_dense_laplacian(N), np.ones(N))
        np.testing.assert_allclose(x_full, x_ref, rtol=tol, atol=tol)


# --------------------------------------------------------------------------- #
#  Error paths in the C++ binding layer
# --------------------------------------------------------------------------- #
@requires_dist
class TestBindingErrors:
    def test_comm_handle_type_error(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        owned = np.arange(start, end, dtype=np.int32)
        with pytest.raises(TypeError, match="mpi4py communicator"):
            dist.vector(
                executor, object(), part, owned, np.ones(owned.size), N, dtype="double"
            )

    def test_matrix_rejects_mismatched_lengths(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        rows = np.array([start], dtype=np.int32)
        cols = np.array([start, start], dtype=np.int32)  # one too many
        vals = np.array([2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="matching lengths"):
            dist.matrix(executor, comm, part, rows, cols, vals, N, dtype="double")

    def test_matrix_rejects_partition_size_mismatch(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        rows, cols, vals = _laplacian_triplets(start, end, N, np.float64)
        with pytest.raises(ValueError, match="partition size"):
            dist.matrix(executor, comm, part, rows, cols, vals, N + 1, dtype="double")

    def test_matrix_rejects_out_of_range_index(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        # A single triplet whose column index lies outside [0, N), injected
        # identically on every rank so the raise happens before read_distributed.
        rows = np.array([start], dtype=np.int32)
        cols = np.array([N], dtype=np.int32)  # out of range
        vals = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="out-of-range global index"):
            dist.matrix(executor, comm, part, rows, cols, vals, N, dtype="double")

    def test_vector_rejects_out_of_range_index(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        rows = np.array([N], dtype=np.int32)  # out of range
        vals = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="out-of-range global index"):
            dist.vector(executor, comm, part, rows, vals, N, dtype="double")

    def test_vector_set_local_rejects_wrong_length(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor, comm, part, owned, np.zeros(owned.size), N, dtype="double"
        )
        with pytest.raises(ValueError, match="must match the vector's local size"):
            dist.vector_set_local(vec, np.ones(owned.size + 1), dtype="double")

    def test_vector_local_rejects_wrong_value_type(self, executor, comm):
        N = 8
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, comm.Get_size())
        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor, comm, part, owned, np.zeros(owned.size), N, dtype="double"
        )
        with pytest.raises(TypeError, match="not a distributed vector"):
            dist.vector_local(vec, dtype="float")


# --------------------------------------------------------------------------- #
#  GPU zero-copy device buffer path (CUDA only -- __cuda_array_interface__)
# --------------------------------------------------------------------------- #
@requires_dist
class TestDeviceBuffer:
    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_vector_local_on_device_matches_host(self, comm, dtype):
        if not _cuda_available():
            pytest.skip("CUDA is not available")
        if not _has_device_binding(dtype):
            pytest.skip("build has no CUDA zero-copy vector_local_device binding")
        cp = pytest.importorskip("cupy")

        executor = pGB.CudaExecutor(0, pGB.ReferenceExecutor())
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        owned = np.arange(start, end, dtype=np.int32)
        owned_vals = (owned + 1).astype(_NP_DTYPE[dtype])
        vec = dist.vector(executor, comm, part, owned, owned_vals, N, dtype=dtype)

        dev = dist.vector_local(vec, dtype=dtype, on_device=True)
        assert hasattr(dev, "__cuda_array_interface__")
        host_from_device = cp.asarray(dev).get()

        host = np.asarray(dist.vector_local(vec, dtype=dtype)).ravel()
        np.testing.assert_allclose(host_from_device.ravel(), host)
        np.testing.assert_allclose(host_from_device.ravel(), owned_vals)

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_vector_set_local_accepts_device_array(self, comm, dtype):
        if not _cuda_available():
            pytest.skip("CUDA is not available")
        if not _has_device_binding(dtype):
            pytest.skip("build has no CUDA zero-copy vector_local_device binding")
        cp = pytest.importorskip("cupy")

        executor = pGB.CudaExecutor(0, pGB.ReferenceExecutor())
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.zeros(owned.size, dtype=_NP_DTYPE[dtype]),
            N,
            dtype=dtype,
        )

        replacement = cp.asarray((100 + np.arange(owned.size)).astype(_NP_DTYPE[dtype]))
        dist.vector_set_local(vec, replacement, dtype=dtype)

        np.testing.assert_allclose(
            np.asarray(dist.vector_local(vec, dtype=dtype)).ravel(),
            (100 + np.arange(owned.size)).astype(_NP_DTYPE[dtype]),
        )

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_device_array_dtype_mismatch_is_rejected(self, comm, dtype):
        # A device array whose dtype does not match the binding must be rejected
        # rather than silently reinterpreted (cai_ptr_and_size validates typestr).
        if not _cuda_available():
            pytest.skip("CUDA is not available")
        if not _has_device_binding(dtype):
            pytest.skip("build has no CUDA zero-copy vector_local_device binding")
        cp = pytest.importorskip("cupy")

        executor = pGB.CudaExecutor(0, pGB.ReferenceExecutor())
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.zeros(owned.size, dtype=_NP_DTYPE[dtype]),
            N,
            dtype=dtype,
        )

        wrong_np = np.float64 if dtype == "float" else np.float32
        bad = cp.asarray(np.zeros(owned.size, dtype=wrong_np))
        with pytest.raises(ValueError, match="dtype mismatch"):
            dist.vector_set_local(vec, bad, dtype=dtype)

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_on_device_rejects_non_cuda_executor(self, comm, dtype):
        # In a CUDA-enabled build, asking for the device buffer of a vector that
        # lives on a CPU executor must raise rather than advertise a host pointer
        # as CUDA memory. Needs only the CUDA binding compiled in, not a GPU.
        if not _has_device_binding(dtype):
            pytest.skip("build has no CUDA zero-copy vector_local_device binding")

        executor = pGB.ReferenceExecutor()
        N, size = 12, comm.Get_size()
        owners, start, end = _block_distribution(N, comm)
        part = dist.build_partition(executor, owners, size)
        owned = np.arange(start, end, dtype=np.int32)
        vec = dist.vector(
            executor,
            comm,
            part,
            owned,
            np.zeros(owned.size, dtype=_NP_DTYPE[dtype]),
            N,
            dtype=dtype,
        )

        with pytest.raises(AttributeError, match="CUDA executors"):
            dist.vector_local(vec, dtype=dtype, on_device=True)

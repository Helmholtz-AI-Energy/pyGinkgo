# SPDX-FileCopyrightText: 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

"""Tests for CuPy interoperability with pyGinkgo.

These tests cover:
    - CuPy dense array ↔ Ginkgo array / dense (via __cuda_array_interface__)
    - CuPy CSR/COO sparse ↔ Ginkgo CSR/COO (zero-copy via from_device_ptrs)
    - End-to-end solver workflow (GMRES / CG with CuPy data)
    - dtype and shape preservation across conversions

All tests are skipped automatically when CuPy or a CUDA device is
unavailable, so they are safe to include in a non-GPU CI pipeline.
"""

import pytest
import numpy as np

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


# ---- helpers -----------------------------------------------------------

def _has_cuda_device() -> bool:
    """Return True if at least one CUDA device is visible to CuPy."""
    if not cupy_avail:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _has_gko_cuda() -> bool:
    """Return True if pyGinkgo was compiled with CUDA support."""
    try:
        import pyGinkgo.pyGinkgoBindings as pGB

        return hasattr(pGB, "CudaExecutor") and pGB.CudaExecutor.get_num_devices() > 0
    except Exception:
        return False


_CUPY_TO_GKO_VALUE = {cupy.float32: "float", cupy.float64: "double"} if cupy_avail else {}
_CUPY_TO_GKO_INDEX = {cupy.int32: "int32", cupy.int64: "int64"} if cupy_avail else {}


skip_no_cupy = pytest.mark.skipif(not cupy_avail, reason="CuPy is not installed")
skip_no_cupy_sparse = pytest.mark.skipif(
    not cupy_sparse_avail, reason="cupyx.scipy.sparse is not installed"
)
skip_no_cuda = pytest.mark.skipif(
    not _has_cuda_device() or not _has_gko_cuda(),
    reason="No CUDA device or pyGinkgo built without CUDA",
)


def _cupy_csr_to_gko(csr, executor, gko_dtype, gko_itype="int32"):
    """Wrap a CuPy CSR as a Ginkgo CSR via the standard constructor."""
    import pyGinkgo.pyGinkgoBindings as pGB

    csr_cls = getattr(pGB.matrix, f"Csr_{gko_dtype}_{gko_itype}")
    return csr_cls(executor, csr)


def _cupy_coo_to_gko(coo, executor, gko_dtype, gko_itype="int32"):
    """Wrap a CuPy COO as a Ginkgo COO via the standard constructor."""
    import pyGinkgo.pyGinkgoBindings as pGB

    coo_cls = getattr(pGB.matrix, f"Coo_{gko_dtype}_{gko_itype}")
    return coo_cls(executor, coo)


# ---- CuPy → Ginkgo array / dense (standard constructors) --------------

@skip_no_cupy
@skip_no_cuda
class TestCuPyToGkoArray:
    """CuPy 1-D array → Ginkgo array via standard constructor (zero-copy)."""

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_basic_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_" + dtype)

        cp_arr = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp_dtype)
        gko_arr = array_cls(executor, cp_arr)
        assert gko_arr.shape == (5,)

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_zero_copy(self, dtype):
        """Verify that the device pointer is the same (zero-copy)."""
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_" + dtype)

        cp_arr = cupy.array([10.0, 20.0, 30.0], dtype=cp_dtype)
        gko_arr = array_cls(executor, cp_arr)

        cp_ptr = cp_arr.__cuda_array_interface__["data"][0]
        gko_ptr = gko_arr.__cuda_array_interface__["data"][0]
        assert cp_ptr == gko_ptr

    def test_roundtrip(self):
        """CuPy → array_cls → cupy.asarray roundtrip."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_double")

        original = cupy.array([1.0, 2.0, 3.0, 4.0], dtype=cupy.float64)
        gko_arr = array_cls(executor, original)
        roundtripped = cupy.asarray(gko_arr)
        cupy.testing.assert_array_almost_equal(original, roundtripped)


@skip_no_cupy
@skip_no_cuda
class TestCuPyToGkoDense:
    """CuPy 1-D / 2-D array → Ginkgo dense via standard constructor (zero-copy)."""

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_1d_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_" + dtype)

        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cp_dtype)
        dense = dense_cls(executor, cp_arr)
        assert dense.shape == (3, 1)

    def test_2d_conversion(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_double")

        cp_arr = cupy.array([[1, 2], [3, 4], [5, 6]], dtype=cupy.float64)
        dense = dense_cls(executor, cp_arr)
        assert dense.shape == (3, 2)

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_zero_copy(self, dtype):
        """Verify that the device pointer is the same (zero-copy)."""
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_" + dtype)

        cp_arr = cupy.array([[1, 2], [3, 4]], dtype=cp_dtype)
        dense = dense_cls(executor, cp_arr)

        cp_ptr = cp_arr.__cuda_array_interface__["data"][0]
        gko_ptr = dense.__cuda_array_interface__["data"][0]
        assert cp_ptr == gko_ptr

    def test_roundtrip(self):
        """CuPy → dense_cls → cupy.asarray roundtrip."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_float")

        original = cupy.array([[1, 2], [3, 4]], dtype=cupy.float32)
        dense = dense_cls(executor, original)
        roundtripped = cupy.asarray(dense)
        cupy.testing.assert_array_almost_equal(
            original.ravel(), roundtripped.ravel()
        )


# ---- Ginkgo → CuPy (dense / array) ------------------------------------

@skip_no_cupy
@skip_no_cuda
class TestGkoToCuPy:
    """Ginkgo → CuPy via cupy.asarray and __cuda_array_interface__."""

    @pytest.mark.parametrize(
        "cupy_dtype,gko_dtype",
        [
            (cupy.float32, "float"),
            (cupy.float64, "double"),
        ],
    )
    def test_array_roundtrip(self, cupy_dtype, gko_dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_" + gko_dtype)

        original = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cupy_dtype)
        gko_arr = array_cls(executor, original)

        roundtripped = cupy.asarray(gko_arr)
        cupy.testing.assert_array_almost_equal(original, roundtripped)

    def test_dense_roundtrip(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_double")

        original = cupy.array([[1, 2], [3, 4]], dtype=cupy.float64)
        dense = dense_cls(executor, original)

        roundtripped = cupy.asarray(dense)
        cupy.testing.assert_array_almost_equal(
            original.ravel(), roundtripped.ravel()
        )

    @pytest.mark.parametrize(
        "cupy_dtype,gko_dtype",
        [
            (cupy.float32, "float"),
            (cupy.float64, "double"),
        ],
    )
    def test_gko_array_has_cuda_interface(self, cupy_dtype, gko_dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_" + gko_dtype)

        cp_arr = cupy.array([1.0, 2.0], dtype=cupy_dtype)
        gko_arr = array_cls(executor, cp_arr)
        assert hasattr(gko_arr, "__cuda_array_interface__")

        cai = gko_arr.__cuda_array_interface__
        assert cai["version"] == 3
        assert cai["shape"] == (2,)
        assert isinstance(cai["data"], tuple)

    def test_gko_dense_has_cuda_interface(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        dense_cls = getattr(pGB.matrix, "dense_double")

        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        dense = dense_cls(executor, cp_arr)
        assert hasattr(dense, "__cuda_array_interface__")

    def test_cupy_asarray_zero_copy(self):
        """cupy.asarray(gko_obj) creates a zero-copy view via CAI."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        array_cls = getattr(pGB.base, "array_double")

        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        gko_arr = array_cls(executor, cp_arr)
        result = cupy.asarray(gko_arr)
        cupy.testing.assert_array_almost_equal(cp_arr, result)

    def test_host_array_no_cuda_interface(self):
        """CPU arrays must NOT expose __cuda_array_interface__."""
        import pyGinkgo.pyGinkgoBindings as pGB

        ref = pGB.ReferenceExecutor()
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        array_cls = getattr(pGB.base, "array_float")
        gko_arr = array_cls(ref, np_arr)
        assert not hasattr(gko_arr, "__cuda_array_interface__")


# ---- CuPy CSR ↔ Ginkgo CSR -------------------------------------------

@skip_no_cupy
@skip_no_cupy_sparse
@skip_no_cuda
class TestCuPyCSR:
    """CuPy CSR sparse ↔ Ginkgo CSR via zero-copy.

    Input:  ``Csr_cls(executor, cupy_csr)`` — the constructor duck-types
            on ``.data``, ``.indices``, ``.indptr``, ``.shape``.
    Output: ``.data``, ``.indices``, ``.indptr`` properties return
            ``gko::array`` views with ``__cuda_array_interface__``.
    """

    @staticmethod
    def _make_cupy_csr(dtype=cupy.float64, itype=cupy.int32):
        """Create a small test CSR matrix on the GPU."""
        # 3x3:  [[1,0,2],[0,3,0],[4,0,5]]
        data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        indices = cupy.array([0, 2, 1, 0, 2], dtype=itype)
        indptr = cupy.array([0, 2, 3, 5], dtype=itype)
        return cupy_sparse.csr_matrix((data, indices, indptr), shape=(3, 3))

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_csr_constructor(self, dtype):
        """Csr_cls(executor, cupy_csr) creates a Ginkgo CSR."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype)
        gko_csr = _cupy_csr_to_gko(
            csr, executor, _CUPY_TO_GKO_VALUE[dtype], "int32",
        )
        assert gko_csr.shape == (3, 3)
        assert gko_csr.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    @pytest.mark.parametrize("itype", [cupy.int32, cupy.int64])
    def test_csr_constructor_dtypes(self, dtype, itype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype, itype)
        gko_csr = _cupy_csr_to_gko(
            csr, executor, _CUPY_TO_GKO_VALUE[dtype], _CUPY_TO_GKO_INDEX[itype],
        )

        assert gko_csr.shape == (3, 3)
        assert gko_csr.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_csr_spmv(self, dtype):
        """CSR matrix–vector product (SpMV) with CuPy data."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype)
        gko_dtype = _CUPY_TO_GKO_VALUE[dtype]
        gko_csr = _cupy_csr_to_gko(csr, executor, gko_dtype, "int32")

        dense_cls = getattr(pGB.matrix, f"dense_{gko_dtype}")

        b_cp = cupy.ones(3, dtype=dtype)
        b_gko = dense_cls(executor, b_cp)

        x_gko = dense_cls(executor, (3, 1))
        x_gko.fill(0.0)

        gko_csr.apply(b_gko, x_gko)

        x_cp = cupy.asarray(x_gko)
        expected = cupy.array([3.0, 3.0, 9.0], dtype=dtype)
        cupy.testing.assert_array_almost_equal(x_cp.ravel(), expected)

    def test_csr_component_properties(self):
        """gko_csr.data / .indices / .indptr return array views with CAI."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr()
        gko_csr = _cupy_csr_to_gko(csr, executor, "double", "int32")

        assert hasattr(gko_csr, "data")
        assert hasattr(gko_csr, "indices")
        assert hasattr(gko_csr, "indptr")

        assert hasattr(gko_csr.data, "__cuda_array_interface__")
        assert gko_csr.data.shape == (5,)
        assert gko_csr.indices.shape == (5,)
        assert gko_csr.indptr.shape == (4,)  # n_rows + 1

    def test_csr_roundtrip_via_properties(self):
        """CuPy CSR → Ginkgo → CuPy CSR roundtrip using properties."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        original = self._make_cupy_csr()
        gko_csr = _cupy_csr_to_gko(original, executor, "double", "int32")

        recovered = cupy_sparse.csr_matrix(
            (
                cupy.asarray(gko_csr.data),
                cupy.asarray(gko_csr.indices),
                cupy.asarray(gko_csr.indptr),
            ),
            shape=gko_csr.shape,
        )
        cupy.testing.assert_array_almost_equal(
            original.toarray(), recovered.toarray()
        )


# ---- CuPy COO ↔ Ginkgo COO -------------------------------------------

@skip_no_cupy
@skip_no_cupy_sparse
@skip_no_cuda
class TestCuPyCOO:
    """CuPy COO sparse ↔ Ginkgo COO via zero-copy.

    Input:  ``Coo_cls(executor, cupy_coo)`` — duck-types on
            ``.data``, ``.col``, ``.row``, ``.shape``.
    Output: ``.data``, ``.col``, ``.row`` properties return
            ``gko::array`` views with ``__cuda_array_interface__``.
    """

    @staticmethod
    def _make_cupy_coo(dtype=cupy.float64, itype=cupy.int32):
        """Create a small test COO matrix on the GPU."""
        data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        row = cupy.array([0, 0, 1, 2, 2], dtype=itype)
        col = cupy.array([0, 2, 1, 0, 2], dtype=itype)
        return cupy_sparse.coo_matrix((data, (row, col)), shape=(3, 3))

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_coo_constructor(self, dtype):
        """Coo_cls(executor, cupy_coo) creates a Ginkgo COO."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        coo = self._make_cupy_coo(dtype)
        gko_coo = _cupy_coo_to_gko(
            coo, executor, _CUPY_TO_GKO_VALUE[dtype], "int32",
        )
        assert gko_coo.shape == (3, 3)
        assert gko_coo.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    @pytest.mark.parametrize("itype", [cupy.int32, cupy.int64])
    def test_coo_constructor_dtypes(self, dtype, itype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        coo = self._make_cupy_coo(dtype, itype)
        gko_coo = _cupy_coo_to_gko(
            coo, executor, _CUPY_TO_GKO_VALUE[dtype], _CUPY_TO_GKO_INDEX[itype],
        )

        assert gko_coo.shape == (3, 3)
        assert gko_coo.get_num_stored_elements() == 5

    def test_coo_component_properties(self):
        """gko_coo.data / .col / .row return array views with CAI."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        coo = self._make_cupy_coo()
        gko_coo = _cupy_coo_to_gko(coo, executor, "double", "int32")

        assert hasattr(gko_coo, "data")
        assert hasattr(gko_coo, "col")
        assert hasattr(gko_coo, "row")

        assert hasattr(gko_coo.data, "__cuda_array_interface__")
        assert gko_coo.data.shape == (5,)
        assert gko_coo.col.shape == (5,)
        assert gko_coo.row.shape == (5,)

    def test_coo_roundtrip_via_properties(self):
        """CuPy COO → Ginkgo → CuPy COO roundtrip using properties."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        original = self._make_cupy_coo()
        gko_coo = _cupy_coo_to_gko(original, executor, "double", "int32")

        recovered = cupy_sparse.coo_matrix(
            (
                cupy.asarray(gko_coo.data),
                (cupy.asarray(gko_coo.row), cupy.asarray(gko_coo.col)),
            ),
            shape=gko_coo.shape,
        )
        cupy.testing.assert_array_almost_equal(
            original.toarray(), recovered.toarray()
        )


# ---- End-to-end solver workflow ----------------------------------------

@skip_no_cupy
@skip_no_cupy_sparse
@skip_no_cuda
class TestSolverWorkflow:
    """End-to-end solver workflow entirely on CuPy / CUDA data."""

    @staticmethod
    def _make_spd_system(n=10, dtype=cupy.float64, itype=cupy.int32):
        """Build a small symmetric positive definite tridiagonal system."""
        diag = 2.0 * cupy.ones(n, dtype=dtype)
        off = -1.0 * cupy.ones(n - 1, dtype=dtype)
        A_dense = cupy.diag(diag) + cupy.diag(off, 1) + cupy.diag(off, -1)
        A_csr = cupy_sparse.csr_matrix(A_dense)
        A_csr = cupy_sparse.csr_matrix(
            (
                A_csr.data.astype(dtype),
                A_csr.indices.astype(itype),
                A_csr.indptr.astype(itype),
            ),
            shape=A_csr.shape,
        )
        b = cupy.ones(n, dtype=dtype)
        return A_csr, b

    @staticmethod
    def _assert_residual_small(A_csr, x_cp, b_cp, tol=1e-6):
        """Check that || A @ x - b || < tol."""
        residual = cupy.linalg.norm(A_csr.dot(x_cp.ravel()) - b_cp)
        assert float(residual) < tol

    def test_solve_gmres_with_cupy_csr(self):
        """GMRES solver using a CuPy CSR matrix (zero-copy)."""
        import pyGinkgo as pg
        import pyGinkgo.pyGinkgoBindings as pGB

        A_csr, b_cp = self._make_spd_system(n=10)
        executor = pGB.CudaExecutor()

        A_gko = _cupy_csr_to_gko(A_csr, executor, "double")
        dense_cls = getattr(pGB.matrix, "dense_double")
        b_gko = dense_cls(executor, b_cp)

        x_gko = dense_cls(executor, (10, 1))
        x_gko.fill(0.0)

        solver_args = {
            "type": "solver::Gmres",
            "criteria": [
                {"type": "Iteration", "max_iters": 200},
                {"type": "ResidualNorm", "reduction_factor": 1e-10},
            ],
        }
        _, x_gko = pg.solve(A_gko, b_gko, x_gko, solver_args=solver_args)

        x_cp = cupy.asarray(x_gko)
        assert x_cp.shape[0] == 10
        self._assert_residual_small(A_csr, x_cp, b_cp)

    def test_solve_cg_with_cupy_csr(self):
        """CG solver using a CuPy CSR matrix (zero-copy, SPD system)."""
        import pyGinkgo as pg
        import pyGinkgo.pyGinkgoBindings as pGB

        A_csr, b_cp = self._make_spd_system(n=10)
        executor = pGB.CudaExecutor()

        A_gko = _cupy_csr_to_gko(A_csr, executor, "double")
        dense_cls = getattr(pGB.matrix, "dense_double")
        b_gko = dense_cls(executor, b_cp)

        x_gko = dense_cls(executor, (10, 1))
        x_gko.fill(0.0)

        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 200},
                {"type": "ResidualNorm", "reduction_factor": 1e-10},
            ],
        }
        _, x_gko = pg.solve(A_gko, b_gko, x_gko, solver_args=solver_args)

        x_cp = cupy.asarray(x_gko)
        self._assert_residual_small(A_csr, x_cp, b_cp)

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_solve_preserves_dtype(self, dtype):
        """The solver output dtype matches the input dtype."""
        import pyGinkgo as pg
        import pyGinkgo.pyGinkgoBindings as pGB

        gko_dtype = _CUPY_TO_GKO_VALUE[dtype]
        A_csr, b_cp = self._make_spd_system(n=5, dtype=dtype)
        executor = pGB.CudaExecutor()

        A_gko = _cupy_csr_to_gko(A_csr, executor, gko_dtype)
        dense_cls = getattr(pGB.matrix, f"dense_{gko_dtype}")
        b_gko = dense_cls(executor, b_cp)

        x_gko = dense_cls(executor, (5, 1))
        x_gko.fill(0.0)

        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 100},
                {"type": "ResidualNorm", "reduction_factor": 1e-5},
            ],
        }
        _, x_gko = pg.solve(A_gko, b_gko, x_gko, solver_args=solver_args)
        x_cp = cupy.asarray(x_gko)
        assert x_cp.dtype == dtype

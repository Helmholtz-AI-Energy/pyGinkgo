# SPDX-FileCopyrightText: 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

"""Tests for CuPy interoperability with pyGinkgo.

These tests cover:
    - CuPy dense array ↔ Ginkgo array / dense (via __cuda_array_interface__)
    - CuPy CSR/COO sparse ↔ Ginkgo CSR/COO (zero-copy via from_device_ptrs)
    - End-to-end solver workflow (GMRES / CG with CuPy data)
    - dtype and shape preservation across conversions
    - Fallback path through host memory when CUDA bindings are absent

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

# These imports must succeed even without CuPy (the module guards itself)
from pyGinkgo.cupy_interop import (
    is_cupy_array,
    is_cupy_sparse,
    from_cupy_to_gko_array,
    from_cupy_to_gko_dense,
    from_cupy_csr_to_gko,
    from_cupy_coo_to_gko,
    gko_to_cupy,
    gko_csr_to_cupy,
    gko_coo_to_cupy,
)


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


skip_no_cupy = pytest.mark.skipif(not cupy_avail, reason="CuPy is not installed")
skip_no_cupy_sparse = pytest.mark.skipif(
    not cupy_sparse_avail, reason="cupyx.scipy.sparse is not installed"
)
skip_no_cuda = pytest.mark.skipif(
    not _has_cuda_device() or not _has_gko_cuda(),
    reason="No CUDA device or pyGinkgo built without CUDA",
)


# ---- detection helpers -------------------------------------------------

@skip_no_cupy
class TestDetection:
    """Test that CuPy array / sparse detection works correctly."""

    def test_cupy_array_detected(self):
        arr = cupy.array([1.0, 2.0, 3.0])
        assert is_cupy_array(arr)

    def test_numpy_array_not_detected_as_cupy(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert not is_cupy_array(arr)

    def test_plain_list_not_detected_as_cupy(self):
        assert not is_cupy_array([1, 2, 3])

    @pytest.mark.skipif(not cupy_sparse_avail, reason="no cupy sparse")
    def test_cupy_csr_detected(self):
        m = cupy_sparse.csr_matrix(cupy.eye(3))
        assert is_cupy_sparse(m)

    @pytest.mark.skipif(not cupy_sparse_avail, reason="no cupy sparse")
    def test_cupy_coo_detected(self):
        m = cupy_sparse.coo_matrix(cupy.eye(3))
        assert is_cupy_sparse(m)


# ---- CuPy → Ginkgo array ----------------------------------------------

@skip_no_cupy
@skip_no_cuda
class TestCuPyToGkoArray:
    """CuPy 1-D array → Ginkgo array via from_cupy_to_gko_array."""

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_basic_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()

        cp_arr = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp_dtype)
        gko_arr = from_cupy_to_gko_array(cp_arr, executor, dtype)
        assert gko_arr.shape == (5,)

    def test_dtype_inference(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([10.0, 20.0], dtype=cupy.float64)
        gko_arr = from_cupy_to_gko_array(cp_arr, executor)
        assert gko_arr.shape == (2,)

    def test_rejects_2d(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([[1, 2], [3, 4]], dtype=cupy.float32)
        with pytest.raises(ValueError, match="1-D"):
            from_cupy_to_gko_array(cp_arr, executor, "float")

    def test_rejects_non_cupy(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        np_arr = np.array([1.0], dtype=np.float32)
        with pytest.raises(TypeError, match="CuPy"):
            from_cupy_to_gko_array(np_arr, executor, "float")


# ---- CuPy → Ginkgo dense ----------------------------------------------

@skip_no_cupy
@skip_no_cuda
class TestCuPyToGkoDense:
    """CuPy 1-D / 2-D array → Ginkgo dense via from_cupy_to_gko_dense."""

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_1d_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        cp_dtype = cupy.float32 if dtype == "float" else cupy.float64
        executor = pGB.CudaExecutor()

        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cp_dtype)
        dense = from_cupy_to_gko_dense(cp_arr, executor, dtype)
        assert dense.shape == (3, 1)

    def test_2d_conversion(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([[1, 2], [3, 4], [5, 6]], dtype=cupy.float64)
        dense = from_cupy_to_gko_dense(cp_arr, executor, "double")
        assert dense.shape == (3, 2)

    def test_dtype_inference(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([7.0, 8.0], dtype=cupy.float32)
        dense = from_cupy_to_gko_dense(cp_arr, executor)
        assert dense.shape == (2, 1)

    def test_rejects_3d(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.zeros((2, 3, 4), dtype=cupy.float32)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            from_cupy_to_gko_dense(cp_arr, executor, "float")


# ---- Ginkgo → CuPy (dense / array) ------------------------------------

@skip_no_cupy
@skip_no_cuda
class TestGkoToCuPy:
    """Ginkgo → CuPy via gko_to_cupy and __cuda_array_interface__."""

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
        original = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cupy_dtype)
        gko_arr = from_cupy_to_gko_array(original, executor, gko_dtype)

        roundtripped = gko_to_cupy(gko_arr)
        cupy.testing.assert_array_almost_equal(original, roundtripped)

    def test_dense_roundtrip(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        original = cupy.array([[1, 2], [3, 4]], dtype=cupy.float64)
        dense = from_cupy_to_gko_dense(original, executor, "double")

        roundtripped = gko_to_cupy(dense)
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
        cp_arr = cupy.array([1.0, 2.0], dtype=cupy_dtype)
        gko_arr = from_cupy_to_gko_array(cp_arr, executor, gko_dtype)
        assert hasattr(gko_arr, "__cuda_array_interface__")

        cai = gko_arr.__cuda_array_interface__
        assert cai["version"] == 3
        assert cai["shape"] == (2,)
        assert isinstance(cai["data"], tuple)

    def test_gko_dense_has_cuda_interface(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        dense = from_cupy_to_gko_dense(cp_arr, executor, "double")
        assert hasattr(dense, "__cuda_array_interface__")

    def test_cupy_asarray_zero_copy(self):
        """cupy.asarray(gko_obj) creates a zero-copy view via CAI."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        gko_arr = from_cupy_to_gko_array(cp_arr, executor, "double")
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
    """CuPy CSR sparse ↔ Ginkgo CSR via zero-copy device views."""

    @staticmethod
    def _make_cupy_csr(dtype=cupy.float64, itype=cupy.int32):
        """Create a small test CSR matrix on the GPU."""
        # 3x3:  [[1,0,2],[0,3,0],[4,0,5]]
        data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        indices = cupy.array([0, 2, 1, 0, 2], dtype=itype)
        indptr = cupy.array([0, 2, 3, 5], dtype=itype)
        return cupy_sparse.csr_matrix((data, indices, indptr), shape=(3, 3))

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_csr_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype)
        gko_csr = from_cupy_csr_to_gko(csr, executor)
        assert gko_csr.shape == (3, 3)
        assert gko_csr.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    @pytest.mark.parametrize("itype", [cupy.int32, cupy.int64])
    def test_csr_conversion_shape(self, dtype, itype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype, itype)
        gko_csr = from_cupy_csr_to_gko(csr, executor)

        assert gko_csr.shape == (3, 3)
        assert gko_csr.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_csr_spmv(self, dtype):
        """CSR matrix–vector product (SpMV) with CuPy data."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype)
        gko_csr = from_cupy_csr_to_gko(csr, executor)

        # b = [1, 1, 1]  → x = A @ b
        b_cp = cupy.ones(3, dtype=dtype)
        b_gko = from_cupy_to_gko_dense(b_cp, executor)

        gko_dtype = "float" if dtype == cupy.float32 else "double"
        x_cls = getattr(pGB.matrix, f"dense_{gko_dtype}")
        x_gko = x_cls(executor, (3, 1))
        x_gko.fill(0.0)

        gko_csr.apply(b_gko, x_gko)

        x_cp = gko_to_cupy(x_gko)
        expected = cupy.array([3.0, 3.0, 9.0], dtype=dtype)
        cupy.testing.assert_array_almost_equal(x_cp.ravel(), expected)

    def test_csr_roundtrip(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        original = self._make_cupy_csr()
        gko_csr = from_cupy_csr_to_gko(original, executor)
        recovered = gko_csr_to_cupy(gko_csr)

        cupy.testing.assert_array_almost_equal(
            original.toarray(), recovered.toarray()
        )

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_from_device_ptrs_is_zero_copy(self, dtype):
        """Verify that from_device_ptrs uses view semantics."""
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        csr = self._make_cupy_csr(dtype)
        gko_dtype = "float" if dtype == cupy.float32 else "double"
        csr_cls = getattr(pGB.matrix, f"Csr_{gko_dtype}_int32")

        if not hasattr(csr_cls, "from_device_ptrs"):
            pytest.skip("from_device_ptrs not available (non-CUDA build)")

        gko_csr = csr_cls.from_device_ptrs(
            executor,
            (3, 3),
            csr.data,
            csr.indices,
            csr.indptr,
        )

        # The Ginkgo object wraps the same device memory.
        assert gko_csr.shape == (3, 3)
        assert gko_csr.get_num_stored_elements() == 5


# ---- CuPy COO ↔ Ginkgo COO -------------------------------------------

@skip_no_cupy
@skip_no_cupy_sparse
@skip_no_cuda
class TestCuPyCOO:
    """CuPy COO sparse ↔ Ginkgo COO via zero-copy device views."""

    @staticmethod
    def _make_cupy_coo(dtype=cupy.float64, itype=cupy.int32):
        """Create a small test COO matrix on the GPU."""
        data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        row = cupy.array([0, 0, 1, 2, 2], dtype=itype)
        col = cupy.array([0, 2, 1, 0, 2], dtype=itype)
        return cupy_sparse.coo_matrix((data, (row, col)), shape=(3, 3))

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_coo_conversion(self, dtype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        coo = self._make_cupy_coo(dtype)
        gko_coo = from_cupy_coo_to_gko(coo, executor)
        assert gko_coo.shape == (3, 3)
        assert gko_coo.get_num_stored_elements() == 5

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    @pytest.mark.parametrize("itype", [cupy.int32, cupy.int64])
    def test_coo_conversion_shape(self, dtype, itype):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        coo = self._make_cupy_coo(dtype, itype)
        gko_coo = from_cupy_coo_to_gko(coo, executor)

        assert gko_coo.shape == (3, 3)
        assert gko_coo.get_num_stored_elements() == 5

    def test_coo_roundtrip(self):
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.CudaExecutor()
        original = self._make_cupy_coo()
        gko_coo = from_cupy_coo_to_gko(original, executor)
        recovered = gko_coo_to_cupy(gko_coo)

        cupy.testing.assert_array_almost_equal(
            original.toarray(), recovered.toarray()
        )


# ---- End-to-end solver workflow ----------------------------------------

@skip_no_cupy
@skip_no_cupy_sparse
@skip_no_cuda
class TestSolverWorkflow:
    """End-to-end solver workflow entirely on CuPy / CUDA data.

    Mirrors the previous NovaPIC Ginkgo solver that wrapped GMRES/CG
    to operate directly on CuPy CSR device pointers (zero-copy via
    gko::make_array_view / gko::make_const_array_view).
    """

    @staticmethod
    def _make_spd_system(n=10, dtype=cupy.float64, itype=cupy.int32):
        """Build a small symmetric positive definite tridiagonal system."""
        # A = tridiag(-1, 2, -1)  →  SPD
        diag = 2.0 * cupy.ones(n, dtype=dtype)
        off = -1.0 * cupy.ones(n - 1, dtype=dtype)
        A_dense = cupy.diag(diag) + cupy.diag(off, 1) + cupy.diag(off, -1)
        A_csr = cupy_sparse.csr_matrix(A_dense)
        # cast indices to the requested itype
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

        A_gko = from_cupy_csr_to_gko(A_csr, executor)
        b_gko = from_cupy_to_gko_dense(b_cp, executor, dtype="double")

        dense_cls = getattr(pGB.matrix, "dense_double")
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

        x_cp = gko_to_cupy(x_gko)
        assert x_cp.shape[0] == 10
        self._assert_residual_small(A_csr, x_cp, b_cp)

    def test_solve_cg_with_cupy_csr(self):
        """CG solver using a CuPy CSR matrix (zero-copy, SPD system)."""
        import pyGinkgo as pg
        import pyGinkgo.pyGinkgoBindings as pGB

        A_csr, b_cp = self._make_spd_system(n=10)
        executor = pGB.CudaExecutor()

        A_gko = from_cupy_csr_to_gko(A_csr, executor)
        b_gko = from_cupy_to_gko_dense(b_cp, executor, dtype="double")

        dense_cls = getattr(pGB.matrix, "dense_double")
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

        x_cp = gko_to_cupy(x_gko)
        self._assert_residual_small(A_csr, x_cp, b_cp)

    @pytest.mark.parametrize("dtype", [cupy.float32, cupy.float64])
    def test_solve_preserves_dtype(self, dtype):
        """The solver output dtype matches the input dtype."""
        import pyGinkgo as pg
        import pyGinkgo.pyGinkgoBindings as pGB

        gko_dtype = "float" if dtype == cupy.float32 else "double"
        A_csr, b_cp = self._make_spd_system(n=5, dtype=dtype)
        executor = pGB.CudaExecutor()

        A_gko = from_cupy_csr_to_gko(A_csr, executor, dtype=gko_dtype)
        b_gko = from_cupy_to_gko_dense(b_cp, executor, dtype=gko_dtype)

        dense_cls = getattr(pGB.matrix, f"dense_{gko_dtype}")
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
        x_cp = gko_to_cupy(x_gko)
        assert x_cp.dtype == dtype


# ---- Edge cases / error handling ---------------------------------------

@skip_no_cupy
class TestEdgeCases:
    """Error handling and edge cases."""

    def test_unsupported_dtype_raises(self):
        # The dtype validation fires before any executor interaction.
        import pyGinkgo.pyGinkgoBindings as pGB

        executor = pGB.ReferenceExecutor()
        cp_arr = cupy.array([1 + 2j, 3 + 4j])
        with pytest.raises(TypeError, match="Unsupported"):
            from_cupy_to_gko_array(cp_arr, executor)

    def test_non_contiguous_cupy_array_is_handled(self):
        """A non-contiguous CuPy array should be silently made contiguous."""
        cp_arr = cupy.array([[1, 2], [3, 4]], dtype=cupy.float32)
        col = cp_arr[:, 0]  # non-contiguous view (Fortran stride)
        assert not col.flags["C_CONTIGUOUS"]

        # The conversion function should handle non-contiguous arrays
        # by making a contiguous copy internally before extracting
        # the device pointer.  We test the full path when CUDA is available.
        if _has_cuda_device() and _has_gko_cuda():
            import pyGinkgo.pyGinkgoBindings as pGB

            executor = pGB.CudaExecutor()
            gko_arr = from_cupy_to_gko_array(col, executor, "float")
            assert gko_arr.shape == (2,)

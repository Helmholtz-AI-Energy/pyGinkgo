# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


class CsrLike:
    def __init__(self, data_dtype=np.float64, index_dtype=np.int32):
        self.data = np.array([1.0, 2.0, 3.0], dtype=data_dtype)
        self.indices = np.array([0, 1, 2], dtype=index_dtype)
        self.indptr = np.array([0, 1, 2, 3], dtype=index_dtype)
        self.shape = (3, 3)


class CooLike:
    def __init__(self, data_dtype=np.float32, index_dtype=np.longlong):
        self.data = np.array([1.0, 2.0, 3.0], dtype=data_dtype)
        self.row = np.array([0, 1, 2], dtype=index_dtype)
        self.col = np.array([0, 1, 2], dtype=index_dtype)
        self.shape = (3, 3)


class TestArrayWrapper:
    def test_infers_double_from_numpy_array(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        arr = pg.array(data)

        assert isinstance(arr, pGB.base.array_double)
        assert arr.shape == data.shape

    def test_explicit_dtype_overrides_inference(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        arr = pg.array(data, dtype="double")

        assert isinstance(arr, pGB.base.array_double)
        assert arr.shape == data.shape

    def test_missing_dtype_for_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.array(3)

    def test_unsupported_dtype_raises_clear_error(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Cannot find dtype *complex64"):
            pg.array(data, dtype="complex64")

    def test_missing_dtype_for_numpy_integer_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.array(np.int64(3))


class TestDenseWrapper:
    def test_infers_float_from_numpy_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        dense = pg.dense(data)

        assert isinstance(dense, pGB.matrix.dense_float)
        assert dense.shape == data.shape

    def test_can_allocate_with_fill(self):
        dense = pg.dense(dim=(3, 1), dtype="float", fill=0.0)

        assert isinstance(dense, pGB.matrix.dense_float)
        assert dense.shape == (3, 1)
        for row in range(3):
            assert dense.at(row, 0) == 0.0

    def test_missing_dtype_for_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.dense(dim=(3, 1))

    def test_unsupported_dtype_raises_clear_error(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)

        with pytest.raises(ValueError, match="Cannot find dtype *int32"):
            pg.dense(data, dtype="int32")


class TestSparseWrappers:
    def test_csr_infers_value_and_index_dtypes(self):
        csr = CsrLike(data_dtype=np.float64, index_dtype=np.int32)

        matrix = pg.Csr(csr)

        assert isinstance(matrix, pGB.matrix.Csr_double_int32)
        assert matrix.shape == csr.shape
        assert matrix.get_num_stored_elements() == csr.data.size

    def test_coo_infers_value_and_index_dtypes(self):
        coo = CooLike(data_dtype=np.float32, index_dtype=np.longlong)

        matrix = pg.Coo(coo)

        assert isinstance(matrix, pGB.matrix.Coo_float_int64)
        assert matrix.shape == coo.shape
        assert matrix.get_num_stored_elements() == coo.data.size

    def test_csr_can_construct_from_component_arrays(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cols = np.array([0, 1, 2], dtype=np.int32)
        rows = np.array([0, 1, 2, 3], dtype=np.int32)

        matrix = pg.Csr(dim=(3, 3), data=data, cols=cols, rows=rows)

        assert isinstance(matrix, pGB.matrix.Csr_float_int32)
        assert matrix.shape == (3, 3)
        assert matrix.get_num_stored_elements() == data.size

    def test_csr_explicit_dtype_and_itype_override_inference(self):
        csr = CsrLike(data_dtype=np.float64, index_dtype=np.int32)

        matrix = pg.Csr(csr, dtype="float", itype="int64")

        assert isinstance(matrix, pGB.matrix.Csr_float_int64)
        assert matrix.shape == csr.shape
        assert matrix.get_num_stored_elements() == csr.data.size

    def test_coo_explicit_dtype_overrides_inference(self):
        coo = CooLike(data_dtype=np.float32, index_dtype=np.int32)

        matrix = pg.Coo(coo, dtype="double", itype="int64")

        assert isinstance(matrix, pGB.matrix.Coo_double_int64)
        assert matrix.shape == coo.shape
        assert matrix.get_num_stored_elements() == coo.data.size

    def test_missing_dtype_for_sparse_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.Csr(itype="int32")

    def test_missing_itype_for_sparse_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer itype.*specify itype"):
            pg.Coo(dtype="float")

    def test_missing_sparse_component_raises_clear_error(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        cols = np.array([0, 1, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="requires dim, data, cols, and rows"):
            pg.Csr(dim=(3, 3), data=data, cols=cols)

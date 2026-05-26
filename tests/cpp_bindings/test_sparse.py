# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


def coo_rows_to_csr_rows(coo_rows: list) -> list:
    # https://stackoverflow.com/a/71339835/8302811
    n_rows = max(coo_rows) + 1
    ret = [0] * n_rows
    for i in coo_rows:
        ret[i] += 1
    for i in range(n_rows - 1):
        ret[i + 1] += ret[i]
    return [0] + ret


@pytest.mark.parametrize("matrix_format", ["Coo", "Csr"])
@pytest.mark.parametrize("value_type", list(pg.gko_types.ValueType))
@pytest.mark.parametrize("index_type", list(pg.gko_types.IndexType))
class TestSparseMatrix:
    # test a 5x5 symmetric matrix
    # A =
    # | 1  10  .  20  .  |
    # | 10  2  11  .  21 |
    # | .  11   3  12  . |
    # | 20  .  12  4  13 |
    # | .  21  .  13   5 |
    #

    coo_rows = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    csr_rows = coo_rows_to_csr_rows(coo_rows)

    cols = [0, 1, 3, 0, 1, 2, 4, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4]
    values = [
        1.0,
        10.0,
        20.0,
        10.0,
        2.0,
        11.0,
        21.0,
        11.0,
        3.0,
        12.0,
        20.0,
        12.0,
        4.0,
        13.0,
        21.0,
        13.0,
        5.0,
    ]
    dense = [1, 1, 1, 1, 1]

    ref = pGB.ReferenceExecutor()

    def get_rows(self, matrix_format) -> list:
        if matrix_format == "Csr":
            return self.csr_rows
        else:
            return self.coo_rows

    def test_can_create_sparse_matrix(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        sparse = matrix_cls(self.ref)
        assert sparse == sparse

    def test_can_create_sparse_from_np_arrays(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        coeffs = np.array(self.values, dtype=value_type.numpy_type)
        rows = np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type)
        cols = np.array(self.cols, dtype=index_type.numpy_type)

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)
        assert sparse == sparse

    def test_can_create_sparse_from_gko_arrays(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        val_arr_cls = getattr(pGB.base, f"array_{value_type}")
        idx_arr_cls = getattr(pGB.base, f"array_{index_type}")
        coeffs = val_arr_cls(
            self.ref, np.array(self.values, dtype=value_type.numpy_type)
        )
        rows = idx_arr_cls(
            self.ref,
            np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type),
        )
        cols = idx_arr_cls(self.ref, np.array(self.cols, dtype=index_type.numpy_type))

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)
        assert sparse == sparse
        assert sparse.get_num_stored_elements() == coeffs.shape[0]

    def test_can_apply_to_dense(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        coeffs = np.array(self.values, dtype=value_type.numpy_type)
        rows = np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type)
        cols = np.array(self.cols, dtype=index_type.numpy_type)

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)

        dense_cls = getattr(pGB.matrix, f"dense_{value_type}")
        dense_b = dense_cls(self.ref, np.array(self.dense, dtype=value_type.numpy_type))
        dense_x = dense_cls(
            self.ref, np.array([0, 0, 0, 0, 0], dtype=value_type.numpy_type)
        )

        sparse.apply(dense_b, dense_x)
        assert dense_x.at(0) == 31.0
        assert dense_x.at(1) == 44.0
        assert dense_x.at(2) == 26.0
        assert dense_x.at(3) == 49.0
        assert dense_x.at(4) == 39.0

    def test_can_read_from_mtx_file(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        reader = getattr(pGB.matrix, f"read_{matrix_format}_{value_type}_{index_type}")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        assert sparse == sparse

    def test_can_get_size(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        reader = getattr(pGB.matrix, f"read_{matrix_format}_{value_type}_{index_type}")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        with pytest.deprecated_call():
            assert sparse.get_size()[0] == 19
            assert sparse.get_size()[1] == 19

    def test_size_property(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        reader = getattr(pGB.matrix, f"read_{matrix_format}_{value_type}_{index_type}")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        with pytest.deprecated_call():
            assert sparse.get_size()[0] == 19
            assert sparse.get_size()[1] == 19
            with pytest.raises(AttributeError):
                sparse.size = pGB.dim2(4, 4)

    def test_shape_property(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        reader = getattr(pGB.matrix, f"read_{matrix_format}_{value_type}_{index_type}")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        assert sparse.shape == (19, 19)
        with pytest.raises(AttributeError):
            sparse.shape = (10, 10)

    def test_can_convert_to_dense(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        coeffs = np.array(self.values, dtype=value_type.numpy_type)
        rows = np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type)
        cols = np.array(self.cols, dtype=index_type.numpy_type)

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)
        dense = sparse.convert_to_dense()
        assert dense == dense

    def test_can_transpose_sparse_matrix(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")
        coeffs = np.array(self.values, dtype=value_type.numpy_type)
        rows = np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type)
        cols = np.array(self.cols, dtype=index_type.numpy_type)

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)

        sparse_T = sparse.T()

        assert sparse_T.shape == (5, 5)

        assert sparse_T.get_num_stored_elements() == sparse.get_num_stored_elements()

        dense = sparse.convert_to_dense()
        dense_T = sparse_T.convert_to_dense()

        for i in range(5):
            for j in range(5):
                assert dense_T.at(i, j) == dense.at(j, i)

    def test_double_transpose_returns_original(
        self,
        matrix_format,
        value_type: pg.gko_types.ValueType,
        index_type: pg.gko_types.IndexType,
    ):
        matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{value_type}_{index_type}")

        coeffs = np.array(self.values, dtype=value_type.numpy_type)
        rows = np.array(self.get_rows(matrix_format), dtype=index_type.numpy_type)
        cols = np.array(self.cols, dtype=index_type.numpy_type)

        sparse = matrix_cls(self.ref, (5, 5), coeffs, cols, rows)

        sparse_TT = sparse.T().T()

        assert sparse_TT.shape == sparse.shape
        assert sparse_TT.get_num_stored_elements() == sparse.get_num_stored_elements()

        dense_orig = sparse.convert_to_dense()
        dense_TT = sparse_TT.convert_to_dense()

        for i in range(5):
            for j in range(5):
                assert dense_TT.at(i, j) == dense_orig.at(i, j)

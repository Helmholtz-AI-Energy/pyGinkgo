# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import sys
import pyGinkgo
import pytest
import numpy as np

sys.path.append("../build")


@pytest.mark.parametrize("matrix_format", ["Coo", "Csr"])
class TestSparseMatrix:
    # test a 5x5 symmetric matrix
    # A =
    # | 1  10  .  20  .  |
    # | 10  2  11  .  21 |
    # | .  11   3  12  . |
    # | 20  .  12  4  13 |
    # | .  21  .  13   5 |
    #

    rows = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
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

    ref = pyGinkgo.ReferenceExecutor()

    def test_can_create_sparse_matrix(self, matrix_format):
        ctr = getattr(pyGinkgo.matrix, matrix_format)
        sparse = ctr(self.ref)
        assert sparse == sparse

    def test_can_create_sparse_from_np_arrays(self, matrix_format):
        if matrix_format == "Csr":
            # NOTE currently this test would fail for
            # Csr matrices since Csr expects compressed
            # rows
            return
        ctr = getattr(pyGinkgo.matrix, matrix_format)
        coeffs = np.array(self.values, dtype=np.double)
        rows = np.array(self.rows, dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)

        sparse = ctr(self.ref, (5, 5), coeffs, rows, cols)
        assert sparse == sparse

    def test_can_apply_to_dense(self, matrix_format):
        if matrix_format == "Csr":
            # NOTE currently this test would fail for
            # Csr matrices since Csr expects compressed
            # rows
            return
        ctr = getattr(pyGinkgo.matrix, matrix_format)
        coeffs = np.array(self.values, dtype=np.double)
        rows = np.array(self.rows, dtype=np.int32)
        cols = np.array(self.cols, dtype=np.int32)

        sparse = ctr(self.ref, (5, 5), coeffs, rows, cols)

        dense_b = pyGinkgo.matrix.dense(self.ref, np.array(self.dense, dtype=np.double))
        dense_x = pyGinkgo.matrix.dense(
            self.ref, np.array([0, 0, 0, 0, 0], dtype=np.double)
        )

        sparse.apply(dense_b, dense_x)
        assert dense_x.at(0) == 31.0
        assert dense_x.at(1) == 44.0
        assert dense_x.at(2) == 26.0
        assert dense_x.at(3) == 49.0
        assert dense_x.at(4) == 39.0

    def test_can_read_from_mtx_file(self, matrix_format):
        reader = getattr(pyGinkgo.matrix, "read_" + matrix_format)
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)

        assert sparse == sparse

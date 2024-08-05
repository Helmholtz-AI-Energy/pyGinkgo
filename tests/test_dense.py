# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import os
import pyGinkgo
import numpy as np

sys.path.append("../build")


def verify_dense_vec(mtx, values):
    """ """
    assert mtx.get_num_stored_elements() == len(values)
    for i in range(len(values)):
        assert mtx.at(i) == values[i]

    assert mtx.at(2, 0) == mtx.at(2)
    # test if it can be called several times
    assert mtx.at(2, 0) == mtx.at(2)


class TestDense:
    values = [1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 5.0, 6.0, -1.0]
    values2d = [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0], [5.0, 6.0, -1.0]]
    ref = pyGinkgo.ReferenceExecutor()

    def test_can_create_dense_linop(self):
        dense = pyGinkgo.matrix.dense(self.ref)
        assert dense == dense

    def test_can_create_dense_linop_with_dim(self):
        dense = pyGinkgo.matrix.dense(self.ref, (len(self.values), 1))
        assert dense == dense

    def test_can_create_dense_from_1D_np_array_with_default_exec(self):
        dense = pyGinkgo.matrix.dense(np.array(self.values))
        verify_dense_vec(dense, self.values)

    def test_can_copy_construct(self):
        dense_a = pyGinkgo.matrix.dense(np.array(self.values))
        dense_b = pyGinkgo.matrix.dense(self.ref, dense_a)
        verify_dense_vec(dense_b, self.values)

    def test_can_create_dense_from_1D_np_array(self):
        dense = pyGinkgo.matrix.dense(self.ref, np.array(self.values))
        verify_dense_vec(dense, self.values)

    def test_can_read_mtx_file(self):
        # Matrix market format stores column major order
        fn = os.path.dirname(os.path.realpath(__file__)) + "/dense_example.mtx"
        dense = pyGinkgo.matrix.read_dense(fn, self.ref)

        assert dense.get_num_stored_elements() == 12
        assert dense.at(0, 2) == 0.2785
        assert dense.at(2) == 0.2785
        assert dense.at(2, 2) == 0.9575

    def test_can_create_dense_from_2D_np_array_with_default_exec(self):
        dense = pyGinkgo.matrix.dense(np.array(self.values2d))
        assert dense.at(0, 1) == 2.0
        assert dense.at(0, 2) == -1.0
        assert dense.at(1, 1) == 4.0

    def test_can_create_dense_from_1D_np_array_with_stride(self):
        dense = pyGinkgo.matrix.dense(self.ref, (3, 3), np.array([self.values]), 3)

        assert dense.get_num_stored_elements() == len(self.values)
        assert dense.at(2) == -1.0
        assert dense.at(0, 2) == dense.at(2)
        assert dense.at(2, 1) == 6

    def test_dense_support_basic_functionality(self):
        dense = pyGinkgo.matrix.dense(self.ref, (3, 3), np.array([self.values]), 3)

        dense.scale(5)
        assert dense.at(2) == -5.0
        assert dense.at(1, 2) == dense.at(5)

        dense.inv_scale(5)
        assert dense.at(2) == -1.0
        assert dense.at(1, 2) == dense.at(5)

    def test_can_apply_to_transpose(self):
        aT = np.array([self.values])
        a = aT.T
        dense_a = pyGinkgo.matrix.dense(aT)
        dense_b = pyGinkgo.matrix.dense(a)
        result = pyGinkgo.matrix.dense(self.ref, (1, 1))
        dense_a.apply(dense_b, result)
        result.at(0) == aT.dot(a)[0, 0]

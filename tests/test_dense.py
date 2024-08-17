# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import sys
import os
import pyGinkgo
import numpy as np

sys.path.append("../build")


class TestDense:
    values = [1.0, 2.0, -1.0, 3.0, 4.0, -1.0, 5.0, 6.0, -1.0]
    ref = pyGinkgo.ReferenceExecutor()

    def test_can_create_dense_linop(self):
        dense = pyGinkgo.matrix.dense(self.ref)
        assert dense == dense

    def test_can_create_dense_linop_with_dim(self):
        dense = pyGinkgo.matrix.dense(self.ref, (len(self.values), 1))
        assert dense == dense

    def test_can_create_dense_from_1D_np_array(self):
        dense = pyGinkgo.matrix.dense(self.ref, np.array([self.values]))

        assert dense.get_num_stored_elements() == len(self.values)
        assert dense.at(2) == -1.0
        assert dense.at(0, 2) == dense.at(2)
        # test if it can be called several times
        assert dense.at(0, 2) == dense.at(2)

    def test_can_read_mtx_file(self):
        # Matrix market format stores column major order
        fn = os.path.dirname(os.path.realpath(__file__)) + "/dense_example.mtx"
        dense = pyGinkgo.matrix.read_dense(fn, self.ref)

        assert dense.get_num_stored_elements() == 12
        assert dense.at(0, 2) == 0.2785
        assert dense.at(2) == 0.2785
        assert dense.at(2, 2) == 0.9575

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

# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import sys

sys.path.append("../../")
import numpy as np
import os
import pytest

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

d_precision_map = {
    "half": 1e-3,
    "float": 1e-6,
    "double": 1e-6,
}


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestSolve:
    executor = pGB.ReferenceExecutor()
    # TODO replace all dense_float
    fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
    mtx = pGB.matrix.read_Coo_float_int32(fn, executor)
    dim = mtx.shape
    rows = dim[0]
    cols = dim[1]
    X = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    AX = [[9.0, 27.0], [12.0, 30.0], [15.0, 33.0]]
    BX = [[1.0, 4.0], [4.0, 10.0], [9.0, 18.0]]
    exphX = [[-0.6748703, -0.7149942], [0.2473979, 0.37255105]]
    expLambda = [1.32522729, 4.07477271]

    def test_can_default_solve(self, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        rhs = dense_cls(self.executor, (self.rows, 1))
        rhs.fill(1.0)
        initial_guess = dense_cls(self.executor, (self.rows, 1))
        initial_guess.fill(0.0)

        denseX = dense_cls(np.array(self.X, dtype=data_type.numpy_type))
        denseAX = dense_cls(np.array(self.AX, dtype=data_type.numpy_type))
        denseBX = dense_cls(np.array(self.BX, dtype=data_type.numpy_type))
        hX, Lambda = pg.RR1(denseX, denseAX, denseBX)
        reshX = np.array(hX, dtype=data_type.numpy_type)

        precision = d_precision_map[data_type]
        assert pytest.approx(Lambda.at(0), precision) == self.expLambda[0]
        assert pytest.approx(Lambda.at(1), precision) == self.expLambda[1]
        assert pytest.approx(reshX[0, 0], precision) == self.exphX[0][0]
        assert pytest.approx(reshX[0, 1], precision) == self.exphX[0][1]
        assert pytest.approx(reshX[1, 0], precision) == self.exphX[1][0]
        assert pytest.approx(reshX[1, 1], precision) == self.exphX[1][1]

# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import sys
import pyGinkgo
import pytest

sys.path.append("../build")


@pytest.mark.parametrize("solver", ["gmres"])
class TestSparseMatrix:
    ref = pyGinkgo.ReferenceExecutor()

    def test_can_create_solver(self, solver_name):
        reader = getattr(pyGinkgo.matrix, "read_coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/sparse_example.mtx"
        sparse = reader(fn, self.ref)
        solver_ctr = getattr(pyGinkgo.solver, solver_name)
        solver = solver_ctr(self.ref, sparse, 100, 10, 1e-06)

        assert solver == solver

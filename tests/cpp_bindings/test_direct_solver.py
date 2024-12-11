# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pgb


@pytest.mark.parametrize("solver_name", ["direct"])
class TestIterativeSolverBinding:
    ref = pgb.ReferenceExecutor()
    values = [1., 0., 0., 0., 2., 0., 0., 0., 3.]
    mtx = pgb.matrix.dense(ref, (3, 3), np.array(values), 3)
    dim = (3, 3)
    exp = np.array([1, 1/2., 1/3.])  

    solver_args = {"direct": {"factorization": "Cholesky"}}

    def test_unpreconditioned_solver(self, solver_name):
        solver_ctr = getattr(pgb.solver, solver_name)
        args = self.solver_args[solver_name]
        solver = solver_ctr(exec=self.ref, system_matrix=self.mtx, **args)

        #dim = self.mtx.get_size()
        #assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(self.mtx.get_executor(), (self.dim[0], 1))
        rhs.fill(1.0)
        x = pgb.matrix.dense(self.mtx.get_executor(), (self.dim[0], 1))
        x.fill(0.0)
        solver.apply(rhs, x)

        res = np.array(x)
        assert (np.all(self.exp) == np.all(res))

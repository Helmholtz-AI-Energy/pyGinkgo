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
    input_mtx = [[ 1,2, 3], [2,1,2], ... ]  #  NOTE one way to implement input matrix, and create a dense nxn matrix from that, see test_dense.py
    exp = np.array([1,2,3,4])  # TODO <- correct and expected result goes here 

    solver_args = {"direct": {"factorization": "Cholesky"}}

    def test_unpreconditioned_solver(self, solver_name):

        # reads from disk
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx" #  TODO replace with something easier for direct solver, also add .mtx to disk
        reader = getattr(pgb.matrix, "read_Coo")
        mtx = reader(fn, self.ref) # create 
        # otherwise
        dense_a = pGB.matrix.dense(np.array(self.values))

        solver_ctr = getattr(pgb.solver, solver_name)
        args = self.solver_args[solver_name]
        solver = solver_ctr(exec=self.ref, system_matrix=mtx, **args)

        dim = mtx.get_size()
        assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        x = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        x.fill(0.0)
        solver.apply(rhs, x)

        res = np.array(x)
        assert (self.exp == approx(res))

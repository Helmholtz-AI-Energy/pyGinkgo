# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest

import pyGinkgo.pyGinkgoBindings as pgb


@pytest.mark.parametrize("solver_name", ["direct"])
class TestIterativeSolverBinding:
    ref = pgb.ReferenceExecutor()

    solver_args = {
        "direct": {
            "factorization": "Cholesky" 
        }
    }

    def test_unpreconditioned_solver(self, solver_name):
        reader = getattr(pgb.matrix, "read_Coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        mtx = reader(fn, self.ref)
        solver_ctr = getattr(pgb.solver, solver_name)
        args = self.solver_args[solver_name]
        solver = solver_ctr(exec=self.ref, system_matrix=mtx, **args)

        dim = mtx.get_size()
        assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        solver.apply(rhs, initial_guess)

# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import os
import pytest

import pyGinkgo.pyGinkgoBindings as pgb


@pytest.mark.parametrize("solver_name", ["gmres"])
class TestIterativeSolverBinding:
    ref = pgb.ReferenceExecutor()

    solver_args = {
        "gmres": {
            "krylov_dim": 10,
            "max_iters": 1000,
            "tolerance": 1e-06,
            "relative_stop_mode": False,
        },
    }

    def test_unpreconditioned_solver(self, solver_name):
        reader = getattr(pgb.matrix, "read_Coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        mtx = reader(fn, self.ref)
        solver_ctr = getattr(pgb.solver, solver_name)
        args = self.solver_args[solver_name]
        solver = solver_ctr(exec=self.ref, system_matrix=mtx, **args)
        logger = solver.initialize_logger()
        assert not logger.has_converged()

        dim = mtx.get_size()
        assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        solver.apply(rhs, initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < args["max_iters"]
        assert logger.get_residual_norm() < args["tolerance"]
        assert logger.get_residual_norm() > 0.0

    def test_ilu_preconditioned_solver(self, solver_name):
        reader = getattr(pgb.matrix, "read_Coo")
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        mtx = reader(fn, self.ref)
        solver_ctr = getattr(pgb.solver, solver_name)
        args = self.solver_args[solver_name]
        ilu = pgb.preconditioner.Ilu(self.ref, mtx)
        solver = solver_ctr(
            exec=self.ref, preconditioner=ilu, system_matrix=mtx, **args
        )
        logger = solver.initialize_logger()
        assert not logger.has_converged()

        dim = mtx.get_size()
        assert dim[0] == dim[1]
        rhs = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = pgb.matrix.dense(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        solver.apply(rhs, initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < args["max_iters"]
        assert logger.get_residual_norm() < args["tolerance"]
        assert logger.get_residual_norm() > 0.0

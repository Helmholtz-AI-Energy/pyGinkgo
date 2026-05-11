# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import pytest

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


@pytest.mark.parametrize("solver_name", ["gmres", "cg"])
@pytest.mark.parametrize(
    "data_type",
    [
        # pg.gko_types.ValueType.half, # TODO: GMRES and ILU GMRES do not converge with half precision
        pg.gko_types.ValueType.float,
        pg.gko_types.ValueType.double,
    ],
)
class TestIterativeSolverBinding:
    ref = pGB.ReferenceExecutor()

    solver_args = {
        "gmres": {
            "krylov_dim": 10,
            "max_iters": 1000,
            "reduction_factor": 1e-06,
            "relative_stop_mode": False,
        },
        "cg": {
            "max_iters": 1000,
            "reduction_factor": 1e-06,
            "relative_stop_mode": False,
        },
    }

    def test_unpreconditioned_solver(
        self, solver_name, data_type: pg.gko_types.ValueType
    ):
        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        reader_cls = getattr(pGB.matrix, f"read_Coo_{data_type}_int32")
        mtx = reader_cls(fn, self.ref)

        solver_cls = getattr(pGB.solver, f"{solver_name}_{data_type}")
        args = self.solver_args[solver_name]
        solver = solver_cls(exec=self.ref, system_matrix=mtx, **args)
        logger = solver.initialize_logger()
        assert not logger.has_converged()

        dim = mtx.shape
        assert dim[0] == dim[1]
        dense_cls = getattr(pGB.matrix, f"dense_{data_type}")
        rhs = dense_cls(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = dense_cls(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        _, result = solver.apply(rhs, initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < args["max_iters"]
        assert logger.get_residual_norm() < args["reduction_factor"]
        assert logger.get_residual_norm() > 0.0
        assert result == initial_guess

    def test_ilu_preconditioned_solver(
        self, solver_name, data_type: pg.gko_types.ValueType
    ):
        if solver_name == "cg":
            pytest.skip(
                "ILU preconditioning is not suitable for CG because ILU is not guaranteed to be SPD"
            )

        fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"
        reader_cls = getattr(pGB.matrix, f"read_Coo_{data_type}_int32")
        mtx = reader_cls(fn, self.ref)

        solver_cls = getattr(pGB.solver, f"{solver_name}_{data_type}")
        args = self.solver_args[solver_name]

        precond_cls = getattr(pGB.preconditioner, f"Ilu_{data_type}_int32")
        ilu = precond_cls(self.ref, mtx)

        solver = solver_cls(
            exec=self.ref, preconditioner=ilu, system_matrix=mtx, **args
        )
        logger = solver.initialize_logger()
        assert not logger.has_converged()

        dim = mtx.shape
        assert dim[0] == dim[1]
        dense_cls = getattr(pGB.matrix, f"dense_{data_type}")
        rhs = dense_cls(mtx.get_executor(), (dim[0], 1))
        rhs.fill(1.0)
        initial_guess = dense_cls(mtx.get_executor(), (dim[0], 1))
        initial_guess.fill(0.0)
        _, result = solver.apply(rhs, initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < args["max_iters"]
        assert logger.get_residual_norm() < args["reduction_factor"]
        assert logger.get_residual_norm() > 0.0
        assert result == initial_guess

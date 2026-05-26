# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import os
import numpy as np
import pytest

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}


class TestCore:
    @pytest.fixture(autouse=True, params=list(d_type_map.keys()))
    def __post_init__(self, request):
        self.data_type = request.param
        self.executor = pGB.ReferenceExecutor()
        self.fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"

        self.reader_cls = getattr(pGB.matrix, f"read_Coo_{self.data_type}_int32")
        self.dense_cls = getattr(pGB.matrix, f"dense_{self.data_type}")

        self.mtx = self.reader_cls(self.fn, self.executor)
        self.rows = self.mtx.shape[0]
        self.cols = self.mtx.shape[1]

    def test_as_tensor_accepts_numpy_array(self):
        data = np.array([[1.0], [2.0], [3.0]], dtype=d_type_map[self.data_type])

        tensor = pg.as_tensor(
            obj=data,
            device=self.executor,
            dtype=self.data_type,
        )

        assert tensor.shape == data.shape

        np_tensor = np.array(tensor.copy_to_host()).reshape(tensor.shape)
        assert np.allclose(np_tensor, data)

    def test_as_tensor_can_create_filled_tensor_from_dim(self):
        tensor = pg.as_tensor(
            dim=(self.rows, 1),
            device=self.executor,
            dtype=self.data_type,
            fill=1.0,
        )

        assert tensor.shape == (self.rows, 1)

        np_tensor = np.array(tensor.copy_to_host()).reshape(tensor.shape)
        assert np.allclose(np_tensor, np.ones((self.rows, 1)))

    def test_as_tensor_requires_obj_or_dim(self):
        with pytest.raises(ValueError, match="Either obj or dim must be provided"):
            pg.as_tensor(
                device=self.executor,
                dtype=self.data_type,
            )

    def test_as_tensor_rejects_invalid_dtype(self):
        with pytest.raises(ValueError, match="Not a valid dtype"):
            pg.as_tensor(
                dim=(self.rows, 1),
                device=self.executor,
                dtype="invalid_dtype",
            )

    def test_read_can_read_sparse_matrix(self):
        matrix = pg.read(
            self.fn,
            format="Coo",
            dtype=self.data_type,
            itype="int32",
            device=self.executor,
        )

        assert matrix.shape == self.mtx.shape
        assert matrix.get_num_stored_elements() == self.mtx.get_num_stored_elements()

    def test_read_rejects_invalid_format(self):
        with pytest.raises(ValueError, match="Not a valid matrix format"):
            pg.read(
                self.fn,
                format="invalid_format",
                dtype=self.data_type,
                itype="int32",
                device=self.executor,
            )

    def test_read_rejects_invalid_dtype(self):
        with pytest.raises(ValueError, match="Not a valid dtype"):
            pg.read(
                self.fn,
                format="Coo",
                dtype="invalid_dtype",
                itype="int32",
                device=self.executor,
            )

    def test_read_rejects_invalid_itype_for_sparse_matrix(self):
        with pytest.raises(ValueError, match="Not a valid itype"):
            pg.read(
                self.fn,
                format="Coo",
                dtype=self.data_type,
                itype="invalid_itype",
                device=self.executor,
            )

    def test_generate_solver_can_apply_solver(self):
        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 2},
            ],
        }

        rhs = self.dense_cls(self.executor, (self.rows, 1))
        rhs.fill(1.0)

        result = self.dense_cls(self.executor, (self.rows, 1))
        result.fill(0.0)

        solver = pg.generate_solver(self.mtx, solver_args=solver_args)
        solver.apply(rhs, result)

        assert result.shape == (self.rows, 1)

        np_result = np.array(result.copy_to_host()).reshape(result.shape)
        assert len(np.nonzero(np_result)[0]) > 0

    def test_config_solve_returns_logger_and_result(self):
        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 2},
            ],
        }

        rhs = self.dense_cls(self.executor, (self.rows, 1))
        rhs.fill(1.0)

        initial_guess = self.dense_cls(self.executor, (self.rows, 1))
        initial_guess.fill(0.0)

        logger, result = pg.config_solve(
            self.mtx,
            rhs,
            initial_guess,
            solver_args=solver_args,
        )

        assert logger.get_num_iterations() == 2
        assert result.shape == (self.rows, 1)

    def test_solve_can_create_default_initial_guess(self):
        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 2},
            ],
        }

        rhs = self.dense_cls(self.executor, (self.rows, 1))
        rhs.fill(1.0)

        logger, result = pg.solve(
            self.mtx,
            rhs,
            solver_args=solver_args,
        )

        assert logger.get_num_iterations() == 2
        assert result.shape == (self.rows, 1)

        np_result = np.array(result.copy_to_host())
        assert len(np.nonzero(np_result)[0]) > 0
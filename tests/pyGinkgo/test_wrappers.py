# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import numpy as np
import pytest

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


class TestArrayWrapper:
    def test_infers_double_from_numpy_array(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        arr = pg.array(data)

        assert isinstance(arr, pGB.base.array_double)
        assert arr.shape == data.shape

    def test_explicit_dtype_overrides_inference(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        arr = pg.array(data, dtype="double")

        assert isinstance(arr, pGB.base.array_double)
        assert arr.shape == data.shape

    def test_missing_dtype_for_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.array(3)

    def test_unsupported_dtype_raises_clear_error(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Not a valid dtype.*complex64"):
            pg.array(data, dtype="complex64")


class TestDenseWrapper:
    def test_infers_float_from_numpy_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        dense = pg.dense(data)

        assert isinstance(dense, pGB.matrix.dense_float)
        assert dense.shape == data.shape

    def test_can_allocate_with_fill(self):
        dense = pg.dense(dim=(3, 1), dtype="float", fill=0.0)

        assert isinstance(dense, pGB.matrix.dense_float)
        assert dense.shape == (3, 1)
        for row in range(3):
            assert dense.at(row, 0) == 0.0

    def test_missing_dtype_for_allocation_raises_clear_error(self):
        with pytest.raises(ValueError, match="Cannot infer dtype.*specify dtype"):
            pg.dense(dim=(3, 1))

    def test_unsupported_dtype_raises_clear_error(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)

        with pytest.raises(ValueError, match="Not a valid dtype.*int32"):
            pg.dense(data, dtype="int32")

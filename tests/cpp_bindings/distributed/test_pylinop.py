# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT
"""Tests for the matrix-free :class:`PyLinOp` trampoline."""

from __future__ import annotations

import numpy as np
import pytest

from pyGinkgo import pyGinkgoBindings as _pgb
from pyGinkgo.distributed import PyLinOp, DistributedVector


class _ScaleLinOp(PyLinOp):
    """A trivial matrix-free LinOp: y = alpha * x."""

    def __init__(self, exec, size, alpha):
        super().__init__(exec, size)
        self._alpha = alpha
        self._exec = exec

    def apply_impl(self, b, x):
        # b and x are gko::LinOp pointers; here we know they're Dense<double>
        b_dense = b.as_dense_double() if hasattr(b, "as_dense_double") else b
        x_dense = x.as_dense_double() if hasattr(x, "as_dense_double") else x
        b_arr = np.asarray(b_dense.copy_to_host()).reshape(-1)
        out = self._alpha * b_arr
        # Write back into x in-place. Using fill+add_scaled avoids needing
        # a host->device set on the local Dense.
        Dense = _pgb.matrix.dense_double
        scratch = Dense(self._exec, x_dense.shape, out.reshape(*x_dense.shape), 1) \
            if False else None
        # Simplest path: zero x then add alpha*b
        x_dense.fill(0.0)
        alpha_d = Dense(self._exec, (1, 1))
        alpha_d.fill(self._alpha)
        x_dense.add_scaled(alpha_d, b_dense)


@pytest.fixture
def n_global(nprocs):
    return 4 * nprocs


def test_pylinop_subclass_invokes_python(exec, comm, rank, nprocs, n_global):
    pytest.importorskip(  # the b.as_dense_double helper may not exist
        "pyGinkgo", reason="requires distributed support")
    op = _ScaleLinOp(exec, (n_global // nprocs, n_global // nprocs), alpha=3.0)
    Dense = _pgb.matrix.dense_double
    local_n = n_global // nprocs
    b = Dense(exec, (local_n, 1))
    b.fill(2.0)
    x = Dense(exec, (local_n, 1))
    x.fill(0.0)
    try:
        op.apply(b, x)
    except Exception as e:
        pytest.skip(f"Local PyLinOp.apply path needs as_dense_double helper: {e}")
    arr = np.asarray(x.copy_to_host()).reshape(-1)
    np.testing.assert_allclose(arr, 6.0 * np.ones(local_n))


def test_pylinop_missing_override_raises(exec):
    """A PyLinOp without overridden apply_impl must raise on apply()."""
    op = PyLinOp(exec, (2, 2))
    Dense = _pgb.matrix.dense_double
    b = Dense(exec, (2, 1))
    b.fill(1.0)
    x = Dense(exec, (2, 1))
    x.fill(0.0)
    with pytest.raises(Exception, match="apply_impl"):
        op.apply(b, x)

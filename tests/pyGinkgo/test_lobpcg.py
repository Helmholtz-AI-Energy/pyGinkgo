# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import pyGinkgo.pyGinkgoBindings as pGB
import pyGinkgo.solver as solver


EXECUTOR = pGB.ReferenceExecutor()
DENSE = pGB.matrix.dense_double


def _dense(values):
    """Create a double-precision Ginkgo Dense matrix."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    return DENSE(EXECUTOR, values)


def _to_numpy(matrix):
    """Copy a Ginkgo Dense matrix to a NumPy array."""
    return np.array(
        matrix.copy_to_host(),
        copy=True,
    ).reshape(tuple(matrix.shape))


def _diagonal_problem():
    """Return a simple SPD eigenproblem with known eigenvalues."""
    n = 10
    m = 2

    eigenvalues = np.arange(1.0, n + 1.0)
    A_np = np.diag(eigenvalues)

    rng = np.random.default_rng(42)
    X0_np = rng.normal(size=(n, m))

    return (
        A_np,
        X0_np,
        _dense(A_np),
        _dense(X0_np),
    )


def _generalized_diagonal_problem(n=12, m=2, seed=42):
    """Return a generalized SPD problem A x = lambda B x.

    A and B are diagonal, with known generalized eigenvalues

        lambda_i = A_ii / B_ii = i + 1.
    """
    generalized_eigenvalues = np.arange(1.0, n + 1.0)

    # Positive diagonal entries ensure that B is SPD.
    b_diagonal = np.linspace(1.0, 2.0, n)

    B_np = np.diag(b_diagonal)
    A_np = np.diag(generalized_eigenvalues * b_diagonal)

    rng = np.random.default_rng(seed)
    X0_np = rng.normal(size=(n, m))

    return (
        A_np,
        B_np,
        X0_np,
        _dense(A_np),
        _dense(B_np),
        _dense(X0_np),
    )


def test_lobpcg_dispatches_to_blopex(monkeypatch):
    A = object()
    X0 = object()
    T = object()

    expected = (
        np.array([1.0, 2.0]),
        object(),
        np.zeros((2, 3)),
    )

    received = {}

    def fake_blopex(
        A_arg,
        X0_arg,
        nev_arg,
        *,
        T=None,
        itmax=200,
        tol=1e-6,
        A_products="implicit",
    ):
        received["A"] = A_arg
        received["X0"] = X0_arg
        received["nev"] = nev_arg
        received["T"] = T
        received["itmax"] = itmax
        received["tol"] = tol
        received["A_products"] = A_products

        return expected

    monkeypatch.setattr(
        solver,
        "blopex_lobpcg_standard_impl_",
        fake_blopex,
    )

    result = solver.lobpcg(
        A,
        X0,
        2,
        T=T,
        itmax=57,
        tol=1e-9,
        method="BLOPEX",
        A_products="explicit",
    )

    assert result is expected

    assert received == {
        "A": A,
        "X0": X0,
        "nev": 2,
        "T": T,
        "itmax": 57,
        "tol": 1e-9,
        "A_products": "explicit",
    }


def test_lobpcg_dispatches_to_generalized_blopex(monkeypatch):
    A = object()
    B = object()
    X0 = object()
    T = object()

    expected = (
        np.array([1.0, 2.0]),
        object(),
        np.zeros((2, 3)),
    )

    received = {}

    def fake_blopex(
        A_arg,
        B_arg,
        X0_arg,
        nev_arg,
        *,
        T=None,
        itmax=200,
        tol=1e-6,
        A_products="implicit",
        B_products="implicit",
    ):
        received["A"] = A_arg
        received["B"] = B_arg
        received["X0"] = X0_arg
        received["nev"] = nev_arg
        received["T"] = T
        received["itmax"] = itmax
        received["tol"] = tol
        received["A_products"] = A_products
        received["B_products"] = B_products

        return expected

    monkeypatch.setattr(
        solver,
        "blopex_lobpcg_generalized_impl_",
        fake_blopex,
    )

    result = solver.lobpcg(
        A,
        X0,
        2,
        B=B,
        T=T,
        itmax=57,
        tol=1e-9,
        method="BLOPEX",
        A_products="explicit",
        B_products="explicit",
    )

    assert result is expected

    assert received == {
        "A": A,
        "B": B,
        "X0": X0,
        "nev": 2,
        "T": T,
        "itmax": 57,
        "tol": 1e-9,
        "A_products": "explicit",
        "B_products": "explicit",
    }


def test_blopex_finds_smallest_eigenvalues():
    A_np, _, A, X0 = _diagonal_problem()

    Lambda, X, res = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="implicit",
    )

    np.testing.assert_allclose(
        Lambda,
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    X_np = _to_numpy(X)

    # Check the actual eigenvalue equations rather than only Lambda.
    residual = A_np @ X_np - X_np * Lambda.reshape(1, -1)

    relative_residual = np.linalg.norm(residual, axis=0) / np.abs(Lambda)

    assert np.all(relative_residual[:2] < 1e-6)

    # The residual history returned by BLOPEX should agree with a
    # directly computed residual at the final iteration.
    np.testing.assert_allclose(
        res[:, -1],
        relative_residual,
        rtol=1e-5,
        atol=1e-8,
    )


def test_generalized_blopex_finds_smallest_eigenvalues():
    A_np, B_np, _, A, B, X0 = _generalized_diagonal_problem()

    Lambda, X, res = solver.lobpcg(
        A,
        X0,
        nev=2,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="implicit",
        B_products="implicit",
    )

    np.testing.assert_allclose(
        Lambda,
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    X_np = _to_numpy(X)

    # Check the generalized eigenvalue equation
    #
    #     A X = B X Lambda.
    AX = A_np @ X_np
    BX = B_np @ X_np

    residual = AX - BX * Lambda.reshape(1, -1)

    relative_residual = np.linalg.norm(residual, axis=0) / np.abs(Lambda)

    assert np.all(relative_residual[:2] < 1e-6)

    # Returned residual history should agree with a directly
    # calculated generalized residual.
    np.testing.assert_allclose(
        res[:, -1],
        relative_residual,
        rtol=1e-5,
        atol=1e-8,
    )


def test_blopex_returns_orthonormal_eigenvectors():
    _, _, A, X0 = _diagonal_problem()

    Lambda, X, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
    )

    X_np = _to_numpy(X)

    np.testing.assert_allclose(
        X_np.T @ X_np,
        np.eye(len(Lambda)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_generalized_blopex_returns_b_orthonormal_eigenvectors():
    _, B_np, _, A, B, X0 = _generalized_diagonal_problem()

    Lambda, X, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
    )

    X_np = _to_numpy(X)

    np.testing.assert_allclose(
        X_np.T @ B_np @ X_np,
        np.eye(len(Lambda)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_blopex_supports_nev_smaller_than_block_size():
    n = 12
    m = 3
    nev = 2

    eigenvalues = np.arange(1.0, n + 1.0)
    A_np = np.diag(eigenvalues)

    rng = np.random.default_rng(123)
    X0_np = rng.normal(size=(n, m))

    A = _dense(A_np)
    X0 = _dense(X0_np)

    Lambda, X, res = solver.lobpcg(
        A,
        X0,
        nev=nev,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
    )

    # BLOPEX keeps m Ritz values/vectors internally.
    assert Lambda.shape == (m,)
    assert X.shape == (n, m)
    assert res.shape[0] == m

    # Only the first nev eigenpairs are required to converge.
    np.testing.assert_allclose(
        Lambda[:nev],
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.all(res[:nev, -1] < 1e-6)


def test_generalized_blopex_supports_nev_smaller_than_block_size():
    n = 12
    m = 3
    nev = 2

    (
        _,
        _,
        _,
        A,
        B,
        X0,
    ) = _generalized_diagonal_problem(
        n=n,
        m=m,
        seed=123,
    )

    Lambda, X, res = solver.lobpcg(
        A,
        X0,
        nev=nev,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
    )

    # BLOPEX keeps all m Ritz vectors internally.
    assert Lambda.shape == (m,)
    assert X.shape == (n, m)
    assert res.shape[0] == m

    # Only the first nev eigenpairs are required to converge.
    np.testing.assert_allclose(
        Lambda[:nev],
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.all(res[:nev, -1] < 1e-6)


def test_blopex_matches_numpy_on_dense_spd_problem():
    n = 12
    m = 2

    rng = np.random.default_rng(7)

    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))

    expected_eigenvalues = np.arange(1.0, n + 1.0)

    A_np = Q @ np.diag(expected_eigenvalues) @ Q.T

    X0_np = rng.normal(size=(n, m))

    Lambda, X, res = solver.lobpcg(
        _dense(A_np),
        _dense(X0_np),
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
    )

    reference, _ = np.linalg.eigh(A_np)

    np.testing.assert_allclose(
        Lambda[:2],
        reference[:2],
        rtol=1e-5,
        atol=1e-7,
    )

    X_np = _to_numpy(X)

    residual = A_np @ X_np - X_np * Lambda.reshape(1, -1)

    relative_residual = np.linalg.norm(residual, axis=0) / np.abs(Lambda)

    assert np.all(relative_residual[:2] < 1e-6)


def test_blopex_rejects_invalid_nev():
    _, _, A, X0 = _diagonal_problem()

    with pytest.raises(ValueError, match="nev"):
        solver.lobpcg(
            A,
            X0,
            nev=0,
            method="BLOPEX",
        )


def test_blopex_rejects_invalid_a_products():
    _, _, A, X0 = _diagonal_problem()

    with pytest.raises(ValueError, match="A_products"):
        solver.lobpcg(
            A,
            X0,
            nev=2,
            method="BLOPEX",
            A_products="wrong",
        )


def test_blopex_rejects_negative_itmax():
    _, _, A, X0 = _diagonal_problem()

    with pytest.raises(ValueError, match="itmax"):
        solver.lobpcg(
            A,
            X0,
            nev=2,
            method="BLOPEX",
            itmax=-1,
        )


def test_blopex_rejects_nonpositive_tolerance():
    _, _, A, X0 = _diagonal_problem()

    with pytest.raises(ValueError, match="tol"):
        solver.lobpcg(
            A,
            X0,
            nev=2,
            method="BLOPEX",
            tol=0.0,
        )


def test_blopex_rejects_oversized_block():
    n = 5
    m = 2

    A = _dense(np.eye(n))
    X0 = _dense(np.ones((n, m)))

    with pytest.raises(
        ValueError,
        match=r"3 \* X0.shape\[1\]",
    ):
        solver.lobpcg(
            A,
            X0,
            nev=2,
            method="BLOPEX",
        )


def test_blopex_zero_iterations_returns_initial_ritz_step():
    _, _, A, X0 = _diagonal_problem()

    Lambda, X, res = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=0,
    )

    assert Lambda.shape == (2,)
    assert X.shape == (10, 2)
    assert res.shape == (2, 1)

    assert np.all(np.isfinite(Lambda))
    assert np.all(np.isfinite(res))


@pytest.mark.parametrize(
    "A_products",
    ["implicit", "explicit"],
)
def test_blopex_implicit_and_explicit_are_correct(A_products):
    _, _, A, X0 = _diagonal_problem()

    Lambda, _, res = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products=A_products,
    )

    np.testing.assert_allclose(
        Lambda,
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.all(res[:2, -1] < 1e-6)


def test_blopex_implicit_matches_explicit():
    _, _, A, X0 = _diagonal_problem()

    Lambda_implicit, X_implicit, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="implicit",
    )

    Lambda_explicit, X_explicit, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="explicit",
    )

    np.testing.assert_allclose(
        Lambda_implicit,
        Lambda_explicit,
        rtol=1e-6,
        atol=1e-8,
    )

    X_implicit = _to_numpy(X_implicit)
    X_explicit = _to_numpy(X_explicit)

    # Eigenvector signs are arbitrary, so compare the subspace
    # projectors instead of comparing vectors entry-by-entry.
    projector_implicit = X_implicit @ X_implicit.T
    projector_explicit = X_explicit @ X_explicit.T

    np.testing.assert_allclose(
        projector_implicit,
        projector_explicit,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("A_products", "B_products"),
    [
        ("implicit", "implicit"),
        ("implicit", "explicit"),
        ("explicit", "implicit"),
        ("explicit", "explicit"),
    ],
)
def test_generalized_blopex_product_modes_are_correct(
    A_products,
    B_products,
):
    _, _, _, A, B, X0 = _generalized_diagonal_problem()

    Lambda, _, res = solver.lobpcg(
        A,
        X0,
        nev=2,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products=A_products,
        B_products=B_products,
    )

    np.testing.assert_allclose(
        Lambda,
        np.array([1.0, 2.0]),
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.all(res[:2, -1] < 1e-6)


def test_generalized_blopex_implicit_matches_explicit():
    _, B_np, _, A, B, X0 = _generalized_diagonal_problem()

    Lambda_implicit, X_implicit, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="implicit",
        B_products="implicit",
    )

    Lambda_explicit, X_explicit, _ = solver.lobpcg(
        A,
        X0,
        nev=2,
        B=B,
        method="BLOPEX",
        itmax=100,
        tol=1e-6,
        A_products="explicit",
        B_products="explicit",
    )

    np.testing.assert_allclose(
        Lambda_implicit,
        Lambda_explicit,
        rtol=1e-6,
        atol=1e-8,
    )

    X_implicit = _to_numpy(X_implicit)
    X_explicit = _to_numpy(X_explicit)

    # Transform the B-orthonormal eigenvectors into an ordinary
    # Euclidean-orthonormal basis.
    B_sqrt = np.diag(np.sqrt(np.diag(B_np)))

    Y_implicit = B_sqrt @ X_implicit
    Y_explicit = B_sqrt @ X_explicit

    # Eigenvector signs are arbitrary, so compare invariant
    # subspace projectors.
    projector_implicit = Y_implicit @ Y_implicit.T

    projector_explicit = Y_explicit @ Y_explicit.T

    np.testing.assert_allclose(
        projector_implicit,
        projector_explicit,
        rtol=1e-5,
        atol=1e-6,
    )


def test_generalized_blopex_rejects_invalid_b_products():
    _, _, _, A, B, X0 = _generalized_diagonal_problem()

    with pytest.raises(
        ValueError,
        match="B_products",
    ):
        solver.lobpcg(
            A,
            X0,
            nev=2,
            B=B,
            method="BLOPEX",
            B_products="wrong",
        )


def test_lobpcg_rejects_unknown_method():
    with pytest.raises(
        ValueError,
        match="Unknown LOBPCG method",
    ):
        solver.lobpcg(
            object(),
            object(),
            nev=2,
            method="SomethingElse",
        )


@pytest.mark.parametrize(
    "method",
    ["Basic", "Ortho", "Skip_ortho"],
)
def test_lobpcg_rejects_unimplemented_methods(method):
    with pytest.raises(
        NotImplementedError,
        match="not implemented",
    ):
        solver.lobpcg(
            object(),
            object(),
            nev=2,
            method=method,
        )

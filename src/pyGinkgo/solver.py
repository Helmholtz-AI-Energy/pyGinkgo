# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from pyGinkgo import pyGinkgoBindings as pGB
from .pyGinkgoBindings.solver import *
import pyGinkgo as pg
from . import gko_types
import numpy as np

def _right_multiply(X, C, out, beta=0.0):
    """Compute out = X @ C + beta * out.

    X and out are executor-local Ginkgo Dense matrices.
    C is a small host-side NumPy matrix.
    """
    rows, inner = tuple(X.shape)
    C = np.asarray(C)

    if C.ndim != 2:
        raise ValueError("C must be a two-dimensional matrix.")

    if C.shape[0] != inner:
        raise ValueError(
            "C.shape[0] must match X.shape[1], "
            f"got {C.shape[0]} and {inner}."
        )

    expected_out_shape = (rows, C.shape[1])
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            f"out must have shape {expected_out_shape}, "
            f"got {tuple(out.shape)}."
        )

    C = np.ascontiguousarray(C)
    C_gko = type(X)(X.get_executor(), C)

    if beta == 0.0:
        X.apply(C_gko, out)
    else:
        X.apply(1.0, C_gko, beta, out)

def _right_solve_cholesky_factor(
    X,
    L,
    work,
    transpose,
):
    """Apply the Cholesky basis transformation using a direct solve.

    Solve

        L @ Y = X.T

    and update

        X <- Y.T.

    This is mathematically equivalent to

        X <- X @ inv(L.T),

    but does not explicitly form the inverse.
    """
    n, m = tuple(X.shape)

    if L.shape != (m, m):
        raise ValueError(
            f"L must have shape {(m, m)}, got {L.shape}."
        )

    if tuple(work.shape) != (n, m):
        raise ValueError(
            f"work must have shape {(n, m)}, got {tuple(work.shape)}."
        )

    if tuple(transpose.shape) != (m, n):
        raise ValueError(
            f"transpose must have shape {(m, n)}, "
            f"got {tuple(transpose.shape)}."
        )

    # Copy X.T to the host.
    X.transpose_into(transpose)

    X_transpose = np.array(
        transpose.copy_to_host(),
        copy=True,
    ).reshape(m, n)

    # Solve L @ Y = X.T instead of explicitly constructing inv(L.T).
    X_transpose = np.linalg.solve(
        L,
        X_transpose,
    )

    work.copy_from(
        type(X)(
            X.get_executor(),
            np.ascontiguousarray(X_transpose.T),
        )
    )
    X.copy_from(work)

def _orthonormalize_blopex_standard(
    W,
    work,
    transpose,
    gram,
    AW=None,
):
    """Cholesky-orthonormalize a BLOPEX block.

    For

        G = W.T @ W = L @ L.T,

    update W by solving

        L @ Y = W.T
        W <- Y.T.

    This is equivalent to W <- W @ inv(L.T), but avoids explicitly
    forming the inverse.

    If AW is provided, apply the same transformation to AW so that it
    remains consistent with W. This is required for implicit A-products.
    """
    n, m = tuple(W.shape)

    if tuple(work.shape) != (n, m):
        raise ValueError(
            f"work must have shape {(n, m)}, got {tuple(work.shape)}."
        )

    if tuple(transpose.shape) != (m, n):
        raise ValueError(
            f"transpose must have shape {(m, n)}, "
            f"got {tuple(transpose.shape)}."
        )

    if tuple(gram.shape) != (m, m):
        raise ValueError(
            f"gram must have shape {(m, m)}, got {tuple(gram.shape)}."
        )

    if AW is not None and tuple(AW.shape) != (n, m):
        raise ValueError(
            f"AW must have shape {(n, m)}, got {tuple(AW.shape)}."
        )

    # G = W.T @ W.
    W.transpose_into(transpose)
    transpose.apply(W, gram)

    G = np.array(
        gram.copy_to_host(),
        copy=True,
    ).reshape(m, m)

    # Remove tiny loss of symmetry from floating-point operations.
    G = 0.5 * (G + G.T)

    # NumPy's linear algebra routines do not support float16.
    if G.dtype == np.float16:
        G = G.astype(np.float32)

    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX block is rank deficient during "
            "Cholesky orthonormalization."
        ) from error

    # Apply W <- W @ inv(L.T) without forming the inverse.
    #
    # Solve
    #
    #     L @ Y = W.T
    #
    # and set W <- Y.T.
    _right_solve_cholesky_factor(
        W,
        L,
        work,
        transpose,
    )
    if AW is not None:
        _right_solve_cholesky_factor(
            AW,
            L,
            work,
            transpose,
        )

def _orthonormalize_blopex_generalized(
    W,
    BW,
    work,
    transpose,
    gram,
    *,
    AW=None,
    update_BW=True,
):
    """B-orthonormalize a BLOPEX block.

    Given BW = B @ W, form

        G = W.T @ BW = L @ L.T.

    Update W by solving

        L @ Y = W.T
        W <- Y.T.

    This is equivalent to W <- W @ inv(L.T), but avoids explicitly
    forming the inverse.

    If ``update_BW`` is True, apply the same transformation to BW.

    If AW is supplied, apply the same transformation to AW.

    Updating AW is required when A-products are maintained implicitly.
    """
    n, m = tuple(W.shape)

    if tuple(BW.shape) != (n, m):
        raise ValueError(
            f"BW must have shape {(n, m)}, got {tuple(BW.shape)}."
        )

    if tuple(work.shape) != (n, m):
        raise ValueError(
            f"work must have shape {(n, m)}, got {tuple(work.shape)}."
        )

    if tuple(transpose.shape) != (m, n):
        raise ValueError(
            f"transpose must have shape {(m, n)}, "
            f"got {tuple(transpose.shape)}."
        )

    if tuple(gram.shape) != (m, m):
        raise ValueError(
            f"gram must have shape {(m, m)}, got {tuple(gram.shape)}."
        )

    if AW is not None and tuple(AW.shape) != (n, m):
        raise ValueError(
            f"AW must have shape {(n, m)}, got {tuple(AW.shape)}."
        )

    # G = W.T @ B @ W = W.T @ BW.
    W.transpose_into(transpose)
    transpose.apply(BW, gram)

    G = np.array(
        gram.copy_to_host(),
        copy=True,
    ).reshape(m, m)

    G = 0.5 * (G + G.T)

    if G.dtype == np.float16:
        G = G.astype(np.float32)

    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX block is rank deficient in the B-inner product, "
            "or B is not positive definite."
        ) from error

    # Apply W <- W @ inv(L.T) without forming the inverse.
    #
    # Solve
    #
    #     L @ Y = W.T
    #
    # and set W <- Y.T.
    _right_solve_cholesky_factor(
        W,
        L,
        work,
        transpose,
    )

    if update_BW:
        _right_solve_cholesky_factor(
            BW,
            L,
            work,
            transpose,
        )

    if AW is not None:
        _right_solve_cholesky_factor(
            AW,
            L,
            work,
            transpose,
        )


def _rayleigh_ritz_blopex_standard(
    blocks,
    a_blocks,
    transpose,
    small,
):
    """Perform the BLOPEX Rayleigh-Ritz step.

    ``blocks`` represents the trial basis V and ``a_blocks`` represents
    A @ V. For example,

        blocks   = (X,)
        blocks   = (X, Z)
        blocks   = (X, Z, P)

    Only the small projected matrices

        H = V.T @ A @ V
        G = V.T @ V

    are copied to the host.

    Returns
    -------
    hX : numpy.ndarray
        Ritz coefficient matrix. Its shape is q*m by m where q is the
        number of trial blocks.
    Lambda : numpy.ndarray
        The m smallest Ritz values.
    """
    blocks = tuple(blocks)
    a_blocks = tuple(a_blocks)

    if not blocks:
        raise ValueError("blocks must not be empty.")

    if len(blocks) != len(a_blocks):
        raise ValueError(
            "blocks and a_blocks must contain the same number of blocks."
        )

    n, m = tuple(blocks[0].shape)

    if tuple(transpose.shape) != (m, n):
        raise ValueError(
            f"transpose must have shape {(m, n)}, "
            f"got {tuple(transpose.shape)}."
        )

    if tuple(small.shape) != (m, m):
        raise ValueError(
            f"small must have shape {(m, m)}, "
            f"got {tuple(small.shape)}."
        )

    for i, (block, a_block) in enumerate(zip(blocks, a_blocks)):
        if tuple(block.shape) != (n, m):
            raise ValueError(
                f"blocks[{i}] must have shape {(n, m)}, "
                f"got {tuple(block.shape)}."
            )

        if tuple(a_block.shape) != (n, m):
            raise ValueError(
                f"a_blocks[{i}] must have shape {(n, m)}, "
                f"got {tuple(a_block.shape)}."
            )

    h_rows = []
    g_rows = []

    # Construct the projected matrices block by block. Only m-by-m
    # matrices cross from the executor to the host.
    for left in blocks:
        left.transpose_into(transpose)

        h_row = []
        g_row = []

        for right_a in a_blocks:
            transpose.apply(right_a, small)
            h_row.append(
                np.array(
                    small.copy_to_host(),
                    copy=True,
                ).reshape(m, m)
            )

        for right in blocks:
            transpose.apply(right, small)
            g_row.append(
                np.array(
                    small.copy_to_host(),
                    copy=True,
                ).reshape(m, m)
            )

        h_rows.append(h_row)
        g_rows.append(g_row)

    H = np.block(h_rows)
    G = np.block(g_rows)

    # H and G are theoretically symmetric.
    H = 0.5 * (H + H.T)
    G = 0.5 * (G + G.T)

    if H.dtype == np.float16:
        H = H.astype(np.float32)
        G = G.astype(np.float32)

    # Solve
    #
    #     H c = G c lambda.
    #
    # With G = L L.T and y = L.T c:
    #
    #     (L^-1 H L^-T) y = y lambda.
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX projected Gram matrix is rank deficient."
        ) from error

    left_reduced = np.linalg.solve(L, H)

    reduced = np.linalg.solve(
        L,
        left_reduced.T,
    ).T

    reduced = 0.5 * (reduced + reduced.T)

    Lambda, reduced_vectors = np.linalg.eigh(reduced)

    # np.linalg.eigh returns eigenvalues in ascending order. BLOPEX needs
    # the m smallest Ritz pairs.
    Lambda = Lambda[:m]
    reduced_vectors = reduced_vectors[:, :m]

    hX = np.linalg.solve(
        L.T,
        reduced_vectors,
    )

    return hX, Lambda

def _rayleigh_ritz_blopex_generalized(
    blocks,
    a_blocks,
    b_blocks,
    transpose,
    small,
):
    """Perform generalized BLOPEX Rayleigh-Ritz.

    ``blocks`` contains the trial basis blocks V,
    ``a_blocks`` contains A @ V, and
    ``b_blocks`` contains B @ V.

    The projected generalized eigenproblem is

        H c = G c lambda,

    with

        H = V.T @ A @ V
        G = V.T @ B @ V.

    Only the small projected matrices are transferred to the host.

    Returns
    -------
    hX : numpy.ndarray
        Ritz coefficient matrix with shape (q*m, m), where q is the
        number of trial blocks.
    Lambda : numpy.ndarray
        The m smallest Ritz values.
    """
    blocks = tuple(blocks)
    a_blocks = tuple(a_blocks)
    b_blocks = tuple(b_blocks)

    if not blocks:
        raise ValueError("blocks must not be empty.")

    if len(blocks) != len(a_blocks):
        raise ValueError(
            "blocks and a_blocks must contain the same number of blocks."
        )

    if len(blocks) != len(b_blocks):
        raise ValueError(
            "blocks and b_blocks must contain the same number of blocks."
        )

    n, m = tuple(blocks[0].shape)

    if tuple(transpose.shape) != (m, n):
        raise ValueError(
            f"transpose must have shape {(m, n)}, "
            f"got {tuple(transpose.shape)}."
        )

    if tuple(small.shape) != (m, m):
        raise ValueError(
            f"small must have shape {(m, m)}, "
            f"got {tuple(small.shape)}."
        )

    for i, (block, a_block, b_block) in enumerate(
        zip(blocks, a_blocks, b_blocks)
    ):
        if tuple(block.shape) != (n, m):
            raise ValueError(
                f"blocks[{i}] must have shape {(n, m)}, "
                f"got {tuple(block.shape)}."
            )

        if tuple(a_block.shape) != (n, m):
            raise ValueError(
                f"a_blocks[{i}] must have shape {(n, m)}, "
                f"got {tuple(a_block.shape)}."
            )

        if tuple(b_block.shape) != (n, m):
            raise ValueError(
                f"b_blocks[{i}] must have shape {(n, m)}, "
                f"got {tuple(b_block.shape)}."
            )

    h_rows = []
    g_rows = []

    for left in blocks:
        left.transpose_into(transpose)

        h_row = []
        g_row = []

        # H_ij = V_i.T @ A @ V_j.
        for right_a in a_blocks:
            transpose.apply(right_a, small)

            h_row.append(
                np.array(
                    small.copy_to_host(),
                    copy=True,
                ).reshape(m, m)
            )

        # G_ij = V_i.T @ B @ V_j.
        for right_b in b_blocks:
            transpose.apply(right_b, small)

            g_row.append(
                np.array(
                    small.copy_to_host(),
                    copy=True,
                ).reshape(m, m)
            )

        h_rows.append(h_row)
        g_rows.append(g_row)

    H = np.block(h_rows)
    G = np.block(g_rows)

    # Theoretically symmetric, but remove small numerical asymmetry.
    H = 0.5 * (H + H.T)
    G = 0.5 * (G + G.T)

    if H.dtype == np.float16:
        H = H.astype(np.float32)

    if G.dtype == np.float16:
        G = G.astype(np.float32)

    # Solve
    #
    #     H c = G c lambda
    #
    # using G = L L.T.
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX projected B-Gram matrix is rank deficient "
            "or not positive definite."
        ) from error

    # reduced = L^-1 H L^-T
    left_reduced = np.linalg.solve(
        L,
        H,
    )

    reduced = np.linalg.solve(
        L,
        left_reduced.T,
    ).T

    reduced = 0.5 * (reduced + reduced.T)

    Lambda, reduced_vectors = np.linalg.eigh(reduced)

    # Smallest m Ritz pairs.
    Lambda = Lambda[:m]
    reduced_vectors = reduced_vectors[:, :m]

    # c = L^-T y.
    hX = np.linalg.solve(
        L.T,
        reduced_vectors,
    )

    return hX, Lambda


def _standard_residual(
    AX,
    X,
    Lambda,
    R,
    work,
    norms,
):
    """Compute the standard-problem relative residuals.

    Computes

        R[:, i] = AX[:, i] - Lambda[i] * X[:, i]

    and returns

        ||R[:, i]||_2 / abs(Lambda[i]).
    """
    n, m = tuple(X.shape)

    if tuple(AX.shape) != (n, m):
        raise ValueError(
            f"AX must have shape {(n, m)}, got {tuple(AX.shape)}."
        )

    if tuple(R.shape) != (n, m):
        raise ValueError(
            f"R must have shape {(n, m)}, got {tuple(R.shape)}."
        )

    if tuple(work.shape) != (n, m):
        raise ValueError(
            f"work must have shape {(n, m)}, got {tuple(work.shape)}."
        )

    if tuple(norms.shape) != (1, m):
        raise ValueError(
            f"norms must have shape {(1, m)}, got {tuple(norms.shape)}."
        )

    Lambda = np.asarray(Lambda).reshape(-1)

    if Lambda.size != m:
        raise ValueError(
            f"Lambda must contain {m} values, got {Lambda.size}."
        )

    lambda_row = type(X)(
        X.get_executor(),
        np.ascontiguousarray(Lambda.reshape(1, m)),
    )

    # work[:, i] = Lambda[i] * X[:, i].
    work.copy_from(X)
    work.scale(lambda_row)

    # R = AX - X @ diag(Lambda).
    R.copy_from(AX)
    R.sub_scaled(1.0, work)

    # Column-wise Euclidean norms.
    R.compute_norm2(norms)

    residual_norms = np.array(
        norms.copy_to_host(),
        copy=True,
    ).reshape(m)

    denominator = np.abs(Lambda)

    # Avoid division by zero for zero Ritz values.
    return np.divide(
        residual_norms,
        denominator,
        out=np.full(
            m,
            np.inf,
            dtype=np.result_type(residual_norms, np.float64),
        ),
        where=denominator != 0,
    )

def _generalized_residual(
    AX,
    BX,
    Lambda,
    R,
    work,
    norms,
):
    """Compute generalized-problem relative residuals.

    Computes

        R[:, i] =
            AX[:, i] - Lambda[i] * BX[:, i]

    for the generalized eigenproblem

        A x = lambda B x,

    and returns

        ||R[:, i]||_2 / abs(Lambda[i]).
    """
    n, m = tuple(AX.shape)

    if tuple(BX.shape) != (n, m):
        raise ValueError(
            f"BX must have shape {(n, m)}, got {tuple(BX.shape)}."
        )

    if tuple(R.shape) != (n, m):
        raise ValueError(
            f"R must have shape {(n, m)}, got {tuple(R.shape)}."
        )

    if tuple(work.shape) != (n, m):
        raise ValueError(
            f"work must have shape {(n, m)}, got {tuple(work.shape)}."
        )

    if tuple(norms.shape) != (1, m):
        raise ValueError(
            f"norms must have shape {(1, m)}, got {tuple(norms.shape)}."
        )

    Lambda = np.asarray(Lambda).reshape(-1)

    if Lambda.size != m:
        raise ValueError(
            f"Lambda must contain {m} values, got {Lambda.size}."
        )

    lambda_row = type(BX)(
        BX.get_executor(),
        np.ascontiguousarray(
            Lambda.reshape(1, m)
        ),
    )

    # work[:, i] = Lambda[i] * BX[:, i].
    work.copy_from(BX)
    work.scale(lambda_row)

    # R = AX - BX @ diag(Lambda).
    R.copy_from(AX)
    R.sub_scaled(
        1.0,
        work,
    )

    R.compute_norm2(norms)

    residual_norms = np.array(
        norms.copy_to_host(),
        copy=True,
    ).reshape(m)

    denominator = np.abs(Lambda)

    return np.divide(
        residual_norms,
        denominator,
        out=np.full(
            m,
            np.inf,
            dtype=np.result_type(
                residual_norms,
                np.float64,
            ),
        ),
        where=denominator != 0,
    )

def lobpcg_basic_standard_impl_(A, X0, nev,
                          T, itmax, tol,
                          A_products):
    """
    Knyazev, A. V. (2001)
    Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate gradient method
    SIAM journal on scientific computing, 23(2), 517-541

    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    B          : right hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of wanted eigenpairs, nev <= m
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of iterations
    tol        : tolerance used for convergence criterion
    A_products : if :implicit, the matrix products with A are updated implicitly
    B_products : if :implicit, the matrix products with B are updated implicitly

    Returns:
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, shape (m, num_iterations + 1)
    Relative residual history, including the initial Ritz step.
    """
    pass

def blopex_lobpcg_standard_impl_(
    A,
    X0,
    nev,
    *,
    T=None,
    itmax=200,
    tol=1e-6,
    A_products="implicit",
):
    """
        Knyazev, A. V., Argentati, M. E., Lashuk, I., & Ovtchinnikov, E. E. (2007)
        Block locally optimal preconditioned eigenvalue Xolvers (BLOPEX) in Hypre and PETSc
        SIAM Journal on Scientific Computing, 29(5), 2224-2239.
    
        Parameters:
        A          : left  hand-side operator, symmetric positive definite, n-by-n
        X0         : initial iterates, n-by-m (m < n)
        nev        : number of wanted eigenpairs, nev <= m
        T          : precondontioner, symmetric positive definite, n-by-n
        itmax      : maximum number of iterations
        tol        : tolerance used for convergence criterion
        A_products : if :implicit, the matrix products with A are updated implicitly
            
        Returns:
        Lambda : last iterates of least dominant eigenvalues, ndarray: shape (m,)
        X      : last iterates of least dominant eigenvectors, n-by-m
        res    : normalized norms of eigenresiduals, ndarray: shape (m, num_iterations + 1)
                 Relative residual history, including the initial Ritz step.
    """
    n, m = tuple(X0.shape)

    if not 1 <= nev <= m:
        raise ValueError(
            "nev must satisfy 1 <= nev <= X0.shape[1]."
        )

    if m >= n:
        raise ValueError(
            "The block size X0.shape[1] must be smaller than "
            "X0.shape[0]."
        )

    if itmax >= 1 and 2 * m > n:
        raise ValueError(
            "BLOPEX requires 2 * X0.shape[1] <= X0.shape[0] "
            "for the first Rayleigh-Ritz step over [X, Z]."
        )

    if itmax >= 2 and 3 * m > n:
        raise ValueError(
            "BLOPEX requires 3 * X0.shape[1] <= X0.shape[0] "
            "for later Rayleigh-Ritz steps over [X, Z, P]."
        )

    if itmax < 0:
        raise ValueError(
            "itmax must be non-negative."
        )

    if tol <= 0.0:
        raise ValueError(
            "tol must be positive."
        )

    if A_products not in {"implicit", "explicit"}:
        raise ValueError(
            "A_products must be 'implicit' or 'explicit'."
        )

    # ------------------------------------------------------------------
    # BLOPEX workspaces
    # ------------------------------------------------------------------

    X = X0.clone()

    R = X0.create_with_config_of()
    Z = X0.create_with_config_of()
    P = X0.create_with_config_of()
    W = X0.create_with_config_of()

    AX = X0.create_with_config_of()
    AZ = X0.create_with_config_of()
    AP = X0.create_with_config_of()

    # Reused small-product workspaces.
    transpose = X0.create_with_type_of((m, n))
    small = X0.create_with_type_of((m, m))
    norms = X0.create_with_type_of((1, m))

    # Residual history is deliberately host-side.
    res = np.full(
        (m, itmax + 1),
        np.nan,
        dtype=np.float64,
    )

    k = 0

    # ==================================================================
    # Initialization
    # ==================================================================
    #
    # Reference:
    #
    #   X = X0
    #   XtX = X.T @ X
    #   U = chol(XtX)
    #   X = solve(U.T, X.T).T
    #
    #   AX = A @ X
    #   hX, Lambda = RR6(X, AX)
    #   X = X @ hX
    #
    #   if implicit:
    #       AX = AX @ hX
    #   else:
    #       AX = A @ X
    # ==================================================================

    _orthonormalize_blopex_standard(
        X,
        W,
        transpose,
        small,
    )

    A.apply(X, AX)

    hX, Lambda = _rayleigh_ritz_blopex_standard(
        (X,),
        (AX,),
        transpose,
        small,
    )

    # X = X @ hX.
    _right_multiply(X, hX, W)
    X.copy_from(W)

    if A_products == "implicit":
        # AX = AX @ hX.
        _right_multiply(AX, hX, W)
        AX.copy_from(W)
    else:
        A.apply(X, AX)

    # R = AX - X @ diag(Lambda).
    res[:, 0] = _standard_residual(
        AX,
        X,
        Lambda,
        R,
        W,
        norms,
    )

    # Preserve the convergence-counting behavior of the reference
    # implementation: only consecutive leading Ritz pairs are counted.
    for i in range(k, nev):
        if res[i, 0] < tol:
            k += 1
        else:
            break

    if k >= nev or itmax == 0:
        return Lambda, X, res[:, :1]

    # ==================================================================
    # BLOPEX iteration
    # ==================================================================

    for j in range(1, itmax + 1):

        # --------------------------------------------------------------
        # Z = T(R), or Z = R without a preconditioner.
        # --------------------------------------------------------------

        if T is not None:
            T.apply(R, Z)
        else:
            Z.copy_from(R)

        # Reference:
        #
        #   ZtZ = Z.T @ Z
        #   U = chol(ZtZ)
        #   Z = solve(U.T, Z.T).T
        #
        _orthonormalize_blopex_standard(
            Z,
            W,
            transpose,
            small,
        )

        A.apply(Z, AZ)

        if j == 1:
            # ==========================================================
            # First BLOPEX iteration:
            #
            #     RR over span{X, Z}
            #
            # Reference:
            #
            #   hX, Lambda = RR_BLOPEX1(X, Z, AX, AZ, Z)
            # ==========================================================

            hX, Lambda = _rayleigh_ritz_blopex_standard(
                (X, Z),
                (AX, AZ),
                transpose,
                small,
            )

            hX_X = hX[:m, :]
            hX_Z = hX[m:2 * m, :]

            # P = Z @ hX_Z.
            _right_multiply(
                Z,
                hX_Z,
                P,
            )

            if A_products == "implicit":
                # AP = AZ @ hX_Z.
                _right_multiply(
                    AZ,
                    hX_Z,
                    AP,
                )
            else:
                A.apply(P, AP)

        else:
            # ==========================================================
            # Later iterations.
            #
            # First normalize P. In implicit mode AP must undergo the
            # same transformation.
            #
            # Reference:
            #
            #   PtP = P.T @ P
            #   U = chol(PtP)
            #   P = solve(U.T, P.T).T
            #
            #   if implicit:
            #       AP = solve(U.T, AP.T).T
            #   else:
            #       AP = A @ P
            # ==========================================================

            if A_products == "implicit":
                _orthonormalize_blopex_standard(
                    P,
                    W,
                    transpose,
                    small,
                    AW=AP,
                )
            else:
                _orthonormalize_blopex_standard(
                    P,
                    W,
                    transpose,
                    small,
                )
                A.apply(P, AP)

            # ----------------------------------------------------------
            # RR over span{X, Z, P}.
            # ----------------------------------------------------------

            hX, Lambda = _rayleigh_ritz_blopex_standard(
                (X, Z, P),
                (AX, AZ, AP),
                transpose,
                small,
            )

            hX_X = hX[:m, :]
            hX_Z = hX[m:2 * m, :]
            hX_P = hX[2 * m:3 * m, :]

            # ----------------------------------------------------------
            # P = Z @ hX_Z + P @ hX_P
            #
            # W is required because P_old must survive until both
            # contributions have been evaluated.
            # ----------------------------------------------------------

            _right_multiply(
                P,
                hX_P,
                W,
            )

            _right_multiply(
                Z,
                hX_Z,
                W,
                beta=1.0,
            )

            P.copy_from(W)

            if A_products == "implicit":
                # ------------------------------------------------------
                # AP = AZ @ hX_Z + AP @ hX_P
                # ------------------------------------------------------

                _right_multiply(
                    AP,
                    hX_P,
                    W,
                )

                _right_multiply(
                    AZ,
                    hX_Z,
                    W,
                    beta=1.0,
                )

                AP.copy_from(W)

            else:
                A.apply(P, AP)

        # ==============================================================
        # X = P + X @ hX_X
        # ==============================================================

        W.copy_from(P)

        _right_multiply(
            X,
            hX_X,
            W,
            beta=1.0,
        )

        X.copy_from(W)

        if A_products == "implicit":
            # ==========================================================
            # AX = AP + AX @ hX_X
            # ==========================================================

            W.copy_from(AP)

            _right_multiply(
                AX,
                hX_X,
                W,
                beta=1.0,
            )

            AX.copy_from(W)

        else:
            A.apply(X, AX)

        # ==============================================================
        # R = AX - X @ diag(Lambda)
        # ==============================================================

        res[:, j] = _standard_residual(
            AX,
            X,
            Lambda,
            R,
            W,
            norms,
        )

        for i in range(k, nev):
            if res[i, j] < tol:
                k += 1
            else:
                break

        if k >= nev:
            return (
                Lambda,
                X,
                res[:, :j + 1],
            )

    return Lambda, X, res

def blopex_lobpcg_generalized_impl_(
    A,
    B,
    X0,
    nev,
    *,
    T=None,
    itmax=200,
    tol=1e-6,
    A_products="implicit",
    B_products="implicit",
):
    """
    Knyazev, A. V., Argentati, M. E., Lashuk, I., & Ovtchinnikov, E. E. (2007)
    Block locally optimal preconditioned eigenvalue Xolvers (BLOPEX) in Hypre and PETSc
    SIAM Journal on Scientific Computing, 29(5), 2224-2239.

    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    B          : right hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of wanted eigenpairs, nev <= m
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of BLOPEX iterations
    tol        : tolerance used for convergence criterion
    A_products : If "implicit", update A-products using the small Ritz
                transformations. If "explicit", recompute them with A.
    B_products : If "implicit", update B-products using the small Ritz
                transformations. If "explicit", recompute them with B.

    Returns:
    Lambda : Last Ritz values, shape (m,)
    X      : Last Ritz vectors, shape (n, m)
    res    : Relative residual history, shape (m, num_iterations + 1)
    """
    n, m = tuple(X0.shape)

    if not 1 <= nev <= m:
        raise ValueError(
            "nev must satisfy 1 <= nev <= X0.shape[1]."
        )

    if m >= n:
        raise ValueError(
            "The block size X0.shape[1] must be smaller than "
            "X0.shape[0]."
        )

    if itmax >= 1 and 2 * m > n:
        raise ValueError(
            "BLOPEX requires 2 * X0.shape[1] <= X0.shape[0] "
            "for the first Rayleigh-Ritz step over [X, Z]."
        )

    if itmax >= 2 and 3 * m > n:
        raise ValueError(
            "BLOPEX requires 3 * X0.shape[1] <= X0.shape[0] "
            "for later Rayleigh-Ritz steps over [X, Z, P]."
        )

    if itmax < 0:
        raise ValueError(
            "itmax must be non-negative."
        )

    if tol <= 0.0:
        raise ValueError(
            "tol must be positive."
        )

    if A_products not in {"implicit", "explicit"}:
        raise ValueError(
            "A_products must be 'implicit' or 'explicit'."
        )

    if B_products not in {"implicit", "explicit"}:
        raise ValueError(
            "B_products must be 'implicit' or 'explicit'."
        )

    # ------------------------------------------------------------------
    # BLOPEX workspaces
    # ------------------------------------------------------------------

    X = X0.clone()

    R = X0.create_with_config_of()
    Z = X0.create_with_config_of()
    P = X0.create_with_config_of()
    W = X0.create_with_config_of()

    AX = X0.create_with_config_of()
    AZ = X0.create_with_config_of()
    AP = X0.create_with_config_of()

    BX = X0.create_with_config_of()
    BZ = X0.create_with_config_of()
    BP = X0.create_with_config_of()

    transpose = X0.create_with_type_of((m, n))
    small = X0.create_with_type_of((m, m))
    norms = X0.create_with_type_of((1, m))

    res = np.full(
        (m, itmax + 1),
        np.nan,
        dtype=np.float64,
    )

    k = 0

    # ==================================================================
    # Initialization
    # ==================================================================
    #
    # Reference:
    #
    #   BX = B @ X
    #   XtBX = X.T @ BX
    #   U = chol(XtBX)
    #   X = solve(U.T, X.T).T
    #
    #   if B_products == "implicit":
    #       BX = solve(U.T, BX.T).T
    #   else:
    #       BX = B @ X
    #
    #   AX = A @ X
    #   Rayleigh-Ritz
    # ==================================================================

    B.apply(X, BX)

    _orthonormalize_blopex_generalized(
        X,
        BX,
        W,
        transpose,
        small,
        update_BW=(B_products == "implicit"),
    )

    if B_products == "explicit":
        B.apply(X, BX)

    A.apply(X, AX)

    hX, Lambda = _rayleigh_ritz_blopex_generalized(
        (X,),
        (AX,),
        (BX,),
        transpose,
        small,
    )

    # X = X @ hX.
    _right_multiply(
        X,
        hX,
        W,
    )
    X.copy_from(W)

    if A_products == "implicit":
        # AX = AX @ hX.
        _right_multiply(
            AX,
            hX,
            W,
        )
        AX.copy_from(W)
    else:
        A.apply(X, AX)

    if B_products == "implicit":
        # BX = BX @ hX.
        _right_multiply(
            BX,
            hX,
            W,
        )
        BX.copy_from(W)
    else:
        B.apply(X, BX)

    # R = AX - BX @ diag(Lambda).
    res[:, 0] = _generalized_residual(
        AX,
        BX,
        Lambda,
        R,
        W,
        norms,
    )

    # Preserve the reference implementation's locking/counting rule:
    # only consecutive leading Ritz pairs count as converged.
    for i in range(k, nev):
        if res[i, 0] < tol:
            k += 1
        else:
            break

    if k >= nev or itmax == 0:
        return (
            Lambda,
            X,
            res[:, :1],
        )

    # ==================================================================
    # BLOPEX iteration
    # ==================================================================

    for j in range(1, itmax + 1):

        # --------------------------------------------------------------
        # Z = T(R), or Z = R.
        # --------------------------------------------------------------

        if T is not None:
            T.apply(R, Z)
        else:
            Z.copy_from(R)

        # --------------------------------------------------------------
        # B-orthonormalize Z.
        #
        # Reference:
        #
        #   BZ = B @ Z
        #   ZtBZ = Z.T @ BZ
        #   U = chol(ZtBZ)
        #   Z = solve(U.T, Z.T).T
        # --------------------------------------------------------------

        B.apply(Z, BZ)

        _orthonormalize_blopex_generalized(
            Z,
            BZ,
            W,
            transpose,
            small,
            update_BW=(B_products == "implicit"),
        )

        if B_products == "explicit":
            B.apply(Z, BZ)

        A.apply(Z, AZ)

        if j == 1:
            # ==========================================================
            # First BLOPEX iteration:
            #
            #     V = [X, Z]
            #
            #     (V.T A V) hX
            #         =
            #     (V.T B V) hX Lambda
            # ==========================================================

            hX, Lambda = _rayleigh_ritz_blopex_generalized(
                (X, Z),
                (AX, AZ),
                (BX, BZ),
                transpose,
                small,
            )

            hX_X = hX[:m, :]
            hX_Z = hX[m:2 * m, :]

            # P = Z @ hX_Z.
            _right_multiply(
                Z,
                hX_Z,
                P,
            )

            if A_products == "implicit":
                # AP = AZ @ hX_Z.
                _right_multiply(
                    AZ,
                    hX_Z,
                    AP,
                )
            else:
                A.apply(P, AP)

            if B_products == "implicit":
                # BP = BZ @ hX_Z.
                _right_multiply(
                    BZ,
                    hX_Z,
                    BP,
                )
            else:
                B.apply(P, BP)

        else:
            # ==========================================================
            # B-orthonormalize P.
            #
            # Reference:
            #
            #   PtBP = P.T @ BP
            #   U = chol(PtBP)
            #   P = solve(U.T, P.T).T
            #
            # The corresponding AP / BP products have to undergo
            # exactly the same transformation in implicit mode.
            #   if A_products == "implicit":
            #       AP = solve(U.T, AP.T).T
            #
            #   if B_products == "implicit":
            #       BP = solve(U.T, BP.T).T
            # ==========================================================

            _orthonormalize_blopex_generalized(
                P,
                BP,
                W,
                transpose,
                small,
                AW=(
                    AP
                    if A_products == "implicit"
                    else None
                ),
                update_BW=(B_products == "implicit"),
            )

            if A_products == "explicit":
                A.apply(P, AP)

            if B_products == "explicit":
                B.apply(P, BP)

            # ----------------------------------------------------------
            # RR over span{X, Z, P}.
            # ----------------------------------------------------------

            hX, Lambda = _rayleigh_ritz_blopex_generalized(
                (X, Z, P),
                (AX, AZ, AP),
                (BX, BZ, BP),
                transpose,
                small,
            )

            hX_X = hX[:m, :]
            hX_Z = hX[m:2 * m, :]
            hX_P = hX[2 * m:3 * m, :]

            # ----------------------------------------------------------
            # P = Z @ hX_Z + P @ hX_P
            # ----------------------------------------------------------

            _right_multiply(
                P,
                hX_P,
                W,
            )

            _right_multiply(
                Z,
                hX_Z,
                W,
                beta=1.0,
            )

            P.copy_from(W)

            # ----------------------------------------------------------
            # AP = AZ @ hX_Z + AP @ hX_P
            # ----------------------------------------------------------

            if A_products == "implicit":
                _right_multiply(
                    AP,
                    hX_P,
                    W,
                )

                _right_multiply(
                    AZ,
                    hX_Z,
                    W,
                    beta=1.0,
                )

                AP.copy_from(W)

            else:
                A.apply(P, AP)

            # ----------------------------------------------------------
            # BP = BZ @ hX_Z + BP @ hX_P
            # ----------------------------------------------------------

            if B_products == "implicit":
                _right_multiply(
                    BP,
                    hX_P,
                    W,
                )

                _right_multiply(
                    BZ,
                    hX_Z,
                    W,
                    beta=1.0,
                )

                BP.copy_from(W)

            else:
                B.apply(P, BP)

        # ==============================================================
        # X = P + X @ hX_X
        # ==============================================================

        W.copy_from(P)

        _right_multiply(
            X,
            hX_X,
            W,
            beta=1.0,
        )

        X.copy_from(W)

        # ==============================================================
        # AX = AP + AX @ hX_X
        # ==============================================================

        if A_products == "implicit":
            W.copy_from(AP)

            _right_multiply(
                AX,
                hX_X,
                W,
                beta=1.0,
            )

            AX.copy_from(W)

        else:
            A.apply(X, AX)

        # ==============================================================
        # BX = BP + BX @ hX_X
        # ==============================================================

        if B_products == "implicit":
            W.copy_from(BP)

            _right_multiply(
                BX,
                hX_X,
                W,
                beta=1.0,
            )

            BX.copy_from(W)

        else:
            B.apply(X, BX)

        # ==============================================================
        # R = AX - BX @ diag(Lambda)
        # ==============================================================

        res[:, j] = _generalized_residual(
            AX,
            BX,
            Lambda,
            R,
            W,
            norms,
        )

        for i in range(k, nev):
            if res[i, j] < tol:
                k += 1
            else:
                break

        if k >= nev:
            return (
                Lambda,
                X,
                res[:, :j + 1],
            )

    return Lambda, X, res

def lobpcg(A, X0, nev,
           B=None, T=None, itmax=200, tol=1e-6,
           method='BLOPEX',
           A_products='implicit',
           B_products='implicit'):
    """
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) method.
      
    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of eigenvalues to compute.
    B          : right hand-side operator, symmetric positive definite, n-by-n
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of iterations
    tol        : tolerance used for convergence criterion
    method     : type of LOBPCG iterations among ('Basic', 'BLOPEX', 'Ortho', 'Skip_ortho')
    A_products : if "implicit", the matrix products with A are updated implicitly
    B_products : if "implicit", the matrix products with B are updated implicitly

    Returns:
    Lambda : ndarray, shape (m,)
    X      : last iterates of least dominant eigenvectors, n-by-m
    res : ndarray, shape (m, num_iterations + 1)
        Relative residual history including the initial Ritz step.
    """
    supported_methods = {
        "Basic",
        "BLOPEX",
        "Ortho",
        "Skip_ortho",
    }

    if method not in supported_methods:
        raise ValueError(
            f"Unknown LOBPCG method {method!r}. "
            f"Expected one of {sorted(supported_methods)}."
        )

    if method == "BLOPEX":
        if B is None:
            return blopex_lobpcg_standard_impl_(
                A,
                X0,
                nev,
                T=T,
                itmax=itmax,
                tol=tol,
                A_products=A_products,
            )

        return blopex_lobpcg_generalized_impl_(
            A,
            B,
            X0,
            nev,
            T=T,
            itmax=itmax,
            tol=tol,
            A_products=A_products,
            B_products=B_products,
        )

    raise NotImplementedError(
        f"LOBPCG method {method!r} is not implemented yet."
    )

def gmres(
    device: gko_types.DeviceType,
    matrix: pGB.LinOp,
    max_iters: int,
    krylov_dim: int,
    reduction_factor: float,
    relative_stop_mode: bool = False,
    preconditioner = None
):
    executor = pg.device(device)

     # TODO: create a better way to check the type of the matrix
    typization = type(matrix).__name__.split('_')[1:]
    if len(typization) > 0:
        gmres_cls = getattr(pGB.solver, "gmres_" + typization[0])
    else:
        raise ValueError(f"Not a known matrix type: {typization}.")
    
    args = [
        executor,
        matrix,
        # Conditionally including the preconditioner, if provided
        *([preconditioner] if preconditioner is not None else []),
        max_iters,
        krylov_dim,
        reduction_factor,
        relative_stop_mode
    ]
    
    return gmres_cls(*args)

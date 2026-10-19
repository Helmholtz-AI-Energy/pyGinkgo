# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from pyGinkgo import pyGinkgoBindings as pGB
from .pyGinkgoBindings.solver import *
import pyGinkgo as pg
from . import gko_types
import numpy as np
from scipy.linalg import solve_triangular

def _right_multiply(X, C, out, beta=0.0, *, one=None):
    """Multiply by executor-local C into independent output storage."""
    if tuple(C.shape)[0] != tuple(X.shape)[1]:
        raise ValueError("Incompatible multiplication dimensions.")

    if tuple(out.shape) != (
        tuple(X.shape)[0], tuple(C.shape)[1]
    ):
        raise ValueError("out has the wrong shape.")

    if X is out or C is out:
        raise ValueError("out must be distinct from both inputs.")

    if beta == 0.0:
        X.apply(C, out)
    elif beta == 1.0 and one is not None:
        X.apply(one, C, one, out)
    else:
        raise ValueError(
            "Use beta=0, or beta=1 with one supplied."
        )

def _apply_cholesky_transform(
    blocks,
    L,
    work,
    *,
    coefficient,
    dtype=None,
):
    """Return (transformed_blocks, replacement_scratch).

    Form the small transform L**(-T) on the host and upload it
    into the reusable coefficient buffer. Large blocks remain
    on their executor.

    Blocks and work must be distinct independent allocations.
    Callers must rebind all returned handles.
    """
    blocks = tuple(blocks)
    n, m = tuple(work.shape)

    if L.shape != (m, m):
        raise ValueError(f"L must have shape {(m, m)}.")

    if tuple(coefficient.shape) != (m, m):
        raise ValueError(
            f"coefficient must have shape {(m, m)}."
        )

    for block in blocks:
        if tuple(block.shape) != (n, m):
            raise ValueError(
                f"Each block must have shape {(n, m)}."
            )

    if len({id(block) for block in (*blocks, work)}) != (
        len(blocks) + 1
    ):
        raise ValueError("Blocks and work must be distinct.")

    C = solve_triangular(
        L,
        np.eye(m, dtype=L.dtype),
        lower=True,
    ).T

    coefficient.copy_from_numpy(
        np.ascontiguousarray(C, dtype=dtype)
    )

    updated = []

    for block in blocks:
        block.apply(coefficient, work)
        updated.append(work)
        work = block

    return tuple(updated), work

def _orthonormalize_blopex_standard(
    W,
    work,
    transpose,
    gram,
    AW=None,
    *,
    coefficient,
):
    """Cholesky-orthonormalize a BLOPEX block.

    Form G = W.T @ W on the executor and factor G = L @ L.T
    on the host. Form C = L**(-T) by solving against the identity,
    upload C into the reusable coefficient buffer, and compute
    W_new = W @ C on the executor.

    If AW is provided, also compute AW_new = AW @ C to preserve
    AW_new = A @ W_new.

    All blocks and workspaces must use the same executor and
    value type. W, work, and AW (if provided) must occupy distinct
    storage. coefficient must have shape (m, m).

    Only m-by-m matrices cross between the host and executor;
    the n-by-m blocks remain on their executor.

    Returns
    -------
    tuple
        (W_new, AW_new, work_new), where AW_new is None when AW
        is not provided. Buffers are recycled, so callers must
        rebind all returned block and scratch handles.
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
    block_dtype = G.dtype

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

    # Form the small transform C = L**(-T) on the host.
    # Apply the same C to W and its supplied companion blocks
    # on their executor, reusing the existing work buffer.
    blocks = (W,) if AW is None else (W, AW)

    updated, work = _apply_cholesky_transform(
        blocks,
        L,
        work,
        coefficient=coefficient,
        dtype=block_dtype,
    )

    return (
        updated[0],
        updated[1] if AW is not None else None,
        work,
    )

def _orthonormalize_blopex_generalized(
    W,
    BW,
    work,
    transpose,
    gram,
    *,
    coefficient,
    AW=None,
    update_BW=True,
):
    """B-orthonormalize a BLOPEX block, given BW = B @ W.

    Form G = W.T @ BW on the executor and factor G = L @ L.T
    on the host. Form C = L**(-T) by solving against the identity,
    upload C into the reusable coefficient buffer, and compute
    W_new = W @ C on the executor.

    If update_BW is True, also compute BW_new = BW @ C to preserve
    BW_new = B @ W_new. Otherwise, return BW unchanged; the caller
    must recompute B @ W_new before reusing it.

    If AW is provided, also compute AW_new = AW @ C to preserve
    AW_new = A @ W_new.

    All blocks and workspaces must use the same executor and
    value type. W, BW, work, and AW (if provided) must occupy
    distinct storage. coefficient must have shape (m, m).

    Only m-by-m matrices cross between the host and executor;
    the n-by-m blocks remain on their executor.

    Returns
    -------
    tuple
        (W_new, BW_new, AW_new, work_new), where AW_new is None
        when AW is not provided. Buffers are recycled, so callers
        must rebind all returned block and scratch handles.
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
    block_dtype = G.dtype

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

    # Form the small transform C = L**(-T) on the host.
    # Apply the same C to W and its supplied companion blocks
    # on their executor, reusing the existing work buffer.
    blocks = (W,)

    if update_BW:
        blocks += (BW,)

    if AW is not None:
        blocks += (AW,)

    updated, work = _apply_cholesky_transform(
        blocks,
        L,
        work,
        coefficient=coefficient,
        dtype=block_dtype,
    )

    it = iter(updated)
    W = next(it)

    if update_BW:
        BW = next(it)

    if AW is not None:
        AW = next(it)

    return W, BW, AW, work

def _project_blopex_upper(
    blocks, a_blocks, transpose, packed, *, b_blocks=None
):
    """Form symmetric H and G with one packed host-copy call."""
    blocks = tuple(blocks)
    a_blocks = tuple(a_blocks)
    metric_blocks = (
        blocks if b_blocks is None else tuple(b_blocks)
    )

    q = len(blocks)
    if q == 0:
        raise ValueError("blocks must not be empty.")

    if len(a_blocks) != q or len(metric_blocks) != q:
        raise ValueError(
            "All block collections must have the same length."
        )

    n, m = tuple(blocks[0].shape)

    if tuple(transpose.shape) != (m, n):
        raise ValueError("transpose has the wrong shape.")

    for collection in (blocks, a_blocks, metric_blocks):
        if any(tuple(block.shape) != (n, m)
               for block in collection):
            raise ValueError("All blocks must have shape (n, m).")

    products = q * (q + 1)
    rows, cols = tuple(packed.shape)

    if rows < products * m or cols != m:
        raise ValueError("packed is too small.")

    slot = 0

    for i, left in enumerate(blocks):
        left.transpose_into(transpose)

        for j in range(i, q):
            for right in (a_blocks[j], metric_blocks[j]):
                target = packed.create_submatrix(
                    slot * m, (slot + 1) * m, 0, m
                )
                transpose.apply(right, target)
                slot += 1

    used = packed.create_submatrix(
        0, products * m, 0, m
    )
    host = np.array(
        used.copy_to_host(), copy=True
    ).reshape(products, m, m)

    H = np.empty((q * m, q * m), dtype=host.dtype)
    G = np.empty_like(H)
    slot = 0

    for i in range(q):
        si = slice(i * m, (i + 1) * m)

        for j in range(i, q):
            sj = slice(j * m, (j + 1) * m)

            for projected in (H, G):
                value = host[slot]
                slot += 1

                if i == j:
                    projected[si, sj] = (
                        0.5 * (value + value.T)
                    )
                else:
                    projected[si, sj] = value
                    projected[sj, si] = value.T

    return H, G


def _rayleigh_ritz_blopex_standard(
    blocks,
    a_blocks,
    transpose,
    packed,
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
    H, G = _project_blopex_upper(
        blocks, a_blocks, transpose, packed
    )
    m = tuple(blocks[0].shape)[1]

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
    packed,
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
    H, G = _project_blopex_upper(
        blocks, a_blocks, transpose, packed,
        b_blocks=b_blocks,
    )
    m = tuple(blocks[0].shape)[1]

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

def _upload_ritz(hX, Lambda, storage):
    """Upload Ritz coefficients and eigenvalues together."""
    hX = np.asarray(hX)
    rows, m = hX.shape

    if m == 0 or rows % m or not 1 <= rows // m <= 3:
        raise ValueError(
            "hX must contain one to three square blocks."
        )

    if np.asarray(Lambda).shape != (m,):
        raise ValueError("Lambda must have shape (m,).")

    payload = np.empty((rows + 1, m), dtype=hX.dtype)
    payload[:rows] = hX
    payload[rows] = Lambda

    storage.create_submatrix(
        0, rows + 1, 0, m
    ).copy_from_numpy(payload)

    coefficients = tuple(
        storage.create_submatrix(i, i + m, 0, m)
        for i in range(0, rows, m)
    )
    lambda_row = storage.create_submatrix(
        rows, rows + 1, 0, m
    )

    return coefficients, lambda_row


def _standard_residual(
    AX,
    X,
    Lambda,
    R,
    lambda_row,
    norms,
):
    """Compute the standard-problem relative residuals.

    Computes

        R[:, i] = AX[:, i] - Lambda[i] * X[:, i]

    and returns

        ||R[:, i]||_2 / abs(Lambda[i]).

    lambda_row is the executor-local eigenvalue row supplied
    by _upload_ritz. No additional eigenvalue upload is needed.
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

    if tuple(lambda_row.shape) != (1, m):
        raise ValueError(
            f"lambda_row must have shape {(1, m)}, "
            f"got {tuple(lambda_row.shape)}."
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

    # R = AX - X @ diag(Lambda).
    R.copy_from(AX)
    R.sub_scaled(lambda_row, X)

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
    lambda_row,
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
    
    lambda_row is the executor-local eigenvalue row supplied
    by _upload_ritz. No additional eigenvalue upload is needed.
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

    if tuple(lambda_row.shape) != (1, m):
        raise ValueError(
            f"lambda_row must have shape {(1, m)}, "
            f"got {tuple(lambda_row.shape)}."
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

    # R = AX - BX @ diag(Lambda).
    R.copy_from(AX)
    R.sub_scaled(lambda_row, BX)

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
        T          : preconditioner, symmetric positive definite, n-by-n
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

    transpose = X0.create_with_type_of((m, n))

    # H and G each have at most six upper-triangle blocks.
    rr_packed = X0.create_with_type_of((12 * m, m))
    small = rr_packed.create_submatrix(0, m, 0, m)

    norms = X0.create_with_type_of((1, m))

    # Reused across orthonormalization calls.
    chol_coefficient = X0.create_with_type_of((m, m))

    # Three Ritz coefficient blocks plus one eigenvalue row.
    ritz_storage = X0.create_with_type_of((3 * m + 1, m))

    # Reused by accumulated GEMMs and block additions.
    one = X0.create_with_type_of((1, 1))
    one.fill(1.0)

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

    X, _, W = _orthonormalize_blopex_standard(
        X,
        W,
        transpose,
        small,
        coefficient=chol_coefficient
    )

    A.apply(X, AX)

    hX, Lambda = _rayleigh_ritz_blopex_standard(
        (X,),
        (AX,),
        transpose,
        rr_packed,
    )
    (C_X,), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

    # X = X @ C_X.
    _right_multiply(X, C_X, W)
    X, W = W, X

    if A_products == "implicit":
        # AX = AX @ C_X.
        _right_multiply(AX, C_X, W)
        AX, W = W, AX
    else:
        A.apply(X, AX)

    # R = AX - X @ diag(Lambda).
    res[:, 0] = _standard_residual(
        AX,
        X,
        Lambda,
        R,
        lambda_row,
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
            Z, R = R, Z

        # Reference:
        #
        #   ZtZ = Z.T @ Z
        #   U = chol(ZtZ)
        #   Z = solve(U.T, Z.T).T
        #
        Z, _, W = _orthonormalize_blopex_standard(
            Z,
            W,
            transpose,
            small,
            coefficient=chol_coefficient
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
                rr_packed,
            )
            (C_X, C_Z), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

            # P = Z @ C_Z.
            _right_multiply(
                Z,
                C_Z,
                P,
                one=one
            )

            if A_products == "implicit":
                # AP = AZ @ C_Z.
                _right_multiply(
                    AZ,
                    C_Z,
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
                P, AP, W = _orthonormalize_blopex_standard(
                    P,
                    W,
                    transpose,
                    small,
                    AW=AP,
                    coefficient=chol_coefficient,
                )
            else:
                P, _, W = _orthonormalize_blopex_standard(
                    P,
                    W,
                    transpose,
                    small,
                    coefficient=chol_coefficient,
                )
                A.apply(P, AP)

            # ----------------------------------------------------------
            # RR over span{X, Z, P}.
            # ----------------------------------------------------------

            hX, Lambda = _rayleigh_ritz_blopex_standard(
                (X, Z, P),
                (AX, AZ, AP),
                transpose,
                rr_packed,
            )
            (C_X, C_Z, C_P), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

            # ----------------------------------------------------------
            # P = Z @ C_Z + P @ C_P
            #
            # W is required because P_old must survive until both
            # contributions have been evaluated.
            # ----------------------------------------------------------

            _right_multiply(
                P,
                C_P,
                W,
            )

            _right_multiply(
                Z,
                C_Z,
                W,
                beta=1.0,
                one=one
            )

            P, W = W, P

            if A_products == "implicit":
                # ------------------------------------------------------
                # AP = AZ @ C_Z + AP @ C_P
                # ------------------------------------------------------

                _right_multiply(
                    AP,
                    C_P,
                    W,
                )

                _right_multiply(
                    AZ,
                    C_Z,
                    W,
                    beta=1.0,
                    one=one
                )

                AP, W = W, AP

            else:
                A.apply(P, AP)

        # ==============================================================
        # X = P + X @ C_X
        # ==============================================================

        _right_multiply(X, C_X, W)
        W.add_scaled(one, P)
        X, W = W, X

        if A_products == "implicit":
            # ==========================================================
            # AX = AP + AX @ C_X
            # ==========================================================

            _right_multiply(AX, C_X, W)
            W.add_scaled(one, AP)
            AX, W = W, AX
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
            lambda_row,
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
    T          : preconditioner, symmetric positive definite, n-by-n
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

    # H and G each have at most six upper-triangle blocks.
    rr_packed = X0.create_with_type_of((12 * m, m))
    small = rr_packed.create_submatrix(0, m, 0, m)

    norms = X0.create_with_type_of((1, m))

    # Reused across orthonormalization calls.
    chol_coefficient = X0.create_with_type_of((m, m))

    # Three Ritz coefficient blocks plus one eigenvalue row.
    ritz_storage = X0.create_with_type_of((3 * m + 1, m))

    # Reused by accumulated GEMMs and block additions.
    one = X0.create_with_type_of((1, 1))
    one.fill(1.0)

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

    X, BX, _, W = _orthonormalize_blopex_generalized(
        X,
        BX,
        W,
        transpose,
        small,
        coefficient=chol_coefficient,
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
        rr_packed
    )
    (C_X,), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

    # X = X @ C_X.
    _right_multiply(
        X,
        C_X,
        W,
    )
    X, W = W, X

    if A_products == "implicit":
        # AX = AX @ C_X.
        _right_multiply(
            AX,
            C_X,
            W,
        )
        AX, W = W, AX
    else:
        A.apply(X, AX)

    if B_products == "implicit":
        # BX = BX @ C_X.
        _right_multiply(
            BX,
            C_X,
            W,
        )
        BX, W = W, BX
    else:
        B.apply(X, BX)

    # R = AX - BX @ diag(Lambda).
    res[:, 0] = _generalized_residual(
        AX,
        BX,
        Lambda,
        R,
        lambda_row,
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
            Z, R = R, Z

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

        Z, BZ, _, W = _orthonormalize_blopex_generalized(
            Z,
            BZ,
            W,
            transpose,
            small,
            coefficient=chol_coefficient,
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
                rr_packed
            )
            (C_X, C_Z), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

            # P = Z @ C_Z.
            _right_multiply(
                Z,
                C_Z,
                P,
                one=one
            )

            if A_products == "implicit":
                # AP = AZ @ C_Z.
                _right_multiply(
                    AZ,
                    C_Z,
                    AP,
                )
            else:
                A.apply(P, AP)

            if B_products == "implicit":
                # BP = BZ @ C_Z.
                _right_multiply(
                    BZ,
                    C_Z,
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

            P, BP, normalized_AP, W = _orthonormalize_blopex_generalized(
                P,
                BP,
                W,
                transpose,
                small,
                coefficient=chol_coefficient,
                AW=AP if A_products == "implicit" else None,
                update_BW=(B_products == "implicit"),
            )

            if A_products == "implicit":
                AP = normalized_AP
            else:
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
                rr_packed,
            )
            (C_X, C_Z, C_P), lambda_row = _upload_ritz(hX, Lambda, ritz_storage)

            # ----------------------------------------------------------
            # P = Z @ C_Z + P @ C_P
            # ----------------------------------------------------------

            _right_multiply(
                P,
                C_P,
                W,
            )

            _right_multiply(
                Z,
                C_Z,
                W,
                beta=1.0,
                one=one
            )

            P, W = W, P

            # ----------------------------------------------------------
            # AP = AZ @ C_Z + AP @ C_P
            # ----------------------------------------------------------

            if A_products == "implicit":
                _right_multiply(
                    AP,
                    C_P,
                    W,
                )

                _right_multiply(
                    AZ,
                    C_Z,
                    W,
                    beta=1.0,
                    one=one
                )

                AP, W = W, AP

            else:
                A.apply(P, AP)

            # ----------------------------------------------------------
            # BP = BZ @ C_Z + BP @ C_P
            # ----------------------------------------------------------

            if B_products == "implicit":
                _right_multiply(
                    BP,
                    C_P,
                    W,
                )

                _right_multiply(
                    BZ,
                    C_Z,
                    W,
                    beta=1.0,
                    one=one
                )

                BP, W = W, BP

            else:
                B.apply(P, BP)

        # ==============================================================
        # X = P + X @ C_X
        # ==============================================================

        _right_multiply(X, C_X, W)
        W.add_scaled(one, P)
        X, W = W, X

        # ==============================================================
        # AX = AP + AX @ C_X
        # ==============================================================

        if A_products == "implicit":
            _right_multiply(AX, C_X, W)
            W.add_scaled(one, AP)
            AX, W = W, AX
        else:
            A.apply(X, AX)

        # ==============================================================
        # BX = BP + BX @ C_X
        # ==============================================================

        if B_products == "implicit":
            _right_multiply(BX, C_X, W)
            W.add_scaled(one, BP)
            BX, W = W, BX
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
            lambda_row,
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

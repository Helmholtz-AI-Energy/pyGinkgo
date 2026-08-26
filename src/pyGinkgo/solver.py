# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from pyGinkgo import pyGinkgoBindings as pGB
from .pyGinkgoBindings.solver import *
import pyGinkgo as pg
from . import gko_types
import numpy as np


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
    res    : normalized norms of eigenresiduals, m-by-it
    """
    pass

def lobpcg_blopex_standard_impl_(A, X0, nev, *, 
                           T=None, itmax=200, tol=1e-6,
                           A_products="explicit"):
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
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, m-by-it
    """
    pass

def lobpcg(A, X0, nev,
           B=None, T=None, itmax=200, tol=1e-6,
           method='Skip_ortho',
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
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, m-by-it
    """
    pass

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

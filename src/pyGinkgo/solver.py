# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

from pyGinkgo import pyGinkgoBindings as pGB
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

  n, m = X0.shape

  X = empty((n, m)) # dense ?
  R = empty((n, m))
  Z = empty((n, m))
  P = empty((n, m))
  W = empty((n, m))
  AX = empty((n, m))
  AZ = empty((n, m))
  AP = empty((n, m))

  res = empty((m, itmax + 1))
  k = 0

  copyto(X, X0)
  # @ - matrix multiply
  # use apply
  AX[:] = A @ X
  # what is RR1 is it from Rayleigh Ritz, what alternative do we have
  hX, Lambda = RR1(X, AX, X)
  # use apply / or dot?
  matmul(X, hX, out=W);
  copyto(X, W) # X[:] = X @ hX
  if A_products == 'implicit':
    matmul(AX, hX, out=W); copyto(AX, W) # AX[:] = AX @ hX
  else:
    AX[:] = A @ X

  # R[:] = AX - X @ diag(Lambda)
  copyto(R, AX)
  R[:] = dgemm(alpha=-1.0, a=X, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
  res[:, 0] = norm(R, axis=0)
  res[:, 0] /= abs(Lambda)
  for i in range(k, nev):
    if res[i, 0] < tol:
      k += 1
    else:
      break
  print("it = 0, k = %d" % k)
  print("extrema(res) =", res[:, 0].min(), res[:, 0].max())

  _AX = AX[:, k:m]
  _R = R[:, k:m]; _W = W[:, k:m]
  _Z = Z[:, k:m]; _AZ = AZ[:, k:m]
  _P = P[:, k:m]; _AP = AP[:, k:m]

  if k < nev:
    for j in range(1, itmax + 1):
      print("it = %d, k = %d" % (j, k))
      if T is not None:
        _Z[:] = T(_R)
      else:
        _Z[:] = _R
      _AZ[:] = A @ _Z
      if j == 1:
        hX, Lambda = RR2(X, _Z, _AX, _AZ, _Z)
        hX_X = hX[:m, :]
        hX_Z = hX[m:2*m-k, :]
        _hX_Z = hX_Z[:, k:m]
        _P[:] = matmul(_Z, _hX_Z) # _P[:] = _Z @ _hX_Z
        if A_products == 'implicit':
          _AP[:] = matmul(_AZ, _hX_Z) # _AP[:] = _AZ @ _hX_Z
        else:
          _AP[:] = A @ _P
      else:
        hX, Lambda = RR3(X, _Z, _P, AX, _AZ, _AP, _Z, _P)
        hX_X = hX[:m, :]
        hX_Z = hX[m:2*m-k, :]
        hX_P = hX[2*m-k:3*m-2*k, :]
        _hX_Z = hX_Z[:, k:m]
        _hX_P = hX_P[:, k:m]
        # _P[:] = _Z @ _hX_Z + _P @ _hX_P
        _W[:] = matmul(_P, _hX_P)
        _W[:] = dgemm(alpha=1.0, a=_Z, b=_hX_Z, c=_W, overwrite_c=True, beta=1.0)
        copyto(_P, _W)
        if A_products == 'implicit':
          # _AP[:] = _AZ @ _hX_Z + _AP @ _hX_P
          _W[:] = matmul(_AP, _hX_P)
          _W[:] = dgemm(alpha=1.0, a=_AZ, b=_hX_Z, c=_W, overwrite_c=True, beta=1.0)
          copyto(_AP, _W)
        else:
          _AP[:] = A @ _P
      if k > 0:
        if j == 1:
          # X[:] = X @ hX_X + _Z @ hX_Z
          matmul(_Z, hX_Z, out=W)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(X, W)
        else:
          # X[:] = X @ hX_X + _Z @ hX_Z + _P @ hX_P
          matmul(_P, hX_P, out=W)
          W[:] = dgemm(alpha=1.0, a=_Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(X, W)
        AX[:] = A @ X
      else:
        # X[:] = P + X @ hX_X
        copyto(W, P)
        W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        copyto(X, W)
        if A_products == 'implicit':
          # AX[:] = AP + AX @ hX_X
          copyto(W, AP)
          W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(AX, W)
        else:
          AX[:] = A @ X

      # R[:] = AX - X @ diag(Lambda)
      copyto(R, AX)
      R[:] = dgemm(alpha=-1.0, a=X, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
      res[:, j] = norm(R, axis=0)
      res[:, j] /= abs(Lambda)
      print("extrema(res) =", res[:, j].min(), res[:, j].max())

      for i in range(k, nev):
        if res[i, j] < tol:
          k += 1
        else:
          break

      if k >= nev:
        return Lambda, X, res[:, :j+1]

      if m - k < _R.shape[1]:
        _AX = AX[:, k:m]
        _R = R[:, k:m]; _W = W[:, k:m]
        _Z = Z[:, k:m]; _AZ = AZ[:, k:m]
        _P = P[:, k:m]; _AP = AP[:, k:m]

  return Lambda, X, res


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
    
  if B is not None:
    # if method == 'Basic':
    #   Lambda, X, res = Basic_LOBPCG_gen(A, B, X0, nev,
    #                                     T=T, itmax=itmax, tol=tol,
    #                                     A_products=A_products,
    #                                     B_products=B_products)
    # elif method == 'BLOPEX':
    #   Lambda, X, res = BLOPEX_LOBPCG_gen(A, B, X0, nev,
    #                                      T=T, itmax=itmax, tol=tol,
    #                                      A_products=A_products,
    #                                      B_products=B_products)
    # elif method == 'Ortho':
    #   Lambda, X, res = Ortho_LOBPCG_gen(A, B, X0, nev,
    #                                     T=T, itmax=itmax, tol=tol,
    #                                     A_products=A_products,
    #                                     B_products=B_products)
    # elif method == 'Skip_ortho':
    #   Lambda, X, res = Skip_ortho_LOBPCG_gen(A, B, X0, nev,
    #                                          T=T, itmax=itmax, tol=tol,
    #                                          A_products=A_products,
    #                                          B_products=B_products)
  else:
    if method == 'Basic':
      return lobpcg_basic_standard_impl_(A, X0, nev,
                                             T=T, itmax=itmax, tol=tol,
                                             A_products=A_products)
  #   elif method == 'BLOPEX':
  #     Lambda, X, res = BLOPEX_LOBPCG_standard(A, X0, nev,
  #                                             T=T, itmax=itmax, tol=tol,
  #                                             A_products=A_products)
  #   elif method == 'Ortho':
  #     Lambda, X, res = Ortho_LOBPCG_standard(A, X0, nev,
  #                                            T=T, itmax=itmax, tol=tol,
  #                                            A_products=A_products)
  #   elif method == 'Skip_ortho':
  #     Lambda, X, res = Skip_ortho_LOBPCG_standard(A, X0, nev,
  #                                                 T=T, itmax=itmax, tol=tol,
  #                                                 A_products=A_products)

  # return Lambda, X, res

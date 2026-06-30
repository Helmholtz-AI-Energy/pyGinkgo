# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import json
import numpy as np
from typing import Optional, Union

from . import gko_types
import pyGinkgo as pg
from pyGinkgo import pyGinkgoBindings as pGB

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False


# TODO: add tests for the functions in this file

def as_array(obj, device: gko_types.DeviceType = "cpu", dtype="float"):
    """create a ginkgo array from a given object"""
    if not dtype in gko_types.dtype:
        raise ValueError(
            f"Not a valid dtype: {dtype}. " +
            "Possible choices are: " +
            ', '.join(t for t in gko_types.dtype)
        )
    
    executor = pg.device(device)
    
    array_cls = getattr(pGB.base, "array_" + dtype)
    return array_cls(executor, obj)


def as_tensor(
    obj = None,
    dim: Optional[tuple] = None,
    device: gko_types.DeviceType = "cpu",
    dtype: Union[gko_types.ValueType, str] = "float",
    fill: Optional[float] = None,
):
    """create a ginkgo array from a given object"""
    dtype = str(dtype)
    if dtype not in gko_types.ValueType.values():
        raise ValueError(
            f"Not a valid dtype: {dtype}. " +
            "Possible choices are: " +
            ', '.join(t for t in gko_types.ValueType)
        )
    
    executor = pg.device(device)

    if torch_avail:
        if isinstance(obj, torch.Tensor):
            obj = obj.__array__()

    array_cls = getattr(pGB.matrix, "dense_" + dtype)
    # Check explicitly for None because obj may contain a multi-element NumPy array.
    if obj is not None:
        return array_cls(executor, obj)

    if dim is None:
        raise ValueError("Either obj or dim must be provided.")
    
    res = array_cls(executor, dim)
    if fill is not None:
        res.fill(fill)
        
    return res


def read(
    path: Union[str, bytes, os.PathLike],
    format: Union[gko_types.MatrixFormat, str] = "dense",
    dtype: Union[gko_types.ValueType, str] = "double",
    itype: Union[gko_types.IndexType, str] = "int32",
    device: gko_types.DeviceType = "cpu",
):
    """Read a matrix from a file

    Parameters: path - The path to the file
                format - The format of the file, eg. dense, Csr, Coo
                dtype - The data type of the matrix, eg. float, double, etc.
                itype - The index type of the matrix, eg. int32, int64, etc.
                device - The device to use for the matrix
    Returns: the matrix
    """

    # Processing filepath
    filepath = os.path.abspath(path)
    
    executor = pg.device(device)

    # Checking if the format is valid
    if format not in gko_types.MatrixFormat.values():
        raise ValueError(
            f"Not a valid matrix format: {format}. " +
            "Possible choices are: " +
            ', '.join(t for t in gko_types.MatrixFormat)
        )

    # Checking if the format is dtype
    if dtype not in gko_types.ValueType.values():
        raise ValueError(
            f"Not a valid dtype: {dtype}. " +
            "Possible choices are: " +
            ', '.join(t for t in gko_types.ValueType)
        )

    # Processing format
    if format == "dense":
        read_func = getattr(pGB.matrix, f"read_dense_{dtype}")
    else:
        # Checking if the itype is valid
        if itype not in gko_types.IndexType.values():
            raise ValueError(
                f"Not a valid itype: {itype}. " +
                "Possible choices are: " +
                ', '.join(t for t in gko_types.IndexType)
            )

        read_func = getattr(pGB.matrix, f"read_{format}_{dtype}_{itype}")

    return read_func(filepath, executor)


def factor(A, kind="Upper", device: Union[str, pGB.Executor] = "cpu"):
    if isinstance(device, str):
        executor = pg.device(device)
    else:
        executor = device
    
    factorization = pGB.factorization.factorization(executor, A)
    if kind == "Upper":
        return factorization.get_upper_factor()
    if kind == "Lower":
        return factorization.get_lower_factor()


def eigen_solve(A,solver_args=None):
    exec_obj = A.get_executor()
    torchA = torch.as_tensor(np.array(A))
    dtype = type(A).__name__.split('_')[1]
    L, Q = torch.linalg.eigh(torchA)
    dense_cls = getattr(pGB.matrix, f"dense_float")
    Lambda = dense_cls(exec_obj, L.__array__())
    hY = dense_cls(exec_obj, Q.__array__())
    return Lambda, hY

def get_solver_default_config():
    """Return the default solver configuration.

    A new dictionary is returned on each call to avoid sharing mutable
    default configuration between function calls.
    """
    return {
        "type": "solver::Gmres",
        "preconditioner": {
            "type": "preconditioner::Ilu",
            "reverse_apply": False,
            "factorization": {"type": "factorization::ParIlu"},
        },
        "criteria": [
            {"type": "Iteration", "max_iters": 1000},
            {"type": "ResidualNorm", "reduction_factor": 1e-7},
        ],
    }

def generate_solver(A, solver_args: Optional[dict] = None):
    """Generate a solver based on the system matrix A

    Parameters: A - The system matrix
                solver_args - An optional dictionary containing 
                    arguments forwarded to the solver, 
                    for example: {"type": "solver::Cg", "criteria": [{"type": "Iteration", "max_iters": 100}]}.
    Returns: the solver
    """

    if solver_args is None or solver_args == {}:
        solver_args = get_solver_default_config()
    
    solver_executor = A.get_executor()
     # TODO: Create a better way to check the dtype of the matrix
    dtype = type(A).__name__.split('_')[1]
    solver_cls = getattr(pGB.solver, "config_solver_" + dtype)
    solver = solver_cls(
        solver_executor, A, json.dumps(solver_args)
    )
    return solver

def config_solve(A, b, x, solver_args: Optional[dict] = None):
    if solver_args is None or solver_args == {}:
        solver_args = get_solver_default_config()

    solver_executor = A.get_executor()
    dtype = type(A).__name__.split('_')[1]
    # TODO: Create a better way to check the dtype of the matrix
    solver_cls = getattr(pGB.solver, "config_solve_" + dtype)
    logger = solver_cls(
        solver_executor, A, b, x, json.dumps(solver_args)
    )

    return logger, x

def triangular_solve(A,b,x,solver_args):
    kind = solver_args["type"]
    dtype = type(A).__name__.split('_')[1]
    itype = type(A).__name__.split('_')[2] # This might fix the TODO
    s = f"{kind}Trs_{dtype}_int32" # TODO why does it fail for + itype
    ctor = getattr(pGB.solver, s)
    exec_obj = A.get_executor()
    trs = ctor(exec_obj, A)
    trs.apply(b, x)
    return None, x

def solve(A, b, initial_guess=None, solver_args: Optional[dict] = None, kind="config"):
    """Solve a given linear system, where A is the system matrix and b the RHS

    Parameters: A - The system matrix
                b - The right hand side vector
                initial_guess - The initial guess
                solver_args - An optional dictionary forwarded to the solver, 
                eg {'type': 'solver::Cg', 'criteria': [{"type": "Iteration", "max_iters": 100}]}
                kind - the underlying solver, eg. config
    Returns: tuple of a logger object and solution vector
    """

    if solver_args is None or solver_args == {}:
        solver_args = get_solver_default_config()

    ctor = globals()[kind+"_solve"]

    if initial_guess is None:
        dtype = type(A).__name__.split('_')[1]
        dense_cls = getattr(pGB.matrix, f"dense_{dtype}")
        dim = (A.shape[1], b.shape[1])
        initial_guess = dense_cls(b.get_executor(), dim)
        initial_guess.fill(0.0)

    return ctor(A,b,x=initial_guess,solver_args=solver_args)

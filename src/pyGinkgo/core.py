# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import json
import copy
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

_STRING_DTYPE_ALIASES = {
    "half": "half",
    "float16": "half",
    "float": "float",
    "float32": "float",
    "single": "float",
    "double": "double",
    "float64": "double",
    "int32": "int32",
    "int64": "int64",
    "longlong": "int64",
}


def _dtype_values(allowed_types):
    if hasattr(allowed_types, "values"):
        return allowed_types.values()
    return [str(dtype) for dtype in allowed_types]


def _dtype_choices(allowed_types):
    return ", ".join(_dtype_values(allowed_types))


def _numpy_to_gko_map(allowed_types):
    values = set(_dtype_values(allowed_types))
    mapping = {}
    if values.intersection(gko_types.ValueType.values()):
        mapping.update(gko_types.NUMPY_TO_GKO_VALUE)
    if values.intersection(gko_types.IndexType.values()):
        mapping.update(gko_types.NUMPY_TO_GKO_INDEX)
    return mapping


def _normalize_numpy_dtype(dtype, allowed_types):
    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return None

    dtype_name = _numpy_to_gko_map(allowed_types).get(np_dtype.type)
    if dtype_name in _dtype_values(allowed_types):
        return dtype_name
    return None


def _normalize_torch_dtype(dtype, allowed_types):
    if not torch_avail:
        return None

    torch_map = {
        torch.float16: "half",
        torch.float32: "float",
        torch.float64: "double",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    dtype_name = torch_map.get(dtype)
    if dtype_name in _dtype_values(allowed_types):
        return dtype_name
    return None


def _normalize_dtype(dtype, allowed_types, name):
    if dtype is None:
        return None

    dtype_values = _dtype_values(allowed_types)
    torch_dtype = _normalize_torch_dtype(dtype, allowed_types)
    if torch_dtype is not None:
        return torch_dtype

    if isinstance(dtype, str):
        dtype_name = _STRING_DTYPE_ALIASES.get(dtype.lower(), dtype)
        if dtype_name in dtype_values:
            return dtype_name

    numpy_dtype = _normalize_numpy_dtype(dtype, allowed_types)
    if numpy_dtype is not None:
        return numpy_dtype

    raise ValueError(
        f"Not a valid {name}: {dtype}. "
        f"Possible choices are: {_dtype_choices(allowed_types)}"
    )


def _try_normalize_dtype(dtype, allowed_types):
    try:
        return _normalize_dtype(dtype, allowed_types, "dtype")
    except ValueError:
        return None


def _try_infer_dtype(obj, allowed_types):
    try:
        return _infer_dtype(obj, allowed_types, name="input")
    except ValueError:
        return None
    

def _normalize_value_dtype(dtype):
    return _normalize_dtype(dtype, gko_types.ValueType, "dtype")


def _normalize_array_dtype(dtype):
    return _normalize_dtype(dtype, gko_types.dtype, "dtype")


def _normalize_index_dtype(dtype):
    return _normalize_dtype(dtype, gko_types.IndexType, "itype")


def _dtype_from_binding_name(obj, allowed_types):
    type_name = type(obj).__name__
    parts = type_name.split("_")
    if len(parts) < 2:
        return None

    dtype_name = parts[1]
    if dtype_name in _dtype_values(allowed_types):
        return dtype_name
    return None


def _dtype_from_array_protocol(obj, allowed_types):
    for protocol_name in ("__cuda_array_interface__", "__array_interface__"):
        if not hasattr(obj, protocol_name):
            continue
        typestr = getattr(obj, protocol_name).get("typestr")
        dtype_name = _normalize_numpy_dtype(typestr, allowed_types)
        if dtype_name is not None:
            return dtype_name

    return None


def _infer_dtype(obj, allowed_types, *, name):
    if obj is None:
        raise ValueError(
            f"Cannot infer dtype for {name}. Please specify dtype. "
            f"Possible choices are: {_dtype_choices(allowed_types)}"
        )

    binding_dtype = _dtype_from_binding_name(obj, allowed_types)
    if binding_dtype is not None:
        return binding_dtype

    if hasattr(obj, "dtype"):
        dtype_name = _try_normalize_dtype(obj.dtype, allowed_types)
        if dtype_name is not None:
            return dtype_name

    protocol_dtype = _dtype_from_array_protocol(obj, allowed_types)
    if protocol_dtype is not None:
        return protocol_dtype

    try:
        np_array = np.asarray(obj)
    except Exception:
        np_array = None

    if (
        np_array is not None
        and np_array.size > 0
        and np_array.dtype != np.dtype("O")
    ):
        dtype_name = _normalize_numpy_dtype(np_array.dtype, allowed_types)
        if dtype_name is not None:
            return dtype_name

    raise ValueError(
        f"Cannot infer dtype for {name}. Please specify dtype. "
        f"Possible choices are: {_dtype_choices(allowed_types)}"
    )


def _infer_sparse_value_dtype(matrix_format, obj, data, dtype):
    if dtype is not None:
        return _normalize_value_dtype(dtype)

    source = data
    if source is None and obj is not None and hasattr(obj, "data"):
        source = obj.data
    if source is None:
        source = obj

    return _infer_dtype(
        source,
        gko_types.ValueType,
        name=f"{matrix_format} value input",
    )


def _sparse_index_sources(matrix_format, obj, cols, rows):
    if cols is not None or rows is not None:
        return [source for source in (cols, rows) if source is not None]

    if obj is None:
        return []

    if matrix_format == "Csr":
        return [
            getattr(obj, name)
            for name in ("indices", "indptr")
            if hasattr(obj, name)
        ]

    return [
        getattr(obj, name)
        for name in ("col", "row")
        if hasattr(obj, name)
    ]


def _infer_sparse_index_dtype(matrix_format, obj, cols, rows, itype):
    if itype is not None:
        return _normalize_index_dtype(itype)

    inferred = [
        dtype
        for dtype in (
            _try_infer_dtype(source, gko_types.IndexType)
            for source in _sparse_index_sources(matrix_format, obj, cols, rows)
        )
        if dtype is not None
    ]

    if not inferred:
        raise ValueError(
            f"Cannot infer itype for {matrix_format} index input. "
            "Please specify itype. "
            f"Possible choices are: {_dtype_choices(gko_types.IndexType)}"
        )

    if len(set(inferred)) > 1:
        raise ValueError(
            f"Cannot infer a single itype for {matrix_format} index input. "
            f"Found: {', '.join(sorted(set(inferred)))}. "
            "Please specify itype."
        )

    return inferred[0]


def _require_sparse_components(matrix_format, dim, data, cols, rows):
    missing = [
        name
        for name, value in (
            ("dim", dim),
            ("data", data),
            ("cols", cols),
            ("rows", rows),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{matrix_format} component construction requires dim, data, "
            f"cols, and rows. Missing: {', '.join(missing)}."
        )


def array(obj, device: gko_types.DeviceType = "cpu", dtype=None):
    """Create a Ginkgo array, inferring dtype from obj when possible."""
    if dtype is None and isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        raise ValueError(
            "Cannot infer dtype for array size allocation. "
            "Please specify dtype."
        )

    dtype = (
        _infer_dtype(obj, gko_types.dtype, name="array input")
        if dtype is None
        else _normalize_array_dtype(dtype)
    )
    executor = pg.device(device)
    array_cls = getattr(pGB.base, "array_" + dtype)
    return array_cls(executor, obj)


def dense(
    obj=None,
    dim: Optional[tuple] = None,
    device: gko_types.DeviceType = "cpu",
    dtype=None,
    fill: Optional[float] = None,
    stride: Optional[int] = None,
):
    """Create a Ginkgo dense matrix, inferring dtype from obj when possible."""
    if dtype is None:
        if obj is None:
            raise ValueError(
                "Cannot infer dtype for dense allocation. "
                "Please specify dtype."
            )
        dtype = _infer_dtype(obj, gko_types.ValueType, name="dense input")
    else:
        dtype = _normalize_value_dtype(dtype)

    executor = pg.device(device)
    dense_cls = getattr(pGB.matrix, "dense_" + dtype)

    if torch_avail and isinstance(obj, torch.Tensor):
        obj = obj.__array__()
    elif isinstance(obj, tuple):
        obj = np.asarray(obj)

    if obj is None:
        if dim is None:
            res = dense_cls(executor)
        elif stride is None:
            res = dense_cls(executor, dim)
        else:
            res = dense_cls(executor, dim, stride)
    elif dim is None:
        res = dense_cls(executor, obj)
    else:
        if stride is None:
            raise ValueError(
                "dense construction with obj and dim requires stride."
            )
        res = dense_cls(executor, dim, obj, stride)

    if fill is not None:
        res.fill(fill)

    return res


def _sparse_matrix(
    matrix_format,
    obj=None,
    *,
    device: gko_types.DeviceType = "cpu",
    dtype=None,
    itype=None,
    dim=None,
    data=None,
    cols=None,
    rows=None,
):
    component_args = (dim, data, cols, rows)
    use_components = any(value is not None for value in component_args)

    if obj is not None and use_components:
        raise ValueError(
            f"Pass either a {matrix_format}-like object or component "
            "arrays, not both."
        )

    if obj is None and not use_components:
        if dtype is None:
            raise ValueError(
                f"Cannot infer dtype for {matrix_format} allocation. "
                "Please specify dtype."
            )
        if itype is None:
            raise ValueError(
                f"Cannot infer itype for {matrix_format} allocation. "
                "Please specify itype."
            )

    if use_components:
        _require_sparse_components(matrix_format, dim, data, cols, rows)

    dtype = _infer_sparse_value_dtype(matrix_format, obj, data, dtype)
    itype = _infer_sparse_index_dtype(matrix_format, obj, cols, rows, itype)

    executor = pg.device(device)
    matrix_cls = getattr(pGB.matrix, f"{matrix_format}_{dtype}_{itype}")

    if use_components:
        return matrix_cls(executor, dim, data, cols, rows)
    if obj is not None:
        return matrix_cls(executor, obj)
    return matrix_cls(executor)


def Csr(
    obj=None,
    *,
    device: gko_types.DeviceType = "cpu",
    dtype=None,
    itype=None,
    dim=None,
    data=None,
    cols=None,
    rows=None,
):
    """Create a Ginkgo CSR matrix, inferring dtype and itype when possible."""
    return _sparse_matrix(
        "Csr",
        obj,
        device=device,
        dtype=dtype,
        itype=itype,
        dim=dim,
        data=data,
        cols=cols,
        rows=rows,
    )


def Coo(
    obj=None,
    *,
    device: gko_types.DeviceType = "cpu",
    dtype=None,
    itype=None,
    dim=None,
    data=None,
    cols=None,
    rows=None,
):
    """Create a Ginkgo COO matrix, inferring dtype and itype when possible."""
    return _sparse_matrix(
        "Coo",
        obj,
        device=device,
        dtype=dtype,
        itype=itype,
        dim=dim,
        data=data,
        cols=cols,
        rows=rows,
    )


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

def generate_solver(A, solver_args: dict = get_solver_default_config()):
    """Generate a solver based on the system matrix A

    Parameters: A - The system matrix
                solver_args - An optional dictionary containing 
                    arguments forwarded to the solver, 
                    for example: {"type": "solver::Cg", "criteria": [{"type": "Iteration", "max_iters": 100}]}.
    Returns: the solver
    """

    if not isinstance(solver_args, dict):
        raise TypeError("solver_args must be a dictionary.")

    if solver_args == {}:
        solver_args = get_solver_default_config()
    else:
        solver_args = copy.deepcopy(solver_args)
    
    solver_executor = A.get_executor()
     # TODO: Create a better way to check the dtype of the matrix
    dtype = type(A).__name__.split('_')[1]
    solver_cls = getattr(pGB.solver, "config_solver_" + dtype)
    solver = solver_cls(
        solver_executor, A, json.dumps(solver_args)
    )
    return solver

def config_solve(A, b, x, solver_args: dict = get_solver_default_config()):
    if not isinstance(solver_args, dict):
        raise TypeError("solver_args must be a dictionary.")

    if solver_args == {}:
        solver_args = get_solver_default_config()
    else:
        solver_args = copy.deepcopy(solver_args)

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

def solve(A, b, initial_guess=None, solver_args: dict = get_solver_default_config(), kind="config"):
    """Solve a given linear system, where A is the system matrix and b the RHS

    Parameters: A - The system matrix
                b - The right hand side vector
                initial_guess - The initial guess
                solver_args - An optional dictionary forwarded to the solver, 
                eg {'type': 'solver::Cg', 'criteria': [{"type": "Iteration", "max_iters": 100}]}
                kind - the underlying solver, eg. config
    Returns: tuple of a logger object and solution vector
    """

    if not isinstance(solver_args, dict):
        raise TypeError("solver_args must be a dictionary.")

    if solver_args == {}:
        solver_args = get_solver_default_config()
    else:
        solver_args = copy.deepcopy(solver_args)

    ctor = globals()[kind+"_solve"]

    if initial_guess is None:
        dtype = type(A).__name__.split('_')[1]
        dense_cls = getattr(pGB.matrix, f"dense_{dtype}")
        dim = (A.shape[1], b.shape[1])
        initial_guess = dense_cls(b.get_executor(), dim)
        initial_guess.fill(0.0)

    return ctor(A,b,x=initial_guess,solver_args=solver_args)

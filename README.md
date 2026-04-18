# PyGinkgo: Python Binding for Ginkgo
![image](https://github.com/Helmholtz-AI-Energy/pyGinkgo/assets/52911730/4d1d9778-1ec2-46c6-a464-ce50d98eb915)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Helmholtz-AI-Energy/pyGinkgo/actions/workflows/build.yml/badge.svg)](https://github.com/Helmholtz-AI-Energy/pyGinkgo//actions)



PyGinkgo is a Python binding for the Ginkgo framework, providing access to Ginkgo's powerful linear algebra capabilities from Python. Ginkgo is a high-performance numerical linear algebra library for sparse systems, primarily designed for developing efficient iterative solvers on complex HPC architectures.

The tests successfully run on the following Python versions:
- 3.8.20
- 3.9.22
- 3.10.17
- 3.11.12
- 3.12.3
- 3.13.3

## Installation

### Prerequisites

- Python 3.8+
- Ginkgo (preinstalled, otherwise it will be cloned during build)
- Pybind11
- Ninja # if you want to use cmake presets
- [pybind11-stubgen](https://pypi.org/project/pybind11-stubgen/) # if you want to use [stubs generation](#stubs-generation)
- [CuPy](https://cupy.dev/) # optional, for zero-copy GPU interoperability (see [CuPy Interoperability](#cupy-interoperability))

### Building the module via CMake

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Helmholtz-AI-Energy/pyGinkgo.git
   ```
2. **Build using CMake**:
   ```bash
   # Make a build directory in the project directory
   mkdir build && cd build

   # Run CMake configuration
   cmake ..

   # Build the project using the specified number of cores (replace "number of cores" with the desired value)
include/FoamAdapter/datastructures/expression.hpp   # (Here we are still within the build directory)
   cmake --build . -j=number_of_cores
   ```
3. **Install the module**:
   ```bash
   # (Here we are still within the build directory)
   cmake --install .
   ```
   - To install in the virtual environment, use `-DPython_ROOT_DIR=path_to_venv_bin_folder` flag during the project configuration.

### Running the tests
You would need to install pytest to be able to run the tests. To run all tests:
```bash
ctest
```
To run a particular test, say 'pyginkgo_import_test':
```bash
ctest -R pyginkgo_import_test
```

### Building the module via pip
You can invoke the build and installation process via pip, this however will require the same dependencies to be present as with the default Cmake installation.
To install pyGinkgo from source use 
```bash
pip install .
```
or alternatively getting it from PyPi
```bash
pip install pyGinkgo
```
**Warning**
Building via pip currently will build Ginkgo, which depending on your system might take a considerable amount of time and memory. An example how to modify the cmake build flags to switch different compute backends on or off and modify the number of threads for compilation is given below. 
```bash
pip install .   --config-settings="override=cmake.args=[-DGINKGO_BUILD_OMP=OFF,-DGINKGO_BUILD_MPI=OFF,-DGINKGO_BUILD_CUDA=OFF,-DGINKGO_BUILD_HIP=OFF,-DGINKGO_BUILD_DPCPP=OFF]"   --config-settings=build_args="-j2"
```

### Stubs generation
From [Python mypy documentation](https://mypy.readthedocs.io/en/stable/stubgen.html):
> A stub file (see [PEP 484](https://peps.python.org/pep-0484/)) contains only type hints for the public interface of a module, with empty function bodies. Mypy can use a stub file instead of the real implementation to provide type information for the module. They are useful for third-party modules whose authors have not yet added type hints (and when no stubs are available in typeshed) and C extension modules (which mypy can’t directly process).

For this project the [pybind11-stubgen](https://github.com/sizmailov/pybind11-stubgen) module was used, [being specifically tailored](https://github.com/sizmailov/pybind11-stubgen/issues/31#issuecomment-1751932149) to work with pybind11.

In order to enable the stubs generation:
1. **Install [pybind11-stubgen](https://pypi.org/project/pybind11-stubgen/) on your local Python installation**:
   ```bash
   pip install pybind11-stubgen
   ```

2. **Set `ENABLE_PYGINKGOBINDINGS_STUBS=ON` when doing CMake configuration**:
   ```bash
   cmake .. -DENABLE_PYGINKGOBINDINGS_STUBS=ON
   ```

3. Now stubs are generated in the build folder and during the library installation. They would allow to see what's inside of the `pyGinkgo.pyGinkgoBindings` module and use autocomplete:
   ```python
   class dense(pyGinkgoBindings.LinOp):
      @typing.overload
      def __init__(self, arg0: typing_extensions.Buffer) -> None:
         ...
      @typing.overload
      def __init__(self, arg0: pyGinkgoBindings.Executor, arg1: typing_extensions.Buffer) -> None:
         ...
      @typing.overload
      def __init__(self, arg0: pyGinkgoBindings.Executor) -> None:
         ...
   ```

#### Development stubs generation
While working on the Python side of the project, it is also useful to have access to the stubs for the C++ code. This can be done by setting `ENABLE_PYGINKGOBINDINGS_DEV_STUBS=ON` when doing CMake configuration:
```bash
cmake .. -DENABLE_PYGINKGOBINDINGS_DEV_STUBS=ON
```
This will generate the stubs for the C++ code in the `pyGinkgoBindings` module inside the `./src/pyGinkgo/pyGinkgoBindings` folder, allowing for autocomplete and type checking by VSCode or other IDEs.

## Usage

Usage examples can be found in [examples](examples) directory. Here's a simple example demonstrating how to use pyGinkgo to perform sparse matrix-vector multiplication:

```python
import pyGinkgo as pg
import numpy as np

# Device initialization
dev = pg.device("cuda")

# Initialize matrix and tensors
fn = 'm1.mtx'

A = pg.read(device=dev, path=fn, dtype="double", format="Csr")
n_rows = A.shape[0]

b = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="double", fill=1.0)

x = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="double", fill=0.0)

# Sparse Matrix Vector Product
A.apply(b, x)
```

## CuPy Interoperability

pyGinkgo supports zero-copy data exchange with [CuPy](https://cupy.dev/) on CUDA devices, eliminating unnecessary device-host-device memory transfers. This is especially useful when you are already working primarily on the GPU with CuPy and want to use Ginkgo's solvers without paying the cost of copying data back and forth.

The interoperability uses the [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) (CAI v3) protocol, which is CuPy's native mechanism for sharing GPU memory. No special wrapper module is needed — the standard constructors and `cupy.asarray()` handle everything.

### Zero-Copy Conversion Paths

| Direction | Mechanism |
|-----------|-----------|
| CuPy array/dense → Ginkgo | Constructor detects `__cuda_array_interface__` |
| CuPy CSR/COO → Ginkgo | Constructor duck-types on `.data`/`.indices`/`.indptr` |
| Ginkgo array/dense → CuPy | `cupy.asarray()` via `__cuda_array_interface__` |
| Ginkgo CSR/COO → CuPy | `.data`/`.indices`/`.indptr` properties + `cupy.asarray()` |

When pyGinkgo is built without CUDA support, conversions fall back transparently to copying through host memory.

### CuPy Examples

#### Dense arrays — zero-copy in both directions

```python
import cupy
import pyGinkgo.pyGinkgoBindings as pGB

executor = pGB.CudaExecutor()

# CuPy → Ginkgo (zero-copy view via __cuda_array_interface__)
cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
gko_arr = pGB.base.array_double(executor, cp_arr)

cp_mat = cupy.array([[1, 2], [3, 4]], dtype=cupy.float64)
gko_dense = pGB.matrix.dense_double(executor, cp_mat)

# Ginkgo → CuPy (zero-copy view via __cuda_array_interface__)
result = cupy.asarray(gko_arr)
```

#### Sparse matrices — zero-copy via constructor

```python
import cupy
import cupyx.scipy.sparse as sp
import pyGinkgo.pyGinkgoBindings as pGB

executor = pGB.CudaExecutor()

# CuPy CSR → Ginkgo CSR (zero-copy, duck-types on .data/.indices/.indptr)
A_cupy = sp.csr_matrix(cupy.eye(3, dtype=cupy.float64))
A_gko = pGB.matrix.Csr_double_int32(executor, A_cupy)

# Ginkgo CSR → CuPy CSR (zero-copy via component array properties)
A_back = sp.csr_matrix(
    (cupy.asarray(A_gko.data), cupy.asarray(A_gko.indices), cupy.asarray(A_gko.indptr)),
    shape=A_gko.shape,
)
```

#### Solving a linear system with GMRES using CuPy data

```python
import cupy
import cupyx.scipy.sparse as sp
import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

# Build a sparse system entirely on the GPU
n = 100
diag = 2.0 * cupy.ones(n, dtype=cupy.float64)
off  = -1.0 * cupy.ones(n - 1, dtype=cupy.float64)
A_cupy = sp.csr_matrix(
    cupy.diag(diag) + cupy.diag(off, 1) + cupy.diag(off, -1)
)
b_cupy = cupy.ones(n, dtype=cupy.float64)

# Wrap CuPy data for Ginkgo — all zero-copy
executor = pGB.CudaExecutor()
A_gko = pGB.matrix.Csr_double_int32(executor, A_cupy)
b_gko = pGB.matrix.dense_double(executor, b_cupy)

# Allocate solution vector on the GPU
x_gko = pGB.matrix.dense_double(executor, (n, 1))
x_gko.fill(0.0)

# Solve with GMRES
solver_args = {
    "type": "solver::Gmres",
    "criteria": [
        {"type": "Iteration", "max_iters": 200},
        {"type": "ResidualNorm", "reduction_factor": 1e-10},
    ],
}
_, x_gko = pg.solve(A_gko, b_gko, x_gko, solver_args=solver_args)

# Get the result back as a CuPy array — zero-copy
x_cupy = cupy.asarray(x_gko)
```

## Benchmarking

The benchmarking results are presented in our [pyGinkgo publication on arXiv](https://arxiv.org/abs/2510.08230).

## Distributed (MPI) bindings

pyGinkgo can optionally expose Ginkgo's distributed-memory types
(`gko::experimental::distributed::{Partition, Vector, Matrix}`,
`Schwarz` preconditioner) plus a `PyLinOp` matrix-free trampoline. This
is gated behind the `pyGinkgo_BUILD_MPI` CMake option so the default
single-process wheel stays MPI-free.

### Building with MPI

```bash
cmake -B build -S . -DpyGinkgo_BUILD_MPI=ON
cmake --build build -j
pip install ".[mpi]"   # adds the mpi4py runtime dependency
```

The build will:

1. Refuse to use a pre-installed Ginkgo that was compiled without MPI
   (the include path must contain `GINKGO_BUILD_MPI=1`).
2. Locate `mpi4py` via Python and pull in its C headers.
3. Bake the `MPI_Get_library_version()` string of the build-time MPI
   into the binary so the Python facade can detect mpich-vs-openmpi
   mismatches at import time.

### Quick start

```python
from mpi4py import MPI
import cupy
from pyGinkgo import device
from pyGinkgo.distributed import (
    Partition, DistributedVector, DistributedMatrix,
)
import pyGinkgo.distributed.communicator as gkc

comm = MPI.COMM_WORLD
exec_ = device("cuda", gkc.map_rank_to_device_id(comm, cupy.cuda.runtime.getDeviceCount()))

# 1D row partition over global_size rows, evenly split across ranks
part = Partition.uniform(exec_, comm.size, global_size=N)

# Distributed RHS from a local cupy slice
b = DistributedVector.from_local_array(
    exec_, comm, global_size=(N, 1), local_array=local_b_cupy)
```

The C++ `solver.{cg,gmres,bicgstab}_*` factories accept any LinOp, so a
distributed Matrix or a `PyLinOp` slots into the existing solver path
unchanged. Each `apply()` returns a `(Convergence, x)` pair so callers
can read final residual and iteration count:

```python
cg = pgb.solver.cg_double(exec_, A.raw, max_iters=200,
                          reduction_factor=1e-10, relative_stop_mode=True)
log, _ = cg.apply(b.raw, x.raw)
print(log.get_num_iterations(), log.get_residual_norm())
```

### Matrix-free distributed operators (`PyLinOp`)

When a `PyLinOp` subclass is used inside a distributed solver, the
`apply_impl(b, x)` callback receives the *distributed* vectors directly;
this rank only owns the local block. **The callback is responsible for
any halo exchange required by the matrix-free matvec** — Ginkgo does not
perform implicit halo communication for user-defined operators. A
typical CuPy callback should call `cupy.cuda.Stream.null.synchronize()`
before returning so Ginkgo (which dispatches on its own executor stream)
sees the writes.

### Diagnostics: gather a distributed vector to host

```python
host = v.raw.gather_on_root(0)  # numpy array on rank 0, None elsewhere
```

### Zero-copy CuPy ↔ distributed.Vector

`DistributedVector.from_local_array(...)` always copies. For a
zero-copy view that aliases a CuPy buffer, use the C++ static method
directly and keep a reference to the source array alive:

```python
v = pgb.distributed.Vector_double.from_local_array_view(
        exec_, comm, (N, 1), local_cupy_array)
```

### MPI ABI safety

Mixing an MPICH-built pyGinkgo with an OpenMPI-built mpi4py is a
common source of mysterious crashes. `pyGinkgo.distributed` performs
three independent checks:

1. CMake-time: warns if the mpi4py library directory disagrees with the
   detected MPI install.
2. Configure-time: a `try_run` probe records the build MPI's
   `MPI_Get_library_version()` string into the compiled extension.
3. Import-time: the first call that takes a communicator compares the
   baked string with `MPI.Get_library_version()` and asks the C++ side
   to round-trip a small MPI op on the user's `comm`.

A mismatch raises `ImportError` with a hint to install a matching
conda variant (`pyginkgo-mpi-{mpich,openmpi}`).


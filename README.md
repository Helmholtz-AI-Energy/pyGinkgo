# PyGinkgo: Python Binding for Ginkgo
![image](https://github.com/Helmholtz-AI-Energy/pyGinkgo/assets/52911730/4d1d9778-1ec2-46c6-a464-ce50d98eb915)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Helmholtz-AI-Energy/pyGinkgo/actions/workflows/build.yml/badge.svg)](https://github.com/Helmholtz-AI-Energy/pyGinkgo//actions)



PyGinkgo is a Python binding for the Ginkgo framework, providing access to Ginkgo's powerful linear algebra capabilities from Python. Ginkgo is a high-performance numerical linear algebra library for sparse systems, primarily designed for developing efficient iterative solvers on complex HPC architectures.

The tests successfully run on the following Python versions:
- 3.9.22
- 3.10.17
- 3.11.12
- 3.12.3
- 3.13.3

## Installation

### Installing via pip (recommended)

Pre-built wheels are the quickest way to get pyGinkgo. They bundle Ginkgo, so no
separate Ginkgo installation or compilation is required.

Three flavours are published, and they are obtained differently: **CPU wheels
come from PyPI**, while the **CUDA and ROCm wheels are attached to GitHub
Releases**.

#### CPU wheels (from PyPI)

```bash
pip install pyGinkgo
```

Available for CPython 3.9–3.13 on:

| Platform | Architectures | Notes |
| --- | --- | --- |
| Linux | x86-64, aarch64 | manylinux, glibc >= 2.28 |
| Windows | AMD64 | |
| macOS | arm64 (Apple Silicon), x86-64 (Intel) | |

Alpine/musl and 32-bit targets are not built. These wheels enable Ginkgo's
**reference backend only** — no OpenMP, MPI, CUDA, HIP or SYCL. If you need a
multi-threaded CPU backend (`OmpExecutor`), SYCL, or MPI support,
build from source as described below.

#### CUDA wheels (from GitHub Releases)

CUDA wheels are *not* published to PyPI: they carry a local version suffix such
as `+cuda128`, and PyPI rejects local version identifiers. They are attached to
the matching [GitHub Release](https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases)
instead, and are installed directly by URL:

Two variants are built, one per CUDA major version. Pick the one matching your
CUDA installation — minor-version compatibility spans a major line but not
across one, so a 12.x wheel will not run against CUDA 13 or the reverse:

| Variant | Suffix | Works with | NVIDIA driver |
| --- | --- | --- | --- |
| CUDA 12.8 | `+cuda128` | any CUDA 12.x | R525+ |
| CUDA 13.1 | `+cuda131` | any CUDA 13.x | R580+ |

```bash
# CUDA 12.8, CPython 3.12, Linux x86-64
pip install https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases/download/v0.0.1/pyGinkgo-0.0.1+cuda128-cp312-cp312-manylinux_2_34_x86_64.whl

# CUDA 13.1, same platform
pip install https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases/download/v0.0.1/pyGinkgo-0.0.1+cuda131-cp312-cp312-manylinux_2_34_x86_64.whl
```

Both variants are otherwise identical, and narrower than the CPU wheels:

| Requirement | Value |
| --- | --- |
| OS | Linux x86-64, glibc >= 2.34 (Ubuntu 22.04+, RHEL/Rocky 9+, Debian 12+) |
| Python | CPython 3.12 only |
| GPU | compute capability 8.0 (Ampere, e.g. A100/A30) and 9.0 (Hopper, e.g. H100) |
| CUDA math libraries | **required on the host**, matching the variant's major version |

The CUDA math libraries are *not* bundled: Ginkgo links cuBLAS, cuSPARSE, cuRAND
and cuFFT, and shipping them produced a 2 GB wheel. The wheel links against the
host's CUDA installation instead, so a matching toolkit must be installed and on
the loader path — on HPC systems this usually means loading the matching module
before importing pyGinkgo:

```bash
module load cuda/12          # or: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

`libcuda.so.1` is excluded too, since it is provided by the installed NVIDIA
driver, while `libcudart` *is* bundled so the CUDA runtime stays matched to the
build. The wheel also embeds PTX, so newer GPU architectures should work through
JIT compilation, but this is not tested.

#### ROCm wheels (from GitHub Releases)

ROCm wheels are published alongside the CUDA ones. As with CUDA, there is one
variant per major version — the ROCm libraries are not bundled, so the wheel
links against the host's sonames and those carry the major version. A ROCm 6
wheel will not load against a ROCm 7 installation:

| Variant | Suffix | Works with |
| --- | --- | --- |
| ROCm 6.4 | `+rocm64` | any ROCm 6.x |
| ROCm 7.0 | `+rocm70` | any ROCm 7.x |

```bash
# ROCm 6.4, CPython 3.12, Linux x86-64, AMD CDNA2 (gfx90a)
pip install https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases/download/v0.0.1/pyGinkgo-0.0.1+rocm64-cp312-cp312-manylinux_2_34_x86_64.whl

# ROCm 7.0, same platform
pip install https://github.com/Helmholtz-AI-Energy/pyGinkgo/releases/download/v0.0.1/pyGinkgo-0.0.1+rocm70-cp312-cp312-manylinux_2_34_x86_64.whl
```

Both variants are otherwise identical:

| Requirement | Value |
| --- | --- |
| OS | Linux x86-64, glibc >= 2.34 (Ubuntu 22.04+, RHEL/Rocky 9+, Debian 12+) |
| Python | CPython 3.12 only |
| GPU | gfx90a (CDNA2, e.g. MI210/MI250X) |
| ROCm runtime | **required on the host**, matching the variant's major version |

Note the difference from the CUDA wheels: *all* the ROCm userspace libraries are
excluded, not just the math ones, because rocBLAS alone ships hundreds of
megabytes of Tensile kernels. The wheel links against the host's ROCm
installation, so a matching ROCm must be installed and on the loader path — on
HPC systems this usually means loading the matching module before importing
pyGinkgo:

```bash
module load rocm/6.4          # or: export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

Because all three flavours share the distribution name `pyGinkgo`, installing a
GPU wheel over an existing CPU install replaces it; pass `--force-reinstall` if
pip reports the requirement as already satisfied.

#### Verifying an installation

```bash
python -c "import pyGinkgo.pyGinkgoBindings as pGB; pGB.ReferenceExecutor().synchronize(); print('pyGinkgo OK')"
```

After installing a CUDA wheel, also check that the GPU backend is present and a
device is visible:

```bash
python -c "import pyGinkgo; print('CUDA available:', pyGinkgo.cuda_available()); pyGinkgo.device('cuda:0').synchronize(); print('CUDA OK')"
```

The equivalent check for a ROCm wheel:

```bash
python -c "import pyGinkgo, pyGinkgo.pyGinkgoBindings as pGB; print('HIP devices:', pGB.HipExecutor.get_num_devices()); pyGinkgo.device('hip:0').synchronize(); print('ROCm OK')"
```

To build from source instead (e.g. to enable a compute backend not covered by
the wheels), follow the sections below.

### Prerequisites

- Python 3.9+
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
   # (Here we are still within the build directory)
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

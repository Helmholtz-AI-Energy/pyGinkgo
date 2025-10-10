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

```
import pyGinkgo as pg
import numpy as np

# Device initialization
dev = pg.device ( "cuda" )

# Initialize matrix and tensors
fn = 'm1.mtx'

A = pg.read ( device = dev , path = fn , dtype = "double" , format = "Csr" )
n_rows = A.shape[0]

b = pg.as_tensor (
device = dev , dim =( n_rows ,1) , dtype = "double" , fill =1.0)

x = pg.as_tensor (
device = dev , dim =( n_rows ,1) , dtype = "double" , fill =0.0)

# Sparse Matrix Vector Product
A.apply (b , x )

```

## Benchmarking

The benchmarking results are presented in our [pyGinkgo publication on arXiv](https://arxiv.org/abs/2510.08230).

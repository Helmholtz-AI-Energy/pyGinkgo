# PyGinkgo: Python Binding for Ginkgo
![image](https://github.com/Helmholtz-AI-Energy/pyGinkgo/assets/52911730/4d1d9778-1ec2-46c6-a464-ce50d98eb915)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Helmholtz-AI-Energy/pyGinkgo/actions/workflows/build.yml/badge.svg)](https://github.com/Helmholtz-AI-Energy/pyGinkgo//actions)



PyGinkgo is a Python binding for the Ginkgo framework, providing access to Ginkgo's powerful linear algebra capabilities from Python. Ginkgo is a high-performance numerical linear algebra library for sparse systems, primarily designed for developing efficient iterative solvers on complex HPC architectures.

## Installation

### Prerequisites

- Python 3.x
- Ginkgo
- Pybind11

### Building the module

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
   cmake --build . -j=number_of_cores
   ```

#### Building on Windows
(That's the notes after the successfull build and passing tests on windows)

* Theoretically PyBind11 requires **Boost** and **pytest** to be installed in system and visible in PATH. It wasn't tested though, whether that's a strong requirement.

* Building of the library is done using the same steps as described above. Yet, during the import of the Python module one might get the `ImportError: DLL load failed while importing pyGinkgo: The specified module could not be found.`. There is a [thread on Stackoverflow](https://stackoverflow.com/questions/59860465/pybind11-importerror-dll-not-found-when-trying-to-import-pyd-in-python-int/78866933) related to this issue, but what worked for me was placing the following `dll`s into the same directory as `pyd` file:
   * `libdll.dll`
   * `libgcc_s_seh-1.dll`
   * `libgomp-1.dll`
   * `libstdc++-6.dll`
   * `libwinpthread-1.dll`

   If the Ginkgo library expands the amount of dependencies, as per the [thread](https://stackoverflow.com/questions/59860465/pybind11-importerror-dll-not-found-when-trying-to-import-pyd-in-python-int/78866933) they could be discovered through the Dependency-Walker.

### Running the tests
You would need to install pytest to be able to run the tests. To run all tests:
```bash
ctest
```
To run a particular test, say 'pyginkgo_import_test':
```bash
ctest -R pyginkgo_import_test
```

## Usage

Here's a simple example demonstrating how to use PyGinkgo to perform matrix-vector multiplication:

## Benchmarking


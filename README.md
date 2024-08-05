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
2. ```bash
   git clone https://github.com/Helmholtz-AI-Energy/pyGinkgo.git
   ```

3. **Build using CMake**:

```bash
# Make a build directory in the project directory
mkdir build && cd build

# Run CMake configuration
cmake ..

# Build the project using the specified number of cores (replace "number of cores" with the desired value)
cmake --build . -j=number_of_cores
```

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


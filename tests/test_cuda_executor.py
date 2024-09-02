# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import sys
import pyGinkgo
import numpy as np

sys.path.append("../build")

@pytest.mark.gpu
def test_array_cuda():
    if pyGinkgo.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pyGinkgo.CudaExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pyGinkgo.base.array(executor, np_array)
    arr_copy = pyGinkgo.base.array(executor, arr)
    assert arr.get_size() == arr_copy.get_size()
    assert pyGinkgo.base.reduce_add(arr, 0.0) == 15.0

@pytest.mark.gpu
def test_device_id():
    if pyGinkgo.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    executor = pyGinkgo.CudaExecutor(0)
    assert executor.device_id == 0
    with pytest.raises(AttributeError):
        executor.device_id = 1

@pytest.mark.gpu
def test_master():
    if pyGinkgo.CudaExecutor.get_num_devices() < 1:
        pytest.skip("CUDA is not available")

    master = pyGinkgo.ReferenceExecutor()
    executor = pyGinkgo.CudaExecutor(0, master)
    assert id(executor.master) == id(master)
    with pytest.raises(AttributeError):
        executor.master = pyGinkgo.ReferenceExecutor()

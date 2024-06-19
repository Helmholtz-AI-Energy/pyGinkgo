# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors


import sys
import pyGinkgo
import numpy as np

# A hack which we should get rid off
pyGinkgo.base.declare_array(pyGinkgo.base)

sys.path.append("../build")


def test_init_empty_array():
    executor = pyGinkgo.ReferenceExecutor()
    size = 5
    # A hack which we should get rid off
    arr = pyGinkgo.base.array(executor, size)
    assert arr.get_size() == size


def test_can_fill_array():
    executor = pyGinkgo.ReferenceExecutor()
    size = 5
    arr = pyGinkgo.base.array(executor, size)
    arr.fill(10.0)
    assert arr.at(0) == 10.0


def test_can_instantiate_array_from_python_list():
    executor = pyGinkgo.ReferenceExecutor()
    lst = [1.0, 2.0, 3.0, 4.0, 5.0]
    arr = pyGinkgo.base.array(executor, lst)
    assert arr.get_size() == len(lst)
    assert arr.at(0) == 1.0
    assert arr.at(1) == 2.0
    assert arr.at(2) == 3.0
    assert arr.at(3) == 4.0
    assert arr.at(4) == 5.0


def test_can_instantiate_array_from_python_int_list():
    executor = pyGinkgo.ReferenceExecutor()
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    arr = pyGinkgo.base.array(executor, np_array)
    assert arr.get_size() == len(np_array)
    assert isinstance(arr.at(0), int)


def test_can_instantiate_array_from_numpy_array():
    executor = pyGinkgo.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pyGinkgo.base.array(executor, np_array)
    assert arr.get_size() == len(np_array)
    assert isinstance(arr.at(0), float)


def test_can_copy_construct_array():
    executor = pyGinkgo.ReferenceExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pyGinkgo.base.array(executor, np_array)
    arr_copy = pyGinkgo.base.array(executor, arr)
    assert arr.get_size() == arr_copy.get_size()


def test_can_construct_np_array_from_gko_array():
    executor = pyGinkgo.ReferenceExecutor()
    lst = [1.0, 2.0, 3.0, 4.0, 5.0]
    arr = pyGinkgo.base.array(executor, lst)
    np_arr = np.array(arr)
    assert arr.get_size() == len(np_arr)
    assert np_arr[0] == 1.0
    assert np_arr[1] == 2.0
    assert np_arr[2] == 3.0
    assert np_arr[3] == 4.0
    assert np_arr[4] == 5.0

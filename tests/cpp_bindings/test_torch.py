# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

try:
    import torch

    torch_avail = True
except ImportError:
    torch = None
    torch_avail = False

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


torch_d_type_map = (
    {
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }
    if torch_avail
    else {}
)


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestSolve:
    def test_can_create_array_from_torch(self, data_type: pg.gko_types.ValueType):
        executor = pGB.ReferenceExecutor()
        array_cls = getattr(pGB.base, "array_" + data_type)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=data_type.numpy_type)
        torch_array = torch.asarray(np_array)
        arr = array_cls(executor, torch_array)
        arr_copy = array_cls(executor, arr)
        assert arr.shape == arr_copy.shape
        assert pGB.base.reduce_add(arr, 0.0) == 15.0

    def test_can_create_torch_array_from_gko_array(
        self, data_type: pg.gko_types.ValueType
    ):
        executor = pGB.ReferenceExecutor()
        array_cls = getattr(pGB.base, "array_" + data_type)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=data_type.numpy_type)
        arr = array_cls(executor, np_array)
        # When receiving Python Buffer protocol object, torch.asarray assumes dtype to be float32
        #   the shape of the array is also lost
        # https://pytorch.org/docs/stable/generated/torch.asarray.html#:~:text=the%20same%20history.-,When%20obj%20is%20not,memory%20with%20the%20buffer.,-When%20obj%20is
        torch_array = torch.asarray(arr, dtype=torch_d_type_map[data_type])
        assert torch_array.size(dim=0) == np_array.size

    def test_can_create_dense_from_torch_tensor(
        self, data_type: pg.gko_types.ValueType
    ):
        executor = pGB.ReferenceExecutor()
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        data = [[1.0, 2.0], [3.0, 4.0]]
        torch_tensor = torch.tensor(data, dtype=torch_d_type_map[data_type])
        dense = dense_cls(executor, torch_tensor.__array__())
        assert dense.get_num_stored_elements() == 4
        assert dense.at(0, 1) == 2.0
        assert dense.at(1, 1) == 4.0
        assert dense.shape[0] == 2
        assert dense.shape[1] == 2

    def test_can_create_torch_tensor_from_dense(
        self, data_type: pg.gko_types.ValueType
    ):
        executor = pGB.ReferenceExecutor()
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=data_type.numpy_type)
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(executor, data)
        torch_tensor = torch.tensor(data, dtype=torch_d_type_map[data_type])
        assert torch_tensor[0][0].item() == 1.0
        assert torch_tensor[0][1].item() == 2.0
        assert torch_tensor[1][0].item() == 3.0
        assert torch_tensor[1][1].item() == 4.0
        torch_tensor = torch.tensor(np.array(dense), dtype=torch_d_type_map[data_type])
        assert torch_tensor[0][0].item() == 1.0
        assert torch_tensor[0][1].item() == 2.0
        assert torch_tensor[1][0].item() == 3.0
        assert torch_tensor[1][1].item() == 4.0

# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


torch_d_type_map = {
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}


@pytest.mark.skipif(not torch_avail, reason="requires pytorch")
@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestTorchInteroperability:
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
        dense = dense_cls(executor, torch_tensor)
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

    def test_dense_accepts_cpu_torch_tensor(self, data_type: pg.gko_types.ValueType):
        data = torch.tensor([[1.0, 2.0]], dtype=torch_d_type_map[data_type])

        dense = pg.dense(data, device="cpu")

        assert dense.shape == (1, 2)
        assert dense.at(0, 0) == 1.0
        assert dense.at(0, 1) == 2.0

    def test_dense_accepts_cpu_torch_tensor_on_omp(
        self, data_type: pg.gko_types.ValueType
    ):
        data = torch.tensor([[1.0, 2.0]], dtype=torch_d_type_map[data_type])

        dense = pg.dense(data, device="omp")

        assert dense.shape == (1, 2)
        assert dense.at(0, 0) == 1.0
        assert dense.at(0, 1) == 2.0

    def test_dense_rejects_cpu_torch_tensor_on_cuda(
        self, data_type: pg.gko_types.ValueType
    ):
        data = torch.tensor([[1.0, 2.0]], dtype=torch_d_type_map[data_type])

        with pytest.raises(ValueError, match="Torch tensor device does not match"):
            pg.dense(data, device="cuda")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not hasattr(pGB, "CudaExecutor"),
        reason="requires CUDA torch and CUDA pyGinkgo bindings",
    )
    def test_dense_rejects_cuda_torch_tensor_on_different_cuda_device(
        self, data_type: pg.gko_types.ValueType
    ):
        if torch.cuda.device_count() < 2:
            pytest.skip("requires at least two CUDA devices")

        data = torch.tensor(
            [[1.0, 2.0]],
            dtype=torch_d_type_map[data_type],
            device="cuda:1",
        )

        with pytest.raises(ValueError, match="Torch tensor device does not match"):
            pg.dense(data, device="cuda:0")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not hasattr(pGB, "CudaExecutor"),
        reason="requires CUDA torch and CUDA pyGinkgo bindings",
    )
    def test_dense_accepts_cuda_torch_tensor_on_same_cuda_device(
        self, data_type: pg.gko_types.ValueType
    ):
        data = torch.tensor(
            [[1.0, 2.0]],
            dtype=torch_d_type_map[data_type],
            device="cuda:0",
        )

        dense = pg.dense(data, device="cuda:0")

        assert dense.shape == (1, 2)

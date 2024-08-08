# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: 2024 pyGinkgo authors

import pytest
import torch


@pytest.mark.gpu
def test_tensor_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    # Create a tensor and move it to GPU
    tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()

    # Check if tensor is on GPU
    assert torch.cuda.is_available()
    # Ensure the device type is 'cuda'
    assert tensor.device.type == "cuda"
    # Perform a simple operation and check the result
    assert tensor.sum().item() == 6.0

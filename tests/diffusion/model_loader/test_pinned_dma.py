# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.model_loader.pinned_dma import _byte_view

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.uint8])
def test_byte_staging_view_preserves_shape_dtype_and_values(dtype):
    source = torch.arange(24, dtype=torch.int64).to(dtype).reshape(4, 6)
    storage = torch.empty(source.numel() * source.element_size(), dtype=torch.uint8)

    staged = _byte_view(source, storage)

    assert staged.shape == source.shape
    assert staged.dtype == source.dtype
    staged.copy_(source)
    torch.testing.assert_close(staged, source)


def test_byte_staging_view_rejects_insufficient_storage():
    source = torch.ones(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="too small"):
        _byte_view(source, torch.empty(3, dtype=torch.uint8))


def test_byte_staging_view_requires_contiguous_source():
    source = torch.ones((4, 4), dtype=torch.float32)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        _byte_view(source, torch.empty(source.numel() * source.element_size(), dtype=torch.uint8))

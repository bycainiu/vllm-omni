# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.minimax_h3_offline_fp8.export import export_checkpoint
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel


def _write_toy_checkpoint(root: Path) -> dict[str, torch.Tensor]:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3DiTModel", "num_layers": 1})
    )
    weights = {
        "blocks.0.attn.qkv_proj.weight": torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16),
        "blocks.0.attn.out_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "blocks.0.mlp.fc1.weight": torch.tensor(
            [
                [1, -2, 3, -4],
                [2, 4, 6, 8],
                [0, 0, 0, 0],
                [-1, -3, -5, -7],
                [4, 2, 0, -2],
                [3, 6, 9, 12],
                [2, 0, -2, -4],
                [1, 3, 5, 7],
            ],
            dtype=torch.bfloat16,
        ),
        "blocks.0.mlp.fc2.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "blocks.0.adaln_proj.linear.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "condition_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(weights, str(transformer / shard_name), metadata={"format": "pt"})
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in weights.values())},
                "weight_map": {name: shard_name for name in weights},
            }
        )
    )
    return weights


def test_export_selectively_quantizes_native_h3_and_keeps_bf16_fallback(tmp_path: Path):
    source_root = tmp_path / "source"
    source_weights = _write_toy_checkpoint(source_root)
    source_transformer = source_root / "transformer"
    output_transformer = tmp_path / "output" / "transformer"

    result = export_checkpoint(source_transformer, output_transformer, write_sha256=False)

    assert result["quantized_matrix_count"] == 4
    expected_quantized_parameters = sum(
        source_weights[name].numel()
        for name in source_weights
        if name.endswith(".weight") and ".adaln_proj." not in name and not name.startswith("condition_proj")
    )
    assert result["quantized_parameter_count"] == expected_quantized_parameters
    config = json.loads((output_transformer / "config.json").read_text())
    quant_config = config["quantization_config"]
    assert quant_config["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN"
    assert quant_config["ignore"] == quant_config["exclude_modules"]

    index = json.loads((output_transformer / "model.safetensors.index.json").read_text())
    shard = output_transformer / "model-00001-of-00001.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as reader:
        for name, source_weight in source_weights.items():
            loaded = reader.get_tensor(name)
            if name.startswith("blocks.0.attn.") or name.startswith("blocks.0.mlp."):
                scale = reader.get_tensor(name[: -len(".weight")] + ".weight_scale")
                assert loaded.dtype == torch.float8_e4m3fn
                assert scale.dtype == torch.float32
                assert tuple(scale.shape) == (source_weight.shape[0],)
                restored = loaded.float() * scale[:, None]
                torch.testing.assert_close(restored, source_weight.float(), rtol=0.12, atol=0.12)
            else:
                assert loaded.dtype == torch.bfloat16
                torch.testing.assert_close(loaded, source_weight)

    assert "blocks.0.attn.qkv_proj.weight_scale" in index["weight_map"]
    assert index["metadata"]["total_size"] > 0
    assert (output_transformer / "offline_fp8_manifest.json").is_file()


def _loader_param(calls: list[tuple[int | None, torch.Tensor]]) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.empty(1), requires_grad=False)

    def weight_loader(_param: torch.Tensor, tensor: torch.Tensor, shard_id: int | None = None) -> None:
        calls.append((shard_id, tensor.detach().clone()))

    param.weight_loader = weight_loader
    return param


def _minimal_loader_context(params: dict[str, torch.nn.Parameter]):
    model = SimpleNamespace(
        adaln_cache=None,
        _diffusers_weights=False,
        arch=SimpleNamespace(num_attention_heads=2, attention_head_dim=2),
        named_parameters=lambda: iter(params.items()),
        named_buffers=lambda: iter(()),
    )
    return model


def test_h3_loader_reorders_qkv_weight_scales_with_weight_rows():
    weight_calls: list[tuple[int | None, torch.Tensor]] = []
    scale_calls: list[tuple[int | None, torch.Tensor]] = []
    model = _minimal_loader_context(
        {
            "blocks.0.attn.qkv_proj.weight": _loader_param(weight_calls),
            "blocks.0.attn.qkv_proj.weight_scale": _loader_param(scale_calls),
        }
    )
    weight = torch.arange(24, dtype=torch.float32).reshape(12, 2).to(torch.float8_e4m3fn)
    scale = torch.arange(12, dtype=torch.float32)

    MiniMaxH3DiTModel.load_weights(
        model,
        [
            ("blocks.0.attn.qkv_proj.weight_scale", scale),
            ("blocks.0.attn.qkv_proj.weight", weight),
        ],
    )

    expected_rows = torch.tensor([0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
    torch.testing.assert_close(scale_calls[0][1], scale[expected_rows])
    torch.testing.assert_close(weight_calls[0][1].float(), weight[expected_rows].float())


def test_h3_loader_splits_fc1_weight_scales_into_gate_up_shards():
    calls: list[tuple[int | None, torch.Tensor]] = []
    model = _minimal_loader_context({"blocks.0.mlp.fc1.weight_scale": _loader_param(calls)})
    scale = torch.arange(8, dtype=torch.float32)

    MiniMaxH3DiTModel.load_weights(model, [("blocks.0.mlp.fc1.weight_scale", scale)])

    assert [shard_id for shard_id, _ in calls] == [0, 1]
    torch.testing.assert_close(calls[0][1], scale[:4])
    torch.testing.assert_close(calls[1][1], scale[4:])


def test_h3_loader_rejects_odd_fc1_scale_vector():
    model = _minimal_loader_context({"blocks.0.mlp.fc1.weight_scale": _loader_param([])})

    with pytest.raises(ValueError, match="weight_scale rows must split evenly"):
        MiniMaxH3DiTModel.load_weights(model, [("blocks.0.mlp.fc1.weight_scale", torch.ones(7))])

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audit a selective MiniMax H3 FP8 checkpoint using CPU-only reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from experiments.minimax_h3_offline_fp8.export import is_target_weight


def verify(candidate: Path, source: Path) -> dict[str, object]:
    candidate = candidate.resolve()
    source = source.resolve()
    candidate_index_path = candidate / "model.safetensors.index.json"
    source_index_path = source / "model.safetensors.index.json"
    index = json.loads(candidate_index_path.read_text())
    source_index = json.loads(source_index_path.read_text())
    manifest = json.loads((candidate / "offline_fp8_manifest.json").read_text())
    config = json.loads((candidate / "config.json").read_text())

    targets = sorted(name for name in index["weight_map"] if is_target_weight(name))
    if len(targets) != manifest["quantized_matrix_count"]:
        raise ValueError(f"found {len(targets)} quantized matrices, manifest says {manifest['quantized_matrix_count']}")
    if config["quantization_config"].get("quant_algo") != "FP8_PER_CHANNEL_PER_TOKEN":
        raise ValueError("candidate config does not declare FP8_PER_CHANNEL_PER_TOKEN")

    sample_names = {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.25.mlp.fc1.weight",
        "blocks.49.attn.out_proj.weight",
    }
    sample_errors: list[dict[str, float | str]] = []
    scale_values = 0
    checked = 0

    for shard_name in sorted({index["weight_map"][name] for name in targets}):
        shard_targets = [name for name in targets if index["weight_map"][name] == shard_name]
        with safe_open(str(candidate / shard_name), framework="pt", device="cpu") as fp8_shard:
            for name in shard_targets:
                weight = fp8_shard.get_tensor(name)
                scale_name = name[: -len(".weight")] + ".weight_scale"
                scale = fp8_shard.get_tensor(scale_name)
                if weight.dtype != torch.float8_e4m3fn:
                    raise TypeError(f"{name}: expected float8_e4m3fn, found {weight.dtype}")
                if scale.dtype != torch.float32 or tuple(scale.shape) != (weight.shape[0],):
                    raise TypeError(f"{scale_name}: expected FP32 [{weight.shape[0]}], found {scale.dtype} {tuple(scale.shape)}")
                if not torch.isfinite(weight.float()).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"non-finite FP8 values or invalid scales in {name}")
                checked += 1
                scale_values += scale.numel()

                if name in sample_names:
                    source_shard = source_index["weight_map"][name]
                    with safe_open(str(source / source_shard), framework="pt", device="cpu") as bf16_shard:
                        bf16 = bf16_shard.get_tensor(name).float()
                    expected_scale = bf16.abs().amax(dim=1) / 448.0
                    if not torch.allclose(scale, expected_scale, rtol=1e-6, atol=1e-8):
                        raise ValueError(f"{scale_name}: does not match max-abs/448 from source BF16")
                    reconstructed = weight.float() * scale[:, None]
                    relative_rmse = (
                        reconstructed.sub(bf16).square().mean().sqrt()
                        / bf16.square().mean().sqrt()
                    ).item()
                    sample_errors.append({"weight": name, "relative_rmse": relative_rmse})

    expected_scales = manifest["scale_bytes_fp32"] // torch.tensor([], dtype=torch.float32).element_size()
    if checked != len(targets) or scale_values != expected_scales:
        raise ValueError(f"checked {checked} matrices/{scale_values} scales; expected {len(targets)}/{expected_scales}")
    if {item["weight"] for item in sample_errors} != sample_names:
        raise ValueError(f"could not verify all representative weight rows: {sample_errors}")

    return {
        "cuda_available": torch.cuda.is_available(),
        "matrices_checked": checked,
        "scale_values_checked": scale_values,
        "quantized_parameter_count": manifest["quantized_parameter_count"],
        "transformer_weight_file_bytes": manifest["all_transformer_weight_file_bytes"],
        "sample_relative_rmse": sample_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_transformer", type=Path)
    parser.add_argument("source_transformer", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.candidate_transformer, args.source_transformer), indent=2))


if __name__ == "__main__":
    main()

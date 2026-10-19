# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build a selective ModelOpt-compatible FP8 checkpoint for native MiniMax H3.

This exporter converts only the four fused DiT linear matrices in each H3
block. It stores E4M3 weights with one FP32 scale per output channel; runtime
activations use ModelOpt's dynamic per-token W8A8 path. It intentionally does
not instantiate the full model and runs entirely on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
QUANT_ALGO = "FP8_PER_CHANNEL_PER_TOKEN"
INDEX_NAME = "model.safetensors.index.json"
TARGET_WEIGHT = re.compile(r"^blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$")
MODEL_OPT_IGNORES = [
    "condition_proj",
    "token_refiner.*",
    "blocks.*.adaln_proj.linear",
    "final_layer.*",
]


def is_target_weight(name: str) -> bool:
    """Return whether a native H3 checkpoint matrix is in the 50-block DiT."""
    return TARGET_WEIGHT.fullmatch(name) is not None


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D [out, in] matrix to E4M3 with max/448 row scales."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D linear weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError(f"expected a floating-point linear weight, got {weight.dtype}")

    value = weight.to(dtype=torch.float32)
    amax = value.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = (value / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.contiguous()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_index(index: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{INDEX_NAME} has no non-empty weight_map")
    targets = sorted(name for name in weight_map if is_target_weight(name))
    if not targets:
        raise ValueError("checkpoint has no native MiniMax H3 fused DiT weights to quantize")
    for name in targets:
        if name[: -len(".weight")] + ".weight_scale" in weight_map:
            raise ValueError(f"source checkpoint already has a scale tensor for {name}")
    return weight_map, targets


def export_checkpoint(
    source_transformer: str | Path,
    output_transformer: str | Path,
    *,
    num_threads: int | None = None,
    write_sha256: bool = True,
) -> dict[str, Any]:
    """Write a new transformer directory and return its conversion manifest.

    The final destination must not exist. A temporary sibling is used so an
    interrupted conversion cannot leave a partially loadable checkpoint at
    the requested output path.
    """
    source = Path(source_transformer).resolve()
    output = Path(output_transformer).absolute()
    if source == output.resolve(strict=False):
        raise ValueError("source and output transformer directories must differ")
    if not (source / "config.json").is_file() or not (source / INDEX_NAME).is_file():
        raise FileNotFoundError(f"expected config.json and {INDEX_NAME} under {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    config = json.loads((source / "config.json").read_text())
    if config.get("_class_name") != "MiniMaxH3DiTModel":
        raise ValueError(
            "this first exporter supports the native H3 tensor layout only; "
            f"found _class_name={config.get('_class_name')!r}"
        )
    index = json.loads((source / INDEX_NAME).read_text())
    weight_map, targets = _validate_index(index)
    expected_layers = int(config.get("num_layers", 0))
    if expected_layers and len(targets) != expected_layers * 4:
        raise ValueError(
            f"expected four fused target matrices per H3 block ({expected_layers * 4}), "
            f"found {len(targets)}"
        )

    target_set = set(targets)
    source_shards = sorted(set(weight_map.values()))
    stage = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    if stage.exists():
        raise FileExistsError(f"temporary output already exists: {stage}")
    stage.mkdir(parents=True)
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)

    new_weight_map = dict(weight_map)
    target_parameters = 0
    source_target_bytes = 0
    scale_bytes = 0
    output_bytes = 0
    output_tensor_bytes = 0
    converted = 0
    started = time.monotonic()
    shard_hashes: dict[str, str] = {}

    try:
        for child in source.iterdir():
            if child.name == INDEX_NAME or child.name in source_shards or child.suffix == ".safetensors":
                continue
            if child.is_file():
                shutil.copy2(child, stage / child.name)

        for shard_name in source_shards:
            shard_path = source / shard_name
            out_tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    tensor = reader.get_tensor(name).contiguous()
                    if name in target_set:
                        if tensor.dtype != torch.bfloat16:
                            raise TypeError(f"target {name} must be BF16, found {tensor.dtype}")
                        fp8_weight, scale = quantize_per_output_channel(tensor)
                        scale_name = name[: -len(".weight")] + ".weight_scale"
                        out_tensors[name] = fp8_weight
                        out_tensors[scale_name] = scale
                        new_weight_map[scale_name] = shard_name
                        target_parameters += tensor.numel()
                        source_target_bytes += tensor.numel() * tensor.element_size()
                        scale_bytes += scale.numel() * scale.element_size()
                        converted += 1
                    else:
                        out_tensors[name] = tensor

            missing_from_shard = {
                name for name in target_set if weight_map[name] == shard_name
            } - {name for name in out_tensors if is_target_weight(name)}
            if missing_from_shard:
                raise ValueError(f"target keys missing in shard {shard_name}: {sorted(missing_from_shard)}")

            out_path = stage / shard_name
            save_file(out_tensors, str(out_path), metadata={"format": "pt"})
            output_bytes += out_path.stat().st_size
            output_tensor_bytes += sum(tensor.numel() * tensor.element_size() for tensor in out_tensors.values())
            if write_sha256:
                shard_hashes[shard_name] = _sha256(out_path)
            print(
                f"wrote {shard_name}: {out_path.stat().st_size} bytes "
                f"({time.monotonic() - started:.1f}s elapsed)",
                flush=True,
            )

        if converted != len(targets):
            raise ValueError(f"converted {converted} matrices, expected {len(targets)}")

        output_index = dict(index)
        output_index["weight_map"] = new_weight_map
        metadata = dict(output_index.get("metadata") or {})
        metadata["total_size"] = output_tensor_bytes
        output_index["metadata"] = metadata
        _atomic_json(stage / INDEX_NAME, output_index)

        output_config = dict(config)
        quantization_config = dict(output_config.get("quantization_config") or {})
        quantization_config.update(
            {
                "producer": {"name": "modelopt", "version": "selective-h3-export-v1"},
                "quant_method": "modelopt",
                "quant_algo": QUANT_ALGO,
                # vLLM ModelOpt's config.json parser consumes `ignore`; keep
                # `exclude_modules` as a compatibility spelling for older pins.
                "ignore": MODEL_OPT_IGNORES,
                "exclude_modules": MODEL_OPT_IGNORES,
            }
        )
        output_config["quantization_config"] = quantization_config
        _atomic_json(stage / "config.json", output_config)

        manifest: dict[str, Any] = {
            "format": "ModelOpt-compatible selective FP8 checkpoint",
            "algorithm": QUANT_ALGO,
            "quantization": "E4M3 per output channel; max-abs/448 weight scale; dynamic per-token activations",
            "source_transformer": str(source),
            "quantized_matrix_count": converted,
            "quantized_parameter_count": target_parameters,
            "target_weight_bytes_bf16": source_target_bytes,
            "scale_bytes_fp32": scale_bytes,
            "all_transformer_weight_file_bytes": output_bytes,
            "target_weight_storage_reduction_bytes": source_target_bytes - target_parameters,
            "modelopt_ignore": MODEL_OPT_IGNORES,
            "sha256": shard_hashes,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        _atomic_json(stage / "offline_fp8_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_transformer", type=Path)
    parser.add_argument("output_transformer", type=Path)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    result = export_checkpoint(
        args.source_transformer,
        args.output_transformer,
        num_threads=args.num_threads,
        write_sha256=not args.skip_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

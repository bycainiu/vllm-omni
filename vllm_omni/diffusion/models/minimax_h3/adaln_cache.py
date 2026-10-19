# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact AdaLN memoization and optional schedule-bound projection tables.

Ordinary memoization reuses outputs only for identical inputs and unchanged
weights. Experimental pruning mode builds a complete table for one request
schedule using the live serving projections, then removes their weights and
locks that model instance to the verified task/schedule. Offline sidecars
without weight identities are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from torch import nn

from vllm_omni.diffusion.cache.exact_projection_cache import (
    ExactProjectionCache,
    tensor_digest,
)
from vllm_omni.diffusion.cache.exact_projection_cache import (
    canonical_json as canonical_json,
)

from .time_request import minimax_h3_time_shift_sigmas

if TYPE_CHECKING:
    from .minimax_h3_transformer import MiniMaxH3DiTArchConfig

FORMAT_VERSION = "4"
MODES = {
    "t2va": ("video", "audio"),
    "fl2va": ("video", "audio", "image"),
    "ref2va-image": ("video", "audio", "image"),
    "ref2va-audio": ("video", "audio", "audio_ref"),
    "ref2va-mixed": ("video", "audio", "image", "audio_ref"),
}
PAYLOAD_NAMES = frozenset({"plan_timesteps", "plan_lengths", "time_embeddings", "block_params", "final_params"})


class MiniMaxH3RuntimeAdalnCache(ExactProjectionCache):
    """Exact projection caching with optional H3 schedule-bound sidecar results."""

    def __init__(
        self,
        *,
        max_bytes: int = 256 * 1024**2,
        offload_weights: bool = False,
        prune_weights: bool = False,
        prune_min_steps: int = 50,
    ) -> None:
        super().__init__(max_bytes=max_bytes)
        if type(prune_weights) is not bool:
            raise ValueError("AdaLN weight pruning must be a boolean")
        if type(prune_min_steps) is not int or prune_min_steps < 1:
            raise ValueError("AdaLN pruning minimum step count must be a positive integer")
        self.offload_weights = offload_weights
        self.prune_weights_requested = prune_weights
        self.prune_min_steps = prune_min_steps
        self.weights_pruned = False
        self.pruned_schedule: dict[str, Any] | None = None
        self._request_math_verified = False
        self._active_pruned_step: int | None = None
        self.sidecar: MiniMaxH3AdalnCache | None = None
        self._sidecar_signatures: dict[str, Any] = {}
        self._sidecar_plan: int | None = None

    def should_prune(self, num_steps: int) -> bool:
        """Keep short warmup schedules on the ordinary projection path."""
        return self.prune_weights_requested and num_steps >= self.prune_min_steps

    def clear(self) -> None:
        if self.weights_pruned:
            raise RuntimeError("Cannot clear a pruned AdaLN cache; its projection weights were released")
        super().clear()
        self._request_math_verified = False
        self._active_pruned_step = None
        self.sidecar = None
        self._sidecar_signatures.clear()
        self._sidecar_plan = None

    def seed(
        self,
        sidecar: MiniMaxH3AdalnCache,
        modules: Mapping[str, nn.Module],
        *,
        frozen: bool = False,
    ) -> None:
        if self.weights_pruned:
            raise RuntimeError("Cannot replace the AdaLN sidecar after projection weights were pruned")
        # A rejected seed must not retain the sidecar's device payload.
        signatures = {}
        for name, module in modules.items():
            try:
                signatures[name] = self._signature(module)
            except RuntimeError:
                if not frozen:
                    raise
                # Inference tensors have no version counter. The runtime-built
                # sidecar is immediately followed by pruning, so module and
                # storage identity are sufficient for this one-way handoff.
                tensors = [*module.named_parameters(), *module.named_buffers()]
                signatures[name] = (
                    id(module),
                    *(
                        (tensor_name, id(value), value.data_ptr(), tuple(value.shape), value.dtype, value.device)
                        for tensor_name, value in tensors
                    ),
                )
        self.sidecar = sidecar
        self._sidecar_signatures = signatures

    @torch.compiler.disable
    def prepare(self, embedding: torch.Tensor) -> None:
        if self.weights_pruned:
            # The serving schedule and rank-local projection weights were
            # validated before pruning. Hashing this small GPU tensor through
            # .cpu() synchronized every denoising step for no runtime benefit.
            if self._active_pruned_step is None:
                raise RuntimeError("Select the pruned AdaLN schedule step before model forward")
            self._input = weakref.ref(embedding)
            self._key = None
            return
        # Hashing and cache bookkeeping stay outside compiled tensor compute.
        # Regional compilation already calls this from the model's outer loop;
        # this boundary also preserves reuse under full-model compilation.
        self._sidecar_plan = None
        super().prepare(embedding)
        if self._key is None or self.sidecar is None:
            return
        if math_identity(embedding.device) != self.sidecar.manifest["math"]:
            if self.weights_pruned:
                raise RuntimeError("Pruned AdaLN cache numerical environment no longer matches its sidecar")
            self.sidecar = None
            return
        self._sidecar_plan = self.sidecar._embedding_plans.get(self._key)

    def select_pruned_step(self, step: int) -> None:
        """Select the already-validated table row using the denoiser loop index."""
        if not self.weights_pruned:
            return
        if self.sidecar is None:
            raise RuntimeError("Pruned AdaLN model lost its schedule cache")
        if type(step) is not int or not 0 <= step < len(self.sidecar._step_plan_indices):
            raise RuntimeError(f"Pruned AdaLN schedule has no denoising step {step}")
        if step == 0:
            device = self.sidecar.block_params.device
            if torch.is_grad_enabled() or torch.compiler.is_compiling():
                raise RuntimeError("Pruned AdaLN projections require eager inference")
            if torch.is_autocast_enabled(device.type) or math_identity(device) != self.sidecar.manifest["math"]:
                raise RuntimeError("Pruned AdaLN cache numerical environment no longer matches its sidecar")
            self._request_math_verified = True
        elif not self._request_math_verified:
            raise RuntimeError("Pruned AdaLN schedule must start at step zero")
        self._active_pruned_step = step
        self._sidecar_plan = self.sidecar._step_plan_indices[step]
        self._input = None
        self._key = None

    def project_pruned(
        self, name: str, embedding: torch.Tensor, *, layer_index: int | None = None
    ) -> torch.Tensor:
        """Read a verified schedule result after its source projection is removed."""
        if not self.weights_pruned:
            raise RuntimeError("AdaLN projections have not been pruned")
        if self._active_pruned_step is None or self._input is None or self._input() is not embedding:
            raise RuntimeError("Pruned AdaLN projection requires a prepared timestep embedding")
        if self.sidecar is None or self._sidecar_plan is None:
            raise RuntimeError("Pruned AdaLN projection does not cover this timestep schedule")
        index, count = self._sidecar_plan, embedding.shape[0]
        if layer_index is not None or name.startswith("blocks."):
            if layer_index is None:
                signature = self._sidecar_signatures.get(name)
                if signature is None:
                    raise RuntimeError(f"Pruned AdaLN projection has no verified weight identity: {name}")
                layer_index = int(name.split(".")[1])
            if self.sidecar._block_major:
                value = self.sidecar.block_params[index, layer_index, :count]
            else:
                # Older serialized v4 sidecars use [plan, timestep, layer, feature].
                value = self.sidecar.block_params[index, :count, layer_index].contiguous()
        else:
            signature = self._sidecar_signatures.get(name)
            if signature is None:
                raise RuntimeError(f"Pruned AdaLN projection has no verified weight identity: {name}")
            value = self.sidecar.final_params[index, :count]

        # Match the collective boundary that the normal projection cache uses,
        # but rendezvous only with the Ulysses peers that exchange A2A tensors.
        # Keep the vote at each block so collective phases cannot drift across
        # the 50-layer loop; the final projection has no later block A2A to align.
        if name.startswith("blocks.") and torch.distributed.is_initialized():
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            sp_group = get_sp_group()
            if sp_group.world_size > 1:
                vote = torch.ones((1,), device=embedding.device, dtype=torch.int32)
                consensus = sp_group.all_reduce(vote)
                if int(consensus.item()) != sp_group.world_size:
                    raise RuntimeError(f"Pruned AdaLN block vote disagrees at step {self._active_pruned_step}")
        return value

    def _lookup_precomputed(self, name: str, embedding: torch.Tensor, signature: Any) -> torch.Tensor | None:
        if (
            self.sidecar is None
            or self._sidecar_plan is None
            or self._sidecar_signatures.get(name) != signature
            or torch.is_autocast_enabled(embedding.device.type)
            or math_identity(embedding.device) != self.sidecar.manifest["math"]
        ):
            return None
        index, count = self._sidecar_plan, embedding.shape[0]
        if name.startswith("blocks."):
            layer = int(name.split(".")[1])
            if self.sidecar._block_major:
                return self.sidecar.block_params[index, layer, :count]
            return self.sidecar.block_params[index, :count, layer].contiguous()
        return self.sidecar.final_params[index, :count].contiguous()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def projection_names(num_layers: int) -> frozenset[str]:
    prefixes = [f"blocks.{i}.adaln_proj.linear" for i in range(num_layers)]
    prefixes.append("final_layer.adaln_proj.linear")
    return frozenset(f"{prefix}.{kind}" for prefix in prefixes for kind in ("weight", "bias"))


def input_names(num_layers: int) -> frozenset[str]:
    return projection_names(num_layers) | {
        f"time_embedder.{module}.{kind}" for module in ("proj_in", "proj_out") for kind in ("weight", "bias")
    }


def architecture(arch: MiniMaxH3DiTArchConfig) -> dict[str, int]:
    return {
        key: getattr(arch, key)
        for key in ("num_layers", "hidden_size", "timestep_input_dim", "time_embed_hidden_size", "time_embed_dim")
    }


def math_identity(device: torch.device) -> dict[str, Any]:
    """Conservatively bind the builder's numerical environment (TP1 only)."""
    result = {
        "torch": str(torch.__version__),
        "device_type": device.type,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        from vllm_omni.platforms import current_omni_platform

        capability = current_omni_platform.get_device_capability(device.index or 0)
        result.update(
            cuda=torch.version.cuda,
            capability=list(capability) if capability is not None else None,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
    return result


def schedule_contract(
    *, mode: str, num_steps: int, base_schedule: Sequence[float] | None, flow_shift: float, audio_flow_shift: float
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported AdaLN cache mode: {mode}")
    if type(num_steps) is not int or num_steps < (2 if base_schedule is None else 1):
        raise ValueError("AdaLN cache requires a positive complete denoising schedule")
    for value in (flow_shift, audio_flow_shift):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("AdaLN cache shifts must be finite and positive")
    if base_schedule is not None:
        from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

        schedule = DMD2SigmaSchedule.from_positions(base_schedule)
        if schedule.num_inference_steps != num_steps:
            raise ValueError("AdaLN cache step count does not match the explicit schedule")
        base_schedule = schedule.base_schedule
    return dict(
        mode=mode,
        num_steps=num_steps,
        base_schedule=list(base_schedule) if base_schedule is not None else None,
        flow_shift=float(flow_shift),
        audio_flow_shift=float(audio_flow_shift),
    )


def timestep_plans(contract: Mapping[str, Any]) -> list[torch.Tensor]:
    """Return distinct serving timestep plans, without interpolation."""
    return [torch.tensor(row, dtype=torch.float32) for row in dict.fromkeys(_timestep_plan_rows(contract))]


def _timestep_plan_rows(contract: Mapping[str, Any]) -> list[tuple[float, ...]]:
    """Return one timestep tuple per denoising step, including repeated plans."""
    from .denoise_loop import MINIMAX_H3_AUDIO_REF_COND_TIMESTEP, MINIMAX_H3_IMGVID_COND_TIMESTEP

    video = minimax_h3_time_shift_sigmas(
        num_steps=contract["num_steps"], shift_scale=contract["flow_shift"], base_schedule=contract["base_schedule"]
    )
    audio = minimax_h3_time_shift_sigmas(
        num_steps=contract["num_steps"],
        shift_scale=contract["audio_flow_shift"],
        base_schedule=contract["base_schedule"],
    )
    plans = []
    for video_sigma, audio_sigma in zip(video[:-1], audio[:-1], strict=True):
        values = {
            "video": 1.0 - video_sigma,
            "audio": 1.0 - audio_sigma,
            "image": max(1.0 - video_sigma, MINIMAX_H3_IMGVID_COND_TIMESTEP),
            "audio_ref": max(1.0 - audio_sigma, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
        }
        plan = torch.tensor([values[key] for key in MODES[contract["mode"]]], dtype=torch.float32).unique(sorted=True)
        plans.append(tuple(plan.tolist()))
    return plans


def timestep_plan_indices(contract: Mapping[str, Any]) -> list[int]:
    """Map each loop step to its deduplicated table row."""
    distinct = {plan: index for index, plan in enumerate(dict.fromkeys(_timestep_plan_rows(contract)))}
    return [distinct[plan] for plan in _timestep_plan_rows(contract)]


class MiniMaxH3AdalnCache(nn.Module):
    """Validate an optional sidecar, then verify its weights at load.

    Hashes are checked against the *post-fusion* stream, before TP conversion.
    They cover all inputs of the cached computation, not unrelated QKV/MLP
    weights. A fixed adapter has an additional complete-file identity.
    """

    block_params: torch.Tensor
    final_params: torch.Tensor
    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    time_embeddings: torch.Tensor

    def __init__(self, arch: MiniMaxH3DiTArchConfig, *, path: str, model_variant: str) -> None:
        super().__init__()
        self.path = str(Path(path).expanduser().resolve())
        self.arch = arch
        self._verified: set[str] = set()
        self._adapter_bound = False
        self._ready = False
        with safe_open(self.path, framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError("Rebuild the AdaLN cache: a weight-bound format-v4 sidecar is required")
            try:
                manifest = json.loads(metadata["manifest"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid AdaLN cache manifest") from exc
            required = {"architecture", "model_variant", "schedule", "adapter_sha256", "weights", "payload", "math"}
            if not isinstance(manifest, dict) or set(manifest) != required:
                raise ValueError("Invalid AdaLN cache manifest fields")
            if any(
                not isinstance(manifest[key], dict)
                for key in ("architecture", "schedule", "weights", "payload", "math")
            ):
                raise ValueError("Invalid AdaLN cache manifest objects")
            if manifest["architecture"] != architecture(arch) or manifest["model_variant"] != model_variant:
                raise ValueError("AdaLN cache model variant or architecture mismatch")
            contract = schedule_contract(**manifest["schedule"])
            variant = "ref2va" if contract["mode"].startswith("ref2va") else "fl2va"
            if variant != model_variant:
                raise ValueError("AdaLN cache task does not match model variant")
            if set(manifest["weights"]) != input_names(arch.num_layers):
                raise ValueError("AdaLN cache must cover exactly the time embedding and AdaLN parameters")
            if set(source.keys()) != PAYLOAD_NAMES or set(manifest["payload"]) != PAYLOAD_NAMES:
                raise ValueError("Invalid AdaLN cache payload keys")
            for digest in [*manifest["weights"].values(), *manifest["payload"].values()]:
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid AdaLN cache SHA256 digest")
            adapter = manifest["adapter_sha256"]
            if adapter is not None and (
                not isinstance(adapter, str) or len(adapter) != 64 or any(c not in "0123456789abcdef" for c in adapter)
            ):
                raise ValueError("Invalid AdaLN cache adapter identity")
            payload = {name: source.get_tensor(name) for name in PAYLOAD_NAMES}
        plans = timestep_plans(contract)
        width = max(len(plan) for plan in plans)
        expected_lengths = torch.tensor([len(plan) for plan in plans], dtype=torch.int64)
        expected_plans = torch.zeros((len(plans), width), dtype=torch.float32)
        for index, plan in enumerate(plans):
            expected_plans[index, : len(plan)] = plan
        shapes = {
            "block_params": ((len(plans), width, arch.num_layers, 18 * arch.hidden_size), torch.bfloat16),
            "final_params": ((len(plans), width, 2 * arch.hidden_size), torch.bfloat16),
            "time_embeddings": ((len(plans), width, arch.time_embed_dim), torch.float32),
            "plan_timesteps": (tuple(expected_plans.shape), torch.float32),
            "plan_lengths": (tuple(expected_lengths.shape), torch.int64),
        }
        for name, value in payload.items():
            shape, dtype = shapes[name]
            if tuple(value.shape) != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid AdaLN cache tensor: {name}")
            if tensor_digest(value) != manifest["payload"][name]:
                raise ValueError(f"AdaLN cache payload checksum mismatch: {name}")
        if not torch.equal(payload["plan_timesteps"], expected_plans) or not torch.equal(
            payload["plan_lengths"], expected_lengths
        ):
            raise ValueError("AdaLN cache timestep plans do not match the declared schedule")
        self.manifest = manifest
        self.projection_names = projection_names(arch.num_layers)
        # Index the validated CPU payload once. Runtime prepare() already hashes
        # the actual embedding, so lookup needs no per-plan device comparisons.
        self._embedding_plans = {
            tensor_digest(payload["time_embeddings"][index, :count]): index
            for index, count in enumerate(expected_lengths.tolist())
        }
        self._step_plan_indices = timestep_plan_indices(contract)
        self._block_major = False
        for name, value in payload.items():
            self.register_buffer(name, value, persistent=False)

    @classmethod
    def from_runtime(
        cls,
        arch: MiniMaxH3DiTArchConfig,
        payload: Mapping[str, torch.Tensor],
        *,
        contract: Mapping[str, Any],
        model_variant: str,
        device: torch.device,
    ) -> MiniMaxH3AdalnCache:
        """Create an in-memory cache from this worker's actual TP projections.

        Unlike the portable offline artifact, this cache is generated after
        weight loading using the serving model's linear layers and collectives.
        It therefore retains the numerical behavior of the active TP/USP
        topology rather than assuming the offline builder's TP1 GEMM.
        """
        checked_contract = schedule_contract(**contract)
        plans = timestep_plans(checked_contract)
        width = max(len(plan) for plan in plans)
        expected_shapes = {
            "plan_timesteps": ((len(plans), width), torch.float32),
            "plan_lengths": ((len(plans),), torch.int64),
            "time_embeddings": ((len(plans), width, arch.time_embed_dim), torch.float32),
            "block_params": ((len(plans), arch.num_layers, width, 18 * arch.hidden_size), torch.bfloat16),
            "final_params": ((len(plans), width, 2 * arch.hidden_size), torch.bfloat16),
        }
        if set(payload) != PAYLOAD_NAMES:
            raise ValueError("Invalid in-memory AdaLN cache payload keys")
        for name, (shape, dtype) in expected_shapes.items():
            value = payload[name]
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
                raise ValueError(f"Invalid in-memory AdaLN cache tensor: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite in-memory AdaLN cache tensor: {name}")

        expected_lengths = torch.tensor([len(plan) for plan in plans], dtype=torch.int64, device=device)
        expected_timesteps = torch.zeros((len(plans), width), dtype=torch.float32, device=device)
        for index, plan in enumerate(plans):
            expected_timesteps[index, : len(plan)] = plan.to(device=device)
        if not torch.equal(payload["plan_lengths"], expected_lengths) or not torch.equal(
            payload["plan_timesteps"], expected_timesteps
        ):
            raise ValueError("In-memory AdaLN cache plans do not match the request schedule")

        cache = cls.__new__(cls)
        nn.Module.__init__(cache)
        cache.path = "<runtime>"
        cache.arch = arch
        cache._verified = set(input_names(arch.num_layers))
        cache._adapter_bound = True
        cache._ready = True
        cache.manifest = {
            "architecture": architecture(arch),
            "model_variant": model_variant,
            "schedule": checked_contract,
            "adapter_sha256": None,
            "weights": {},
            "payload": {},
            "math": math_identity(device),
        }
        cache.projection_names = projection_names(arch.num_layers)
        cache._embedding_plans = {
            tensor_digest(payload["time_embeddings"][index, : len(plan)]): index
            for index, plan in enumerate(plans)
        }
        cache._step_plan_indices = timestep_plan_indices(checked_contract)
        cache._block_major = True
        for name in PAYLOAD_NAMES:
            cache.register_buffer(name, payload[name], persistent=False)
        return cache

    def bind_adapter(self, adapter_sha256: str | None) -> None:
        if adapter_sha256 != self.manifest["adapter_sha256"]:
            raise ValueError("AdaLN cache adapter mismatch; rebuild for the active fixed adapter")
        self._adapter_bound = True

    def verify_weight(self, name: str, weight: torch.Tensor) -> None:
        if name not in self.manifest["weights"]:
            return
        if self._ready or name in self._verified:
            raise ValueError(f"AdaLN input was loaded more than once: {name}")
        if tensor_digest(weight) != self.manifest["weights"][name]:
            raise ValueError(f"AdaLN cache effective weight mismatch: {name}")
        self._verified.add(name)

    def finish_loading(self, device: torch.device) -> None:
        if not self._adapter_bound:
            raise ValueError("AdaLN cache requires the active adapter identity before loading")
        missing = input_names(self.arch.num_layers) - self._verified
        if missing:
            raise ValueError(f"Missing AdaLN cache source weights: {sorted(missing)}")
        if math_identity(device) != self.manifest["math"]:
            raise ValueError("AdaLN cache numerical environment mismatch; rebuild on the serving configuration")
        self.to(device=device)
        self._ready = True

    def check_request(
        self,
        *,
        mode: str,
        num_steps: int,
        base_schedule: Sequence[float] | None,
        flow_shift: float,
        audio_flow_shift: float,
    ) -> None:
        contract = schedule_contract(
            mode=mode,
            num_steps=num_steps,
            base_schedule=base_schedule,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        if contract != self.manifest["schedule"]:
            raise ValueError("AdaLN cache request schedule/task mismatch; rebuild or disable the cache")

    def lookup(self, timesteps: torch.Tensor) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, ...]]:
        if not self._ready:
            raise RuntimeError("AdaLN cache has not verified its source weights")
        if timesteps.dtype != torch.float32:
            raise ValueError("AdaLN cache requires exact FP32 timesteps")
        values = timesteps.reshape(-1).to(self.plan_timesteps.device)
        count = values.numel()
        if not 0 < count <= self.plan_timesteps.shape[1]:
            raise ValueError("AdaLN cache does not cover this timestep plan")
        matches = self.plan_lengths.eq(count) & self.plan_timesteps[:, :count].eq(values).all(dim=1)
        if not bool(matches.any()):
            raise ValueError("AdaLN cache does not cover this timestep plan")
        index = matches.to(torch.int64).argmax()
        blocks = self.block_params.permute(2, 0, 1, 3)[:, index, :count]
        blocks = blocks.reshape(self.arch.num_layers, -1, 6, self.arch.hidden_size)
        final = self.final_params[index, :count].reshape(-1, 2, self.arch.hidden_size)
        return tuple(tuple(layer.unbind(1)) for layer in blocks), tuple(final.unbind(1))


@torch.inference_mode()
def build_cache(
    arch: MiniMaxH3DiTArchConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    contract: dict[str, Any],
    model_variant: str,
    adapter_sha256: str | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stream one projection at a time; do not instantiate the full DiT.

    The four time-embedder tensors must precede the AdaLN projections. Callers
    fuse any fixed adapter before supplying this stream, just like serving.
    """
    from .minimax_h3_transformer import MiniMaxH3TimeEmbedder

    contract = schedule_contract(**contract)
    plans = timestep_plans(contract)
    width = max(len(plan) for plan in plans)
    payload = {
        "plan_timesteps": torch.zeros((len(plans), width), dtype=torch.float32),
        "plan_lengths": torch.tensor([len(plan) for plan in plans], dtype=torch.int64),
        "block_params": torch.zeros((len(plans), width, arch.num_layers, 18 * arch.hidden_size), dtype=torch.bfloat16),
        "final_params": torch.zeros((len(plans), width, 2 * arch.hidden_size), dtype=torch.bfloat16),
        "time_embeddings": torch.zeros((len(plans), width, arch.time_embed_dim), dtype=torch.float32),
    }
    for i, plan in enumerate(plans):
        payload["plan_timesteps"][i, : len(plan)] = plan

    # Use the runtime embedding implementation, including FP32 boundaries. A
    # lightweight module supplies ordinary Linear wrappers with TP1 semantics.
    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    embedder = MiniMaxH3TimeEmbedder.__new__(MiniMaxH3TimeEmbedder)
    nn.Module.__init__(embedder)
    embedder.frequency_embedding_size = arch.timestep_input_dim
    embedder.proj_in = Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, device=device, dtype=torch.float32)
    embedder.proj_out = Linear(arch.time_embed_hidden_size, arch.time_embed_dim, device=device, dtype=torch.float32)
    fingerprints: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    inputs: list[torch.Tensor] | None = None
    for name, weight in weights:
        if name not in input_names(arch.num_layers) or name in fingerprints:
            raise ValueError(f"Unexpected or duplicate AdaLN builder input: {name}")
        dtype = torch.float32 if name.startswith("time_embedder.") else torch.bfloat16
        if weight.dtype != dtype or not bool(torch.isfinite(weight).all()):
            raise ValueError(f"Invalid AdaLN builder input dtype/values: {name}")
        fingerprints[name] = tensor_digest(weight)
        if name.startswith("time_embedder."):
            target = dict(embedder.named_parameters())[name.removeprefix("time_embedder.")]
            if weight.shape != target.shape:
                raise ValueError(f"Invalid AdaLN time embedding shape: {name}")
            target.copy_(weight)
            continue
        if inputs is None:
            if len([key for key in fingerprints if key.startswith("time_embedder.")]) != 4:
                raise ValueError("Supply all four time embedding weights before the AdaLN projections")
            embeddings = [embedder(plan.to(device)) for plan in plans]
            for index, embedding in enumerate(embeddings):
                payload["time_embeddings"][index, : len(embedding)].copy_(embedding.cpu())
            inputs = [nn.functional.silu(embedding).to(torch.bfloat16) for embedding in embeddings]
        pending[name] = weight.to(device)
        prefix = name.rsplit(".", 1)[0]
        if f"{prefix}.weight" not in pending or f"{prefix}.bias" not in pending:
            continue
        w, bias = pending.pop(f"{prefix}.weight"), pending.pop(f"{prefix}.bias")
        is_final = prefix.startswith("final_layer.")
        out_dim = (2 if is_final else 18) * arch.hidden_size
        if w.shape != (out_dim, arch.time_embed_dim) or bias.shape != (out_dim,):
            raise ValueError(f"Invalid AdaLN projection shape: {prefix}")
        for i, x in enumerate(inputs):
            result = nn.functional.linear(x, w, bias).cpu()
            if is_final:
                payload["final_params"][i, : len(x)].copy_(result)
            else:
                payload["block_params"][i, : len(x), int(prefix.split(".")[1])].copy_(result)
        del w, bias
    if set(fingerprints) != input_names(arch.num_layers) or pending:
        raise ValueError("Incomplete AdaLN builder input weights")
    if any(not bool(torch.isfinite(value).all()) for value in payload.values()):
        raise ValueError("Non-finite AdaLN builder output")
    manifest = dict(
        architecture=architecture(arch),
        model_variant=model_variant,
        schedule=contract,
        adapter_sha256=adapter_sha256,
        weights=fingerprints,
        payload={name: tensor_digest(value) for name, value in payload.items()},
        math=math_identity(device),
    )
    return payload, manifest

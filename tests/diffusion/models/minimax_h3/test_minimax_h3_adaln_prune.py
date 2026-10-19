# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    MiniMaxH3RuntimeAdalnCache,
    schedule_contract,
    timestep_plans,
    timestep_plan_indices,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.offloader.config import OffloadStrategy
from vllm_omni.diffusion.offloader import distributed_layerwise_backend as dlo_backend
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import PinnedResidentLayerGroup
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _build_adaln_schedule_contract,
    _prepare_and_prune_adaln_weights,
)

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion]


def test_pruned_projection_uses_only_verified_schedule_result(monkeypatch):
    calls = []

    class FakeSPGroup:
        world_size = 2

        def all_reduce(self, value):
            calls.append(tuple(value.shape))
            return value * 2

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_sp_group", lambda: FakeSPGroup()
    )
    cache = MiniMaxH3RuntimeAdalnCache(prune_weights=True)
    embedding = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    result = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    name = "blocks.0.adaln_proj.linear"
    cache.weights_pruned = True
    cache.sidecar = SimpleNamespace(
        block_params=result.reshape(1, 1, 1, 2),
        final_params=torch.empty(1, 1, 2),
        _block_major=True,
    )
    cache._sidecar_signatures[name] = ("verified",)
    cache._sidecar_plan = 0
    cache._active_pruned_step = 0
    cache._input = weakref.ref(embedding)

    with torch.inference_mode():
        actual = cache.project_pruned(name, embedding)

    torch.testing.assert_close(actual, result)
    assert actual.is_contiguous()
    assert calls == [(1,)]
    with torch.inference_mode(), pytest.raises(RuntimeError, match="prepared timestep"):
        cache.project_pruned(name, embedding.clone())


def test_prune_adaln_weights_removes_every_projection_parameter():
    cache = SimpleNamespace(
        prune_weights_requested=True,
        weights_pruned=False,
        sidecar=SimpleNamespace(_ready=True, manifest={"schedule": {"num_steps": 50}}),
        _sidecar_signatures={
            "blocks.0.adaln_proj.linear": ("block",),
            "final_layer.adaln_proj.linear": ("final",),
        },
    )
    block_projection = SimpleNamespace(linear=nn.Linear(4, 6, bias=True), _weights_pruned=False)
    final_projection = SimpleNamespace(linear=nn.Linear(4, 2, bias=True), _weights_pruned=False)
    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.adaln_cache = cache
    model.blocks = [SimpleNamespace(adaln_proj=block_projection)]
    model.final_layer = SimpleNamespace(adaln_proj=final_projection)

    removed = MiniMaxH3DiTModel.prune_adaln_weights(model)

    expected = (4 * 6 + 6) + (4 * 2 + 2)
    assert removed == expected
    assert block_projection.linear is None
    assert final_projection.linear is None
    assert block_projection._weights_pruned and final_projection._weights_pruned
    assert cache.weights_pruned
    assert cache.pruned_schedule == {"num_steps": 50}


def test_pruned_cache_cannot_be_cleared():
    cache = MiniMaxH3RuntimeAdalnCache(prune_weights=True)
    cache.weights_pruned = True
    with pytest.raises(RuntimeError, match="projection weights were released"):
        cache.clear()


def test_short_warmup_does_not_lock_pruning_schedule():
    cache = MiniMaxH3RuntimeAdalnCache(prune_weights=True, prune_min_steps=50)
    assert not cache.should_prune(2)
    assert cache.should_prune(50)


def test_pipeline_builds_one_contract_for_validation_and_pruning():
    cache = SimpleNamespace(sidecar=None, prune_weights_requested=True)
    contract = _build_adaln_schedule_contract(
        cache,
        task="ref2va",
        num_steps=50,
        base_schedule=None,
        video_shift=12.0,
        audio_shift=3.0,
        visual_shapes=[(124, 20, 30)],
        audio_lengths=[1000],
    )
    assert contract == {
        "mode": "ref2va-mixed",
        "num_steps": 50,
        "base_schedule": None,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }


def test_dlo_adaln_precompute_stages_before_prune_and_full_dlo_load():
    events = []
    block_projection = SimpleNamespace(linear=nn.Linear(2, 3))
    final_projection = SimpleNamespace(linear=nn.Linear(2, 2))
    cache = SimpleNamespace(sidecar=None)
    transformer = SimpleNamespace(
        adaln_cache=cache,
        blocks=[SimpleNamespace(adaln_proj=block_projection)],
        final_layer=SimpleNamespace(adaln_proj=final_projection),
    )

    def prepare(_contract, *, parameter_stager):
        events.append("prepare")
        parameters = tuple(block_projection.linear.parameters())
        parameter_stager.stage_resident_parameters(parameters)
        parameter_stager.offload_resident_parameters(parameters)
        cache.sidecar = SimpleNamespace(check_request=lambda **_kwargs: None)

    def prune():
        events.append("prune")
        block_projection.linear = None
        final_projection.linear = None
        return 10

    transformer.prepare_adaln_pruning = prepare
    transformer.prune_adaln_weights = prune
    controller = SimpleNamespace(
        stage_resident_parameters=lambda params: events.append(("stage", len(params))) or 123,
        offload_resident_parameters=lambda params: events.append(("offload", len(params))),
        release_pruned_resident_parameters=lambda params: events.append(("release", len(params))) or 6,
    )

    removed = _prepare_and_prune_adaln_weights(
        transformer,
        {"num_steps": 50},
        residency_controller=controller,
    )

    assert removed == 10
    assert events == ["prepare", ("stage", 2), ("offload", 2), "prune", ("release", 4)]


def test_dlo_adaln_precompute_rolls_back_selective_staging_on_failure():
    events = []
    block_projection = SimpleNamespace(linear=nn.Linear(2, 3))

    def fail_during_precompute(_contract, *, parameter_stager):
        parameter_stager.stage_resident_parameters(tuple(block_projection.linear.parameters()))
        raise RuntimeError("precompute failed")

    transformer = SimpleNamespace(
        adaln_cache=SimpleNamespace(sidecar=None),
        blocks=[SimpleNamespace(adaln_proj=block_projection)],
        final_layer=SimpleNamespace(adaln_proj=SimpleNamespace(linear=nn.Linear(2, 2))),
        prepare_adaln_pruning=fail_during_precompute,
    )
    controller = SimpleNamespace(
        stage_resident_parameters=lambda params: events.append(("stage", len(params))) or 1,
        offload_resident_parameters=lambda params: events.append(("rollback", len(params))),
    )

    with pytest.raises(RuntimeError, match="precompute failed"):
        _prepare_and_prune_adaln_weights(transformer, {}, residency_controller=controller)

    assert events == [("stage", 2), ("rollback", 2)]


def test_all_resident_dlo_is_compatible_with_adaln_pruning():
    compatible = h3._adaln_pruning_conflicts_with_offload
    assert not compatible(
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        offloads_dit=True,
        resident_layers=50,
        num_layers=50,
        use_hsdp=False,
    )
    assert compatible(
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        offloads_dit=True,
        resident_layers=49,
        num_layers=50,
        use_hsdp=False,
    )
    assert compatible(
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        offloads_dit=True,
        resident_layers=50,
        num_layers=50,
        use_hsdp=True,
    )


def test_resident_group_prunes_cpu_metadata_without_copying_host_backing():
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    expected_numel = first.numel()
    second = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))
    group = object.__new__(PinnedResidentLayerGroup)
    group.loaded = False
    group.rank_local_mmap = False
    group.pin_memory = False
    group._states = [
        {
            "targets": {"first": first, "second": second},
            "cpu_shards": {torch.float32: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])},
            "cpu_sources": {},
            "metadata": {
                torch.float32: [
                    {"name": "first", "offset": 0, "numel": 2, "shape": (2,), "stride": (1,)},
                    {"name": "second", "offset": 2, "numel": 3, "shape": (3,), "stride": (1,)},
                ]
            },
        }
    ]

    released = group.exclude_parameters({id(first)})

    assert released == expected_numel
    assert first.numel() == 0
    assert set(group._states[0]["targets"]) == {"second"}
    assert torch.equal(
        group._states[0]["cpu_shards"][torch.float32],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    assert group._states[0]["metadata"][torch.float32][0]["offset"] == 0
    assert group._states[0]["metadata"][torch.float32][0]["cpu_offset"] == 2


def test_rank_local_mmap_resident_group_releases_pruned_sources():
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    expected_numel = first.numel()
    second = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))
    group = object.__new__(PinnedResidentLayerGroup)
    group.loaded = False
    group.rank_local_mmap = True
    group.pin_memory = False
    group._cpu_staging_buffers = [{torch.float32: torch.empty(5)}, {torch.float32: torch.empty(5)}]
    group._cpu_staging_events = [None, None]
    group._states = [
        {
            "targets": {"first": first, "second": second},
            "cpu_shards": {},
            "cpu_sources": {
                torch.float32: [
                    {"name": "first", "tensor": first.detach(), "transform": None},
                    {"name": "second", "tensor": second.detach(), "transform": None},
                ]
            },
            "metadata": {
                torch.float32: [
                    {"name": "first", "offset": 0, "numel": 2, "shape": (2,), "stride": (1,)},
                    {"name": "second", "offset": 2, "numel": 3, "shape": (3,), "stride": (1,)},
                ]
            },
        }
    ]

    released = group.exclude_parameters({id(first)})

    assert released == expected_numel
    assert first.numel() == 0
    assert [source["name"] for source in group._states[0]["cpu_sources"][torch.float32]] == ["second"]
    assert group._states[0]["metadata"][torch.float32][0]["offset"] == 0
    assert group._cpu_staging_buffers[0][torch.float32].numel() == 3


def test_loaded_resident_group_compacts_device_buffer_without_reloading(monkeypatch):
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))
    expected_numel = first.numel()
    device_buffer = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    group = object.__new__(PinnedResidentLayerGroup)
    group.loaded = True
    group.rank_local_mmap = False
    group.pin_memory = False
    group.device = torch.device("cpu")
    group._gpu_buffers = [{torch.float32: device_buffer}]
    group._states = [
        {
            "targets": {"first": first, "second": second},
            "cpu_shards": {torch.float32: device_buffer.clone()},
            "cpu_sources": {},
            "metadata": {
                torch.float32: [
                    {"name": "first", "offset": 0, "numel": 2, "shape": (2,), "stride": (1,)},
                    {"name": "second", "offset": 2, "numel": 3, "shape": (3,), "stride": (1,)},
                ]
            },
        }
    ]
    monkeypatch.setattr(dlo_backend.current_omni_platform, "synchronize", lambda: None)

    released = group.exclude_parameters({id(first)})

    assert released == expected_numel
    assert group.loaded
    assert group._gpu_buffers[0][torch.float32].numel() == second.numel()
    assert torch.equal(second, torch.tensor([3.0, 4.0, 5.0]))
    assert first.numel() == 0


def test_live_schedule_precompute_supports_tp_and_prunes_exact_outputs(monkeypatch):
    class FakeTimeEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = nn.Linear(1, 2)

        def forward(self, timesteps):
            return torch.stack((timesteps, timesteps.square()), dim=-1).to(torch.float32)

    class FakeProjection:
        def __init__(self, out_features):
            self.linear = nn.Linear(2, out_features, dtype=torch.bfloat16)
            self._weights_pruned = False

        def _compute_projection(self, embedding):
            return torch.nn.functional.linear(
                torch.nn.functional.silu(embedding).to(torch.bfloat16),
                self.linear.weight,
                self.linear.bias,
            )

    architecture = h3.MiniMaxH3DiTArchConfig(
        num_layers=1,
        hidden_size=2,
        timestep_input_dim=2,
        time_embed_hidden_size=2,
        time_embed_dim=2,
    )
    block_projection = FakeProjection(18 * architecture.hidden_size)
    final_projection = FakeProjection(2 * architecture.hidden_size)
    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = architecture
    model.adaln_cache = MiniMaxH3RuntimeAdalnCache(prune_weights=True, prune_min_steps=1)
    model.time_embedder = FakeTimeEmbedder()
    model.blocks = [SimpleNamespace(adaln_proj=block_projection)]
    model.final_layer = SimpleNamespace(adaln_proj=final_projection)
    model.parallel_config = SimpleNamespace(ulysses_degree=2)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 4)
    contract = schedule_contract(
        mode="t2va",
        num_steps=4,
        base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0),
        flow_shift=12.0,
        audio_flow_shift=3.0,
    )

    model.prepare_adaln_pruning(contract)
    sidecar = model.adaln_cache.sidecar
    assert sidecar is not None
    embeddings = []
    with torch.inference_mode():
        for index, plan in enumerate(timestep_plans(contract)):
            embedding = model.time_embedder(plan)
            embeddings.append(embedding)
            model.adaln_cache.prepare(embedding)
            for name, projection, expected_values in (
                (
                    "blocks.0.adaln_proj.linear",
                    block_projection,
                    sidecar.block_params[index, 0, : len(plan)],
                ),
                (
                    "final_layer.adaln_proj.linear",
                    final_projection,
                    sidecar.final_params[index, : len(plan)],
                ),
            ):
                signature = model.adaln_cache._sidecar_signatures[name]
                actual = model.adaln_cache._lookup_precomputed(name, embedding, signature)
                expected = projection._compute_projection(embedding)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected_values, rtol=0, atol=0)

    model.prune_adaln_weights()
    assert block_projection.linear is None
    assert final_projection.linear is None
    with torch.inference_mode():
        for step, plan_index in enumerate(model.adaln_cache.sidecar._step_plan_indices):
            embedding = embeddings[plan_index]
            model.adaln_cache.select_pruned_step(step)
            model.adaln_cache.prepare(embedding)
            actual = model.adaln_cache.project_pruned("blocks.0.adaln_proj.linear", embedding)
            assert actual.shape == (embedding.shape[0], 18 * architecture.hidden_size)
            assert actual.is_contiguous()


def test_timestep_plan_indices_preserve_repeated_loop_steps():
    contract = schedule_contract(
        mode="t2va",
        num_steps=8,
        base_schedule=(0.999, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.0),
        flow_shift=12.0,
        audio_flow_shift=3.0,
    )
    plans = timestep_plans(contract)
    indices = timestep_plan_indices(contract)
    assert len(indices) == contract["num_steps"]
    assert all(0 <= index < len(plans) for index in indices)

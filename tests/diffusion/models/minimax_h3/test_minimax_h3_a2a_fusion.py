# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from vllm_omni.diffusion.attention.parallel.ulysses import (
    UlyssesParallelAttention,
)
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_fused_ulysses_case(
    local_rank: int,
    world_size: int,
    mode: str,
    local_seq_lens: tuple[int, ...],
    head_counts: tuple[int, int, int],
    fusion_setting: str | None,
    master_port: int,
) -> None:
    configure_cpu_collective_worker()
    os.environ["MINIMAX_H3_QKV_A2A_FUSION"] = fusion_setting
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    try:
        init_distributed_environment(backend="gloo")
        initialize_model_parallel(ulysses_degree=world_size)
        sp_group = get_sp_group()
        group = sp_group.ulysses_group

        batch, head_dim = 1, 3
        local_seq_len = local_seq_lens[local_rank]
        qkv = tuple(
            torch.arange(batch * local_seq_len * heads * head_dim, dtype=torch.float32).reshape(
                batch, local_seq_len, heads, head_dim
            )
            + 1000 * local_rank
            + 100000 * index
            for index, heads in enumerate(head_counts)
        )

        # Establish a real CPU/Gloo oracle using the original three-collective path.
        expected = tuple(SeqAllToAll4D.apply(group, tensor, 2, 1, False) for tensor in qkv)

        calls = {"all_to_all_single": 0, "all_gather": 0}
        original_all_to_all_single = dist.all_to_all_single
        original_all_gather = dist.all_gather

        def counted_all_to_all_single(*args, **kwargs):
            calls["all_to_all_single"] += 1
            return original_all_to_all_single(*args, **kwargs)

        def counted_all_gather(*args, **kwargs):
            calls["all_gather"] += 1
            return original_all_gather(*args, **kwargs)

        dist.all_to_all_single = counted_all_to_all_single
        dist.all_gather = counted_all_gather
        context = ForwardContext(
            omni_diffusion_config=SimpleNamespace(
                parallel_config=SimpleNamespace(ulysses_mode=mode, sequence_parallel_size=world_size)
            )
        )
        context.minimax_h3_qkv_fusion = True
        strategy = UlyssesParallelAttention(sp_group, 2, 1, use_sync=False)
        same_qkv_heads = head_counts[0] == head_counts[1] == head_counts[2]
        expected_qkv_a2as = 1 if fusion_setting == "1" and same_qkv_heads else 3
        try:
            with override_forward_context(context):
                fused = strategy.pre_attention(*qkv, attn_metadata=None)[:3]
                for actual, reference in zip(fused, expected, strict=True):
                    assert torch.equal(actual, reference)
                assert calls["all_to_all_single"] == expected_qkv_a2as
                assert calls["all_gather"] == 0
        finally:
            dist.all_to_all_single = original_all_to_all_single
            dist.all_gather = original_all_gather
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize(
    ("mode", "local_seq_lens", "head_counts", "fusion_setting"),
    [
        ("strict", (4, 4), (4, 4, 4), "1"),
        ("strict", (4, 4), (4, 2, 2), "1"),
        ("strict", (4, 4), (4, 4, 4), "0"),
    ],
)
def test_h3_qkv_fusion_matches_three_a2as_and_is_opt_out(
    mode: str,
    local_seq_lens: tuple[int, ...],
    head_counts: tuple[int, int, int],
    fusion_setting: str | None,
) -> None:
    start_collective_workers(
        _run_fused_ulysses_case,
        args=(2, mode, local_seq_lens, head_counts, fusion_setting, _open_port()),
        nprocs=2,
        device_kind="cpu",
    )

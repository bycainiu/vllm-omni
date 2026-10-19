# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Four-rank CUDA correctness check for MiniMax-H3 communication codecs."""

import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.parallel.h3_comm_quant import (
    output_all_to_all_fp8,
    qkv_all_to_all_int8,
)


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, limit: float) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    error = (actual_f - expected_f).abs()
    max_abs = error.max().item()
    rms_rel = (error.square().mean().sqrt() / expected_f.square().mean().sqrt()).item()
    print(f"{name}: max_abs={max_abs:.6g} rms_rel={rms_rel:.6g}", flush=True)
    if rms_rel > limit:
        raise AssertionError(f"{name}: rms relative error {rms_rel:.6g} exceeds {limit}")


def native_qkv(pg, q, k, v):
    rank = dist.get_rank(pg)
    world = dist.get_world_size(pg)
    batch, rows, heads, dim = q.shape
    heads_local = heads // world
    packed = torch.stack((q, k, v), dim=2)
    send = packed.reshape(batch, rows, 3, world, heads_local, dim)
    send = send.permute(3, 1, 0, 2, 4, 5).contiguous().flatten(0, 1)
    recv = torch.empty((world * rows, batch, 3, heads_local, dim), device=q.device, dtype=q.dtype)
    dist.all_to_all_single(recv, send, group=pg)
    recv = recv.permute(1, 0, 2, 3, 4).contiguous()
    return recv.unbind(dim=2)


def native_reverse(pg, x):
    world = dist.get_world_size(pg)
    batch, global_rows, heads_local, dim = x.shape
    rows = global_rows // world
    send = x.reshape(batch, world, rows, heads_local, dim).permute(1, 2, 0, 3, 4).contiguous()
    send = send.flatten(0, 1)
    recv = torch.empty((world * rows, batch, heads_local, dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(recv, send, group=pg)
    return recv.reshape(world, rows, batch, heads_local, dim).permute(2, 1, 0, 3, 4).reshape(
        batch, rows, world * heads_local, dim
    )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    pg = dist.group.WORLD
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(1000 + rank)
    device = torch.device("cuda", local_rank)

    batch, rows, heads, dim = 2, 8, 8, 128
    q, k, v = (torch.randn((batch, rows, heads, dim), device=device, dtype=torch.bfloat16) for _ in range(3))
    expected_qkv = native_qkv(pg, q, k, v)
    got_qkv = qkv_all_to_all_int8(
        pg,
        q,
        k,
        v,
        seq_lens=[rows] * world,
        padded_head_cnt=heads,
        use_sync=False,
    )
    for name, got, expected in zip(("q", "k", "v"), got_qkv, expected_qkv):
        check_close(f"QKV INT8 {name}", got, expected, limit=0.018)

    local_rows = rows
    global_rows = world * local_rows
    heads_local = heads // world
    output = torch.randn((batch, global_rows, heads_local, dim), device=device, dtype=torch.bfloat16)
    expected_output = native_reverse(pg, output)
    got_output = output_all_to_all_fp8(
        pg,
        output,
        seq_lens=[local_rows] * world,
        local_seq_len=local_rows,
        orig_head_cnt=heads,
        use_sync=False,
    )
    check_close("attention output FP8", got_output, expected_output, limit=0.035)
    if rank == 0:
        print("COMM_QUANT_SMOKE_PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 low-precision Ulysses wire formats.

The kernels fuse destination-major packing with QKV quantization, and fuse
FP8 output decoding with the reverse Ulysses head merge. The values used by
attention remain BF16; only the inter-GPU payload is quantized.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


QKV_GROUP = 32
QKV_VECTOR = 128
QKV_PACKET = 408  # 3 * 128 signed INT8 bytes + 3 * 4 FP16 scales


@triton.jit
def _pack_qkv_int8_kernel(
    q_ptr, k_ptr, v_ptr, packet_ptr,
    rows, batch, heads, heads_local, world,
    q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    k_stride_b, k_stride_s, k_stride_h, k_stride_d,
    v_stride_b, v_stride_s, v_stride_h, v_stride_d,
    RECORDS: tl.constexpr,
    VECTOR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    PACKET_SIZE: tl.constexpr,
):
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    dim = tl.arange(0, VECTOR)
    local_head = record % heads_local
    batch_index = (record // heads_local) % batch
    row = (record // (heads_local * batch)) % rows
    destination = record // (heads_local * batch * rows)
    global_head = destination * heads_local + local_head
    valid_record = record < world * rows * batch * heads_local
    valid_head = valid_record & (global_head < heads)
    q_mask = valid_head[:, None] & (dim[None, :] < VECTOR)

    q = tl.load(
        q_ptr + batch_index[:, None] * q_stride_b + row[:, None] * q_stride_s
        + global_head[:, None] * q_stride_h + dim[None, :] * q_stride_d,
        mask=q_mask, other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + batch_index[:, None] * k_stride_b + row[:, None] * k_stride_s
        + global_head[:, None] * k_stride_h + dim[None, :] * k_stride_d,
        mask=q_mask, other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + batch_index[:, None] * v_stride_b + row[:, None] * v_stride_s
        + global_head[:, None] * v_stride_h + dim[None, :] * v_stride_d,
        mask=q_mask, other=0.0,
    ).to(tl.float32)

    q_groups = tl.reshape(q, (RECORDS, GROUP_COUNT, GROUP_SIZE))
    k_groups = tl.reshape(k, (RECORDS, GROUP_COUNT, GROUP_SIZE))
    v_groups = tl.reshape(v, (RECORDS, GROUP_COUNT, GROUP_SIZE))
    q_scale = tl.max(tl.abs(q_groups), axis=2) / 127.0
    k_scale = tl.max(tl.abs(k_groups), axis=2) / 127.0
    v_scale = tl.max(tl.abs(v_groups), axis=2) / 127.0
    q_scale = tl.where(q_scale > 0.0, q_scale, 1.0)
    k_scale = tl.where(k_scale > 0.0, k_scale, 1.0)
    v_scale = tl.where(v_scale > 0.0, v_scale, 1.0)

    q_scale_by_dim = tl.reshape(tl.broadcast_to(q_scale[:, :, None], (RECORDS, GROUP_COUNT, GROUP_SIZE)), (RECORDS, VECTOR))
    k_scale_by_dim = tl.reshape(tl.broadcast_to(k_scale[:, :, None], (RECORDS, GROUP_COUNT, GROUP_SIZE)), (RECORDS, VECTOR))
    v_scale_by_dim = tl.reshape(tl.broadcast_to(v_scale[:, :, None], (RECORDS, GROUP_COUNT, GROUP_SIZE)), (RECORDS, VECTOR))
    # Round to nearest with ties away from zero, then clamp to symmetric INT8.
    q_round = tl.where(q >= 0.0, tl.floor(q / q_scale_by_dim + 0.5), tl.ceil(q / q_scale_by_dim - 0.5))
    k_round = tl.where(k >= 0.0, tl.floor(k / k_scale_by_dim + 0.5), tl.ceil(k / k_scale_by_dim - 0.5))
    v_round = tl.where(v >= 0.0, tl.floor(v / v_scale_by_dim + 0.5), tl.ceil(v / v_scale_by_dim - 0.5))
    q_int = tl.maximum(-127.0, tl.minimum(127.0, q_round)).to(tl.int8)
    k_int = tl.maximum(-127.0, tl.minimum(127.0, k_round)).to(tl.int8)
    v_int = tl.maximum(-127.0, tl.minimum(127.0, v_round)).to(tl.int8)

    byte_ptr = tl.cast(packet_ptr, tl.pointer_type(tl.int8))
    byte_base = record[:, None] * PACKET_SIZE
    tl.store(byte_ptr + byte_base + dim[None, :], q_int, mask=valid_record[:, None])
    tl.store(byte_ptr + byte_base + 128 + dim[None, :], k_int, mask=valid_record[:, None])
    tl.store(byte_ptr + byte_base + 256 + dim[None, :], v_int, mask=valid_record[:, None])

    scale_ptr = tl.cast(packet_ptr, tl.pointer_type(tl.float16))
    group = tl.arange(0, GROUP_COUNT)
    scale_base = record[:, None] * (PACKET_SIZE // 2) + 192
    tl.store(scale_ptr + scale_base + group[None, :], q_scale.to(tl.float16), mask=valid_record[:, None])
    tl.store(scale_ptr + scale_base + 4 + group[None, :], k_scale.to(tl.float16), mask=valid_record[:, None])
    tl.store(scale_ptr + scale_base + 8 + group[None, :], v_scale.to(tl.float16), mask=valid_record[:, None])


@triton.jit
def _decode_qkv_int8_kernel(
    packet_ptr, q_ptr, k_ptr, v_ptr,
    rows, batch, heads_local, elements,
    VECTOR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKET_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    dim = offsets % VECTOR
    record = offsets // VECTOR
    # Output is contiguous as (batch, rows, heads, dim), while the received
    # packet is contiguous as (rows, batch, heads, packet). Translate between
    # those two record orders explicitly.
    batch_index = record // (rows * heads_local)
    row = (record // heads_local) % rows
    local_head = record % heads_local
    packet_record = row * batch * heads_local + batch_index * heads_local + local_head

    byte_ptr = tl.cast(packet_ptr, tl.pointer_type(tl.int8))
    base = packet_record * PACKET_SIZE
    q = tl.load(byte_ptr + base + dim, mask=valid, other=0).to(tl.float32)
    k = tl.load(byte_ptr + base + 128 + dim, mask=valid, other=0).to(tl.float32)
    v = tl.load(byte_ptr + base + 256 + dim, mask=valid, other=0).to(tl.float32)

    scale_ptr = tl.cast(packet_ptr, tl.pointer_type(tl.float16))
    scale_base = packet_record * (PACKET_SIZE // 2) + 192 + dim // GROUP_SIZE
    q_scale = tl.load(scale_ptr + scale_base, mask=valid, other=1.0).to(tl.float32)
    k_scale = tl.load(scale_ptr + scale_base + 4, mask=valid, other=1.0).to(tl.float32)
    v_scale = tl.load(scale_ptr + scale_base + 8, mask=valid, other=1.0).to(tl.float32)

    out = ((batch_index * rows + row) * heads_local + local_head) * VECTOR + dim
    tl.store(q_ptr + out, (q * q_scale).to(tl.bfloat16), mask=valid)
    tl.store(k_ptr + out, (k * k_scale).to(tl.bfloat16), mask=valid)
    tl.store(v_ptr + out, (v * v_scale).to(tl.bfloat16), mask=valid)


@triton.jit
def _encode_output_fp8_kernel(
    input_ptr, packet_ptr, rows, batch, heads, elements,
    stride_b, stride_s, stride_h, stride_d,
    VECTOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    dim = offsets % VECTOR
    head = (offsets // VECTOR) % heads
    batch_index = (offsets // (VECTOR * heads)) % batch
    row = offsets // (VECTOR * heads * batch)
    source = batch_index * stride_b + row * stride_s + head * stride_h + dim * stride_d
    value = tl.load(input_ptr + source, mask=valid, other=0.0).to(tl.float32)
    value = tl.maximum(-448.0, tl.minimum(448.0, value))
    encoded = value.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(packet_ptr + offsets, encoded, mask=valid)


@triton.jit
def _decode_merge_output_fp8_kernel(
    packet_ptr, output_ptr, batch, rows, heads_local, world, heads_out, elements,
    VECTOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    dim = offsets % VECTOR
    head = (offsets // VECTOR) % heads_out
    row = (offsets // (VECTOR * heads_out)) % rows
    batch_index = offsets // (VECTOR * heads_out * rows)
    source = head // heads_local
    local_head = head % heads_local
    packet_offset = (((source * rows + row) * batch + batch_index) * heads_local + local_head) * VECTOR + dim
    encoded = tl.load(packet_ptr + packet_offset, mask=valid, other=0).to(tl.float8e4nv, bitcast=True)
    tl.store(output_ptr + offsets, encoded.to(tl.bfloat16), mask=valid)


def can_transport_qkv_int8(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        query.is_cuda
        and query.dtype == torch.bfloat16
        and query.ndim == 4
        and query.shape == key.shape == value.shape
        and query.device == key.device == value.device
        and query.shape[-1] == QKV_VECTOR
    )


def qkv_all_to_all_int8(
    pg,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    seq_lens: list[int],
    padded_head_cnt: int,
    use_sync: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-32 symmetric INT8 QKV transport with per-group FP16 scales."""
    world = torch.distributed.get_world_size(pg)
    batch, local_rows, original_heads, head_dim = query.shape
    if head_dim != QKV_VECTOR or padded_head_cnt % world:
        raise ValueError("INT8 QKV transport requires head_dim=128 and padded heads divisible by Ulysses size")
    if len(seq_lens) != world:
        raise ValueError(f"Expected {world} Ulysses sequence lengths, got {len(seq_lens)}")
    heads_local = padded_head_cnt // world
    send = torch.empty(
        (world * local_rows, batch, heads_local, QKV_PACKET),
        dtype=torch.uint8,
        device=query.device,
    )
    total_records = world * local_rows * batch * heads_local
    if total_records:
        _pack_qkv_int8_kernel[(triton.cdiv(total_records, 4),)](
            query, key, value, send,
            local_rows, batch, original_heads, heads_local, world,
            *query.stride(), *key.stride(), *value.stride(),
            RECORDS=4,
            VECTOR=QKV_VECTOR,
            GROUP_SIZE=QKV_GROUP,
            GROUP_COUNT=QKV_VECTOR // QKV_GROUP,
            PACKET_SIZE=QKV_PACKET,
            num_warps=4,
        )

    received = torch.empty(
        (sum(seq_lens), batch, heads_local, QKV_PACKET),
        dtype=torch.uint8,
        device=query.device,
    )
    torch.distributed.all_to_all_single(
        received,
        send,
        output_split_sizes=seq_lens,
        input_split_sizes=[local_rows] * world,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    global_rows = sum(seq_lens)
    output_shape = (batch, global_rows, heads_local, head_dim)
    q_out = torch.empty(output_shape, dtype=torch.bfloat16, device=query.device)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    elements = global_rows * batch * heads_local * head_dim
    if elements:
        _decode_qkv_int8_kernel[(triton.cdiv(elements, 1024),)](
            received, q_out, k_out, v_out,
            global_rows, batch, heads_local, elements,
            VECTOR=QKV_VECTOR,
            GROUP_SIZE=QKV_GROUP,
            PACKET_SIZE=QKV_PACKET,
            BLOCK=1024, num_warps=4,
        )
    return q_out, k_out, v_out


def can_transport_output_fp8(output: torch.Tensor) -> bool:
    return (
        output.is_cuda
        and output.dtype == torch.bfloat16
        and output.ndim == 4
        and output.shape[-1] == QKV_VECTOR
    )


def output_all_to_all_fp8(
    pg,
    output: torch.Tensor,
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_head_cnt: int,
    use_sync: bool,
) -> torch.Tensor:
    """E4M3 output transport; decode and merge heads in one kernel."""
    world = torch.distributed.get_world_size(pg)
    batch, global_rows, heads_local, head_dim = output.shape
    if head_dim != QKV_VECTOR or len(seq_lens) != world:
        raise ValueError("FP8 output transport requires head_dim=128 and one sequence length per Ulysses rank")
    send = torch.empty(
        (global_rows, batch, heads_local, head_dim),
        dtype=torch.uint8,
        device=output.device,
    )
    elements = global_rows * batch * heads_local * head_dim
    if elements:
        _encode_output_fp8_kernel[(triton.cdiv(elements, 1024),)](
            output, send, global_rows, batch, heads_local, elements,
            *output.stride(), VECTOR=QKV_VECTOR, BLOCK=1024, num_warps=4,
        )

    received = torch.empty(
        (world * local_seq_len, batch, heads_local, head_dim),
        dtype=torch.uint8,
        device=output.device,
    )
    torch.distributed.all_to_all_single(
        received,
        send,
        output_split_sizes=[local_seq_len] * world,
        input_split_sizes=seq_lens,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    padded_heads = world * heads_local
    out_shape = (batch, local_seq_len, padded_heads, head_dim)
    merged = torch.empty(out_shape, dtype=torch.bfloat16, device=output.device)
    merged_elements = batch * local_seq_len * padded_heads * head_dim
    if merged_elements:
        _decode_merge_output_fp8_kernel[(triton.cdiv(merged_elements, 1024),)](
            received, merged, batch, local_seq_len, heads_local, world, padded_heads, merged_elements,
            VECTOR=QKV_VECTOR, BLOCK=1024, num_warps=4,
        )
    if orig_head_cnt > 0 and orig_head_cnt != padded_heads:
        merged = merged[:, :, :orig_head_cnt, :].contiguous()
    return merged

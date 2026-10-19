# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused indexed modulation with FP32 accumulation."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

# Ascend CANN limits the launch coreDim to uint16.
_MAX_1D_GRID_SIZE = 65535


def _iter_row_chunks(rows: int):
    for row_offset in range(0, rows, _MAX_1D_GRID_SIZE):
        yield row_offset, min(rows - row_offset, _MAX_1D_GRID_SIZE)


def _launch_row_chunks(kernel, rows: int, device_type: str, *args, **kwargs) -> None:
    if device_type != "npu":
        kernel[(rows,)](*args, 0, **kwargs)
        return
    for row_offset, chunk_rows in _iter_row_chunks(rows):
        kernel[(chunk_rows,)](*args, row_offset, **kwargs)


@triton.jit
def _indexed_scale_shift_kernel(
    output_ptr,
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_ptr + row * stride_x_row + columns,
        x * (1.0 + scale) + shift,
        mask=mask,
    )


@triton.jit
def _indexed_gate_kernel(
    output_ptr,
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_output_row,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + row * stride_other_row + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        output_ptr + row * stride_output_row + columns,
        x + gate * other,
        mask=mask,
    )


@triton.jit
def _rms_norm_indexed_scale_shift_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    # Keep RMSNorm and AdaLN intermediates in FP32 on every architecture.
    # Rounding the normalized value to BF16 before the affine transform
    # changes the fused computation, including values at rounding boundaries.
    normalized = x * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    output = normalized * (1.0 + scale) + shift
    tl.store(
        output_ptr + row * hidden_size + columns,
        output,
        mask=mask,
    )


@triton.jit
def _row_rms_inverse_kernel(
    output_ptr,
    x_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_x_row,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    x = tl.load(x_ptr + row * stride_x_row + columns, mask=columns < hidden_size, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    tl.store(output_ptr + row, tl.rsqrt(variance + eps))


@triton.jit
def _rms_adaln_qkv_kernel(
    output_ptr,
    x_ptr,
    norm_weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    inverse_rms_ptr,
    qkv_weight_ptr,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    stride_weight_out,
    stride_weight_in,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_m = pid_m * block_m + tl.arange(0, block_m)
    columns_n = pid_n * block_n + tl.arange(0, block_n)
    columns_k = tl.arange(0, block_k)
    row_mask = rows_m < rows
    indices = tl.load(indices_ptr + rows_m * stride_indices, mask=row_mask, other=0)
    inverse_rms = tl.load(inverse_rms_ptr + rows_m, mask=row_mask, other=0.0).to(tl.float32)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(tl.cdiv(hidden_size, block_k)):
        k = k_start * block_k + columns_k
        k_mask = k < hidden_size
        x = tl.load(
            x_ptr + rows_m[:, None] * stride_x_row + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        norm_weight = tl.load(norm_weight_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        scale = tl.load(
            scale_ptr + indices[:, None] * stride_scale_row + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            shift_ptr + indices[:, None] * stride_shift_row + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        modulated = (x * inverse_rms[:, None] * norm_weight[None, :]) * (1.0 + scale) + shift
        modulated = modulated.to(tl.bfloat16)
        weight = tl.load(
            qkv_weight_ptr + columns_n[None, :] * stride_weight_out + k[:, None] * stride_weight_in,
            mask=(columns_n[None, :] < output_size) & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(modulated, weight)

    tl.store(
        output_ptr + rows_m[:, None] * output_size + columns_n[None, :],
        accumulator,
        mask=row_mask[:, None] & (columns_n[None, :] < output_size),
    )


@triton.jit
def _indexed_gate_rms_norm_scale_shift_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_residual_row,
    stride_gate_row,
    stride_branch_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    row_offset,
    block_n: tl.constexpr,
):
    row = tl.program_id(0) + row_offset
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    residual = tl.load(residual_ptr + row * stride_residual_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + row * stride_branch_row + columns, mask=mask, other=0.0).to(tl.float32)
    # Match the BF16 residual value consumed by the unfused RMSNorm path.
    updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
    tl.store(residual_out_ptr + row * hidden_size + columns, updated, mask=mask)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(updated * updated, axis=0) / hidden_size
    normalized = updated * tl.rsqrt(variance + eps) * weight
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    output = normalized * (1.0 + scale) + shift
    tl.store(
        modulated_out_ptr + row * hidden_size + columns,
        output,
        mask=mask,
    )


def indexed_scale_shift_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply indexed scale/shift in-place to a disposable contiguous input."""
    if x.is_cpu:
        x.copy_((x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype))
        return x
    rows, hidden_size = x.shape
    if rows == 0:
        return x
    _launch_row_chunks(
        _indexed_scale_shift_kernel,
        rows,
        x.device.type,
        x,
        x,
        shift,
        scale,
        indices,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return x


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return ``x + gate[indices] * other`` without indexed temporaries."""
    if x.is_cpu:
        return (x + gate.index_select(0, indices) * other).to(x.dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows == 0:
        return output
    _launch_row_chunks(
        _indexed_gate_kernel,
        rows,
        x.device.type,
        output,
        x,
        gate,
        other,
        indices,
        hidden_size,
        output.stride(0),
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return output


def rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse RMSNorm with an indexed affine transform."""
    if x.is_cpu:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        return (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(input_dtype)
    output = torch.empty_like(x)
    rows, hidden_size = x.shape
    if rows:
        _launch_row_chunks(
            _rms_norm_indexed_scale_shift_kernel,
            rows,
            x.device.type,
            output,
            x,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            eps,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            block_n=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return output


def rms_norm_indexed_scale_shift_qkv(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    qkv_weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse H3 RMSNorm/AdaLN input transform with a BF16 QKV projection."""
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError("fused RMSNorm/AdaLN-QKV requires CUDA BF16 activations")
    rows, hidden_size = x.shape
    if rows == 0:
        return torch.empty((0, qkv_weight.shape[0]), device=x.device, dtype=x.dtype)
    if not x.is_contiguous() or not norm_weight.is_contiguous() or not qkv_weight.is_contiguous():
        raise ValueError("fused RMSNorm/AdaLN-QKV requires contiguous activation and weights")

    inverse_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
    output = torch.empty((rows, qkv_weight.shape[0]), device=x.device, dtype=x.dtype)
    _row_rms_inverse_kernel[(rows,)](
        inverse_rms,
        x,
        hidden_size,
        eps,
        x.stride(0),
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8 if hidden_size >= 2048 else 4,
    )

    # 64x64x64 lowers register pressure on RTX 5090 for this shape. The
    # additional N tiles and K loop iterations were faster than wider tiles
    # in the shape-matched kernel tuning.
    block_m, block_n, block_k = 64, 64, 64
    _rms_adaln_qkv_kernel[(triton.cdiv(rows, block_m), triton.cdiv(qkv_weight.shape[0], block_n))](
        output,
        x,
        norm_weight,
        shift,
        scale,
        indices,
        inverse_rms,
        qkv_weight,
        rows,
        hidden_size,
        qkv_weight.shape[0],
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        qkv_weight.stride(0),
        qkv_weight.stride(1),
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _rms_adaln_quant_fp8_kernel(
    output_ptr,
    output_scale_ptr,
    x_ptr,
    norm_weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    block_n: tl.constexpr,
):
    """Apply H3 RMSNorm/AdaLN and quantize one activation row to FP8."""
    row = tl.program_id(0)
    columns = tl.arange(0, block_n)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    norm_weight = tl.load(norm_weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    normalized = x * tl.rsqrt(variance + eps) * norm_weight
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)

    # Match the ordinary path's BF16 modulation output before dynamic FP8
    # activation quantization; the QKV GEMM still consumes the checkpoint's
    # FP8 weights and its existing per-shard weight scales.
    modulated = ((normalized * (1.0 + scale) + shift).to(tl.bfloat16)).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(modulated), axis=0), 1.0e-10)
    input_scale = absmax * (1.0 / fp8_max)
    quantized = tl.clamp(modulated / input_scale, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)

    tl.store(output_ptr + row * hidden_size + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, input_scale)


def rms_norm_indexed_scale_shift_fp8(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse H3 RMSNorm/AdaLN with dynamic per-token E4M3 activation quantization.

    The returned FP8 activation and row scales can be passed directly to an
    FP8 scaled-MM kernel. This avoids materializing the intermediate BF16
    modulation tensor while keeping the projection weights quantized.
    """
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError("fused RMSNorm/AdaLN-FP8 requires CUDA BF16 activations")
    if not x.is_contiguous() or not norm_weight.is_contiguous():
        raise ValueError("fused RMSNorm/AdaLN-FP8 requires contiguous activation and norm weights")
    if not shift.is_contiguous() or not scale.is_contiguous() or not indices.is_contiguous():
        raise ValueError("fused RMSNorm/AdaLN-FP8 requires contiguous modulation tensors")

    rows, hidden_size = x.shape
    output = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows == 0:
        return output, output_scale

    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    _rms_adaln_quant_fp8_kernel[(rows,)](
        output,
        output_scale,
        x,
        norm_weight,
        shift,
        scale,
        indices,
        hidden_size,
        eps,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        fp8_min=float(torch.finfo(torch.float8_e4m3fn).min),
        fp8_max=fp8_max,
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8 if hidden_size >= 2048 else 4,
        num_stages=1,
    )
    return output, output_scale


def indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated residual, the following RMSNorm, and AdaLN affine."""
    if residual.is_cpu:
        input_dtype = residual.dtype
        residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
        normalized = residual_out.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + eps)
        normalized = (weight.float() * normalized).to(input_dtype)
        modulated_out = (normalized * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(
            input_dtype
        )
        return residual_out, modulated_out
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    rows, hidden_size = residual.shape
    if rows:
        _launch_row_chunks(
            _indexed_gate_rms_norm_scale_shift_kernel,
            rows,
            residual.device.type,
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            hidden_size,
            eps,
            residual.stride(0),
            gate.stride(0),
            branch.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            block_n=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return residual_out, modulated_out


__all__ = [
    "indexed_gate",
    "indexed_gate_rms_norm_scale_shift",
    "indexed_scale_shift_",
    "rms_norm_indexed_scale_shift",
    "rms_norm_indexed_scale_shift_qkv",
    "rms_norm_indexed_scale_shift_fp8",
]

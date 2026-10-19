# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned, double-buffered host-to-CUDA staging for checkpoint tensors."""

from __future__ import annotations

import torch


def _byte_view(source: torch.Tensor, byte_storage: torch.Tensor) -> torch.Tensor:
    """View a contiguous byte buffer with ``source``'s dtype and shape."""
    if source.device.type != "cpu":
        raise ValueError("staged checkpoint source must be on CPU")
    if not source.is_contiguous():
        raise ValueError("staged checkpoint source must be contiguous")
    size = source.numel() * source.element_size()
    if byte_storage.dtype is not torch.uint8 or byte_storage.numel() < size:
        raise ValueError("byte storage is too small for the checkpoint tensor")
    return byte_storage[:size].view(source.dtype).view(source.shape)


class DoubleBufferedPinnedStager:
    """Copy CPU checkpoint tensors into CUDA parameters with two pinned slots.

    The caller supplies the ordinary vLLM ``weight_loader`` as ``write``. Its
    CUDA copies run on one private stream while the host prepares the next
    tensor in the other pinned slot. An event protects each slot from reuse
    until its previous H2D work has finished.
    """

    def __init__(self, device: torch.device, *, initial_buffer_bytes: int = 1 << 20) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DoubleBufferedPinnedStager requires a CUDA target")
        if initial_buffer_bytes <= 0:
            raise ValueError("initial_buffer_bytes must be positive")
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = [
            torch.empty(initial_buffer_bytes, dtype=torch.uint8, pin_memory=True),
            torch.empty(initial_buffer_bytes, dtype=torch.uint8, pin_memory=True),
        ]
        self.events: list[torch.cuda.Event | None] = [None, None]
        self.slot = 0

    @staticmethod
    def _required_bytes(source: torch.Tensor) -> int:
        return source.numel() * source.element_size()

    def _buffer_for(self, slot: int, required_bytes: int) -> torch.Tensor:
        buffer = self.buffers[slot]
        if required_bytes > buffer.numel():
            # Growth happens only after the slot's prior DMA event is complete.
            capacity = max(required_bytes, buffer.numel() * 2)
            buffer = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
            self.buffers[slot] = buffer
        return buffer

    def submit(self, source: torch.Tensor, write) -> object:
        """Stage ``source`` and invoke ``write`` before its slot is reused."""
        if source.device.type != "cpu":
            return write(source)
        contiguous = source.detach().contiguous()
        size = self._required_bytes(contiguous)
        if size == 0:
            return write(contiguous)

        slot = self.slot
        previous = self.events[slot]
        if previous is not None:
            previous.synchronize()

        buffer = self._buffer_for(slot, size)
        staged = _byte_view(contiguous, buffer)
        staged.copy_(contiguous)
        with torch.cuda.stream(self.stream):
            result = write(staged)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.events[slot] = event
        self.slot ^= 1
        return result

    def synchronize(self) -> None:
        """Wait until all queued weight writes are visible to the caller."""
        self.stream.synchronize()
        self.events = [None, None]


__all__ = ["DoubleBufferedPinnedStager", "_byte_view"]

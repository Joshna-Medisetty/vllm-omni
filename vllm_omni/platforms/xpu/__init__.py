# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

__all__ = ["XPUOmniPlatform"]


def _xpu_get_memory_info(
    device: int | str | torch.device | None = None,
) -> tuple[int, int]:
    """Query Level Zero driver for actual device free/total memory.

    Overrides the broken monkey-patch from vllm/platforms/xpu.py which
    uses torch.ops._C_cache_ops.getMemoryInfo (reports caching allocator
    pool state, not real device memory).
    """
    if device is None:
        device_idx = torch.xpu.current_device()
    elif isinstance(device, torch.device):
        if device.type != "xpu":
            raise RuntimeError(f"Expected 'xpu' device, got '{device.type}'")
        device_idx = (
            device.index if device.index is not None
            else torch.xpu.current_device()
        )
    elif isinstance(device, str):
        if not device.startswith("xpu"):
            raise RuntimeError(
                f"Expected 'xpu' device string, got '{device}'"
            )
        parts = device.split(":")
        if len(parts) == 1:
            device_idx = torch.xpu.current_device()
        elif len(parts) == 2:
            device_idx = int(parts[1])
        else:
            raise RuntimeError(f"Invalid device string format: '{device}'")
    elif isinstance(device, int):
        device_idx = device
    else:
        raise TypeError(
            f"device must be int, str, torch.device, or None, "
            f"got {type(device)}"
        )

    free, total = torch.xpu.mem_get_info(device_idx)
    return free, total


# Override the broken monkey-patch applied by vllm/vllm/platforms/xpu.py
# which assigns torch.ops._C_cache_ops.getMemoryInfo (caching allocator
# pool state) to torch.accelerator.get_memory_info.
torch.accelerator.get_memory_info = _xpu_get_memory_info

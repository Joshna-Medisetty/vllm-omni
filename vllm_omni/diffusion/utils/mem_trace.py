# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional XPU/CUDA memory logging for diffusion OOM diagnosis.

Enable with ``VLLM_OMNI_DIFFUSION_MEM_TRACE=1`` or Helios ``end2end.py --diffusion-mem-trace``.
"""

from __future__ import annotations

import logging
import os

import torch
from vllm.utils.mem_utils import GiB_bytes

from vllm_omni.platforms import current_omni_platform

_ENV_VAR = "VLLM_OMNI_DIFFUSION_MEM_TRACE"


def diffusion_mem_trace_enabled() -> bool:
    v = os.environ.get(_ENV_VAR, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def log_diffusion_device_memory(
    logger: logging.Logger,
    device: torch.device,
    stage: str,
    *,
    extra: str = "",
) -> None:
    if not diffusion_mem_trace_enabled():
        return
    try:
        if hasattr(current_omni_platform, "synchronize"):
            current_omni_platform.synchronize()
    except Exception:
        pass
    driver_used_gib = torch_alloc_gib = torch_res_gib = None
    try:
        if device.type == "xpu":
            free_b, total_b = torch.xpu.mem_get_info(device)
            driver_used_gib = (total_b - free_b) / GiB_bytes
            torch_alloc_gib = torch.xpu.memory_allocated(device) / GiB_bytes
            torch_res_gib = torch.xpu.memory_reserved(device) / GiB_bytes
        elif device.type == "cuda":
            free_b, total_b = torch.cuda.mem_get_info(device)
            driver_used_gib = (total_b - free_b) / GiB_bytes
            torch_alloc_gib = torch.cuda.memory_allocated(device) / GiB_bytes
            torch_res_gib = torch.cuda.memory_reserved(device) / GiB_bytes
        else:
            return
    except Exception as exc:
        logger.warning("[DiffusionMemTrace] %s: snapshot unavailable (%s)", stage, exc)
        return
    suffix = f" {extra}" if extra else ""
    logger.info(
        "[DiffusionMemTrace] stage=%s device=%s driver_used_gib=%.4f "
        "torch_allocated_gib=%.4f torch_reserved_gib=%.4f%s",
        stage,
        device,
        driver_used_gib,
        torch_alloc_gib,
        torch_res_gib,
        suffix,
    )

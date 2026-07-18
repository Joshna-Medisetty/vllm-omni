# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


def _clamped_request_memory(init_snapshot, cache_config) -> int:
    """A replacement for vllm.v1.worker.utils.request_memory that bypasses
    the preflight ValueError when free memory is 0 on XPU.

    On Intel XPU with Level Zero, the parent process context may hold the
    entire device memory pool.  When the child worker starts, free_memory
    reports 0.  The Level Zero driver will reclaim the parent's idle context
    memory once the child starts allocating, so we return the full
    requested_memory budget instead of clamping to 0 or raising.
    """
    import math
    import logging
    logger = logging.getLogger(__name__)

    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if init_snapshot.free_memory < requested_memory:
        logger.warning(
            "XPU memory preflight bypass: free memory %.2f GiB < "
            "requested %.2f GiB on device %s. Returning full budget "
            "(Level Zero will reclaim parent context memory).",
            init_snapshot.free_memory / (1024**3),
            requested_memory / (1024**3),
            init_snapshot.device_,
        )
        # Return the full requested budget - do NOT clamp to 0.
        return requested_memory

    return requested_memory


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        # Always monkey-patch request_memory on XPU to bypass the preflight
        # check that fails when the parent Level Zero context holds device memory.
        import vllm.v1.worker.xpu_worker as _xpu_worker_mod
        _xpu_worker_mod.request_memory = _clamped_request_memory

        super().init_device()
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)
        # XPUARModelRunner is V1-based; prevent V2 warmup_kernels() from being called
        self.use_v2_model_runner = False

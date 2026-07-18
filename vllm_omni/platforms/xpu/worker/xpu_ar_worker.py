# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = logging.getLogger(__name__)


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        try:
            super().init_device()
        except ValueError as e:
            err_msg = str(e)
            # XPU memory reporting bug: subprocess reports 0 free bytes on a
            # card that was fully free at probe time. Fall back to using
            # total_memory * gpu_memory_utilization as requested_memory.
            if "Free memory on device" in err_msg and "0.0/" in err_msg:
                logger.warning(
                    "Suspected XPU memory reporting bug (0 free bytes in "
                    "subprocess). Falling back to total_memory * "
                    "gpu_memory_utilization for requested_memory. "
                    "Original error: %s", err_msg)
                # super().init_device() already set self.device,
                # self.init_gpu_memory, and initialized distributed before
                # request_memory() raised. We need to complete the remaining
                # initialization that was skipped after the raise.
                import math
                from vllm.v1.worker.workspace import init_workspace_manager
                self.requested_memory = math.ceil(
                    self.init_gpu_memory
                    * self.cache_config.gpu_memory_utilization
                )
                logger.info(
                    "Fallback requested_memory: %.2f GiB",
                    self.requested_memory / (1024**3))
                num_ubatches = (
                    2 if self.vllm_config.parallel_config.enable_dbo else 1
                )
                init_workspace_manager(self.device, num_ubatches)
            else:
                raise
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)
        # XPUARModelRunner is V1-based; override the flag so
        # compile_or_warm_up_model uses the V1 warmup path.
        self.use_v2_model_runner = False

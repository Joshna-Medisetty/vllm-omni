# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        super().init_device()
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # XPUARModelRunner is V1-based and lacks V2 attributes
        # (e.g. num_speculative_steps) required by warmup_kernels().
        # Force the V1 warmup path by temporarily clearing the flag.
        orig = self.use_v2_model_runner
        self.use_v2_model_runner = False
        try:
            return super().compile_or_warm_up_model()
        finally:
            self.use_v2_model_runner = orig

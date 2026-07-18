# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        super().init_device()
        # OMNI: v2 model runner does not include omni AR hooks;
        # force v1 path for compile_or_warm_up_model.
        self.use_v2_model_runner = False
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)

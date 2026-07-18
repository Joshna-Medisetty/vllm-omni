# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.memory_utils import request_memory_tolerant
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        # Replicate XPUWorker.init_device() but use request_memory_tolerant
        # to handle multi-stage GPU sharing without raising ValueError.
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            tp_pp_world_size = (
                parallel_config.pipeline_parallel_size
                * parallel_config.tensor_parallel_size
            )
            self.local_rank += dp_local_rank * tp_pp_world_size

            visible_device_count = torch.accelerator.device_count()
            assert self.local_rank < visible_device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds. "
            )
            assert parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            self.device = torch.device(f"xpu:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # global all_reduce needed for overall oneccl warm up
        if torch.distributed.is_xccl_available():
            torch.distributed.all_reduce(torch.zeros(1).xpu())

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after distributed init
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot — use tolerant version for multi-stage
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory_tolerant(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # The Omni XPU model runners inherit from the v1 GPUModelRunner
        # (vllm.v1.worker.gpu_model_runner) which lacks v2 attributes like
        # num_speculative_steps. Force v1 warmup path.
        self.use_v2_model_runner = False

        # Construct the model runner
        self.model_runner = XPUARModelRunner(self.vllm_config, self.device)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Override to clamp negative KV cache memory to 0 as a safety net.

        On XPU the upstream Worker.determine_available_memory() may compute a
        negative available_kv_cache_memory_bytes when the memory reporting is
        inaccurate at subprocess startup. Clamp to 0 to avoid ValueError in
        downstream KV cache initialization.
        """
        result = super().determine_available_memory()
        if result < 0:
            logger.warning(
                "Clamping negative KV cache memory %s GiB to 0 on XPU "
                "(likely due to inaccurate memory reporting at startup).",
                format_gib(result),
            )
            self.available_kv_cache_memory_bytes = 0
            return 0
        return result

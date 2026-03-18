import os
import sys
from pathlib import Path

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 1, "rocm": 2, "xpu": 2})
def test_nextstep_text2img(run_level):
    if run_level == "core_model":
        pytest.skip()

    m = None
    try:
        omni_kwargs = {
            "model": "stepfun-ai/NextStep-1.1",
            "model_class_name": "NextStep11Pipeline",
        }
        if current_omni_platform.is_xpu():
            omni_kwargs["parallel_config"] = DiffusionParallelConfig(tensor_parallel_size=2)
        m = Omni(**omni_kwargs)
        # high resolution may cause OOM on L4
        height = 256
        width = 256
        outputs = m.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=4,
                guidance_scale=7.5,
                guidance_scale_2=1.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
            ),
        )
        first_output = outputs[0]
        assert first_output.final_output_type == "image"
        assert getattr(first_output, "request_output", None), "no request_output on NextStep output"
        req_out = first_output.request_output[0]
        assert isinstance(req_out, OmniRequestOutput), "request_output[0] is not OmniRequestOutput"
        assert getattr(req_out, "images", None), "no images in NextStep request_output"
        images = req_out.images
        assert len(images) == 1, f"expected 1 image, got {len(images)}"
        assert images[0].width == width
        assert images[0].height == height
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()

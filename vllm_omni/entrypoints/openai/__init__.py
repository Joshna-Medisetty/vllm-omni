# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

def __getattr__(name):
    if name in ("build_async_omni", "omni_init_app_state", "omni_run_server"):
        from vllm_omni.entrypoints.openai.api_server import (
            build_async_omni,
            omni_init_app_state,
            omni_run_server,
        )
        return locals()[name]
    if name == "OmniOpenAIServingChat":
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
        return OmniOpenAIServingChat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Server functions
    "omni_run_server",
    "build_async_omni",
    "omni_init_app_state",
    # Serving classes
    "OmniOpenAIServingChat",
]

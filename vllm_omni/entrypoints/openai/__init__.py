# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

try:
    from vllm_omni.entrypoints.openai.api_server import (
        build_async_omni,
        omni_init_app_state,
        omni_run_server,
    )
except (ImportError, ModuleNotFoundError):
    build_async_omni = None  # type: ignore[assignment,misc]
    omni_init_app_state = None  # type: ignore[assignment,misc]
    omni_run_server = None  # type: ignore[assignment,misc]

try:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
except (ImportError, ModuleNotFoundError):
    OmniOpenAIServingChat = None  # type: ignore[assignment,misc]

__all__ = [n for n in ("omni_run_server", "build_async_omni", "omni_init_app_state", "OmniOpenAIServingChat") if vars().get(n) is not None]

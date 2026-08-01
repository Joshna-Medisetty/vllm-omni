# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.api_server import (
        build_async_omni,
        omni_init_app_state,
        omni_run_server,
    )
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

# Lazy imports to avoid triggering api_server.py's heavy import chain
# when only submodules like stage_params are needed.
_LAZY_IMPORTS = {
    "build_async_omni": "vllm_omni.entrypoints.openai.api_server",
    "omni_init_app_state": "vllm_omni.entrypoints.openai.api_server",
    "omni_run_server": "vllm_omni.entrypoints.openai.api_server",
    "OmniOpenAIServingChat": "vllm_omni.entrypoints.openai.serving_chat",
}

__all__ = [
    # Server functions
    "omni_run_server",
    "build_async_omni",
    "omni_init_app_state",
    # Serving classes
    "OmniOpenAIServingChat",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

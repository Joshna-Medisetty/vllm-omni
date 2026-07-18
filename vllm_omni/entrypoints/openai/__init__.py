# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
OpenAI-compatible API entrypoints for vLLM-Omni.

Provides:
- omni_run_server: Main server entry point (auto-detects model type)
- OmniOpenAIServingChat: Unified chat completion handler for both LLM and diffusion models
"""

import importlib

_LAZY_IMPORTS = {
    'build_async_omni': 'vllm_omni.entrypoints.openai.api_server',
    'omni_init_app_state': 'vllm_omni.entrypoints.openai.api_server',
    'omni_run_server': 'vllm_omni.entrypoints.openai.api_server',
    'OmniOpenAIServingChat': 'vllm_omni.entrypoints.openai.serving_chat',
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())

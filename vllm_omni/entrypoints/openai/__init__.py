# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenAI-compatible API entrypoints for vLLM-Omni."""

__all__ = [
    "omni_run_server",
    "build_async_omni",
    "omni_init_app_state",
    "OmniOpenAIServingChat",
]

def __getattr__(name):
    if name in ("build_async_omni", "omni_init_app_state", "omni_run_server"):
        from vllm_omni.entrypoints.openai.api_server import (
            build_async_omni,
            omni_init_app_state,
            omni_run_server,
        )
        globals().update({"build_async_omni": build_async_omni, "omni_init_app_state": omni_init_app_state, "omni_run_server": omni_run_server})
        return globals()[name]
    if name == "OmniOpenAIServingChat":
        from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
        globals()["OmniOpenAIServingChat"] = OmniOpenAIServingChat
        return OmniOpenAIServingChat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

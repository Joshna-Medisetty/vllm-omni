# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def build_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    h = height or 1024
    w = width or 1024
    result: dict[str, Any] = {
        "prompt": prompt,
        "mm_processor_kwargs": {"target_h": h, "target_w": w},
    }
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result

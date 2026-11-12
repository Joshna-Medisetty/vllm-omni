# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.patch import (
    _patch_xpu_uva_moe_weight_transpose,
    _wrap_unquantized_moe_pwal,
    _xpu_pwal_undoes_uva,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _XPUBackend:
    name = "XPU"


class _CudaBackend:
    name = "TRITON"


def _buggy_xpu_pwal(self, layer) -> None:
    # Mirrors the pin that undoes UVA: UnquantizedMoeBackend.XPU then .contiguous().
    if self.unquantized_backend.name == "XPU":
        for name in ("w13_weight", "w2_weight"):
            param = getattr(layer, name)
            param.data = param.transpose(-1, -2).contiguous()


def _current_pwal(self, layer) -> None:
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        param.data = param.transpose(-1, -2).contiguous()


def test_xpu_contiguous_branch_is_detected_and_current_main_is_not():
    assert _xpu_pwal_undoes_uva(_buggy_xpu_pwal)
    assert not _xpu_pwal_undoes_uva(_current_pwal)


def _layer(flag_w13: bool):
    layer = nn.Module()
    layer.w13_weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(1, 2, 3))
    layer.w2_weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(1, 3, 2))
    if flag_w13:
        layer.w13_weight._vllm_is_uva_offloaded = True
    return layer


def test_uva_xpu_transpose_uses_rehost_result(monkeypatch):
    calls = []

    def fake_rehost(weight):
        calls.append(weight)
        return weight.detach().transpose(-1, -2).contiguous()

    monkeypatch.setattr("vllm_omni.patch._rehost_transposed_expert_weight", fake_rehost)
    layer = _layer(flag_w13=True)
    expected_w13 = layer.w13_weight.detach().transpose(-1, -2).contiguous()
    expected_w2 = layer.w2_weight.detach().transpose(-1, -2).contiguous()
    transpose = torch.Tensor.transpose
    contiguous = torch.Tensor.contiguous
    method = _wrap_unquantized_moe_pwal(_buggy_xpu_pwal)

    method(_XPUBackendHolder(), layer)

    assert len(calls) == 1
    assert torch.equal(layer.w13_weight.data, expected_w13)
    assert torch.equal(layer.w2_weight.data, expected_w2)
    assert layer.w13_weight._vllm_is_uva_offloaded is True
    assert torch.Tensor.transpose is transpose
    assert torch.Tensor.contiguous is contiguous


class _XPUBackendHolder:
    unquantized_backend = _XPUBackend()


class _CudaBackendHolder:
    unquantized_backend = _CudaBackend()


def test_non_uva_and_non_xpu_keep_the_original_transpose(monkeypatch):
    def fail_rehost(weight):
        raise AssertionError(weight)

    monkeypatch.setattr("vllm_omni.patch._rehost_transposed_expert_weight", fail_rehost)

    def always_transpose(self, layer):
        for name in ("w13_weight", "w2_weight"):
            param = getattr(layer, name)
            param.data = param.transpose(-1, -2).contiguous()

    for holder, flag in ((_XPUBackendHolder(), False), (_CudaBackendHolder(), True)):
        layer = _layer(flag_w13=flag)
        expected_w13 = layer.w13_weight.detach().transpose(-1, -2).contiguous()
        method = _wrap_unquantized_moe_pwal(always_transpose)
        method(holder, layer)
        assert torch.equal(layer.w13_weight.data, expected_w13)


def test_tensor_methods_are_restored_when_the_original_method_raises(monkeypatch):
    def fake_rehost(weight):
        return weight.detach().transpose(-1, -2).contiguous()

    monkeypatch.setattr("vllm_omni.patch._rehost_transposed_expert_weight", fake_rehost)
    transpose = torch.Tensor.transpose
    contiguous = torch.Tensor.contiguous

    def explode(self, layer):
        if self.unquantized_backend.name == "XPU":
            layer.w13_weight.data = layer.w13_weight.transpose(-1, -2).contiguous()
            raise RuntimeError("load failed")

    method = _wrap_unquantized_moe_pwal(explode)
    try:
        method(_XPUBackendHolder(), _layer(flag_w13=True))
    except RuntimeError as exc:
        assert str(exc) == "load failed"
    assert torch.Tensor.transpose is transpose
    assert torch.Tensor.contiguous is contiguous


def test_install_is_idempotent_and_matches_the_installed_vllm_pin():
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    before = UnquantizedFusedMoEMethod.process_weights_after_loading
    _patch_xpu_uva_moe_weight_transpose()
    after = UnquantizedFusedMoEMethod.process_weights_after_loading
    assert before is after
    original = getattr(after, "_vllm_omni_wrapped_pwal", after)
    if _xpu_pwal_undoes_uva(original):
        assert getattr(after, "_vllm_omni_wrapped_pwal", None) is original
    else:
        assert not hasattr(after, "_vllm_omni_wrapped_pwal")

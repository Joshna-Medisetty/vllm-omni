# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class LayerwiseOffloadHook(ModelHook):
    """Hook for layerwise (transformer-block-wise) CPU offloading.

    The hook instance retains parameters for both the current registered block
    module and those for the next block, as well as flattened CPU tensors which
    record the parameters of the current block module, so that these parameters
    could be re-materialized on device in an overlapping way.
    This hook should be registered to each of the transformer blocks in DiT
    module(s) of the target pipeline.

    Based on implementations from:
    https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py
    """

    _HOOK_NAME = "layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        stream: current_omni_platform.Stream | None = None,
        pin_memory: bool = True,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.copy_stream = stream or current_omni_platform.current_stream()
        self.pin_memory = pin_memory

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: current_omni_platform.Event | None = None

        # Backward link to the hook that is responsible for prefetching *this* block's weights
        self._prev_hook: LayerwiseOffloadHook | None = None

        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}
        self.dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        self.dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

    @staticmethod
    def _is_dtensor(t: torch.Tensor) -> bool:
        return isinstance(t, DTensor)

    @staticmethod
    def _set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
        if LayerwiseOffloadHook._is_dtensor(target):
            target._local_tensor = value
        else:
            target.data = value

    @staticmethod
    def _make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
        if LayerwiseOffloadHook._is_dtensor(tensor):
            local_shape = tuple(tensor.to_local().shape)
            return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
        return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _is_materialized_tensor(t: torch.Tensor) -> bool:
        if LayerwiseOffloadHook._is_dtensor(t):
            local_t = t.to_local()
            return not local_t.is_meta
        return not t.is_meta and t.data.numel() > 0

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        # This all happen during the hook instance being registered to hook registry;
        # the input module is kept intact
        module = super().initialize_hook(module)

        self.block_parameters: dict[str, nn.Parameter] = dict(module.named_parameters())
        self.block_buffers: dict[str, torch.Tensor] = dict(module.named_buffers())

        self.next_block_parameters: dict[str, nn.Parameter] = dict(self.next_block.named_parameters())
        self.next_block_buffers: dict[str, torch.Tensor] = dict(self.next_block.named_buffers())

        # Pre-allocate gpu tensors in a flattened way
        self.dtype_cpu_flattened_weights, self.dtype_metadata = LayerwiseOffloadHook._to_cpu(
            self.next_block_parameters,
            self.next_block_buffers,
            self.device,
            self.pin_memory,
        )

        return module

    @staticmethod
    def _to_cpu(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        device: torch.device,
        pin_memory: bool = True,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Helper method to move block parameters and buffers to CPU, flattening by dtype.

        Consolidates parameters and buffers into contiguous CPU tensors grouped by dtype
        for GPU transfers. Replaces original tensors with empty placeholders.

        Returns:
            Tuple of
                flattened CPU tensors by dtype,
                metadata for reconstruction by dtype
        """
        dtype_grouped_weights: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_cpu_flattened_weights: dict[torch.dtype, torch.Tensor] = {}
        # NOTE: order does matter
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped_weights:
                dtype_grouped_weights[dtype] = {}
            dtype_grouped_weights[dtype][name] = param_or_buf

        for dtype, name2weights in dtype_grouped_weights.items():
            # total # of parameters + buffers
            weights_with_local = []
            for name, t in name2weights.items():
                local_t = t.to_local() if hasattr(t, "to_local") else t
                stride = local_t.stride()
                storage_numel = (
                    0
                    if local_t.numel() == 0
                    else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(local_t.shape, stride))
                )
                weights_with_local.append((name, t, local_t, storage_numel, stride))
            total_numel = sum(storage_numel for _, _, _, storage_numel, _ in weights_with_local)
            cpu_tensor = torch.empty(total_numel, dtype=dtype, device="cpu", pin_memory=pin_memory)

            current_offset = 0
            for (
                name,
                original_tensor,
                local_tensor,
                storage_numel,
                stride,
            ) in weights_with_local:
                if local_tensor.is_contiguous():
                    flat_storage = local_tensor.flatten()
                else:
                    # Cutlass FP8 weights use a transposed physical layout.
                    # Preserve it across the flattened CPU staging buffer.
                    flat_storage = torch.zeros(
                        storage_numel,
                        dtype=dtype,
                        device=local_tensor.device,
                    )
                    physical_view = torch.as_strided(
                        flat_storage,
                        size=local_tensor.shape,
                        stride=stride,
                    )
                    physical_view.copy_(local_tensor)
                cpu_tensor[current_offset : current_offset + storage_numel].copy_(flat_storage)
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": storage_numel,
                        "shape": local_tensor.shape,
                        "stride": stride,
                    }
                )

                LayerwiseOffloadHook._set_tensor_storage(
                    original_tensor, LayerwiseOffloadHook._make_offload_placeholder(original_tensor)
                )
                current_offset += storage_numel

            dtype_cpu_flattened_weights[dtype] = cpu_tensor

        return dtype_cpu_flattened_weights, dtype_metadata

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        for param in self.block_parameters.values():
            return LayerwiseOffloadHook._is_materialized_tensor(param)

        return True

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking: bool = True) -> None:
        """Copy layer weights from CPU -> GPU.

        Pre-fetch target block in an asynchronous way with compute - memory copy overlap,
        with non_blocking set to True.
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        layer_params = self.next_block_parameters
        layer_bufs = self.next_block_buffers

        evt = current_omni_platform.Event()
        gpu_weights: dict[torch.dtype, torch.Tensor] = {}

        with current_omni_platform.stream(self.copy_stream):
            for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
                gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                gpu_weights[dtype] = gpu_weight

            evt.record(self.copy_stream)

        for dtype, ordered_metadata in self.dtype_metadata.items():
            # ordered_metadata: list[dict[str, Any]]
            gpu_weight = gpu_weights[dtype]

            for metadata in ordered_metadata:
                target_name = metadata["name"]
                target_param_or_buf = (
                    layer_params[target_name] if target_name in layer_params else layer_bufs[target_name]
                )

                LayerwiseOffloadHook._set_tensor_storage(
                    target_param_or_buf,
                    torch.as_strided(
                        gpu_weight[metadata["offset"] : metadata["offset"] + metadata["numel"]],
                        size=metadata["shape"],
                        stride=metadata["stride"],
                    ),
                )

        self._prefetch_done = evt

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for layer by replacing tensors with empty placeholders.
        This function does not actually offload weights from GPU back to CPU.
        """
        evt = self._prefetch_done
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)

        self._prefetch_done = None

        # free GPU residency
        for _, param in self.block_parameters.items():
            LayerwiseOffloadHook._set_tensor_storage(param, LayerwiseOffloadHook._make_offload_placeholder(param))
        for _, buf in self.block_buffers.items():
            LayerwiseOffloadHook._set_tensor_storage(buf, LayerwiseOffloadHook._make_offload_placeholder(buf))

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # if the previous hook was skipped and the weights are not on device,
        # (e.g. by cache-dit block caching), ask the previous hook to
        # synchronously prefetch *this* block's weights before computation
        if not self.is_materialized and self._prev_hook is not None:
            self._prev_hook.prefetch_layer(non_blocking=False)

        self.prefetch_layer(non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()

        return output


def apply_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    stream: current_omni_platform.Stream | None = None,
    pin_memory: bool = True,
) -> LayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = LayerwiseOffloadHook(next_block, device, stream, pin_memory)
    registry.register_hook(LayerwiseOffloadHook._HOOK_NAME, hook)

    return hook


def remove_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(LayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed offload hook from %s", module.__class__.__name__)


class LayerWiseOffloadBackend(OffloadBackend):
    """Layer-wise (block-level) offloading backend.

    Implements sliding window offloading where only a small number of transformer
    blocks reside on GPU at a time. Blocks are prefetched asynchronously while
    previous blocks compute, and freed after use.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self._blocks: list[list[nn.Module]] = []

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("LayerWiseOffloadBackend already enabled")
            return

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping layer-wise offloading")
            return

        # Move encoders to GPU (they stay resident), or apply layerwise
        # offload if the encoder is too large to fit in GPU memory.
        for enc_name, enc in zip(modules.encoder_names, modules.encoders):
            enc_size = sum(p.nbytes for p in enc.parameters()) + sum(
                b.nbytes for b in enc.buffers()
            )
            try:
                free_mem = current_omni_platform.get_free_memory()
            except Exception:
                free_mem = float("inf")

            if enc_size > free_mem:
                # Try layerwise offload on encoder blocks
                blocks = self._discover_encoder_blocks(enc)
                if blocks and len(blocks) > 1:
                    # Move non-block children to GPU (small: embeddings, norms)
                    block_parent, block_attr_name = self._encoder_block_parent_and_attr(enc)
                    if block_parent is not None:
                        for child_name, child in block_parent.named_children():
                            if child_name != block_attr_name:
                                child.to(self.device)
                        for p in block_parent._parameters.values():
                            if p is not None:
                                p.data = p.data.to(self.device, non_blocking=True)
                        for b in block_parent._buffers.values():
                            if b is not None:
                                b.data = b.data.to(self.device, non_blocking=True)
                    # Move any top-level children of enc that are NOT ancestors of blocks
                    # (e.g. enc has language_model -> model -> layers; move siblings of language_model)
                    if block_parent is not enc:
                        for child_name, child in enc.named_children():
                            # Skip if this child is an ancestor of block_parent
                            is_ancestor = any(m is block_parent for m in child.modules())
                            if not is_ancestor:
                                child.to(self.device)
                        for p in enc._parameters.values():
                            if p is not None:
                                p.data = p.data.to(self.device, non_blocking=True)
                        for b in enc._buffers.values():
                            if b is not None:
                                b.data = b.data.to(self.device, non_blocking=True)
                        # Also move intermediate wrappers' own params/buffers
                        # (e.g. language_model's own params, language_model.model's own params)
                        self._move_intermediate_params_to_device(enc, block_parent)

                    # Apply block hooks for streaming
                    last_block, first_block = blocks[-1], blocks[0]
                    last_hook = apply_block_hook(
                        last_block,
                        first_block,
                        self.device,
                        self.copy_stream,
                        self.config.pin_cpu_memory,
                    )
                    last_hook.prefetch_layer(non_blocking=False)

                    block_hooks: list[LayerwiseOffloadHook] = [last_hook]
                    for i, block in enumerate(blocks[:-1]):
                        next_block = blocks[(i + 1) % len(blocks)]
                        hook = apply_block_hook(
                            block,
                            next_block,
                            self.device,
                            self.copy_stream,
                            self.config.pin_cpu_memory,
                        )
                        block_hooks.append(hook)

                    for i in range(len(block_hooks)):
                        block_hooks[i]._prev_hook = block_hooks[i - 1]

                    self._blocks.append(blocks)
                    logger.info(
                        "Encoder '%s' too large (%.2f GiB > %.2f GiB free), "
                        "applied layerwise offload on %d blocks",
                        enc_name,
                        enc_size / (1024**3),
                        free_mem / (1024**3),
                        len(blocks),
                    )
                else:
                    # No blocks found; try to move anyway (may OOM)
                    logger.warning(
                        "Encoder '%s' is %.2f GiB but no streamable blocks found; "
                        "attempting full GPU placement",
                        enc_name,
                        enc_size / (1024**3),
                    )
                    enc.to(self.device)
            else:
                enc.to(self.device)

        # Move VAE(s) to GPU if available
        for vae in modules.vaes:
            try:
                vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying layer-wise offloading on %s", modules.dit_names)

        # Apply block-wise offloading hook for each of the blocks in DiT model(s)
        # Note that there might exist multiple DiT models in specific pipelines
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)

            if not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            num_blocks = len(blocks)
            if num_blocks <= 1:
                logger.warning(
                    "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

            # Move non-block modules to GPU (they stay resident)
            for name, m in dit_module.named_children():
                if name not in blocks_attr_names:
                    m.to(self.device)
                    logger.debug(f"Moved {name} to device {self.device}")
                else:
                    logger.debug(f"Skipped blocks module {name}")

            # Move top-level params/buffers to GPU (dit_module's own, not sub-modules)
            for param in dit_module._parameters.values():
                if param is not None:
                    param.data = param.data.to(self.device, non_blocking=True)

            for buffer in dit_module._buffers.values():
                if buffer is not None:
                    buffer.data = buffer.data.to(self.device, non_blocking=True)

            # Pre-fetch the first layer by manually calling the hook function on the last layer;
            # For subsequent requests, the first layer/block will be pre-fetched
            # during the last layer compute of the previous request.
            last_block, first_block = blocks[-1], blocks[0]
            last_hook = apply_block_hook(
                last_block,
                first_block,
                self.device,
                self.copy_stream,
                self.config.pin_cpu_memory,
            )
            last_hook.prefetch_layer(non_blocking=False)

            block_hooks: list[LayerwiseOffloadHook] = [last_hook]
            # Register hook for each of blocks
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[(i + 1) % num_blocks]
                hook = apply_block_hook(
                    block,
                    next_block,
                    self.device,
                    self.copy_stream,
                    self.config.pin_cpu_memory,
                )
                block_hooks.append(hook)

            # NOTE(yuanheng-zhao): We make each hook gets a backward reference to the hook
            # that is responsible for prefetching its block's weights. This is specifically a
            # workaround for that arbitrary blocks are skipped by caching systems (e.g., cache-dit)
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            logger.info(f"Layer-wise offloading enabled on {num_blocks} layers (blocks)")

            # Track hooked blocks for cleanup
            self._blocks.append(blocks)

        if len(self._blocks) > 0 and len(self._blocks[0]) > 0:
            self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return

        for blocks in self._blocks:
            for block in blocks:
                remove_block_hook(block)

        self._blocks.clear()
        self.enabled = False
        logger.info("Layer-wise offloading disabled")

    # ---- Encoder layerwise offload helpers ----

    # Common dotted paths where encoder models store their block layers.
    _ENCODER_BLOCK_SEARCH_PATHS = [
        "language_model.model.layers",
        "model.layers",
        "layers",
        "encoder.layers",
        "transformer.layers",
        "blocks",
        "encoder.block",
        "model.encoder.layers",
    ]

    def _discover_encoder_blocks(self, enc: nn.Module) -> list[nn.Module] | None:
        """Find a streamable nn.ModuleList of blocks inside an encoder module.

        Searches common attribute paths used by text/vision encoders.
        Returns the list of block modules, or None if not found.
        """
        from operator import attrgetter

        for attr_path in self._ENCODER_BLOCK_SEARCH_PATHS:
            try:
                candidate = attrgetter(attr_path)(enc)
            except AttributeError:
                continue
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                # Store the successful path for later use
                self._last_encoder_block_path = attr_path
                return list(candidate)
        return None

    def _encoder_block_parent_and_attr(
        self, enc: nn.Module
    ) -> tuple[nn.Module | None, str | None]:
        """Return (parent_module, attr_name) for the discovered block list.

        E.g. for path 'language_model.model.layers', returns
        (enc.language_model.model, 'layers').
        """
        from operator import attrgetter

        attr_path = getattr(self, "_last_encoder_block_path", None)
        if attr_path is None:
            return None, None

        parts = attr_path.split(".")
        attr_name = parts[-1]
        if len(parts) > 1:
            parent_path = ".".join(parts[:-1])
            try:
                parent = attrgetter(parent_path)(enc)
            except AttributeError:
                return None, None
        else:
            parent = enc
        return parent, attr_name

    def _move_intermediate_params_to_device(
        self, enc: nn.Module, block_parent: nn.Module
    ) -> None:
        """Move own parameters/buffers of intermediate wrapper modules to GPU.

        For a path like enc.language_model.model.layers, this moves the own
        params/buffers of enc.language_model and enc.language_model.model
        (but NOT their children, which are handled separately).
        """
        from operator import attrgetter

        attr_path = getattr(self, "_last_encoder_block_path", None)
        if attr_path is None:
            return

        parts = attr_path.split(".")
        # Walk intermediate modules (excluding the last part which is the blocks)
        current = enc
        for part in parts[:-1]:
            try:
                current = getattr(current, part)
            except AttributeError:
                break
            # Move this intermediate module's own params/buffers (not children)
            for p in current._parameters.values():
                if p is not None:
                    p.data = p.data.to(self.device, non_blocking=True)
            for b in current._buffers.values():
                if b is not None:
                    b.data = b.data.to(self.device, non_blocking=True)

    @staticmethod
    def get_blocks_attr_names(model: nn.Module) -> list[str]:
        """Get block attribute names from model class."""
        attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

        if not attrs:
            old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
            if old_attr is not None:
                logger.warning(
                    "'_layerwise_offload_blocks_attr' is deprecated, "
                    "please use '_layerwise_offload_blocks_attrs' instead. "
                    "Example: _layerwise_offload_blocks_attrs = ['blocks']"
                )
                attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

        return attrs

    @staticmethod
    def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
        if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
            setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)

    @staticmethod
    def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
        """
        Retrieve blocks and attribute names from provided DiT model. Blocks attribute names
        are found by `_layerwise_offload_blocks_attrs` set to DiT models. For example,

        ```
        class WanTransformer3DModel(nn.Module):
            _layerwise_offload_blocks_attrs = ["blocks"]
        ```

        Returns:
            Tuple of (blocks_attr_names, blocks)
        """
        blocks_attr_names = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        if not blocks_attr_names:
            logger.warning(
                f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
                "skipping layerwise offloading"
            )
            return [], []

        blocks = []
        for name in blocks_attr_names:
            attr = getattr(model, name, None)
            if attr is None:
                raise AttributeError(
                    f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                    f"does not exist on model {model.__class__.__name__}"
                )
            try:
                attr_iter = iter(attr)
            except TypeError:
                if isinstance(attr, nn.Module):
                    logger.warning(
                        "Attribute '%s' on %s is not iterable; treating it as one block.",
                        name,
                        model.__class__.__name__,
                    )
                    blocks.append(attr)
                    continue

                logger.warning(
                    "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                    name,
                    model.__class__.__name__,
                    type(attr).__name__,
                )
            else:
                blocks.extend(attr_iter)

        if not blocks:
            logger.warning(
                "No blocks found in %s for %s, skipping layerwise offloading",
                blocks_attr_names,
                model.__class__.__name__,
            )
            return [], []

        return blocks_attr_names, blocks

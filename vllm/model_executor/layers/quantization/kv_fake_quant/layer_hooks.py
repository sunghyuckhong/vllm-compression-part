# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-Attention-layer KV-quant state + the two hot-path entry points.

``LayerKVQuantState`` is an ``nn.Module`` so the SmoothKV calib buffers
(``s_k``, ``s_v``) auto-migrate when ``module.to(device)`` is called on the
parent Attention layer.

  attach_kv_quant_to_layer(layer, prefix)        called from Attention.__init__
                                                 builds + attaches state
  apply_kv_quant(layer, key, value) -> (K, V)    called from Attention.forward
                                                 reads state + dispatches to kernels
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import KVCacheQuantConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger

from .kernels import (
    fake_quantize_fp8,
    fake_quantize_nvfp4,
    fake_quantize_pertoken,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-layer state object
# ---------------------------------------------------------------------------

class LayerKVQuantState(nn.Module):
    """Per-Attention-layer KV-quant state.

    Single source of truth: read by ``apply_kv_quant`` and by the inline
    check in ``Attention.forward``. Subclassing ``nn.Module`` (rather than
    using a plain dataclass) makes the smoothkv calib buffers auto-migrate
    with the parent's ``module.to(device)``.

    Attributes:
        method: "fp8" / "pertoken" / "smoothkv" / "nvfp4" / "smkv_nvfp4".
            Note: when the user picked "smoothkv_fused", this is "pertoken"
            because the s_K/s_V scaling was already folded into the projection
            weights at load time (see ``fusion.py``).
        group_size: per-group size for the quant kernels.
        bits: bit width (4 for pertoken/smoothkv, 8 for fp8 -- ignored on
            fp8/nvfp4).
        s_k / s_v: SmoothKV per-(kv_head, channel) scales, registered as
            non-persistent buffers. Only present when method ==
            "smoothkv" or "smkv_nvfp4".
        gs_k / gs_v: NVFP4 per-tensor (per-layer) FP32 global scales,
            registered as non-persistent buffers. Only present when method ==
            "nvfp4" or "smkv_nvfp4".
    """

    def __init__(
        self,
        method: str,
        group_size: int,
        bits: int,
        s_k: torch.Tensor | None = None,
        s_v: torch.Tensor | None = None,
        gs_k: torch.Tensor | None = None,
        gs_v: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.group_size = group_size
        self.bits = bits
        if s_k is not None:
            assert s_v is not None, "s_k/s_v must be provided together"
            # Non-persistent buffers auto-migrate with module.to(device); won't
            # be saved with state_dict.
            self.register_buffer("s_k", s_k, persistent=False)
            self.register_buffer("s_v", s_v, persistent=False)
        if gs_k is not None:
            assert gs_v is not None, "gs_k/gs_v must be provided together"
            self.register_buffer("gs_k", gs_k, persistent=False)
            self.register_buffer("gs_v", gs_v, persistent=False)


# ---------------------------------------------------------------------------
# Calib cache (one entry per calib_path, populated lazily inside the worker
# the first time a smoothkv layer is constructed)
# ---------------------------------------------------------------------------

_CALIB_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_GLOBAL_SCALES_CACHE: dict[
    tuple[str, bool], tuple[torch.Tensor, torch.Tensor]
] = {}


def _str_to_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        name, torch.bfloat16
    )


def _load_calib_cached(
    calib_path: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (s_K, s_V) from the calib .pt and keep them on CPU; cache by path
    so workers only pay the IO cost once."""
    cached = _CALIB_CACHE.get(calib_path)
    if cached is not None:
        return cached
    calib = torch.load(calib_path, weights_only=True)
    sk = calib["s_K"].to(dtype)
    sv = calib["s_V"].to(dtype)
    _CALIB_CACHE[calib_path] = (sk, sv)
    return sk, sv


def _load_global_scales_cached(
    path: str, smooth: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-layer NVFP4 global scales from the derived .pt, kept FP32
    on CPU and cached by (path, smooth). ``smooth=True`` selects the
    SmoothKV+NVFP4 variant (gs_K_smooth / gs_V_smooth); ``smooth=False``
    selects plain NVFP4 (gs_K_raw / gs_V_raw)."""
    key = (path, smooth)
    cached = _GLOBAL_SCALES_CACHE.get(key)
    if cached is not None:
        return cached
    blob = torch.load(path, weights_only=True)
    if smooth:
        gk, gv = blob["gs_K_smooth"], blob["gs_V_smooth"]
    else:
        gk, gv = blob["gs_K_raw"], blob["gs_V_raw"]
    gk = gk.to(torch.float32)
    gv = gv.to(torch.float32)
    _GLOBAL_SCALES_CACHE[key] = (gk, gv)
    return gk, gv


def get_active_kv_quant_config() -> KVCacheQuantConfig | None:
    """Read the active KV-quant config from the current VllmConfig, or None
    if no LLM is constructed / no kv_cache_quant_config was set."""
    vllm_cfg = get_current_vllm_config_or_none()
    if vllm_cfg is None:
        return None
    cfg = vllm_cfg.kv_cache_quant_config
    if cfg is None or not cfg.is_active():
        return None
    return cfg


# ---------------------------------------------------------------------------
# Per-layer registration (called from Attention.__init__)
# ---------------------------------------------------------------------------

def attach_kv_quant_to_layer(layer, prefix: str) -> None:
    """Build a ``LayerKVQuantState`` from the active config and attach it as
    ``layer.kv_quant_state``. No-op unless ``LLM(kv_cache_quant_config=...)``
    was set.

    For ``smoothkv_fused``: at runtime the layer behaves as plain ``pertoken``
    (the s_K / s_V scaling is already folded into qkv_proj / o_proj weights at
    load time by ``maybe_run_post_load_fusion``). So we set the per-layer
    method to ``"pertoken"`` here.
    """
    cfg = get_active_kv_quant_config()
    if cfg is None:
        return

    runtime_method = "pertoken" if cfg.method == "smoothkv_fused" else cfg.method
    s_k, s_v = None, None
    gs_k, gs_v = None, None
    if cfg.method in ("smoothkv", "smkv_nvfp4"):
        s_k, s_v = _resolve_smoothkv_scales(layer, prefix, cfg)
    if cfg.method in ("nvfp4", "smkv_nvfp4"):
        gs_k, gs_v = _resolve_nvfp4_global_scales(layer, prefix, cfg)

    layer.kv_quant_state = LayerKVQuantState(
        method=runtime_method,
        group_size=cfg.group_size,
        bits=cfg.bits,
        s_k=s_k,
        s_v=s_v,
        gs_k=gs_k,
        gs_v=gs_v,
    )


def _resolve_smoothkv_scales(
    layer, prefix: str, cfg: KVCacheQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the full calib (num_layers, full_num_kv_heads, head_dim) down to
    this layer + this TP worker's kv-head shard.

    Returns scales on the layer's device. The calib .pt is loaded onto CPU;
    the runtime smoothkv kernel divides the on-device key/value by these
    scales, so they must move to GPU here. (LayerKVQuantState's
    register_buffer would do this automatically on `module.to(device)` —
    but vLLM's v1 engine constructs Attention modules with the device
    context already set, so the buffers are never explicitly migrated.)
    """
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.models.utils import extract_layer_index
    try:
        layer_idx = extract_layer_index(prefix)
    except Exception as e:
        raise RuntimeError(
            f"[kv_fake_quant] SmoothKV could not extract layer_idx "
            f"from prefix={prefix!r}"
        ) from e
    try:
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_rank = 0
    s_k_full, s_v_full = _load_calib_cached(
        cfg.calib_path, _str_to_dtype(cfg.dtype)
    )
    # Pin the slice to the same device as the layer's projection weights.
    # Falls back to current CUDA device when the layer has no parameters yet.
    try:
        device = next(layer.parameters()).device
    except StopIteration:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu"
        )
    full_kv_heads = s_k_full.shape[1]
    per_worker = layer.num_kv_heads
    if per_worker == full_kv_heads:
        return (s_k_full[layer_idx].clone().to(device),
                s_v_full[layer_idx].clone().to(device))
    lo = tp_rank * per_worker
    hi = lo + per_worker
    return (s_k_full[layer_idx, lo:hi].clone().to(device),
            s_v_full[layer_idx, lo:hi].clone().to(device))


def _resolve_nvfp4_global_scales(
    layer, prefix: str, cfg: KVCacheQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the per-layer NVFP4 global scale tensor (num_layers,) down to
    this layer. Returns FP32 scalars (shape [1]) on the layer's device.

    Global scale is per-tensor (per-layer) by NVFP4 spec, so it is the same
    across all TP ranks and head shards -- no kv-head splitting required.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    try:
        layer_idx = extract_layer_index(prefix)
    except Exception as e:
        raise RuntimeError(
            f"[kv_fake_quant] NVFP4 could not extract layer_idx "
            f"from prefix={prefix!r}"
        ) from e
    smooth = (cfg.method == "smkv_nvfp4")
    gs_k_full, gs_v_full = _load_global_scales_cached(
        cfg.global_scales_path, smooth=smooth,
    )
    try:
        device = next(layer.parameters()).device
    except StopIteration:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu"
        )
    return (gs_k_full[layer_idx].clone().reshape(1).to(device),
            gs_v_full[layer_idx].clone().reshape(1).to(device))


# ---------------------------------------------------------------------------
# Forward dispatch (called from Attention.forward)
# ---------------------------------------------------------------------------

def apply_kv_quant(
    layer, key: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake-quantized (K, V) per the layer's configured method.

    Reads ``layer.kv_quant_state``. torch.compile specializes per
    (method, group_size, bits) value at trace time.
    """
    state: LayerKVQuantState = layer.kv_quant_state
    method = state.method
    gs = state.group_size
    bits = state.bits
    nh = layer.num_kv_heads
    hd = layer.head_size

    if method == "fp8":
        key = fake_quantize_fp8(key, nh, hd, gs)
        value = fake_quantize_fp8(value, nh, hd, gs)
    elif method == "pertoken":
        key = fake_quantize_pertoken(key, nh, hd, gs, bits)
        value = fake_quantize_pertoken(value, nh, hd, gs, bits)
    elif method == "smoothkv":
        sk_flat = state.s_k.reshape(-1)
        sv_flat = state.s_v.reshape(-1)
        k_s = key / sk_flat
        k_s = fake_quantize_pertoken(k_s, nh, hd, gs, bits)
        key = k_s * sk_flat
        v_s = value / sv_flat
        v_s = fake_quantize_pertoken(v_s, nh, hd, gs, bits)
        value = v_s * sv_flat
    elif method == "nvfp4":
        key = fake_quantize_nvfp4(key, nh, hd, state.gs_k)
        value = fake_quantize_nvfp4(value, nh, hd, state.gs_v)
    elif method == "smkv_nvfp4":
        sk_flat = state.s_k.reshape(-1)
        sv_flat = state.s_v.reshape(-1)
        k_s = key / sk_flat
        k_s = fake_quantize_nvfp4(k_s, nh, hd, state.gs_k)
        key = k_s * sk_flat
        v_s = value / sv_flat
        v_s = fake_quantize_nvfp4(v_s, nh, hd, state.gs_v)
        value = v_s * sv_flat
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return key, value

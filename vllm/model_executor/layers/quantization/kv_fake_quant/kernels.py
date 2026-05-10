# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fake-quantization kernels for KV cache.

``@torch.library.custom_op`` ops that appear by name in inductor's FX graph
dump (``computation_graph.py``):

    vllm_kv_quant::fake_quantize_dequantize_fp8(x, group_size) -> Tensor
    vllm_kv_quant::fake_quantize_dequantize_nvfp4(x, global_scale) -> Tensor
    vllm_kv_quant::quant_and_pack_vcache(v, group_size, bits) -> (code, scale, mn)
    vllm_kv_quant::unpack_and_dequant_vcache(code, scale, mn, group_size, bits) -> Tensor

Plus high-level shape-handling wrappers:

    fake_quantize_fp8(x, num_kv_heads, head_dim, group_size) -> Tensor
    fake_quantize_nvfp4(x, num_kv_heads, head_dim, global_scale) -> Tensor
    fake_quantize_pertoken(x, num_kv_heads, head_dim, group_size, bits) -> Tensor

The custom_ops are opaque to torch.compile -- the compiler doesn't try to
inline or rewrite the body. Their names appear verbatim in inductor's
computation_graph.py, which is what verify_compiled_graph.py greps for.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# FP8 E4M3 quant-dequant
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0


def _round_to_fp8e4m3(x_fp32: torch.Tensor) -> torch.Tensor:
    """sm_80-compatible E4M3 round (matches hardware cast for normal range).

    On A100, Triton's inductor codegen cannot lower torch.float8_e4m3fn casts
    inside cudagraphs; this fp32-arithmetic emulation does and is bit-identical
    for normals.
    """
    sign = torch.sign(x_fp32)
    abs_x = x_fp32.abs().clamp(max=_FP8_MAX)
    eps_floor = 2.0 ** -9
    abs_safe = abs_x.clamp(min=eps_floor)
    exp = torch.floor(torch.log2(abs_safe))
    exp_clamped = exp.clamp(min=-6.0, max=8.0)
    pow2_exp = torch.pow(2.0, exp_clamped)
    mantissa_q = torch.round(abs_x / pow2_exp * 8.0) / 8.0
    out = sign * mantissa_q * pow2_exp
    return torch.where(x_fp32 == 0, torch.zeros_like(out), out)


def _is_sm89_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


@torch.library.custom_op(
    "vllm_kv_quant::fake_quantize_dequantize_fp8", mutates_args=()
)
def _fake_quantize_dequantize_fp8(
    data: torch.Tensor, group_size: int
) -> torch.Tensor:
    """FP8 E4M3 quant-dequant round-trip on (B, nh, T, D) input."""
    B, nh, T, D = data.shape
    num_groups = D // group_size
    grouped = data.view(B, nh, T, num_groups, group_size)

    amax = grouped.abs().amax(dim=-1).to(torch.float32)
    scale = (amax / _FP8_MAX).clamp(min=1e-4)
    scaled = grouped.to(torch.float32) / scale.unsqueeze(-1)

    if _is_sm89_or_newer():
        rounded = scaled.to(_FP8_DTYPE).to(torch.float32)
    else:
        rounded = _round_to_fp8e4m3(scaled)

    out = (rounded * scale.unsqueeze(-1)).view(B, nh, T, D)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(data.dtype)


@_fake_quantize_dequantize_fp8.register_fake
def _fake_quantize_dequantize_fp8_meta(
    data: torch.Tensor, group_size: int
) -> torch.Tensor:
    return torch.empty_like(data)


# ---------------------------------------------------------------------------
# NVFP4 quant-dequant (per-tensor static global scale, per-group dynamic
# FP8 E4M3 block scale, per-element FP4 E2M1)
# ---------------------------------------------------------------------------

_FP4_MAX = 6.0     # FP4_E2M1_DATA.max
_NVFP4_GROUP_SIZE = 16


def _round_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest FP4 E2M1 grid value:
        ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

    Bit-identical to ``compressed_tensors.quantization.quant_args.FP4_E2M1_DATA
    .cast_to_fp4`` — including its asymmetric tie-breaking at exact midpoints
    {0.25, 1.25, 2.5, 5.0} which round DOWN to the smaller grid value
    (compressed_tensors closes its smaller-magnitude intervals on both sides:
    [0,0.25], [0.75,1.25], [1.75,2.5], [3.5,5.0]).

    Practically these midpoints are measure-zero in real-valued inputs, but
    matching the canonical reference avoids spurious skew at the few tie cases.
    """
    sign = torch.sign(x)
    a = x.abs()
    # Midpoints between adjacent positive grid values.
    # Inequalities chosen so a==0.25→0, a==1.25→1, a==2.5→2, a==5→4
    # (matches compressed_tensors round-half-toward-smaller).
    out = torch.where(a <= 0.25, torch.zeros_like(a), torch.full_like(a, 0.5))
    out = torch.where(a >= 0.75, torch.full_like(a, 1.0), out)
    out = torch.where(a >  1.25, torch.full_like(a, 1.5), out)
    out = torch.where(a >= 1.75, torch.full_like(a, 2.0), out)
    out = torch.where(a >  2.5,  torch.full_like(a, 3.0), out)
    out = torch.where(a >= 3.5,  torch.full_like(a, 4.0), out)
    out = torch.where(a >  5.0,  torch.full_like(a, 6.0), out)
    return sign * out


@torch.library.custom_op(
    "vllm_kv_quant::fake_quantize_dequantize_nvfp4", mutates_args=()
)
def _fake_quantize_dequantize_nvfp4(
    data: torch.Tensor, global_scale: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 quant-dequant on (B, nh, T, D) input.

    Bit-identical to ``vllm.model_executor.layers.quantization.utils.
    nvfp4_emulation_utils.ref_nvfp4_quant`` for fp32 ``global_scale``
    (verified at file head). Compute order matches the reference:

        scale          = round_fp8(amax_per_group * global_scale / FP4_MAX)
        output_scale   = 1 / (scale * (1 / global_scale))
        x_fp4          = round_fp4(clamp(x * output_scale, -6, 6))
        x_dequantized  = x_fp4 * (scale / global_scale)

    The reciprocal-then-multiply ordering matters at fp32 ULP-level for
    bf16 inputs near the FP4 grid boundaries; using `x / (scale/gs)` directly
    introduces ~0.1% disagreements with the reference for those edge cases.
    """
    B, nh, T, D = data.shape
    G = _NVFP4_GROUP_SIZE
    assert D % G == 0, f"head_dim {D} must be divisible by NVFP4 group_size {G}"
    num_groups = D // G

    grouped = data.view(B, nh, T, num_groups, G).to(torch.float32)
    gs = global_scale.to(torch.float32).reshape(1)

    # 1. per-group amax → FP8 local scale
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    local_scale_unrounded = gs * (amax * (1.0 / _FP4_MAX))
    if _is_sm89_or_newer():
        local_scale = local_scale_unrounded.clamp(max=_FP8_MAX, min=-_FP8_MAX).to(_FP8_DTYPE).to(torch.float32)
    else:
        local_scale = _round_to_fp8e4m3(local_scale_unrounded)

    # 2. quantize using ref's reciprocal-then-multiply ordering (bit-identical
    # to ref_nvfp4_quant: output_scale = 1 / (scale * (1/global_scale)))
    gs_reciprocal = 1.0 / (gs + (gs == 0) * 1e8)
    inner = local_scale * gs_reciprocal
    output_scale = 1.0 / (inner + (inner == 0) * 1e8)

    # 3. quantize: x * output_scale, clamp to ±FP4_MAX, round to FP4 grid
    scaled = grouped * output_scale
    scaled = scaled.clamp(min=-_FP4_MAX, max=_FP4_MAX)
    fp4 = _round_to_fp4_e2m1(scaled)

    # 4. dequantize using ref's order: fp4 * (scale / global_scale)
    # (direct division — must NOT use scale * (1/gs), it differs at fp32 ULP)
    dequant_scale = local_scale / gs
    out = (fp4 * dequant_scale).view(B, nh, T, D)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(data.dtype)


@_fake_quantize_dequantize_nvfp4.register_fake
def _fake_quantize_dequantize_nvfp4_meta(
    data: torch.Tensor, global_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(data)


# ---------------------------------------------------------------------------
# int{2,4} per-token pack/unpack (groups along D = head_dim)
# ---------------------------------------------------------------------------

def _pack_tensor(data: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack along the last dim: feat_per_int (32 // bits) elements per int32."""
    feat_per_int = 32 // bits
    out_shape = data.shape[:-1] + (data.shape[-1] // feat_per_int,)
    code = torch.zeros(out_shape, dtype=torch.int32, device=data.device)
    for j in range(feat_per_int):
        code |= data[..., j::feat_per_int] << (bits * j)
    return code


def _unpack_tensor(code: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack the int32-packed last dim back to per-element int16."""
    feat_per_int = 32 // bits
    new_len = code.shape[-1] * feat_per_int
    j = torch.arange(new_len, device=code.device) % feat_per_int
    i = torch.arange(new_len, device=code.device) // feat_per_int
    mask = 0xFF >> (8 - bits)
    return ((code[..., i] >> (j * bits)).to(torch.int16)) & mask


@torch.library.custom_op(
    "vllm_kv_quant::quant_and_pack_vcache", mutates_args=()
)
def _quant_and_pack_vcache(
    v: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token V quant: groups along last dim (head_dim). (B, nh, T, D)."""
    shape = v.shape
    num_groups = shape[-1] // group_size
    new_shape = shape[:-1] + (num_groups, group_size)
    max_int = 2 ** bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = (data - mn) / scale
    data = data.clamp(0, max_int).round().to(torch.int32).view(shape)
    code = _pack_tensor(data, bits)
    return code, scale, mn


@_quant_and_pack_vcache.register_fake
def _quant_and_pack_vcache_meta(
    v: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, nh, T, D = v.shape
    feat_per_int = 32 // bits
    num_groups = D // group_size
    code = torch.empty(B, nh, T, D // feat_per_int, dtype=torch.int32, device=v.device)
    scale = torch.empty(B, nh, T, num_groups, 1, dtype=v.dtype, device=v.device)
    mn = torch.empty_like(scale)
    return code, scale, mn


@torch.library.custom_op(
    "vllm_kv_quant::unpack_and_dequant_vcache", mutates_args=()
)
def _unpack_and_dequant_vcache(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    data = _unpack_tensor(code, bits)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:-1] + (num_groups, group_size)).to(scale.dtype)
    data = data * scale + mn
    return data.view(shape)


@_unpack_and_dequant_vcache.register_fake
def _unpack_and_dequant_vcache_meta(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    feat_per_int = 32 // bits
    B, nh, T, packed_D = code.shape
    return torch.empty(B, nh, T, packed_D * feat_per_int,
                       dtype=scale.dtype, device=code.device)


# ---------------------------------------------------------------------------
# Shape helpers + high-level fake-quant entry points
# ---------------------------------------------------------------------------

def _to_bnhtd(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape (T, num_kv_heads*head_dim) or (T, num_kv_heads, head_dim)
    -> (B=1, nh, T, D)."""
    if x.dim() == 2:
        T = x.shape[0]
        return x.view(1, T, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    if x.dim() == 3:
        return x.unsqueeze(0).transpose(1, 2).contiguous()
    raise ValueError(f"unexpected KV shape {tuple(x.shape)}")


def _from_bnhtd(x4: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    return x4.transpose(1, 2).contiguous().view(*orig_shape)


def fake_quantize_fp8(
    x: torch.Tensor, num_kv_heads: int, head_dim: int, group_size: int
) -> torch.Tensor:
    """FP8 E4M3 round-trip with reshape into KIVI's canonical layout."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = torch.ops.vllm_kv_quant.fake_quantize_dequantize_fp8(x4, group_size)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


def fake_quantize_nvfp4(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 round-trip with reshape into KIVI's canonical layout.

    ``global_scale`` is a per-tensor (per-layer) FP32 scalar derived offline
    from the K (or V) cache amax, see ``scripts/derive_nvfp4_global_scales.py``.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = torch.ops.vllm_kv_quant.fake_quantize_dequantize_nvfp4(x4, global_scale)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


def fake_quantize_pertoken(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
    group_size: int, bits: int,
) -> torch.Tensor:
    """Per-token int{2,4} quant: groups along the last dim (head_dim).

    Used for both K and V in pertoken / smoothkv methods (the kernel is
    symmetric in K vs V).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    code, scale, mn = torch.ops.vllm_kv_quant.quant_and_pack_vcache(
        x4, group_size, bits
    )
    out = torch.ops.vllm_kv_quant.unpack_and_dequant_vcache(
        code, scale, mn, group_size, bits
    )
    return _from_bnhtd(out, orig_shape).to(orig_dtype)

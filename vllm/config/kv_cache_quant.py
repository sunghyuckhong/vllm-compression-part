# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-cache fake-quantization config.

Picklable dataclass that flows through ``VllmConfig`` to every worker. The
heavy SmoothKV calib tensors are NOT stored here -- only the file path is.
Each worker lazy-loads the calib (with caching) inside
``attach_kv_quant_to_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheQuantConfig:
    """Configuration for KV-cache fake-quantization.

    Pass to ``LLM(...)`` via ``kv_cache_quant_config=KVCacheQuantConfig(...)``.

    Attributes:
        method: One of bf16 / fp16 / fp8 / pertoken / smoothkv / smoothkv_fused
            / nvfp4 / smkv_nvfp4. "bf16" / "fp16" are no-op baselines.
        group_size: Per-group size for the int{2,4} quant kernels (default 128).
            Ignored for nvfp4 (group_size is always 16, set inside the kernel).
        bits: Bit width for int{2,4} pertoken / smoothkv (default 4).
            Ignored for fp8/nvfp4.
        calib_path: Required for smoothkv / smoothkv_fused / smkv_nvfp4.
            Points to a SmoothKV `.pt` with keys "s_K" / "s_V" of shape
            ``(num_layers, num_kv_heads, head_dim)``.
        global_scales_path: Required for nvfp4 / smkv_nvfp4. Points to a
            `.pt` produced by ``KIVI/scripts/derive_nvfp4_global_scales.py``
            with keys ``gs_K_raw`` / ``gs_V_raw`` (for nvfp4) and
            ``gs_K_smooth`` / ``gs_V_smooth`` (for smkv_nvfp4), each of
            shape ``(num_layers,)`` in FP32.
        dtype: dtype the calib scales are cast to before being held on CPU.
            "bfloat16" or "float16". (NVFP4 global scales stay FP32 per spec.)
    """

    method: str = "bf16"
    group_size: int = 128
    bits: int = 4
    calib_path: str | None = None
    global_scales_path: str | None = None
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        valid = {
            "bf16", "fp16", "fp8", "pertoken",
            "smoothkv", "smoothkv_fused",
            "nvfp4", "smkv_nvfp4",
        }
        if self.method not in valid:
            raise ValueError(
                f"Unknown method {self.method!r}; expected one of {sorted(valid)}"
            )
        if self.method in ("smoothkv", "smoothkv_fused", "smkv_nvfp4") \
                and not self.calib_path:
            raise ValueError(f"method={self.method!r} requires calib_path")
        if self.method in ("nvfp4", "smkv_nvfp4") and not self.global_scales_path:
            raise ValueError(
                f"method={self.method!r} requires global_scales_path "
                f"(produced by KIVI/scripts/derive_nvfp4_global_scales.py)"
            )

    def is_active(self) -> bool:
        """Returns True if this config requires per-step quantization work."""
        return self.method not in ("bf16", "fp16")

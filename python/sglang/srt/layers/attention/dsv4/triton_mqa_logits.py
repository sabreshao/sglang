# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""64 KiB-LDS Triton FP8 paged MQA-logits kernel for gfx942 (MI300X/MI325X).

Vendored from vLLM PR #42893 ("[ROCm][DSv4] Functional fixes for DeepSeek V4
on MI300X (gfx942)", merged as vllm-project/vllm#45681), file
``vllm/v1/attention/ops/triton_fp8_mqa_logits.py``. That module vendors
AITER's ``fp8_mqa_logits`` Triton kernel together with the launch-time
tile-size selection from ROCm/aiter#3257.

Why this kernel (and not the AITER paged kernel used by
``_aiter_fp8_paged_mqa_logits``): on gfx942 the default
``(BLOCK_KV=128, num_stages=2)`` tile for the DSv4 indexer shape
``(NUM_HEADS=64, HEAD_SIZE=128)`` double-buffers a 32 KiB KV tile + 32 KiB
fp32 scores accumulator + 8 KiB Q tile, which Triton routes into LDS and
requests ~96 KiB. The MI300X CU only has 64 KiB of LDS, so that kernel
JIT-aborts with ``OutOfResources: shared memory``. Selecting
``(BLOCK_KV=64, num_stages=1)`` needs only ~33 KiB and clears the 64 KiB
budget with margin.

Two pieces live here:

* ``_fp8_mqa_logits_kernel`` / ``fp8_mqa_logits_gfx942`` -- the ragged
  (non-paged) kernel exactly as vendored from PR #42893, kept verbatim for
  provenance and for a potential prefill indexer fallback.
* ``_fp8_paged_mqa_logits_kernel`` / ``triton_fp8_paged_mqa_logits`` -- a
  paged decode adaptation that reuses the *same* per-tile math (fp8 dot ->
  kv-scale -> ReLU -> per-head weight -> head-sum) and the *same*
  ``(BLOCK_KV=64, num_stages=1)`` 64 KiB-LDS launch config, but addresses a
  per-batch gathered KV buffer and writes row-local logits so its Python
  entry point matches ``fp8_paged_mqa_logits_torch``'s call signature.

CUDA-graph / HIP-graph capture: the wrapper performs its paged gather with
plain device-side tensor ops (no ``.item()``/``.cpu()``/host allocation),
allocates the output on-device, and launches a single Triton kernel. Unlike
the TileLang/TVM path (which failed capture with InjectSoftwarePipeline
"canceled"), Triton emits a static kernel with no host tensors at call time,
so it is capture-safe. FP8 casts use ``torch.float8_e4m3fnuz`` on ROCm
(``torch.version.hip is not None``) to match the fnuz cache layout and avoid
the e4m3fn-vs-fnuz mismatch seen in the TileLang attempt.
"""

from typing import Any

import torch

import triton
import triton.language as tl

# ROCm (HIP) uses the fnuz FP8 encoding; CUDA uses fn. The DSv4 indexer KV
# cache is stored in the platform-native fp8, so reinterpret casts must use
# the matching dtype or the bytes decode incorrectly (the e4m3fn-vs-fnuz
# mismatch that broke the TileLang attempt / T36).
FP8_DTYPE = (
    torch.float8_e4m3fnuz if torch.version.hip is not None else torch.float8_e4m3fn
)

# gfx942 (MI300X) has 64 KiB of LDS per CU. We accept the default
# (BLOCK_KV=128, num_stages=2) tile only when *both* of these hold:
#
# 1. Occupancy gate. With waves_per_eu=2 and num_warps=4 we target two
#    workgroups co-resident on a CU -> per-WG LDS budget = 32 KiB. Triton
#    keeps Q in registers (loop-invariant) and the fp32 scores accumulator
#    in VGPRs (heavy VALU), so only the double-buffered KV tile is
#    expected to live in LDS. A 0.9 safety factor leaves headroom for any
#    LDS overhead the compiler may add.
#
# 2. Hardware ceiling. Defensive upper bound that also counts Q and
#    scores against the 64 KiB CU limit, in case a Triton version (older
#    or future) decides to spill them to LDS. False positives here only
#    shrink the tile; false negatives are JIT-aborts, so we lean
#    conservative.
_GFX942_CU_LDS_BYTES = 64 * 1024
_GFX942_PER_WG_LDS_BUDGET_BYTES = _GFX942_CU_LDS_BYTES * 9 // 20  # ~28.8 KiB


def _gfx942_default_tile_fits_lds(num_heads: int, head_size: int) -> bool:
    """Return True iff (BLOCK_KV=128, num_stages=2) fits in MI300X LDS."""
    BLOCK_KV = 128
    NUM_STAGES = 2
    kv_bytes = head_size * BLOCK_KV * NUM_STAGES
    scores_bytes = num_heads * BLOCK_KV * 4
    q_bytes = num_heads * head_size
    fits_occupancy = kv_bytes < _GFX942_PER_WG_LDS_BUDGET_BYTES
    fits_hardware = q_bytes + kv_bytes + scores_bytes <= _GFX942_CU_LDS_BYTES
    return fits_occupancy and fits_hardware


def _select_block_kv(num_heads: int, head_size: int) -> tuple[int, int]:
    """Pick (BLOCK_KV, num_stages) for gfx942's 64 KiB LDS budget."""
    if _gfx942_default_tile_fits_lds(num_heads, head_size):
        return 128, 2
    # DSv4 sparse indexer (NUM_HEADS=64, HEAD_SIZE=128) lands here:
    # default tile spills past gfx942's 64 KiB LDS budget. (64, 1)
    # needs ~33 KiB and clears the per-WG budget with margin.
    return 64, 1


# ---------------------------------------------------------------------------
# Ragged (non-paged) kernel -- vendored verbatim from vLLM PR #42893.
# The @triton.jit body is byte-for-byte equivalent to AITER's
# aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits.
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales[None, :]
        # ReLU
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_gfx942(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on MI300X (gfx942) using the vendored kernel.

    Drop-in replacement for ``aiter.ops.triton.attention.fp8_mqa_logits.
    fp8_mqa_logits`` on MI300X. Selects ``(BLOCK_KV, num_stages)`` based on
    whether the default tile fits within the 64 KiB LDS budget of a gfx942
    CU (see module docstring).
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    kv_scales_1d = kv_scales.reshape(-1)

    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    block_kv, num_stages = _select_block_kv(num_heads, head_size)

    # heuristic for MFMA instruction shape, identical to AITER's choice
    matrix_instr_nonkdim = 32
    if seq_len <= 1024:
        matrix_instr_nonkdim = 16

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=k_fp8,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits


# ---------------------------------------------------------------------------
# Paged decode adaptation.
# Same per-tile math and same (BLOCK_KV=64, num_stages=1) 64 KiB-LDS launch
# config as the vendored ragged kernel, but each program handles one batch
# row: it reads that row's gathered KV (batch-strided) and writes row-local
# logits [batch, max_seq_len]. This lets the Python entry point match
# ``fp8_paged_mqa_logits_torch``'s call signature exactly.
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr,  # fp8 [batch, H, D]
    KV_ptr,  # fp8 [batch, padded_seq, D]
    kv_scales_ptr,  # fp32 [batch, padded_seq]
    weights_ptr,  # [batch, H]
    seq_lens_ptr,  # int32 [batch]
    logits_ptr,  # fp32 [batch, max_seq_len]
    max_seq_len,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_b: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_b: tl.int64,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_ks_b: tl.int64,
    stride_ks_s: tl.constexpr,
    stride_w_b: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_b: tl.int64,
    stride_logits_k: tl.int64,
    BLOCK_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    tl.assume(batch_id >= 0)
    tl.assume(stride_q_b > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_b > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_b > 0)
    tl.assume(stride_w_h > 0)

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # Q[H, D] and per-head weights for this batch row (loop-invariant).
    q_ptrs = (
        Q_ptr + batch_id * stride_q_b + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + batch_id * stride_w_b + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Local KV window for this row: [0, end_ind).
    end_ind = tl.load(seq_lens_ptr + batch_id)
    end_ind = tl.maximum(end_ind, 0)
    end_ind = tl.minimum(end_ind, max_seq_len)
    unmasked_end = end_ind // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV)
    kv_base = KV_ptr + batch_id * stride_kv_b
    kv_ptrs = kv_base + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    kv_scales_ptrs = kv_scales_ptr + batch_id * stride_ks_b + kv_col_offsets * stride_ks_s
    logits_ptrs = (
        logits_ptr + batch_id * stride_logits_b + kv_col_offsets * stride_logits_k
    )

    for _ in tl.range(0, unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV * stride_ks_s
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked tail tile
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    scores = scores * kv_scales[None, :]
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)
    tl.store(logits_ptrs, scores, mask=kv_col_mask)


def triton_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """64 KiB-LDS Triton FP8 paged MQA-logits, drop-in for the torch path.

    Signature and return value match
    ``fp8_paged_mqa_logits_torch``: returns a ``[batch, max_seq_len]`` fp32
    tensor whose row ``b`` holds the (ReLU'd, weighted, head-summed,
    kv-scaled) logits for KV positions ``[0, seq_lens[b])`` and ``0.0``
    elsewhere.

    The paged KV gather is done with device-side tensor ops only (no host
    sync / allocation), so the call is HIP/CUDA-graph capture-safe.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "DSV4 indexer head_dim is 128"
    assert block_size == 64, "DSV4 indexer cache block_size is 64"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size

    device = q_fp8.device
    SCALE_OFFSET = block_size * head_dim
    total_dim = block_size * (head_dim + 4)

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_pages = min(max_pages, page_table.shape[1])
    padded_seq = max_pages * block_size

    # Paged gather (device-side, capture-safe). Mirrors fp8_paged_mqa_logits_torch.
    kvcache_flat = kvcache_fp8.view(-1, total_dim)
    page_ids = page_table[:, :max_pages].clamp(min=0)
    kvcache_gathered = kvcache_flat[page_ids]  # [batch, max_pages, total_dim]

    kv_values = (
        kvcache_gathered[..., :SCALE_OFFSET]
        .contiguous()
        .view(dtype=FP8_DTYPE)
        .reshape(batch_size, padded_seq, head_dim)
    )
    kv_scales = (
        kvcache_gathered[..., SCALE_OFFSET:]
        .contiguous()
        .view(dtype=torch.float32)
        .reshape(batch_size, padded_seq)
    )

    q = q_fp8[:, 0].contiguous()  # [batch, num_heads, head_dim], fp8
    weights = weight.contiguous().to(torch.float32)
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()

    logits = torch.zeros((batch_size, max_seq_len), dtype=torch.float32, device=device)

    block_kv, num_stages = _select_block_kv(num_heads, head_dim)

    matrix_instr_nonkdim = 32
    if max_seq_len <= 1024:
        matrix_instr_nonkdim = 16

    stride_q_b, stride_q_h, stride_q_d = q.stride()
    stride_kv_b, stride_kv_s, stride_kv_d = kv_values.stride()
    stride_ks_b, stride_ks_s = kv_scales.stride()
    stride_w_b, stride_w_h = weights.stride()
    stride_logits_b, stride_logits_k = logits.stride()

    _fp8_paged_mqa_logits_kernel[(batch_size,)](
        Q_ptr=q,
        KV_ptr=kv_values,
        kv_scales_ptr=kv_scales,
        weights_ptr=weights,
        seq_lens_ptr=seq_lens_i32,
        logits_ptr=logits,
        max_seq_len=max_seq_len,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_dim,
        stride_q_b=stride_q_b,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_b=stride_kv_b,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_ks_b=stride_ks_b,
        stride_ks_s=stride_ks_s,
        stride_w_b=stride_w_b,
        stride_w_h=stride_w_h,
        stride_logits_b=stride_logits_b,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits

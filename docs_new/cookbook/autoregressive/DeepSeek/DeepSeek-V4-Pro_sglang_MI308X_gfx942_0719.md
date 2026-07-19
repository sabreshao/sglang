# DeepSeek-V4-Pro (DSV4) on 8× MI308X — sglang

*sglang bringup report · AMD Instinct MI308X (gfx942)*

First passing DeepSeek-V4-Pro sglang result on gfx942: MXFP4→int4 MoE conversion + a hybrid-SWA chunked-prefill fix, with EAGLE speculative decode, CUDA graph, and fp8 KV cache all enabled.

**Badges:** GATE PASS · gsm8k 0.97 ≥ 0.92 | sglang 0.5.15 | tp8 · EAGLE + CUDA graph + fp8 KV + int4 MoE

## Executive summary

**DeepSeek-V4-Pro** (`DeepseekV4ForCausalLM`) was brought up on
 **8× AMD Instinct MI308X (gfx942)** under **sglang**. The acceptance gate is
 **gsm8k 8-shot exact_match ≥ 0.92** (temperature 0), and the run
 **PASSES at 0.97** (num_concurrent=64, the official `acc.sh` level),
 corroborated at 0.96 (concurrency 8).

The hard constraint — **EAGLE + CUDA graph + fp8 KV cache all ON** — is met, and the
 MoE runs on a numerically-exact **int4 (a16wi4)** path. Two changes made this work on gfx942:
 (a) the model's MXFP4 (E2M1 + E8M0 microscale) routed experts are converted to
 **int4 + bf16 groupwise scale** so the FlyDSL `int4_bf16` kernel runs
 instead of the broken/absent gfx942 fp4 MoE GEMM paths (isolated `logits_diff ~3e-5`);
 and (b) a one-line config fix, `--chunked-prefill-size 1024`, resolves a
 hybrid-SWA prefill-admission wedge so concurrent serving completes cleanly with no hang.

> **Result:** gsm8k 8-shot exact_match **0.97** @ num_concurrent=64 (temp=0), gate 0.92 →
 **PASS** (margin +0.05). Client throughput 188 tok/s at 12k in / 2k out. 0 failed requests.

## Results

- **gsm8k exact_match:** 0.97 @ c64
- **Total throughput:** 188 tok/s
- **Mean TTFT:** 12.9 s
- **Mean TPOT:** 30.75 ms

### Accuracy — gsm8k 8-shot, temperature 0

| Harness / concurrency | exact_match | Gate 0.92 | Notes |
| --- | --- | --- | --- |
| **lm_eval, num_concurrent=64** (official acc.sh level) | **0.97** | PASS | headline; flexible-extract AND strict-match, ±0.0171; ~7 min @ ~99-100% GPU, no hang |
| lm_eval, num_concurrent=8 | 0.96 | PASS | ±0.0197 |

All at limit=100, max_gen_toks=2048 (lm_eval). Correctness cross-check: an isolated MoE probe measured the
 MXFP4→int4 a16wi4 path at `logits_diff ~3e-5` vs a torch reference, versus ~0.98
 (garbage) for the gfx942 CK-Tile fp4 A16W4 kernel — the accuracy comes from a numerically correct MoE
 path, not luck. Failed requests: 0.

### Client benchmark — sglang.bench_serving (client_fit.sh)

| Metric | Value | Notes |
| --- | --- | --- |
| Configuration | random-input 12000, random-output 2000, num-prompts 4, max-concurrency 1 | largest size that fits the KV pool (12k+2k = 14k < 16896) |
| Successful requests | 4 / 4 | 0 failed |
| Benchmark duration | 297.44 s | — |
| **Total token throughput** | **188.28 tok/s** | — |
| Input token throughput | 161.38 tok/s | — |
| Output token throughput | 26.90 tok/s | peak 36.00 |
| **Mean TTFT** | **12879 ms** | 12k-token prefill, chunked 12×1024; median 12945 ms |
| **Mean TPOT** | **30.75 ms** | median 31.12 ms; mean ITL 30.75 ms |
| Mean E2E latency | 74354 ms/req | — |
| EAGLE accept length | 2.69 | — |

> **Note:** the example client's `--random-input 40000` does **not**
 fit the current KV pool (max_total_num_tokens = 16896): a request with
 input+output+page ≥ 16896 is rejected at admission. The 12k/2k benchmark is the largest safe size.

## Scripts

Full, verbatim contents of the server, accuracy, and client scripts.

### server.sh

The final working server config. tp8, --attention-backend dsv4, EAGLE + CUDA graph + fp8 KV + int4 MoE all on, --chunked-prefill-size 1024 (the hang fix), --mem-fraction-static 0.92, per-rank numactl binding.

```bash
# DeepSeek-V4-Pro (MXFP4->int4 MoE) server on 8xMI308X (gfx942)
# EAGLE + cuda graph + fp8 KV cache all ON. int4 MoE conversion ON.
#
# STABILITY FIX (T26): the concurrent/long-context decode HANG (0% GPU, scheduler stalls
# after ~2 prefill batches) is caused by SGLANG_OPT_USE_MULTI_STREAM_OVERLAP (defaults ON
# in this sglang 0.5.15 build; known issue #25662 for DSV4). The proven-stable reference
# container 20260611_ttest_ds sets it to 0. Ported that single guard here.
#
# NOTE: the reference also set SGLANG_OPT_USE_COMPRESSOR_V2=false, but on THIS newer sglang
# the V1 compressor path is BROKEN (fp32 dst vs bf16 src dtype crash in
# deepseek_v4_compress_state.__setitem__ during cuda graph capture). compressor_v2=true is
# the only working path here, so we keep it at its default (ON). Likewise the other
# reference "=false" guards are left at their working defaults to avoid the newer sglang's
# less-maintained fallback paths.

# --- NUMA balancing OFF (must be before any GPU work) ---
# NUMA auto-balancing migrates CPU-adjacent pages used by HIP graph capture, corrupting
# captured addresses on replay -> garbled tokens. Reference root-caused this 2026-06-17.
echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true

# --- THE hang fix: disable multi-stream overlap (DSV4 issue #25662) ---
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # defrag int4-load temporaries (benign)

export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
# gfx942: enable aiter fused_clamp_act_mul for shared-experts MLP (else falls through to a
# CUDA-only silu_and_mul_clamp JIT kernel that needs cuda_fp8.h -> not found on ROCm). T2/T3.
export SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL=true
# gfx942: MoE runner uses explicit clamp_ instead of the CUDA-only fused swiglu-clamp kernel.
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_OPT_FP8_WO_A_GEMM=false

# --- gfx942 RMSNorm+FP8-group-quant fusion ---
export SGLANG_GFX942_FUSE_RMSNORM_QUANT=0

# --- gfx942 MXFP4 MoE -> int4 (a16wi4 FlyDSL) conversion (proven correct, logits_diff 3e-5) ---
export SGLANG_GFX942_MXFP4_INT4=true
export AITER_GFX942_MXFP4_A4=0
export AITER_KSPLIT=2

MODEL=/mnt/md0/models/DeepSeek-V4-Pro/

# --- NUMA per-rank binding (TPOT recovery). numactl installed. GPU0-3->NUMA0, GPU4-7->NUMA1 ---
sglang serve \
 --model-path ${MODEL} \
 --trust-remote-code \
 --tp 8 \
 --disable-radix-cache \
 --disable-overlap-schedule \
 --attention-backend dsv4 \
 --max-running-request 32 \
 --page-size 128 \
 --chunked-prefill-size 1024 \
 --mem-fraction-static 0.92 \
 --port 8000 \
 --disable-shared-experts-fusion \
 --tool-call-parser deepseekv4 \
 --reasoning-parser deepseek-v4 \
 --cuda-graph-max-bs-decode 64 \
 --numa-node 0 0 0 0 1 1 1 1 \
 --speculative-attention-mode decode \
 --speculative-algorithm EAGLE \
 --speculative-num-steps 2 \
 --speculative-eagle-topk 1 \
 --speculative-num-draft-tokens 3
```

### acc.sh

The official lm_eval gsm8k gate: gsm8k 8-shot, temperature=0, num_concurrent=64, limit=100, max_gen_toks=2048. This is the level the headline 0.97 (and the 0.96 at concurrency 8) was measured at.

```bash
lm_eval --model local-completions --model_args model=/mnt/md0/models/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=64,tokenized_requests=False,tokenizer_backend=none,max_retries=3,max_gen_toks=2048,timeout=7200,trust_remote_code=True,eos_string="Question:" --tasks gsm8k --num_fewshot 8 --limit 100 --gen_kwargs temperature=0
```

### client_fit.sh

The KV-fitting client benchmark that was actually run (12k in / 2k out, sglang.bench_serving), sized to the DSV4 int4 KV pool.

```bash
#!/usr/bin/env bash
# Client throughput benchmark (sglang.bench_serving), sized to the DSV4 int4 KV pool.
# The requested random-input 40000 does NOT fit: full-attention KV pool = max_total_num_tokens
# = 16896 tokens, and a request is rejected at admission when input+output+page >= 16896.
# So we use the largest input that safely fits: 12000 in + 2000 out = 14000 < 16896.
IN=${1:-12000}
OUT=${2:-2000}
NP=${3:-4}
MC=${4:-1}
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://localhost:8000 \
  --model /mnt/md0/models/DeepSeek-V4-Pro \
  --tokenizer /mnt/md0/models/DeepSeek-V4-Pro \
  --dataset-name random \
  --random-input ${IN} \
  --random-output ${OUT} \
  --random-range-ratio 1 \
  --num-prompts ${NP} \
  --max-concurrency ${MC} \
  --warmup-requests 1 \
  --extra-request-body '{"temperature": 0}'
```

## Key engineering fixes

### a. MXFP4 MoE → int4 conversion — the core correctness fix

DSV4's routed experts ship as **MXFP4** (E2M1 mantissa + E8M0 microscale). On gfx942 every fp4
 MoE route is broken or absent: the aiter fp4x2 activation-quant kernels are unimplemented
 (quant_kernels.cu:944/1912 "not support output type: fp4x2"), the afp8_wfp4 FlyDSL
 kernels are gfx950-only (LLVM "expand operand" codegen error), and the CK-Tile fp4 A16W4 instances are
 compiled out (#ifndef __gfx942__). Force-enabling the CK-Tile fp4 kernel produced
 *garbage* output — the exclusion was correctness-motivated, not just a build guard (isolated
 probe: logits_diff ~0.98).

The fix (in sglang `fp8.py`): convert MXFP4 → **int4 [-7,7] + bf16
 groupwise scale** (mxfp4_to_f32 × e8m0_to_f32 → per_1x32_i4_quant,
 chunked over experts to bound the f32 temporary), then run the FlyDSL **a16wi4**
 (int4_bf16) path, which is well-supported on gfx942. Isolated probe:
 logits_diff ~3e-5 (numerically exact). The converted weights are viewed as
 i4x2 (not re-viewed as fp4x2) so dispatch keeps the int4 path.

### b. SwiGLU-clamp / cuda_fp8.h routing fix

The shared-experts MLP fell into a CUDA-only silu_and_mul_clamp JIT kernel that
 #includes cuda_fp8.h — not present on ROCm, so CUDA
 graph capture failed to build. Root cause: the example set
 `USE_FUSED_CLAMP_ACT_MUL=false`, which on HIP *disabled* the aiter fused path
 and fell through to the CUDA-only kernel (backwards for gfx942). Fix:
 `SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL=true` (aiter HIP fused clamp) +
 `SGLANG_OPT_SWIGLU_CLAMP_FUSION=false` (MoE runner uses an explicit
 clamp_ instead of the CUDA-only fused kernel).

### c. The concurrency hang — a hybrid-SWA prefill-admission wedge (NOT a scheduler bug)

An earlier diagnosis blamed a "sglang 0.5.15 TP-scheduler desync" that was "not fixable via env". **That
 was wrong** and is corrected here. Fresh py-spy of all 8 TP schedulers + live isolation proved the
 real cause is a **hybrid-SWA prefill-admission wedge**: the int4 conversion leaves only ~13.6 GB/GPU
 free, so the KV pool is tiny (full=16896, swa=1536). The PrefillAdder computes
 swa_needed = max(min(extend_input_len, rem_chunk_tokens), sliding_window) + page_size
 and returns NO_TOKEN when swa_needed ≥ rem_swa_tokens. With
 --chunked-prefill-size 2048 > 1536, any prompt over ~1280 tokens can never be
 admitted → _get_new_batch_prefill_raw returns None every loop → all 8 ranks
 symmetric idle-spin at 0% GPU (a livelock, not a gloo/broadcast deadlock).

**Fix (config only):** `--chunked-prefill-size 1024` (must be ≤ SWA
 pool 1536 − page_size). Every prefill chunk now fits the SWA pool, so arbitrarily long prompts prefill
 chunk-by-chunk up to the full 16896 pool. No scheduler source patch, no sglang downgrade. Proof: the original
 4×1723-tok hang trigger completes in 56.6 s @ ~100% GPU; lm_eval gsm8k at concurrency 8/64 completes
 cleanly (0.96 / 0.97).

### Supporting stability fixes

- **numactl install + per-rank binding** (`--numa-node 0 0 0 0 1 1 1 1`): the flag wraps each TP rank in `numactl`; without it rank 0 died at startup. Kept as a TPOT optimization.
- **`--mem-fraction-static 0.92`**: weights are ~158 GB/GPU; 0.86 left no room for the KV cache. 0.92 is the working point.
- **`numa_balancing = 0`**: NUMA auto-balancing migrates CPU-adjacent pages used by HIP graph capture, corrupting captured addresses on replay → garbled tokens.
- **`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`**: ported from the proven-stable reference container (DSV4 issue #25662).

---
DeepSeek-V4-Pro sglang bringup · MI308X (gfx942) · container 20260707_ttest_glm_ds ·
 gsm8k 8-shot exact_match 0.97 ≥ 0.92 (PASS). Self-contained report; no external dependencies.

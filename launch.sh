#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export HF_HOME="${HF_HOME:-/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/models}"
PORT="${PORT:-8012}"

# DSV4 prefill graph is incompatible with LogitsProcessorOutput here.
exec env SGLANG_DSV4_FP4_DEQUANT=1 sglang serve \
  --trust-remote-code \
  --model-path "${MODEL_NAME}" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --moe-runner-backend auto \
  --fp8-gemm-backend aiter \
  --attention-backend dsv4 \
  --page-size 256 \
  --mem-fraction-static 0.90 \
  --context-length 4096 \
  --swa-full-tokens-ratio 0.1 \
  --disable-shared-experts-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --host 0.0.0.0 \
  --port "${PORT}"

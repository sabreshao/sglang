import json
import os
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = sequence[-(ngram_size - 1) :]
            else:
                current_prefix = array("q")

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits


def _spec_ntok(logits, n):
    try:
        nt = logits.shape[0] // n
        return nt if nt >= 1 else 1
    except Exception:
        return 1


class NoRepeatNGramFixed(CustomLogitProcessor):
    """Sliding-window n-gram ban, correct across all spec rows. Opt-in via ngram_size."""

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits
        from array import array as _arr

        nreq = len(custom_param_list)
        ntok = _spec_ntok(logits, nreq)
        for j, params in enumerate(custom_param_list):
            if not params:
                continue
            req = params.get("__req__")
            if req is None:
                continue
            ng = int(params.get("ngram_size") or 0)
            win = int(params.get("window_size") or 0)
            if ng <= 0 or win <= 0:
                continue
            seq = req.origin_input_ids + req.output_ids
            if len(seq) < ng:
                continue
            start = max(0, len(seq) - win)
            end = len(seq) - ng + 1
            if end <= start:
                continue
            prefix = seq[-(ng - 1):] if ng > 1 else _arr("q")
            banned = set()
            for idx in range(start, end):
                gram = seq[idx: idx + ng]
                if ng == 1 or gram[:-1] == prefix:
                    banned.add(gram[-1])
            wl = params.get("whitelist_token_ids") or []
            banned.difference_update(int(t) for t in wl)
            if not banned:
                continue
            logits[j * ntok:(j + 1) * ntok, list(banned)] = -float("inf")
        return logits


class DRYLogitProcessor(CustomLogitProcessor):
    """DRY variable-length repetition penalty (long-period loops). Opt-in via dry_multiplier."""

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits
        nreq = len(custom_param_list)
        ntok = _spec_ntok(logits, nreq)
        for j, params in enumerate(custom_param_list):
            if not params or "dry_multiplier" not in params:
                continue
            req = params.get("__req__")
            if req is None:
                continue
            mult = float(params.get("dry_multiplier") or 0.0)
            if mult <= 0.0:
                continue
            base = float(params.get("dry_base", 1.75) or 1.75)
            allowed = int(params.get("dry_allowed_length", 2) or 2)
            rng = int(params.get("dry_range", 2048) or 2048)
            breakers = set(params.get("dry_sequence_breaker_ids") or [])
            seq = list(req.origin_input_ids) + list(req.output_ids)
            n = len(seq)
            if n < 2:
                continue
            m = min(rng, n)
            w = seq[n - m:]
            last = w[-1]
            if last in breakers:
                continue
            L = len(w)
            best = {}
            for i in range(L - 1):
                if w[i] != last:
                    continue
                match = 1
                while (match <= i and match <= L - 1
                       and w[i - match] == w[L - 1 - match]
                       and w[i - match] not in breakers):
                    match += 1
                nxt = w[i + 1]
                if match > best.get(nxt, 0):
                    best[nxt] = match
            r0, r1 = j * ntok, (j + 1) * ntok
            for tok, ml in best.items():
                if ml >= allowed:
                    expo = min(ml - allowed, 30)  # cap: base**expo can overflow float
                    logits[r0:r1, tok] -= mult * (base ** expo)
        return logits


class EntropyCollapseLogitProcessor(CustomLogitProcessor):
    """Confidence-collapse early-stop (force EOS). Opt-in via ent_enable."""

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list:
            return logits
        import torch as _torch

        nreq = len(custom_param_list)
        ntok = _spec_ntok(logits, nreq)
        for j, params in enumerate(custom_param_list):
            if not params or not params.get("ent_enable"):
                continue
            req = params.get("__req__")
            if req is None:
                continue
            window = int(params.get("ent_window", 128) or 128)
            thr = float(params.get("ent_threshold", 0.35) or 0.35)
            min_tokens = int(params.get("ent_min_tokens", 1200) or 1200)
            eos_ids = params.get("eos_token_ids") or [1]
            r0 = j * ntok
            with _torch.no_grad():
                logp = _torch.log_softmax(logits[r0].float(), dim=-1)
                ent = float(-(logp.exp() * logp).sum().item())
            hist = getattr(req, "_ent_hist", None)
            if hist is None:
                hist = []
                try:
                    req._ent_hist = hist
                except Exception:
                    pass
            hist.append(ent)
            if len(hist) > window:
                del hist[: len(hist) - window]
            if len(req.output_ids) >= min_tokens and len(hist) >= window and max(hist) < thr:
                logits[r0:r0 + ntok, :] = -float("inf")
                for e in eos_ids:
                    logits[r0:r0 + ntok, int(e)] = 0.0
        return logits


class LoopGuardLogitProcessor(CustomLogitProcessor):
    """Composition: (fixed) no-repeat-ngram + DRY + entropy-collapse; each opt-in."""

    _ngram = NoRepeatNGramFixed()
    _dry = DRYLogitProcessor()
    _ent = EntropyCollapseLogitProcessor()

    def __call__(self, logits, custom_param_list=None):
        logits = self._ngram(logits, custom_param_list)
        logits = self._dry(logits, custom_param_list)
        logits = self._ent(logits, custom_param_list)
        return logits


# === ng20 baked-in server-side default ===
# Bake the FIXED no-repeat n-gram block (NoRepeatNGramFixed, correct across all
# DSpark draft-token rows) into every request as a server-side default, so the
# client no longer has to attach a per-request custom_logit_processor payload.
# Defaults: ngram_size=20, window_size=90. Fully overridable per request (see
# maybe_inject_default_ngram below). Disable build-wide with env
# SGLANG_DEFAULT_NO_REPEAT_NGRAM=0.
DEFAULT_NO_REPEAT_NGRAM_SIZE = 20
DEFAULT_NO_REPEAT_WINDOW_SIZE = 90


def _default_ngram_enabled() -> bool:
    return os.environ.get("SGLANG_DEFAULT_NO_REPEAT_NGRAM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


@lru_cache(maxsize=1)
def _default_ngram_clp_str() -> str:
    """Serialized NoRepeatNGramFixed processor (cached)."""
    return NoRepeatNGramFixed.to_str()


def maybe_inject_default_ngram(custom_logit_processor, sampling_params):
    """Compute the ng20 server-side default for a request.

    Returns ``(clp_str, custom_params_dict)`` when the default no-repeat n-gram
    block should be applied, or ``(None, None)`` when it should NOT (i.e. the
    request already handles it or opted out). The caller is responsible for
    setting these onto the Req / SamplingParams.

    Override / opt-out rules (all preserved):
      * If the request already supplies its own ``custom_logit_processor`` ->
        no-op (full override, the request's processor wins).
      * If disabled build-wide via ``SGLANG_DEFAULT_NO_REPEAT_NGRAM=0`` -> no-op.
      * If the request passes ``custom_params={"no_repeat_ngram": False}`` -> no-op.
      * If the request passes ``custom_params={"ngram_size": <=0}`` -> no-op.
      * If the request passes its own ``ngram_size`` / ``window_size`` in
        ``custom_params``, those values are respected (defaults only fill gaps).
    """
    if custom_logit_processor is not None:
        return None, None
    if not _default_ngram_enabled():
        return None, None
    cp = sampling_params.custom_params
    if cp is None:
        cp = {}
    elif not isinstance(cp, dict):
        return None, None
    if cp.get("no_repeat_ngram") is False:
        return None, None
    ng = cp.get("ngram_size")
    if ng is not None:
        try:
            if int(ng) <= 0:
                return None, None
        except (TypeError, ValueError):
            return None, None
    merged = dict(cp)
    merged.setdefault("ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)
    merged.setdefault("window_size", DEFAULT_NO_REPEAT_WINDOW_SIZE)
    return _default_ngram_clp_str(), merged

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def _extract_prob(text: str) -> Optional[float]:
    """
    Best-effort probability parsing.

    Strategy:
    - Prefer JSON-like: {"p_yes": 0.123}
    - Else, take the last numeric token and clamp to [0,1] if plausible.
    """
    # JSON-ish / key-value (allow quotes or not)
    m = re.search(r'["\']?p_yes["\']?\s*:\s*([0-9]*\.?[0-9]+)\s*%?', text, flags=re.I)
    if m:
        try:
            p = float(m.group(1))
            if 0.0 <= p <= 1.0:
                return p
        except ValueError:
            pass

    # Try common phrasing near the end (avoid picking up numbers in the prompt context).
    tail = text[-600:]
    m = re.search(
        r"(?:p\s*\(?\s*yes\s*\)?|prob(?:ability)?(?:\s*of)?\s*yes)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%)?",
        tail,
        flags=re.I,
    )
    if m:
        try:
            p = float(m.group(1))
            if m.group(2):  # percent
                p = p / 100.0
            if 0.0 <= p <= 1.0:
                return p
        except ValueError:
            pass

    nums = _FLOAT_RE.findall(tail)
    if not nums:
        return None
    try:
        p = float(nums[-1])
    except ValueError:
        return None
    if 0.0 <= p <= 1.0:
        return p
    # Sometimes models output percent.
    if 0.0 <= p <= 100.0:
        return p / 100.0
    return None


@dataclass(frozen=True)
class ARCoTSpec:
    model_name_or_path: str
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    device_map: Optional[str] = None  # e.g. "auto" for multi-GPU sharding (requires accelerate)
    trust_remote_code: bool = True

    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256

    # If True, ask for brief reasoning; we still parse the final probability.
    include_cot: bool = True
    aggregate: str = "mean"  # "mean" or "median"


class ARCoTPredictor:
    """
    Minimal AR+CoT wrapper around HuggingFace Transformers.

    This is designed for evaluation-only use (no finetuning).
    """

    def __init__(self, spec: ARCoTSpec):
        self.spec = spec
        self._model = None
        self._tok = None

    def _lazy_load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.spec.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = None
        if self.spec.dtype == "float16":
            dtype = torch.float16
        elif self.spec.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.spec.dtype == "float32":
            dtype = torch.float32
        elif self.spec.dtype == "auto":
            # Conservative defaults: bf16 on CUDA (H100-friendly), else fp32.
            if device == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

        tok = AutoTokenizer.from_pretrained(
            self.spec.model_name_or_path, trust_remote_code=self.spec.trust_remote_code
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.spec.device_map is not None:
            # Shard across multiple GPUs if available (common for 14B+).
            load_kwargs["device_map"] = self.spec.device_map
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(self.spec.model_name_or_path, **load_kwargs)
        model.eval()
        if self.spec.device_map is None:
            model.to(device)

        self._tok = tok
        self._model = model
        self._device = str(model.device) if self.spec.device_map is not None else device

    def _prompt(self, info: str, L: int) -> str:
        if self.spec.include_cot:
            return (
                "You are a probabilistic forecaster.\n"
                "Given the information below, estimate P(YES) as a number in [0,1].\n"
                f"Think step-by-step in at most {max(L, 1)} short steps, then output ONLY a JSON object "
                'of the form {"p_yes": <number>}.\n\n'
                f"INFORMATION:\n{info}\n\n"
                "OUTPUT:\n"
            )
        return (
            "You are a probabilistic forecaster.\n"
            "Given the information below, output ONLY a JSON object of the form "
            '{"p_yes": <number in [0,1]>}.\n\n'
            f"INFORMATION:\n{info}\n\n"
            "OUTPUT:\n"
        )

    @staticmethod
    def _depth_to_max_new_tokens(L: int, base: int = 64, per_step: int = 32) -> int:
        # A rough knob to vary “reasoning budget” without requiring any special model support.
        return int(base + max(L, 0) * per_step)

    def predict_proba(
        self,
        infos: List[str],
        *,
        L: int = 4,
        K: int = 1,
        seed: int = 0,
        max_examples: Optional[int] = None,
        aggregate: Optional[str] = None,
        batch_size: int = 8,  # NEW: batched inference for 4-8x speedup
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns:
          probs: (n,) float32 array
          meta: diagnostics (parse failures etc)
        """
        self._lazy_load()

        import torch

        rng = np.random.default_rng(seed)
        max_new = min(self.spec.max_new_tokens, self._depth_to_max_new_tokens(L))

        infos_run = infos if max_examples is None else infos[: max_examples]
        n_total = len(infos_run)

        tok = self._tok
        model = self._model
        
        # Enable left-padding for batch generation
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        all_probs: List[List[float]] = [[] for _ in range(n_total)]  # K samples per input
        n_fail = 0

        # For K > 1, we need multiple passes (can't easily batch different K samples)
        for k in range(K):
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
            
            # Build all prompts
            prompts = [self._prompt(info, L=L) for info in infos_run]
            
            # Process in batches
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                batch_prompts = prompts[batch_start:batch_end]
                
                # Tokenize with padding
                inputs = tok(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,  # Reasonable context limit
                ).to(self._device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=(K > 1),
                        temperature=float(self.spec.temperature),
                        top_p=float(self.spec.top_p),
                        max_new_tokens=int(max_new),
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                
                # Decode each output in the batch
                for i, output in enumerate(outputs):
                    text = tok.decode(output, skip_special_tokens=True)
                    tail = text.split("OUTPUT:", 1)[-1]
                    p = _extract_prob(tail)
                    if p is None:
                        n_fail += 1
                        p = 0.5
                    all_probs[batch_start + i].append(float(np.clip(p, 0.0, 1.0)))
                
                # Progress logging for long runs
                if (batch_end % 500 == 0) or (batch_end == n_total):
                    print(f"[ARCoT] Processed {batch_end}/{n_total} (K={k+1}/{K})")

        # Aggregate K samples per input
        agg = (aggregate or self.spec.aggregate).lower()
        probs: List[float] = []
        for per_sample in all_probs:
            if agg == "median":
                probs.append(float(np.median(per_sample)))
            elif agg == "mean":
                probs.append(float(np.mean(per_sample)))
            else:
                raise ValueError(f"Unknown aggregate={agg!r} (expected mean|median)")

        meta = {
            "n": len(infos_run),
            "K": int(K),
            "L": int(L),
            "max_new_tokens_used": int(max_new),
            "parse_failures": int(n_fail),
            "aggregate": agg,
            "batch_size": int(batch_size),
        }
        return np.asarray(probs, dtype=np.float32), meta



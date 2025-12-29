from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from forecastbench.models.ar_cot import _extract_prob
from forecastbench.train.rlvr import build_prompt


def _auto_device(device: str, torch) -> str:
    device = str(device)
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _torch_dtype(dtype: str, torch):
    dt = str(dtype)
    if dt == "float16":
        return torch.float16
    if dt == "bfloat16":
        return torch.bfloat16
    if dt == "float32":
        return torch.float32
    if dt == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError("dtype must be auto|float16|bfloat16|float32")


def _bnb_compute_dtype(name: str, torch):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("bnb_4bit_compute_dtype must be bfloat16|float16|float32")


@dataclass(frozen=True)
class ARLoRAPredictorSpec:
    """
    AR predictor that loads a base HF causal LM plus a PEFT LoRA adapter directory.
    """

    base_model_name_or_path: str
    adapter_path: str

    device: str = "auto"
    dtype: str = "auto"
    device_map: Optional[str] = None
    trust_remote_code: bool = True

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    include_cot: bool = True
    cot_max_steps: int = 4

    aggregate: str = "median"  # mean|median


class ARLoRAPredictor:
    def __init__(self, spec: ARLoRAPredictorSpec):
        self.spec = spec
        self._tok = None
        self._model = None
        self._input_device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = self.spec
        device = _auto_device(spec.device, torch)
        dtype = _torch_dtype(spec.dtype, torch)

        tok = AutoTokenizer.from_pretrained(spec.base_model_name_or_path, trust_remote_code=bool(spec.trust_remote_code))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        load_kwargs = dict(torch_dtype=dtype, trust_remote_code=bool(spec.trust_remote_code))
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_dt = _bnb_compute_dtype(spec.bnb_4bit_compute_dtype, torch)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_dt,
            )
            load_kwargs["device_map"] = "auto" if spec.device_map is None else str(spec.device_map)
        elif spec.device_map is not None:
            load_kwargs["device_map"] = str(spec.device_map)
            load_kwargs["low_cpu_mem_usage"] = True

        base = AutoModelForCausalLM.from_pretrained(spec.base_model_name_or_path, **load_kwargs)
        base.eval()
        if spec.device_map is None and not spec.load_in_4bit:
            base.to(device)

        try:
            from peft import PeftModel
        except Exception as e:
            raise ImportError(
                "ARLoRAPredictor requires peft to load adapters. Install extras: `pip install -e .[rlvr]`"
            ) from e

        model = PeftModel.from_pretrained(base, spec.adapter_path)
        model.eval()

        try:
            self._input_device = str(next(model.parameters()).device)
        except StopIteration:
            self._input_device = device

        self._tok = tok
        self._model = model

    def predict_proba(
        self,
        infos: List[str],
        *,
        K: int = 5,
        seed: int = 0,
        aggregate: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        self._lazy_load()
        import torch

        tok = self._tok
        model = self._model
        rng = np.random.default_rng(int(seed))

        agg = (aggregate or self.spec.aggregate).lower()
        if agg not in {"mean", "median"}:
            raise ValueError("aggregate must be mean|median")

        probs: List[float] = []
        n_fail = 0

        for info in infos:
            prompt = build_prompt(
                info, include_cot=bool(self.spec.include_cot), cot_max_steps=int(self.spec.cot_max_steps)
            )
            enc = tok(
                prompt,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self._input_device)

            ps: List[float] = []
            for _k in range(int(K)):
                torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        do_sample=(int(K) > 1),
                        temperature=float(self.spec.temperature),
                        top_p=float(self.spec.top_p),
                        max_new_tokens=int(self.spec.max_new_tokens),
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                text = tok.decode(out[0], skip_special_tokens=True)
                tail = text.split("OUTPUT:", 1)[-1]
                p = _extract_prob(tail)
                if p is None:
                    n_fail += 1
                    p = 0.5
                ps.append(float(np.clip(p, 0.0, 1.0)))

            probs.append(float(np.median(ps) if agg == "median" else np.mean(ps)))

        return np.asarray(probs, dtype=np.float32), {"n": int(len(infos)), "K": int(K), "parse_failures": int(n_fail)}



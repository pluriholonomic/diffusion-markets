from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class HFTextEmbedderSpec:
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"  # cpu|cuda|mps|auto
    dtype: str = "auto"  # float32|float16|bfloat16|auto
    # Some encoder models (and many LLM backbones) require trust_remote_code=True.
    trust_remote_code: bool = True
    # Optional sharding across devices (e.g. "auto" for big Qwen/Llama backbones).
    # If set, we will NOT call model.to(device); Accelerate will place modules.
    device_map: Optional[str] = None
    max_length: int = 256
    normalize: bool = True


class HFTextEmbedder:
    """
    Lightweight sentence embedding using HuggingFace AutoModel + mean pooling.

    This intentionally avoids an extra dependency on sentence-transformers.
    """

    def __init__(self, spec: HFTextEmbedderSpec):
        self.spec = spec
        self._tok = None
        self._model = None
        self._device = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

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
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

        tok = AutoTokenizer.from_pretrained(
            self.spec.model_name_or_path, trust_remote_code=bool(self.spec.trust_remote_code)
        )
        if tok.pad_token_id is None:
            # Common for decoder-only backbones used as embedders.
            tok.pad_token = tok.eos_token

        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=bool(self.spec.trust_remote_code),
        )
        if self.spec.device_map is not None:
            load_kwargs["device_map"] = str(self.spec.device_map)
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModel.from_pretrained(self.spec.model_name_or_path, **load_kwargs)
        model.eval()
        if self.spec.device_map is None:
            model.to(device)

        self._tok = tok
        self._model = model
        # For sharded models, put input tensors on the device of the first parameter.
        try:
            self._device = str(next(model.parameters()).device)
        except StopIteration:
            self._device = device

    def encode(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:
        self._lazy_load()
        import torch

        tok = self._tok
        model = self._model

        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=int(self.spec.max_length),
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                out = model(**enc)
            last = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1)  # (B,T,1)
            summed = torch.sum(last * mask, dim=1)
            denom = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            emb = summed / denom
            if self.spec.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)



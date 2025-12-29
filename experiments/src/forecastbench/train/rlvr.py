from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from forecastbench.models.ar_cot import _extract_prob


@dataclass(frozen=True)
class RLVRHybridRewardSpec:
    """
    Hybrid reward: proper scoring + trading PnL.

    We maximize:
      R = alpha * logscore(p, y) + beta * pnl(p, q, y)

    where:
      logscore(p, y) = y log p + (1-y) log(1-p)
      pnl(...) is a simple bounded linear trading proxy against price q.
    """

    alpha_logscore: float = 1.0
    beta_pnl: float = 0.1

    # Trading proxy params
    B: float = 1.0
    transaction_cost: float = 0.0
    trading_mode: str = "linear"  # linear|sign


@dataclass(frozen=True)
class RLVRTrainSpec:
    # Model
    model_name_or_path: str
    device: str = "auto"  # cpu|cuda|mps|auto
    dtype: str = "auto"  # auto|float16|bfloat16|float32
    device_map: Optional[str] = None  # e.g. "auto" (accelerate)
    trust_remote_code: bool = True

    # Optional quantization (QLoRA-style). Requires bitsandbytes.
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16|float16|float32

    # LoRA (requires peft). If False, we attempt to finetune full model (usually infeasible for 14B).
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Prompting / decoding
    include_cot: bool = True
    cot_max_steps: int = 4
    max_prompt_tokens: int = 512
    max_new_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.95

    # RL training
    steps: int = 200
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0

    # KL regularization to reference (reference = base model with LoRA disabled)
    kl_coef: float = 0.02

    # Variance reduction / stability
    reward_clip: float = 10.0
    baseline_ema: float = 0.95

    # Logging / checkpointing
    log_every: int = 10
    save_every: int = 50

    reward: RLVRHybridRewardSpec = RLVRHybridRewardSpec()


def _logscore(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


def _pnl_proxy(
    *,
    p: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    B: float,
    transaction_cost: float,
    mode: str,
) -> np.ndarray:
    """
    Simple bounded PnL proxy against traded price q:
      pnl = b * (y - q) - c|b|

    where b is a function of (p-q):
      - sign:  b = B * sign(p-q)
      - linear: b = clip(B*(p-q), -B, B)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gap = p - q
    if mode == "sign":
        b = float(B) * np.sign(gap)
    elif mode == "linear":
        b = np.clip(float(B) * gap, -float(B), float(B))
    else:
        raise ValueError("mode must be sign|linear")
    return b * (y - q) - float(transaction_cost) * np.abs(b)


def build_prompt(info: str, *, include_cot: bool, cot_max_steps: int) -> str:
    if include_cot:
        return (
            "You are a probabilistic forecaster.\n"
            "Given the information below, estimate P(YES) as a number in [0,1].\n"
            f"Think step-by-step in at most {max(int(cot_max_steps), 1)} short steps, then output ONLY a JSON object "
            'of the form {"p_yes": <number>}.\n\n'
            f"INFORMATION:\n{info}\n\n"
            "OUTPUT:\n"
        )
    return (
        "You are a probabilistic forecaster.\n"
        'Given the information below, output ONLY a JSON object of the form {"p_yes": <number in [0,1]>}.\n\n'
        f"INFORMATION:\n{info}\n\n"
        "OUTPUT:\n"
    )


def _auto_device(spec: RLVRTrainSpec, torch) -> str:
    device = str(spec.device)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device


def _torch_dtype(spec: RLVRTrainSpec, torch):
    dt = str(spec.dtype)
    if dt == "float16":
        return torch.float16
    if dt == "bfloat16":
        return torch.bfloat16
    if dt == "float32":
        return torch.float32
    if dt == "auto":
        # Conservative: bf16 on CUDA, else fp32.
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


def _compute_logps_for_generated(
    *,
    model,
    input_len: int,
    sequences,
    pad_token_id: int,
) -> "tuple[object, object]":
    """
    Compute per-sample logp sums over generated tokens and per-sample token counts.

    Args:
      input_len: padded prompt length (same for all items in batch)
      sequences: (B, L) token ids

    Returns:
      (logp_sum: (B,), gen_token_count: (B,))
    """
    torch = __import__("torch")
    # attention mask: treat pad tokens as masked
    attn = (sequences != int(pad_token_id)).to(torch.long)
    out = model(sequences, attention_mask=attn)
    logits = out.logits  # (B, L, V)
    logp = torch.log_softmax(logits, dim=-1)

    # token logp for positions 1..L-1 (predict token at position t using logits at t-1)
    target = sequences[:, 1:]  # (B, L-1)
    logp_next = logp[:, :-1, :]  # (B, L-1, V)
    tok_lp = torch.gather(logp_next, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # Mask to keep only generated tokens (positions >= input_len) and non-pad targets.
    # tok_lp index i corresponds to token position (i+1).
    pos = torch.arange(1, sequences.shape[1], device=sequences.device).unsqueeze(0)  # (1, L-1)
    gen_mask = (pos >= int(input_len)) & (target != int(pad_token_id))
    tok_lp = tok_lp * gen_mask.to(tok_lp.dtype)
    lp_sum = torch.sum(tok_lp, dim=1)
    gen_count = torch.sum(gen_mask.to(torch.long), dim=1)
    return lp_sum, gen_count


class RLVRTrainer:
    def __init__(self, spec: RLVRTrainSpec):
        self.spec = spec
        self._model = None
        self._tok = None
        self._device = None
        self._input_device = None
        self._opt = None
        self._baseline = 0.0
        self._step = 0
        self._init()

    def _init(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = self.spec
        device = _auto_device(spec, torch)
        dtype = _torch_dtype(spec, torch)

        tok = AutoTokenizer.from_pretrained(spec.model_name_or_path, trust_remote_code=bool(spec.trust_remote_code))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # Left padding is recommended for batched generation with decoder-only models.
        tok.padding_side = "left"

        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=bool(spec.trust_remote_code),
        )

        if spec.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise ImportError("load_in_4bit=True requires bitsandbytes + transformers BitsAndBytesConfig") from e

            bnb_dt = _bnb_compute_dtype(spec.bnb_4bit_compute_dtype, torch)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_dt,
            )
            # With quantization, device_map is typically required/expected.
            if spec.device_map is None:
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = str(spec.device_map)
        elif spec.device_map is not None:
            load_kwargs["device_map"] = str(spec.device_map)
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(spec.model_name_or_path, **load_kwargs)
        model.train()
        if spec.device_map is None and not spec.load_in_4bit:
            model.to(device)

        # LoRA adapter
        if spec.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise ImportError(
                    "use_lora=True requires `peft`. Install extras (recommended): `pip install -e .[rlvr]`"
                ) from e

            lora_cfg = LoraConfig(
                r=int(spec.lora_r),
                lora_alpha=int(spec.lora_alpha),
                lora_dropout=float(spec.lora_dropout),
                target_modules=list(spec.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        else:
            # Full finetune (rarely feasible at 14B); at least warn.
            print("[rlvr] WARNING: use_lora=False => full finetune; this is likely infeasible for 14B on 1 GPU.")

        # Optimizer (only trainable params)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(spec.lr), weight_decay=float(spec.weight_decay))

        self._tok = tok
        self._model = model
        self._device = device
        try:
            self._input_device = str(next(model.parameters()).device)
        except StopIteration:
            self._input_device = device
        self._opt = opt
        self._baseline = 0.0
        self._step = 0

    @property
    def tokenizer(self):
        return self._tok

    @property
    def model(self):
        return self._model

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save adapter if present; otherwise save full model.
        try:
            self._model.save_pretrained(str(out_dir))
        except Exception:
            # fallback: state dict
            import torch

            torch.save(self._model.state_dict(), str(out_dir / "model_state_dict.pt"))
        self._tok.save_pretrained(str(out_dir))
        (out_dir / "rlvr_spec.json").write_text(json.dumps(asdict(self.spec), indent=2, sort_keys=True) + "\n")

    def train_step(self, *, infos: List[str], y: np.ndarray, q: np.ndarray) -> dict:
        """
        One RLVR update from a minibatch.

        infos: list of prompt info strings (length B)
        y: (B,) 0/1 outcomes
        q: (B,) market prices in [0,1]
        """
        import torch

        spec = self.spec
        tok = self._tok
        model = self._model

        # prompts
        prompts = [
            build_prompt(info, include_cot=bool(spec.include_cot), cot_max_steps=int(spec.cot_max_steps))
            for info in infos
        ]
        enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(spec.max_prompt_tokens),
            return_tensors="pt",
        )
        in_dev = self._input_device or model.device
        input_ids = enc["input_ids"].to(in_dev)
        attn = enc["attention_mask"].to(in_dev)
        input_len = int(input_ids.shape[1])  # padded prompt length (left padding)

        # sample a completion
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=float(spec.temperature),
            top_p=float(spec.top_p),
            max_new_tokens=int(spec.max_new_tokens),
            pad_token_id=int(tok.pad_token_id),
            eos_token_id=int(tok.eos_token_id),
        )

        # decode + parse probabilities
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        ps = []
        n_fail = 0
        for t in texts:
            tail = t.split("OUTPUT:", 1)[-1]
            p = _extract_prob(tail)
            if p is None:
                n_fail += 1
                p = 0.5
            ps.append(float(np.clip(p, 0.0, 1.0)))
        p_arr = np.asarray(ps, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)

        # base reward
        logscore = _logscore(p_arr, y_arr)  # (B,)
        pnl = _pnl_proxy(
            p=p_arr,
            q=q_arr,
            y=y_arr,
            B=float(spec.reward.B),
            transaction_cost=float(spec.reward.transaction_cost),
            mode=str(spec.reward.trading_mode),
        )
        R = float(spec.reward.alpha_logscore) * logscore + float(spec.reward.beta_pnl) * pnl

        # clip rewards
        clip = float(spec.reward_clip)
        if clip is not None and clip > 0:
            R = np.clip(R, -clip, clip)

        # compute logp policy + logp reference (LoRA disabled) on the generated sequence
        seq = gen.to(in_dev)
        logp_pol, n_gen = _compute_logps_for_generated(
            model=model, input_len=input_len, sequences=seq, pad_token_id=int(tok.pad_token_id)
        )
        # reference logp: LoRA disabled (peft) if available, else treat as identical (kl=0)
        try:
            with model.disable_adapter():
                logp_ref, _n2 = _compute_logps_for_generated(
                    model=model, input_len=input_len, sequences=seq, pad_token_id=int(tok.pad_token_id)
                )
        except Exception:
            logp_ref = logp_pol.detach()

        kl = (logp_pol - logp_ref).detach()  # (B,)
        R_t = torch.from_numpy(R.astype(np.float32)).to(model.device)
        eff = R_t - float(spec.kl_coef) * kl.to(R_t.dtype)

        # baseline (EMA) to reduce variance
        eff_mean = float(torch.mean(eff).detach().cpu().item())
        self._baseline = float(spec.baseline_ema) * float(self._baseline) + (1.0 - float(spec.baseline_ema)) * eff_mean
        adv = eff - float(self._baseline)

        # REINFORCE loss: -adv * logp
        loss = -torch.mean(adv.detach() * logp_pol)

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if spec.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.grad_clip))
        self._opt.step()

        self._step += 1
        return {
            "step": int(self._step),
            "loss": float(loss.detach().cpu().item()),
            "reward_mean": float(np.mean(R)),
            "logscore_mean": float(np.mean(logscore)),
            "pnl_mean": float(np.mean(pnl)),
            "eff_mean": float(eff_mean),
            "baseline": float(self._baseline),
            "kl_mean": float(torch.mean(kl).detach().cpu().item()),
            "gen_tokens_mean": float(torch.mean(n_gen.to(torch.float32)).detach().cpu().item()),
            "parse_failures": int(n_fail),
        }


def train_rlvr(
    *,
    infos: List[str],
    y: np.ndarray,
    q: np.ndarray,
    spec: RLVRTrainSpec,
    out_dir: Path,
) -> Dict[str, object]:
    """
    High-level training harness.
    """
    rng = np.random.default_rng(int(spec.seed))
    trainer = RLVRTrainer(spec)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[dict] = []

    n = int(len(infos))
    bs = int(spec.batch_size)
    if n <= 0:
        raise ValueError("Empty training set")
    if y.shape != (n,) or q.shape != (n,):
        raise ValueError("y and q must have shape (n,)")

    t0 = time.time()
    for step in range(int(spec.steps)):
        idx = rng.integers(0, n, size=(bs,))
        batch_infos = [infos[int(i)] for i in idx.tolist()]
        batch_y = y[idx]
        batch_q = q[idx]
        rec = trainer.train_step(infos=batch_infos, y=batch_y, q=batch_q)
        metrics.append(rec)
        if spec.log_every and (step + 1) % int(spec.log_every) == 0:
            dt = time.time() - t0
            print(
                f"[rlvr] step {step+1}/{spec.steps} "
                f"loss={rec['loss']:.4f} R={rec['reward_mean']:.4f} "
                f"kl={rec['kl_mean']:.4f} gen_toks={rec['gen_tokens_mean']:.1f} "
                f"fails={rec['parse_failures']} time={dt:.1f}s"
            )
        if spec.save_every and (step + 1) % int(spec.save_every) == 0:
            trainer.save(out_dir / f"ckpt_step_{step+1:06d}")

    trainer.save(out_dir / "final")
    return {"n": int(n), "steps": int(spec.steps), "metrics": metrics, "out_dir": str(out_dir)}



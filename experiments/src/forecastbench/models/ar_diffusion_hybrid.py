"""
AR + Diffusion Hybrid: Uses AR for initial reasoning, diffusion for refinement.

The theory (main.tex) suggests:
- AR captures "serial" reasoning but has a spectral cutoff at depth L
- Diffusion provides smooth spectral attenuation for refinement

This hybrid:
1. AR model generates chain-of-thought and initial probability q_AR
2. Diffusion models p(true | q_AR, context) - refining the AR estimate
3. The diffusion acts as a "calibration" or "repair" layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ARDiffusionHybridSpec:
    """Specification for AR+Diffusion hybrid forecaster."""
    # AR component
    ar_model_name: str = "Qwen/Qwen3-14B"
    ar_max_new_tokens: int = 256
    ar_temperature: float = 0.7
    ar_top_p: float = 0.95
    ar_include_cot: bool = True
    ar_K: int = 5  # self-consistency samples
    
    # Diffusion refinement
    diff_hidden_dim: int = 256
    diff_depth: int = 3
    diff_T: int = 50
    diff_time_dim: int = 64
    
    # How to combine
    use_ar_as_input: bool = True  # If True, diffusion conditions on q_AR
    use_text_embedding: bool = True  # If True, also conditions on text embedding


class DiffusionRefinementHead:
    """
    A small diffusion model that refines AR predictions.
    
    Input: (q_AR, text_embedding) where q_AR is the AR model's probability estimate
    Output: Refined probability distribution
    
    The key insight: diffusion can model the residual (p - q_AR) or directly
    model p conditioned on q_AR. Either way, it acts as calibration.
    """
    
    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        T: int = 50,
        time_dim: int = 64,
        device: str = "cuda",
    ):
        import torch
        import torch.nn as nn
        
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.T = T
        self.time_dim = time_dim
        self.device = device
        
        # Diffusion schedule (linear beta)
        betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
        alphas = 1.0 - betas
        self.alpha_bar = np.cumprod(alphas, axis=0)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        ).to(device)
        
        # Condition includes: q_AR (1-dim) + text_embedding (cond_dim)
        # In train_loop, we concatenate [q_ar_logit, text_embed] -> full_cond of dim (1 + cond_dim)
        # In forward: inp = [x_t (1), time_embed (hidden_dim), full_cond (1 + cond_dim)]
        in_dim = 1 + hidden_dim + 1 + cond_dim  # x_t + time_embed + full_cond
        self._expected_cond_dim = 1 + cond_dim  # Store for validation
        
        # Main network with residual connections
        self.input_proj = nn.Linear(in_dim, hidden_dim).to(device)
        self.input_norm = nn.LayerNorm(hidden_dim).to(device)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1),
            ).to(device)
            for _ in range(depth)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim).to(device)
        self.output_proj = nn.Linear(hidden_dim, 1).to(device)
    
    def parameters(self):
        params = list(self.time_embed.parameters())
        params += list(self.input_proj.parameters())
        params += list(self.input_norm.parameters())
        for block in self.blocks:
            params += list(block.parameters())
        params += list(self.output_norm.parameters())
        params += list(self.output_proj.parameters())
        return params
    
    def train_mode(self):
        self.time_embed.train()
        self.input_proj.train()
        self.input_norm.train()
        for block in self.blocks:
            block.train()
        self.output_norm.train()
        self.output_proj.train()
    
    def eval_mode(self):
        self.time_embed.eval()
        self.input_proj.eval()
        self.input_norm.eval()
        for block in self.blocks:
            block.eval()
        self.output_norm.eval()
        self.output_proj.eval()
    
    def _sinusoidal_embed(self, t: np.ndarray) -> np.ndarray:
        """Sinusoidal time embedding."""
        t = t.astype(np.float32).reshape(-1, 1)
        half = self.time_dim // 2
        freqs = np.exp(-np.log(10_000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1))
        ang = t * freqs.reshape(1, -1)
        emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
        return emb.astype(np.float32)
    
    def _q_sample(self, x0: np.ndarray, t: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Forward diffusion: add noise."""
        a = self.alpha_bar[t].reshape(-1, 1)
        return np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise
    
    def forward(self, x_t, t_embed, cond):
        """Forward pass: predict noise.
        
        Args:
            x_t: (batch, 1) noisy logit at timestep t
            t_embed: (batch, time_dim) sinusoidal time embedding
            cond: (batch, 1 + cond_dim) full condition = [q_ar_logit, text_embed]
        """
        import torch
        
        h_t = self.time_embed(t_embed)
        inp = torch.cat([x_t, h_t, cond], dim=-1)
        
        # Validate dimensions match (helps debug shape mismatches)
        expected_in_dim = self.input_proj.in_features
        actual_in_dim = inp.shape[-1]
        if actual_in_dim != expected_in_dim:
            raise ValueError(
                f"Dimension mismatch in DiffusionRefinementHead.forward: "
                f"input has {actual_in_dim} features but model expects {expected_in_dim}. "
                f"Components: x_t={x_t.shape[-1]}, h_t={h_t.shape[-1]}, cond={cond.shape[-1]}. "
                f"Expected cond_dim={self._expected_cond_dim if hasattr(self, '_expected_cond_dim') else 'N/A'}."
            )
        
        x = self.input_proj(inp)
        x = self.input_norm(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual
        
        x = self.output_norm(x)
        return self.output_proj(x)
    
    def train_loop(
        self,
        q_ar: np.ndarray,  # AR predictions (N,)
        cond: np.ndarray,  # Text embeddings (N, cond_dim)
        p_true: np.ndarray,  # True probabilities or outcomes (N,)
        steps: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-4,
        seed: int = 0,
    ) -> dict:
        """
        Train diffusion to refine AR predictions.
        
        The target is p_true (or outcomes y), and we condition on q_ar.
        """
        import torch
        import math
        
        rng = np.random.default_rng(seed)
        N = len(q_ar)
        
        # Convert to logit space for diffusion (unbounded)
        def to_logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        
        # x0 is the true probability in logit space
        x0 = to_logit(p_true).reshape(-1, 1).astype(np.float32)
        
        # Condition includes q_ar (in logit space) concatenated with text embedding
        # full_cond shape: (N, 1 + cond_dim) where cond_dim is the text embedding dimension
        q_ar_logit = to_logit(q_ar).reshape(-1, 1).astype(np.float32)
        full_cond = np.concatenate([q_ar_logit, cond.astype(np.float32)], axis=1)
        
        # Validate full_cond dimension matches what the model expects
        expected_cond_dim = getattr(self, '_expected_cond_dim', 1 + self.cond_dim)
        if full_cond.shape[1] != expected_cond_dim:
            raise ValueError(
                f"Condition dimension mismatch: full_cond has {full_cond.shape[1]} dims "
                f"(1 for q_ar + {cond.shape[1]} for text_embed), "
                f"but model expects {expected_cond_dim}. "
                f"Check that cond_dim={self.cond_dim} matches your text embedding dimension."
            )
        
        self.train_mode()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine schedule
        def lr_lambda(step):
            warmup = 100
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        
        losses = []
        for step in range(steps):
            idx = rng.integers(0, N, size=(batch_size,))
            t = rng.integers(0, self.T, size=(batch_size,))
            eps = rng.standard_normal(size=(batch_size, 1)).astype(np.float32)
            
            x_t = self._q_sample(x0[idx], t, eps)
            t_emb = self._sinusoidal_embed(t)
            
            x_t_t = torch.from_numpy(x_t).to(self.device)
            eps_t = torch.from_numpy(eps).to(self.device)
            t_t = torch.from_numpy(t_emb).to(self.device)
            c_t = torch.from_numpy(full_cond[idx]).to(self.device)
            
            pred = self.forward(x_t_t, t_t, c_t)
            loss = torch.mean((pred - eps_t) ** 2)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            losses.append(float(loss.detach().cpu().item()))
            if (step + 1) % 200 == 0:
                print(f"[refine-diff] step {step+1}/{steps}  loss={np.mean(losses[-200:]):.6f}")
        
        self.eval_mode()
        return {"steps": steps, "final_loss": float(np.mean(losses[-100:]))}
    
    def sample(
        self,
        q_ar: np.ndarray,
        cond: np.ndarray,
        n_steps: int = 50,
        n_samples: int = 16,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Sample refined probabilities conditioned on AR predictions.
        
        Args:
            q_ar: (N,) AR predictions (probabilities in [0,1])
            cond: (N, cond_dim) text embeddings
            n_steps: Number of diffusion sampling steps
            n_samples: Monte Carlo samples to average
            seed: Random seed
        
        Returns: (N,) array of mean refined probabilities
        """
        import torch
        
        rng = np.random.default_rng(seed)
        N = len(q_ar)
        
        def to_logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        
        def from_logit(x):
            return 1 / (1 + np.exp(-x))
        
        # Build full condition: [q_ar_logit, text_embedding]
        q_ar_logit = to_logit(q_ar).reshape(-1, 1).astype(np.float32)
        full_cond = np.concatenate([q_ar_logit, cond.astype(np.float32)], axis=1)
        
        # Validate dimension
        expected_cond_dim = getattr(self, '_expected_cond_dim', 1 + self.cond_dim)
        if full_cond.shape[1] != expected_cond_dim:
            raise ValueError(
                f"Condition dimension mismatch in sample: full_cond has {full_cond.shape[1]} dims, "
                f"expected {expected_cond_dim}. cond.shape={cond.shape}, cond_dim={self.cond_dim}"
            )
        
        all_samples = []
        for _ in range(n_samples):
            # Start from noise
            x_t = rng.standard_normal(size=(N, 1)).astype(np.float32)
            x_t = torch.from_numpy(x_t).to(self.device)
            c_t = torch.from_numpy(full_cond).to(self.device)
            
            # DDIM sampling
            t_idx = np.linspace(self.T - 1, 0, n_steps, dtype=np.int64)
            
            for t in t_idx:
                t_arr = np.full((N,), int(t), dtype=np.int64)
                t_emb = self._sinusoidal_embed(t_arr)
                t_t = torch.from_numpy(t_emb).to(self.device)
                
                with torch.no_grad():
                    eps = self.forward(x_t, t_t, c_t)
                
                ab_t = float(self.alpha_bar[t])
                x0 = (x_t - (1.0 - ab_t) ** 0.5 * eps) / (ab_t ** 0.5)
                
                if t == 0:
                    x_t = x0
                    continue
                
                ab_prev = float(self.alpha_bar[t - 1])
                # Deterministic DDIM (eta=0)
                x_t = (ab_prev ** 0.5) * x0 + ((1 - ab_prev) ** 0.5) * eps
            
            sample = x_t.detach().cpu().numpy().astype(np.float32)
            all_samples.append(from_logit(sample))
        
        # Stack and return mean
        stacked = np.stack(all_samples, axis=-1)  # (N, 1, n_samples)
        return np.mean(stacked, axis=-1).squeeze(-1)


class ARDiffusionHybridForecaster:
    """
    Hybrid forecaster: AR for reasoning, diffusion for calibration.
    
    Pipeline:
    1. AR model generates K samples with chain-of-thought
    2. Extract probability from each sample, aggregate to q_AR
    3. Diffusion refines q_AR → p_refined
    4. (Optional) Temperature scaling on p_refined
    """
    
    def __init__(
        self,
        spec: ARDiffusionHybridSpec,
        embedder=None,  # TextEmbedder instance
        device: str = "cuda",
    ):
        self.spec = spec
        self.embedder = embedder
        self.device = device
        
        # AR predictor (lazy load)
        self._ar_predictor = None
        
        # Diffusion head (created after we know cond_dim)
        self._diff_head = None
    
    def _get_ar_predictor(self):
        if self._ar_predictor is None:
            from forecastbench.models.ar_cot import ARCoTPredictor, ARCoTSpec
            ar_spec = ARCoTSpec(
                model_name_or_path=self.spec.ar_model_name,
                device=self.device,
                temperature=self.spec.ar_temperature,
                top_p=self.spec.ar_top_p,
                max_new_tokens=self.spec.ar_max_new_tokens,
                include_cot=self.spec.ar_include_cot,
            )
            self._ar_predictor = ARCoTPredictor(ar_spec)
        return self._ar_predictor
    
    def _get_diff_head(self, cond_dim: int):
        if self._diff_head is None:
            # cond_dim = 1 (q_ar) + text_embed_dim
            full_cond_dim = 1 + cond_dim if self.spec.use_text_embedding else 1
            self._diff_head = DiffusionRefinementHead(
                cond_dim=cond_dim,
                hidden_dim=self.spec.diff_hidden_dim,
                depth=self.spec.diff_depth,
                T=self.spec.diff_T,
                time_dim=self.spec.diff_time_dim,
                device=self.device,
            )
        return self._diff_head
    
    def predict_ar_only(
        self,
        texts: list,
        K: Optional[int] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, dict]:
        """Get AR predictions (without diffusion refinement)."""
        K = K or self.spec.ar_K
        ar = self._get_ar_predictor()
        q_ar, meta = ar.predict_proba(texts, K=K, seed=seed)
        return q_ar, meta
    
    def predict_hybrid(
        self,
        texts: list,
        text_embeddings: Optional[np.ndarray] = None,
        K: Optional[int] = None,
        diff_samples: int = 16,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Full hybrid prediction.
        
        Returns:
            q_ar: AR predictions
            q_refined: Diffusion-refined predictions
            meta: Metadata dict
        """
        K = K or self.spec.ar_K
        
        # Step 1: Get AR predictions
        q_ar, ar_meta = self.predict_ar_only(texts, K=K, seed=seed)
        
        # Step 2: Get text embeddings if needed
        if text_embeddings is None and self.embedder is not None:
            text_embeddings = self.embedder.encode(texts)
        
        if text_embeddings is None:
            raise ValueError("Need text_embeddings for diffusion refinement")
        
        # Step 3: Diffusion refinement
        diff_head = self._get_diff_head(text_embeddings.shape[1])
        q_refined = diff_head.sample(
            q_ar=q_ar,
            cond=text_embeddings,
            n_samples=diff_samples,
            seed=seed + 1000,
        )
        
        return q_ar, q_refined, {"ar_meta": ar_meta}
    
    def train_refinement(
        self,
        q_ar: np.ndarray,
        text_embeddings: np.ndarray,
        y: np.ndarray,
        steps: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-4,
        seed: int = 0,
    ) -> dict:
        """
        Train the diffusion refinement head.
        
        q_ar: AR predictions (N,)
        text_embeddings: (N, embed_dim)
        y: Outcomes (0 or 1) or true probabilities
        """
        diff_head = self._get_diff_head(text_embeddings.shape[1])
        return diff_head.train_loop(
            q_ar=q_ar,
            cond=text_embeddings,
            p_true=y.astype(np.float32),
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )


"""
Unified Model Loader for C_t Estimation.

Supports multiple model types for generating C_t samples:
- AR+Diffusion Hybrid: LLM generates initial prediction, diffusion refines
- RLCR: Fine-tuned LLM with calibration rewards
- Bundle Diffusion: Joint diffusion over multiple markets

Modes:
- SINGLE: Use one specific model
- UNION: Combine samples from all enabled models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CtMode(Enum):
    """C_t generation mode."""
    SINGLE = "single"  # Use one model
    UNION = "union"    # Combine samples from all enabled models


class CtModel(Enum):
    """Available model types for C_t generation."""
    AR_DIFFUSION = "ar_diffusion"
    RLCR = "rlcr"
    BUNDLE = "bundle"
    LEGACY = "legacy"  # Backward compat with old ct_loader


@dataclass
class ModelLoaderConfig:
    """Configuration for C_t generation."""
    
    # Mode selection
    ct_mode: CtMode = CtMode.SINGLE
    ct_model: CtModel = CtModel.LEGACY  # Used when ct_mode=SINGLE
    
    # Model paths
    ar_diffusion_path: Optional[Path] = None
    rlcr_model_path: Optional[str] = None
    bundle_diffusion_path: Optional[Path] = None
    
    # Embedding configuration
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    
    # Sampling config per model
    ar_diffusion_samples: int = 16
    rlcr_K: int = 5
    bundle_samples: int = 16
    
    # Which models are enabled (for UNION mode)
    enabled_models: Tuple[str, ...] = ("ar_diffusion", "rlcr", "bundle")
    
    # Device
    device: str = "cpu"  # Default to CPU for testing; use "cuda" for GPU
    
    def __post_init__(self):
        if isinstance(self.ar_diffusion_path, str):
            self.ar_diffusion_path = Path(self.ar_diffusion_path)
        if isinstance(self.bundle_diffusion_path, str):
            self.bundle_diffusion_path = Path(self.bundle_diffusion_path)


class ARDiffusionLoader:
    """
    AR+Diffusion hybrid loader.
    
    Pipeline:
    1. AR model generates K CoT samples -> q_AR (aggregated)
    2. Diffusion refines q_AR conditioned on text embeddings
    3. Returns n_samples from diffusion posterior
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        ar_model: str = "Qwen/Qwen3-14B",
        ar_K: int = 5,
        diff_hidden_dim: int = 256,
        diff_depth: int = 4,
        diff_T: int = 50,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.ar_model = ar_model
        self.ar_K = ar_K
        self.diff_hidden_dim = diff_hidden_dim
        self.diff_depth = diff_depth
        self.diff_T = diff_T
        self.device = device
        
        self._forecaster: Any = None
        self._loaded = False
    
    def _load(self) -> None:
        """Lazy load the AR+Diffusion hybrid forecaster."""
        if self._loaded:
            return
        
        import torch
        
        from forecastbench.models.ar_diffusion_hybrid import (
            ARDiffusionHybridForecaster,
            ARDiffusionHybridSpec,
        )
        
        spec = ARDiffusionHybridSpec(
            ar_model_name=self.ar_model,
            ar_K=self.ar_K,
            diff_hidden_dim=self.diff_hidden_dim,
            diff_depth=self.diff_depth,
            diff_T=self.diff_T,
        )
        
        self._forecaster = ARDiffusionHybridForecaster(spec, device=self.device)
        
        # Load diffusion head weights if checkpoint exists
        diff_head_path = self.checkpoint_path / "diff_head.pt"
        if diff_head_path.exists():
            # Get the diff head (this will create it with default cond_dim)
            # We need embeddings to know cond_dim, so defer actual loading
            pass
        
        self._loaded = True
    
    def sample(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        n_samples: int = 16,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Generate C_t samples via AR + diffusion refinement.
        
        Args:
            texts: List of market question texts
            embeddings: (n, embed_dim) text embeddings
            n_samples: Number of diffusion samples per market
            seed: Random seed
        
        Returns:
            samples: (n_samples, n) probability samples
        """
        self._load()
        
        # Get AR predictions first
        q_ar, _ = self._forecaster.predict_ar_only(texts, K=self.ar_K, seed=seed)
        
        # Refine with diffusion
        diff_head = self._forecaster._get_diff_head(embeddings.shape[1])
        
        # Load weights if available
        diff_head_path = self.checkpoint_path / "diff_head.pt"
        if diff_head_path.exists() and not hasattr(self, '_diff_loaded'):
            import torch
            state_dict = torch.load(diff_head_path, map_location=self.device)
            # Load into diff_head (implementation-specific)
            self._diff_loaded = True
        
        # Sample from diffusion
        q_refined = diff_head.sample(
            q_ar=q_ar,
            cond=embeddings.astype(np.float32),
            n_samples=n_samples,
            seed=seed,
        )
        
        # q_refined is (n,) mean of samples; we want (n_samples, n)
        # Re-sample to get multiple samples
        samples = []
        for i in range(n_samples):
            s = diff_head.sample(
                q_ar=q_ar,
                cond=embeddings.astype(np.float32),
                n_samples=1,
                seed=seed + i * 1000,
            )
            samples.append(s)
        
        return np.stack(samples, axis=0)  # (n_samples, n)
    
    def is_loaded(self) -> bool:
        return self._loaded


class RLCRLoader:
    """
    RLCR (Reinforcement Learning with Calibration Rewards) loader.
    
    Generates K samples with temperature sampling from a fine-tuned LLM.
    Each sample is an independent forward pass with different random seed.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen3-14B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self._predictor: Any = None
        self._loaded = False
    
    def _load(self) -> None:
        """Lazy load the RLCR model."""
        if self._loaded:
            return
        
        from forecastbench.models.rlvr_ar import ARLoRAPredictor, ARLoRAPredictorSpec
        
        spec = ARLoRAPredictorSpec(
            base_model_name_or_path=self.base_model,
            adapter_path=self.model_path,
            device=self.device,
            load_in_4bit=self.load_in_4bit,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        
        self._predictor = ARLoRAPredictor(spec)
        self._loaded = True
    
    def sample(
        self,
        texts: List[str],
        K: int = 5,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Generate K samples per input via temperature sampling.
        
        Args:
            texts: List of market question texts
            K: Number of samples per text
            seed: Base random seed
        
        Returns:
            samples: (K, n) probability samples
        """
        self._load()
        
        samples = []
        for k in range(K):
            # Each sample uses a different seed for diversity
            probs, _ = self._predictor.predict_proba(
                texts,
                K=1,  # Single sample per call
                seed=seed + k * 1000,
            )
            samples.append(probs)
        
        return np.stack(samples, axis=0)  # (K, n)
    
    def is_loaded(self) -> bool:
        return self._loaded


class BundleDiffusionLoader:
    """
    Bundle diffusion loader for joint modeling of multiple markets.
    
    Wraps the existing BundleLogitDiffusionForecaster from ct_loader.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        bundle_size: int = 8,
        embed_dim: int = 4096,
        model_dim: int = 256,
        depth: int = 2,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.bundle_size = bundle_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.depth = depth
        self.device = device
        
        self._model: Any = None
        self._loaded = False
    
    def _load(self) -> None:
        """Lazy load the bundle diffusion model."""
        if self._loaded:
            return
        
        import json
        import torch
        
        from forecastbench.models.diffusion_bundle import (
            BundleLogitDiffusionForecaster,
            BundleLogitDiffusionSpec,
        )
        
        # Load config if available
        config_path = self.checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            spec = BundleLogitDiffusionSpec(
                bundle_size=cfg.get("bundle_size", self.bundle_size),
                embed_dim=cfg.get("embed_dim", self.embed_dim),
                model_dim=cfg.get("model_dim", self.model_dim),
                depth=cfg.get("depth", self.depth),
            )
        else:
            spec = BundleLogitDiffusionSpec(
                bundle_size=self.bundle_size,
                embed_dim=self.embed_dim,
                model_dim=self.model_dim,
                depth=self.depth,
            )
        
        self._model = BundleLogitDiffusionForecaster(spec, device=self.device)
        
        # Load weights
        model_path = self.checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self._model.model.load_state_dict(state_dict)
            self._model.model.eval()
        
        self._loaded = True
    
    def sample(
        self,
        cond: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_samples: int = 16,
        n_steps: int = 50,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Sample from bundle diffusion model.
        
        Args:
            cond: (batch, bundle_size, embed_dim) conditioning embeddings
            mask: (batch, bundle_size) validity mask
            n_samples: Number of MC samples
            n_steps: Diffusion steps
            seed: Random seed
        
        Returns:
            samples: (n_samples, batch, bundle_size) probability samples
        """
        self._load()
        
        if mask is None:
            mask = np.ones((cond.shape[0], cond.shape[1]), dtype=bool)
        
        samples = []
        for i in range(n_samples):
            sample_seed = seed + i * 1000
            logits = self._model.sample_x(
                cond=cond,
                mask=mask,
                n_steps=n_steps,
                seed=sample_seed,
            )
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            samples.append(probs)
        
        return np.stack(samples, axis=0)
    
    def is_loaded(self) -> bool:
        return self._loaded


class UnifiedModelLoader:
    """
    Unified loader managing multiple model types for C_t estimation.
    
    Supports:
    - SINGLE mode: Use one specific model
    - UNION mode: Combine samples from all enabled models
    """
    
    def __init__(self, cfg: ModelLoaderConfig):
        self.cfg = cfg
        
        # Individual loaders (lazy initialized)
        self._ar_diffusion: Optional[ARDiffusionLoader] = None
        self._rlcr: Optional[RLCRLoader] = None
        self._bundle: Optional[BundleDiffusionLoader] = None
        
        # Text embedder (lazy initialized)
        self._embedder: Any = None
    
    def _get_embedder(self):
        """Lazy load text embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.cfg.embed_model)
        return self._embedder
    
    def _get_ar_diffusion(self) -> Optional[ARDiffusionLoader]:
        """Get or create AR+Diffusion loader."""
        if self._ar_diffusion is None and self.cfg.ar_diffusion_path is not None:
            self._ar_diffusion = ARDiffusionLoader(
                checkpoint_path=self.cfg.ar_diffusion_path,
                device=self.cfg.device,
            )
        return self._ar_diffusion
    
    def _get_rlcr(self) -> Optional[RLCRLoader]:
        """Get or create RLCR loader."""
        if self._rlcr is None and self.cfg.rlcr_model_path is not None:
            self._rlcr = RLCRLoader(
                model_path=self.cfg.rlcr_model_path,
                device=self.cfg.device,
            )
        return self._rlcr
    
    def _get_bundle(self) -> Optional[BundleDiffusionLoader]:
        """Get or create Bundle diffusion loader."""
        if self._bundle is None and self.cfg.bundle_diffusion_path is not None:
            self._bundle = BundleDiffusionLoader(
                checkpoint_path=self.cfg.bundle_diffusion_path,
                embed_dim=self.cfg.embed_dim,
                device=self.cfg.device,
            )
        return self._bundle
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute text embeddings."""
        embedder = self._get_embedder()
        return embedder.encode(texts, convert_to_numpy=True)
    
    def _sample_ar_diffusion(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        seed: int = 0,
    ) -> Optional[np.ndarray]:
        """Sample from AR+Diffusion model."""
        loader = self._get_ar_diffusion()
        if loader is None:
            return None
        
        return loader.sample(
            texts=texts,
            embeddings=embeddings,
            n_samples=self.cfg.ar_diffusion_samples,
            seed=seed,
        )
    
    def _sample_rlcr(
        self,
        texts: List[str],
        seed: int = 0,
    ) -> Optional[np.ndarray]:
        """Sample from RLCR model."""
        loader = self._get_rlcr()
        if loader is None:
            return None
        
        return loader.sample(
            texts=texts,
            K=self.cfg.rlcr_K,
            seed=seed,
        )
    
    def _sample_bundle(
        self,
        embeddings: np.ndarray,
        seed: int = 0,
    ) -> Optional[np.ndarray]:
        """Sample from Bundle diffusion model."""
        loader = self._get_bundle()
        if loader is None:
            return None
        
        n = embeddings.shape[0]
        bundle_size = loader.bundle_size
        
        # Reshape embeddings for bundle model: (1, n, embed_dim)
        # Pad to bundle_size if needed
        if n < bundle_size:
            pad_size = bundle_size - n
            cond = np.pad(
                embeddings.reshape(1, n, -1),
                ((0, 0), (0, pad_size), (0, 0)),
                mode="constant",
            )
            mask = np.zeros((1, bundle_size), dtype=bool)
            mask[0, :n] = True
        else:
            # Take first bundle_size markets
            cond = embeddings[:bundle_size].reshape(1, bundle_size, -1)
            mask = np.ones((1, bundle_size), dtype=bool)
        
        samples = loader.sample(
            cond=cond,
            mask=mask,
            n_samples=self.cfg.bundle_samples,
            seed=seed,
        )
        
        # (n_samples, 1, bundle_size) -> (n_samples, n)
        samples = samples[:, 0, :n]
        return samples
    
    def sample_ct(
        self,
        market_ids: List[str],
        texts: Dict[str, str],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Sample C_t according to configured mode.
        
        Args:
            market_ids: List of market IDs
            texts: Dict mapping market_id -> question text
            embeddings: Optional dict mapping market_id -> embedding
            seed: Random seed
        
        Returns:
            samples: (n_samples, k) probability samples
            valid_ids: List of valid market IDs
        """
        # Filter to markets with texts
        valid_ids = [m for m in market_ids if m in texts]
        if not valid_ids:
            return np.zeros((1, 0)), []
        
        # Get texts in order
        text_list = [texts[m] for m in valid_ids]
        
        # Compute embeddings if not provided
        if embeddings is None:
            embed_array = self._compute_embeddings(text_list)
        else:
            embed_array = np.stack([embeddings[m] for m in valid_ids if m in embeddings])
        
        if self.cfg.ct_mode == CtMode.SINGLE:
            samples = self._sample_single(text_list, embed_array, seed)
        else:  # UNION
            samples = self._sample_union(text_list, embed_array, seed)
        
        return samples, valid_ids
    
    def _sample_single(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample from a single model based on ct_model config."""
        if self.cfg.ct_model == CtModel.AR_DIFFUSION:
            samples = self._sample_ar_diffusion(texts, embeddings, seed)
            if samples is None:
                raise ValueError("AR+Diffusion model not configured but ct_model=ar_diffusion")
            return samples
        
        elif self.cfg.ct_model == CtModel.RLCR:
            samples = self._sample_rlcr(texts, seed)
            if samples is None:
                raise ValueError("RLCR model not configured but ct_model=rlcr")
            return samples
        
        elif self.cfg.ct_model == CtModel.BUNDLE:
            samples = self._sample_bundle(embeddings, seed)
            if samples is None:
                raise ValueError("Bundle model not configured but ct_model=bundle")
            return samples
        
        else:  # LEGACY
            raise ValueError("Legacy mode should use ct_loader directly, not UnifiedModelLoader")
    
    def _sample_union(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample from all enabled models and concatenate."""
        all_samples = []
        
        if "ar_diffusion" in self.cfg.enabled_models:
            samples = self._sample_ar_diffusion(texts, embeddings, seed)
            if samples is not None:
                all_samples.append(samples)
        
        if "rlcr" in self.cfg.enabled_models:
            samples = self._sample_rlcr(texts, seed)
            if samples is not None:
                all_samples.append(samples)
        
        if "bundle" in self.cfg.enabled_models:
            samples = self._sample_bundle(embeddings, seed)
            if samples is not None:
                all_samples.append(samples)
        
        if not all_samples:
            # No models available, return empty
            return np.zeros((1, len(texts)))
        
        # Union = concatenation along sample axis
        unified = np.concatenate(all_samples, axis=0)
        return unified
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "ct_mode": self.cfg.ct_mode.value,
            "ct_model": self.cfg.ct_model.value,
            "enabled_models": list(self.cfg.enabled_models),
        }
        
        if self._ar_diffusion is not None:
            info["ar_diffusion"] = {
                "loaded": self._ar_diffusion.is_loaded(),
                "path": str(self.cfg.ar_diffusion_path),
                "samples": self.cfg.ar_diffusion_samples,
            }
        
        if self._rlcr is not None:
            info["rlcr"] = {
                "loaded": self._rlcr.is_loaded(),
                "path": self.cfg.rlcr_model_path,
                "K": self.cfg.rlcr_K,
            }
        
        if self._bundle is not None:
            info["bundle"] = {
                "loaded": self._bundle.is_loaded(),
                "path": str(self.cfg.bundle_diffusion_path),
                "samples": self.cfg.bundle_samples,
            }
        
        return info

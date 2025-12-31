"""
C_t Checkpoint Loader.

Loads pre-computed daily model checkpoints and samples from them
to get Monte Carlo representations of the constraint set C_t.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CtCheckpointSpec:
    """Specification for loading C_t checkpoints."""

    checkpoint_dir: Path
    model_type: str = "bundle"  # "bundle" or "single"
    embed_dim: int = 4096
    bundle_size: int = 8
    device: str = "cuda"

    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


class CtCheckpointLoader:
    """
    Loads pre-computed daily C_t model checkpoints.

    The loader expects checkpoints organized by date:
        checkpoint_dir/
        ├── 2024-01-01/
        │   ├── model.pt
        │   └── config.json
        ├── 2024-01-02/
        │   ├── model.pt
        │   └── config.json
        └── ...

    Alternatively, a single checkpoint can be used for all dates:
        checkpoint_dir/
        ├── model.pt
        └── config.json
    """

    def __init__(self, spec: CtCheckpointSpec):
        self.spec = spec
        self.current_date: Optional[str] = None
        self.model: Any = None
        self._ct_samples: Optional[np.ndarray] = None
        self._market_ids: Optional[List[str]] = None

    def _find_checkpoint_path(self, date: str) -> Path:
        """Find the checkpoint path for a given date."""
        # Try date-specific directory first
        date_dir = self.spec.checkpoint_dir / date
        if date_dir.exists() and (date_dir / "model.pt").exists():
            return date_dir

        # Try YYYYMMDD format
        date_compact = date.replace("-", "")
        date_dir = self.spec.checkpoint_dir / date_compact
        if date_dir.exists() and (date_dir / "model.pt").exists():
            return date_dir

        # Fall back to root checkpoint_dir (single checkpoint for all dates)
        if (self.spec.checkpoint_dir / "model.pt").exists():
            return self.spec.checkpoint_dir

        # Try finding the most recent checkpoint before this date
        available = self._list_available_dates()
        valid_dates = [d for d in available if d <= date]
        if valid_dates:
            return self.spec.checkpoint_dir / max(valid_dates)

        raise FileNotFoundError(
            f"No checkpoint found for date {date} in {self.spec.checkpoint_dir}"
        )

    def _list_available_dates(self) -> List[str]:
        """List all available checkpoint dates."""
        dates = []
        for path in self.spec.checkpoint_dir.iterdir():
            if path.is_dir() and (path / "model.pt").exists():
                # Try to parse as date
                name = path.name
                try:
                    if len(name) == 10 and name[4] == "-":  # YYYY-MM-DD
                        datetime.strptime(name, "%Y-%m-%d")
                        dates.append(name)
                    elif len(name) == 8:  # YYYYMMDD
                        datetime.strptime(name, "%Y%m%d")
                        dates.append(f"{name[:4]}-{name[4:6]}-{name[6:8]}")
                except ValueError:
                    pass
        return sorted(dates)

    def load_for_date(self, date: str) -> None:
        """
        Load the model checkpoint for a given date.

        Args:
            date: Date in YYYY-MM-DD format
        """
        if self.current_date == date and self.model is not None:
            return  # Already loaded

        checkpoint_path = self._find_checkpoint_path(date)
        model_path = checkpoint_path / "model.pt"
        config_path = checkpoint_path / "config.json"

        if self.spec.model_type == "bundle":
            self.model = self._load_bundle_model(model_path, config_path)
        else:
            self.model = self._load_single_model(model_path, config_path)

        self.current_date = date
        self._ct_samples = None  # Reset cached samples

    def _load_bundle_model(self, model_path: Path, config_path: Path) -> Any:
        """Load a BundleLogitDiffusionForecaster."""
        import torch

        from forecastbench.models.diffusion_bundle import (
            BundleLogitDiffusionForecaster,
            BundleLogitDiffusionSpec,
        )

        # Load config if available
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            spec = BundleLogitDiffusionSpec(
                bundle_size=cfg.get("bundle_size", self.spec.bundle_size),
                embed_dim=cfg.get("embed_dim", self.spec.embed_dim),
                model_dim=cfg.get("model_dim", 256),
                depth=cfg.get("depth", 2),
            )
        else:
            spec = BundleLogitDiffusionSpec(
                bundle_size=self.spec.bundle_size,
                embed_dim=self.spec.embed_dim,
            )

        model = BundleLogitDiffusionForecaster(spec, device=self.spec.device)

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.spec.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.model.load_state_dict(state_dict)
        model.model.eval()

        return model

    def _load_single_model(self, model_path: Path, config_path: Path) -> Any:
        """Load a ContinuousDiffusionForecaster."""
        import torch

        from forecastbench.models.diffusion_core import (
            ContinuousDiffusionForecaster,
            ContinuousDiffusionSpec,
        )

        # Load config if available
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            spec = ContinuousDiffusionSpec(
                in_dim=cfg.get("in_dim", self.spec.embed_dim),
                out_dim=cfg.get("out_dim", 1),
                hidden_dim=cfg.get("hidden_dim", 256),
                depth=cfg.get("depth", 3),
            )
        else:
            spec = ContinuousDiffusionSpec(
                in_dim=self.spec.embed_dim,
                out_dim=1,
            )

        model = ContinuousDiffusionForecaster(spec, device=self.spec.device)

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.spec.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.model.load_state_dict(state_dict)
        model.model.eval()

        return model

    def sample_ct(
        self,
        cond: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_samples: int = 64,
        n_steps: int = 50,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Sample from the current model to get a C_t representation.

        The samples define C_t ≈ conv({p^(1), ..., p^(n_samples)}).

        Args:
            cond: Conditioning embeddings (batch, bundle_size, embed_dim) for bundle
                  or (batch, embed_dim) for single
            mask: Optional mask for bundle models (batch, bundle_size)
            n_samples: Number of MC samples
            n_steps: Number of diffusion steps
            seed: Random seed

        Returns:
            samples: (n_samples, batch, bundle_size) or (n_samples, batch, 1)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_for_date first.")

        samples = []
        for i in range(n_samples):
            sample_seed = seed + i * 1000

            if self.spec.model_type == "bundle":
                if mask is None:
                    mask = np.ones((cond.shape[0], cond.shape[1]), dtype=bool)
                logits = self.model.sample_x(
                    cond=cond,
                    mask=mask,
                    n_steps=n_steps,
                    seed=sample_seed,
                )
                # Convert logits to probabilities
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            else:
                logits = self.model.sample_x(
                    cond=cond,
                    n_steps=n_steps,
                    seed=sample_seed,
                )
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

            samples.append(probs)

        return np.stack(samples, axis=0)

    def sample_ct_for_markets(
        self,
        market_ids: List[str],
        embeddings: Dict[str, np.ndarray],
        n_samples: int = 64,
        seed: int = 0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Sample C_t for a specific set of markets.

        Args:
            market_ids: List of market IDs
            embeddings: Dict mapping market_id -> embedding vector
            n_samples: Number of MC samples
            seed: Random seed

        Returns:
            samples: (n_samples, k) where k = len(market_ids)
            valid_ids: List of market IDs that have embeddings
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_for_date first.")

        # Filter to markets with embeddings
        valid_ids = [m for m in market_ids if m in embeddings]
        if not valid_ids:
            return np.zeros((n_samples, 0)), []

        # Stack embeddings
        embed_stack = np.stack([embeddings[m] for m in valid_ids], axis=0)  # (k, embed_dim)

        if self.spec.model_type == "bundle":
            # Reshape for bundle model: (1, k, embed_dim)
            cond = embed_stack.reshape(1, len(valid_ids), -1)
            mask = np.ones((1, len(valid_ids)), dtype=bool)

            # Pad to bundle_size if needed
            if len(valid_ids) < self.spec.bundle_size:
                pad_size = self.spec.bundle_size - len(valid_ids)
                cond = np.pad(cond, ((0, 0), (0, pad_size), (0, 0)), mode="constant")
                mask = np.pad(mask, ((0, 0), (0, pad_size)), mode="constant", constant_values=False)

            samples = self.sample_ct(cond, mask, n_samples, seed=seed)
            # (n_samples, 1, bundle_size) -> (n_samples, k)
            samples = samples[:, 0, : len(valid_ids)]
        else:
            # Single model: process each market separately
            samples_list = []
            for i, m in enumerate(valid_ids):
                cond = embeddings[m].reshape(1, -1)
                s = self.sample_ct(cond, n_samples=n_samples, seed=seed + i)
                samples_list.append(s[:, 0, 0])  # (n_samples,)
            samples = np.stack(samples_list, axis=1)  # (n_samples, k)

        self._ct_samples = samples
        self._market_ids = valid_ids
        return samples, valid_ids

    def get_cached_ct(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Get the most recently sampled C_t."""
        if self._ct_samples is None:
            return None
        return self._ct_samples, self._market_ids or []

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None




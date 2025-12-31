"""
Tests for the multi-model C_t loader.

Tests cover:
- CtMode and CtModel enums
- ModelLoaderConfig validation
- UnifiedModelLoader with mocked models
- Single mode (each model type)
- Union mode (combining samples)
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from backtest.model_loader import (
    CtMode,
    CtModel,
    ModelLoaderConfig,
    UnifiedModelLoader,
    ARDiffusionLoader,
    RLCRLoader,
    BundleDiffusionLoader,
)


class TestCtModeEnum:
    """Test CtMode enum."""
    
    def test_single_mode(self):
        assert CtMode.SINGLE.value == "single"
    
    def test_union_mode(self):
        assert CtMode.UNION.value == "union"
    
    def test_from_string(self):
        assert CtMode("single") == CtMode.SINGLE
        assert CtMode("union") == CtMode.UNION


class TestCtModelEnum:
    """Test CtModel enum."""
    
    def test_all_model_types(self):
        assert CtModel.AR_DIFFUSION.value == "ar_diffusion"
        assert CtModel.RLCR.value == "rlcr"
        assert CtModel.BUNDLE.value == "bundle"
        assert CtModel.LEGACY.value == "legacy"


class TestModelLoaderConfig:
    """Test ModelLoaderConfig dataclass."""
    
    def test_default_values(self):
        cfg = ModelLoaderConfig()
        assert cfg.ct_mode == CtMode.SINGLE
        assert cfg.ct_model == CtModel.LEGACY
        assert cfg.ar_diffusion_samples == 16
        assert cfg.rlcr_K == 5
        assert cfg.bundle_samples == 16
    
    def test_path_conversion(self):
        cfg = ModelLoaderConfig(
            ar_diffusion_path="/some/path",
            bundle_diffusion_path="/another/path",
        )
        assert isinstance(cfg.ar_diffusion_path, Path)
        assert isinstance(cfg.bundle_diffusion_path, Path)
    
    def test_enabled_models_default(self):
        cfg = ModelLoaderConfig()
        assert "ar_diffusion" in cfg.enabled_models
        assert "rlcr" in cfg.enabled_models
        assert "bundle" in cfg.enabled_models


class TestUnifiedModelLoader:
    """Test UnifiedModelLoader with mocked models."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock sentence transformer."""
        embedder = Mock()
        embedder.encode = Mock(return_value=np.random.randn(5, 384).astype(np.float32))
        return embedder
    
    @pytest.fixture
    def basic_config(self):
        """Create basic config for testing."""
        return ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.RLCR,
            rlcr_model_path="mock/rlcr/path",
            ar_diffusion_samples=4,
            rlcr_K=3,
            bundle_samples=4,
        )
    
    def test_initialization(self, basic_config):
        loader = UnifiedModelLoader(basic_config)
        assert loader.cfg == basic_config
        assert loader._ar_diffusion is None
        assert loader._rlcr is None
        assert loader._bundle is None
    
    def test_get_model_info_empty(self, basic_config):
        loader = UnifiedModelLoader(basic_config)
        info = loader.get_model_info()
        assert info["ct_mode"] == "single"
        assert info["ct_model"] == "rlcr"
    
    def test_compute_embeddings(self, basic_config):
        """Test that embeddings are computed via embedder."""
        loader = UnifiedModelLoader(basic_config)
        
        # Mock the embedder directly
        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.random.randn(3, 384)
        loader._embedder = mock_embedder
        
        texts = ["question 1", "question 2", "question 3"]
        embeddings = loader._compute_embeddings(texts)
        
        assert embeddings.shape == (3, 384)
        mock_embedder.encode.assert_called_once()


class TestSingleModeSampling:
    """Test single mode sampling for each model type."""
    
    def test_single_mode_rlcr_requires_model(self):
        """RLCR model must be configured when ct_model=rlcr."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.RLCR,
            rlcr_model_path=None,  # Not configured
        )
        loader = UnifiedModelLoader(cfg)
        
        with pytest.raises(ValueError, match="RLCR model not configured"):
            loader._sample_single(
                texts=["test"],
                embeddings=np.random.randn(1, 384).astype(np.float32),
            )
    
    def test_single_mode_ar_diffusion_requires_model(self):
        """AR+Diffusion model must be configured when ct_model=ar_diffusion."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.AR_DIFFUSION,
            ar_diffusion_path=None,  # Not configured
        )
        loader = UnifiedModelLoader(cfg)
        
        with pytest.raises(ValueError, match="AR\\+Diffusion model not configured"):
            loader._sample_single(
                texts=["test"],
                embeddings=np.random.randn(1, 384).astype(np.float32),
            )
    
    def test_single_mode_bundle_requires_model(self):
        """Bundle model must be configured when ct_model=bundle."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.BUNDLE,
            bundle_diffusion_path=None,  # Not configured
        )
        loader = UnifiedModelLoader(cfg)
        
        with pytest.raises(ValueError, match="Bundle model not configured"):
            loader._sample_single(
                texts=["test"],
                embeddings=np.random.randn(1, 384).astype(np.float32),
            )
    
    def test_legacy_mode_raises_error(self):
        """Legacy mode should not use UnifiedModelLoader."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.LEGACY,
        )
        loader = UnifiedModelLoader(cfg)
        
        with pytest.raises(ValueError, match="Legacy mode"):
            loader._sample_single(
                texts=["test"],
                embeddings=np.random.randn(1, 384).astype(np.float32),
            )


class TestUnionModeSampling:
    """Test union mode sampling."""
    
    def test_union_mode_empty_when_no_models(self):
        """Union mode returns zeros when no models are enabled."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.UNION,
            enabled_models=(),  # No models enabled
        )
        loader = UnifiedModelLoader(cfg)
        
        samples = loader._sample_union(
            texts=["test1", "test2"],
            embeddings=np.random.randn(2, 384).astype(np.float32),
        )
        
        # Should return zeros
        assert samples.shape == (1, 2)
        assert np.allclose(samples, 0)
    
    def test_union_mode_combines_samples(self):
        """Union mode concatenates samples from all enabled models."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.UNION,
            rlcr_model_path="mock/path",
            ar_diffusion_path=Path("mock/ar_path"),
            enabled_models=("rlcr", "ar_diffusion"),
            rlcr_K=3,
            ar_diffusion_samples=5,
        )
        loader = UnifiedModelLoader(cfg)
        
        # Mock the individual samplers
        loader._sample_rlcr = Mock(return_value=np.random.randn(3, 2).astype(np.float32))
        loader._sample_ar_diffusion = Mock(return_value=np.random.randn(5, 2).astype(np.float32))
        
        samples = loader._sample_union(
            texts=["test1", "test2"],
            embeddings=np.random.randn(2, 384).astype(np.float32),
        )
        
        # Should have 3 + 5 = 8 samples
        assert samples.shape == (8, 2)


class TestSampleCt:
    """Test the main sample_ct method."""
    
    def test_sample_ct_filters_markets(self):
        """sample_ct should filter to markets with texts."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.SINGLE,
            ct_model=CtModel.RLCR,
            rlcr_model_path="mock/path",
        )
        loader = UnifiedModelLoader(cfg)
        
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.random.randn(2, 384)
        loader._embedder = mock_embedder
        
        # Mock RLCR sampler
        loader._sample_rlcr = Mock(return_value=np.random.randn(5, 2).astype(np.float32))
        
        samples, valid_ids = loader.sample_ct(
            market_ids=["m1", "m2", "m3"],
            texts={"m1": "question 1", "m2": "question 2"},  # m3 has no text
        )
        
        assert len(valid_ids) == 2
        assert "m1" in valid_ids
        assert "m2" in valid_ids
        assert "m3" not in valid_ids
    
    def test_sample_ct_empty_texts(self):
        """sample_ct should return empty when no texts provided."""
        cfg = ModelLoaderConfig()
        loader = UnifiedModelLoader(cfg)
        
        samples, valid_ids = loader.sample_ct(
            market_ids=["m1", "m2"],
            texts={},  # No texts
        )
        
        assert samples.shape == (1, 0)
        assert valid_ids == []


class TestARDiffusionLoader:
    """Test ARDiffusionLoader."""
    
    def test_initialization(self, tmp_path):
        loader = ARDiffusionLoader(
            checkpoint_path=tmp_path,
            ar_model="test/model",
            ar_K=5,
        )
        assert loader.checkpoint_path == tmp_path
        assert loader.ar_K == 5
        assert not loader.is_loaded()


class TestRLCRLoader:
    """Test RLCRLoader."""
    
    def test_initialization(self):
        loader = RLCRLoader(
            model_path="test/model",
            base_model="Qwen/Qwen3-14B",
            load_in_4bit=True,
        )
        assert loader.model_path == "test/model"
        assert loader.load_in_4bit
        assert not loader.is_loaded()


class TestBundleDiffusionLoader:
    """Test BundleDiffusionLoader."""
    
    def test_initialization(self, tmp_path):
        loader = BundleDiffusionLoader(
            checkpoint_path=tmp_path,
            bundle_size=8,
            embed_dim=4096,
        )
        assert loader.checkpoint_path == tmp_path
        assert loader.bundle_size == 8
        assert not loader.is_loaded()


# Integration test (requires mocking heavy dependencies)
class TestIntegration:
    """Integration tests with mocked dependencies."""
    
    def test_full_pipeline_union_mode(self):
        """Test full pipeline in union mode with mocked models."""
        cfg = ModelLoaderConfig(
            ct_mode=CtMode.UNION,
            rlcr_model_path="mock/rlcr",
            enabled_models=("rlcr",),
            rlcr_K=5,
        )
        loader = UnifiedModelLoader(cfg)
        
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.encode.return_value = np.random.randn(3, 384)
        loader._embedder = mock_embedder
        
        # Mock RLCR sampler
        mock_samples = np.clip(np.random.randn(5, 3) * 0.2 + 0.5, 0.01, 0.99)
        loader._sample_rlcr = Mock(return_value=mock_samples.astype(np.float32))
        
        samples, valid_ids = loader.sample_ct(
            market_ids=["m1", "m2", "m3"],
            texts={"m1": "q1", "m2": "q2", "m3": "q3"},
        )
        
        assert samples.shape == (5, 3)
        assert len(valid_ids) == 3
        # Samples should be probabilities in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

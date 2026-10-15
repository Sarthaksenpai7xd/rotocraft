"""Unit tests for Rotocraft model registry and ModelManager."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from rotocraft.models import (
    AVAILABLE_MODELS,
    ModelInfo,
    ModelManager,
    ModelNotFoundError,
    get_device,
    resolve_model_name,
)


class TestModels:
    """Test suite for model management, device detection, and cache storage."""

    def test_registered_models_exist(self) -> None:
        """Verify all 4 required model variants are registered."""
        required = [
            "sam2_hiera_tiny",
            "sam2_hiera_small",
            "sam2_hiera_base_plus",
            "sam2_hiera_large",
        ]
        for name in required:
            assert name in AVAILABLE_MODELS
            info = AVAILABLE_MODELS[name]
            assert isinstance(info, ModelInfo)
            assert info.checkpoint_url.startswith("https://")
            assert info.checkpoint_filename.endswith(".pt")
            assert info.config_filename.endswith(".yaml")
            assert info.size_mb > 0

    def test_alias_resolution(self) -> None:
        """Test alias mappings (e.g. 'tiny' -> 'sam2_hiera_tiny')."""
        assert resolve_model_name("tiny") == "sam2_hiera_tiny"
        assert resolve_model_name("t") == "sam2_hiera_tiny"
        assert resolve_model_name("small") == "sam2_hiera_small"
        assert resolve_model_name("s") == "sam2_hiera_small"
        assert resolve_model_name("base") == "sam2_hiera_base_plus"
        assert resolve_model_name("base_plus") == "sam2_hiera_base_plus"
        assert resolve_model_name("b+") == "sam2_hiera_base_plus"
        assert resolve_model_name("large") == "sam2_hiera_large"
        assert resolve_model_name("l") == "sam2_hiera_large"

        with pytest.raises(ModelNotFoundError):
            resolve_model_name("non_existent_model_xyz")

    def test_get_device(self) -> None:
        """Verify device detection returns valid device identifier."""
        dev = get_device()
        assert dev in ("cuda", "mps", "cpu")

        # Explicit CPU request should always return cpu
        assert get_device(preferred="cpu") == "cpu"

    def test_model_manager_list_models(self) -> None:
        """Verify ModelManager lists all models with status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelManager(cache_dir=tmp_dir)
            models = manager.list_models()
            assert len(models) == 4
            for m in models:
                assert "name" in m
                assert "downloaded" in m
                assert m["downloaded"] is False
                assert m["path"] is None

    def test_model_manager_cache_detection(self) -> None:
        """Verify detection when model file is present in cache."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_p = Path(tmp_dir)
            manager = ModelManager(cache_dir=cache_p)

            # Create mock weights file for tiny
            fake_ckpt = cache_p / "sam2_hiera_tiny.pt"
            fake_ckpt.write_bytes(b"dummy_weights_data_1234567890")

            assert manager.is_downloaded("sam2_hiera_tiny") is True
            assert manager.is_downloaded("sam2_hiera_small") is False

            # get_model_path with download=False should return existing path
            found_path = manager.get_model_path("tiny", download=False)
            assert found_path == fake_ckpt

            # delete_model should remove file
            assert manager.delete_model("tiny") is True
            assert manager.is_downloaded("tiny") is False
            assert not fake_ckpt.exists()

"""Model management and hardware device detection for Rotocraft.

Manages downloading, local caching, and verification of Meta's SAM 2 (Segment
Anything Model 2) checkpoints and configuration files, with automated device
selection (CUDA > MPS > CPU).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import urllib.request

from tqdm import tqdm

logger = logging.getLogger("rotocraft.models")


class RotocraftModelError(Exception):
    """Base exception for Rotocraft model management errors."""


class ModelNotFoundError(RotocraftModelError):
    """Raised when an unknown model name or variant is requested."""


class ModelDownloadError(RotocraftModelError):
    """Raised when downloading a model checkpoint fails."""


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """Metadata describing a SAM 2 checkpoint variant."""

    name: str
    display_name: str
    checkpoint_url: str
    checkpoint_filename: str
    config_filename: str
    size_mb: float
    description: str
    aliases: tuple[str, ...] = ()
    sha256: Optional[str] = None


AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "sam2_hiera_tiny": ModelInfo(
        name="sam2_hiera_tiny",
        display_name="SAM 2 Hiera Tiny",
        checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        checkpoint_filename="sam2_hiera_tiny.pt",
        config_filename="sam2_hiera_t.yaml",
        size_mb=156.4,
        description="Fastest, lightest SAM 2 model; ideal for real-time preview and low-VRAM GPUs.",
        aliases=("tiny", "t", "sam2_tiny"),
    ),
    "sam2_hiera_small": ModelInfo(
        name="sam2_hiera_small",
        display_name="SAM 2 Hiera Small",
        checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        checkpoint_filename="sam2_hiera_small.pt",
        config_filename="sam2_hiera_s.yaml",
        size_mb=184.8,
        description="Balanced speed and edge fidelity; default model for interactive rotoscoping.",
        aliases=("small", "s", "sam2_small"),
    ),
    "sam2_hiera_base_plus": ModelInfo(
        name="sam2_hiera_base_plus",
        display_name="SAM 2 Hiera Base+",
        checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        checkpoint_filename="sam2_hiera_base_plus.pt",
        config_filename="sam2_hiera_b+.yaml",
        size_mb=323.2,
        description="High precision segmentation with enhanced mask refinement for fine details.",
        aliases=("base_plus", "base", "b+", "b", "sam2_base", "sam2_base_plus"),
    ),
    "sam2_hiera_large": ModelInfo(
        name="sam2_hiera_large",
        display_name="SAM 2 Hiera Large",
        checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        checkpoint_filename="sam2_hiera_large.pt",
        config_filename="sam2_hiera_l.yaml",
        size_mb=897.6,
        description="Maximum accuracy model; recommended for final export and complex multi-subject tracking.",
        aliases=("large", "l", "sam2_large"),
    ),
}

# Lookup map for model aliases
_ALIAS_LOOKUP: dict[str, str] = {}
for _canonical, _info in AVAILABLE_MODELS.items():
    _ALIAS_LOOKUP[_canonical.lower()] = _canonical
    for _alias in _info.aliases:
        _ALIAS_LOOKUP[_alias.lower()] = _canonical


def resolve_model_name(name: str) -> str:
    """Resolve a user-provided model name or alias to canonical model identifier.

    Args:
        name: Model name or alias (e.g., 'small', 'tiny', 'sam2_hiera_large').

    Returns:
        Canonical model name.

    Raises:
        ModelNotFoundError: If the name does not match any registered model.
    """
    normalized = name.strip().lower()
    if normalized in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[normalized]
    valid = ", ".join(sorted(AVAILABLE_MODELS.keys()))
    raise ModelNotFoundError(
        f"Unknown model variant '{name}'. Available models: {valid}"
    )


def get_default_cache_dir() -> Path:
    """Return the default root directory for cached model weights.

    Can be customized via the ``ROTOCRAFT_CACHE_DIR`` environment variable.
    Defaults to ``~/.cache/rotocraft/models/`` (Windows: ``%USERPROFILE%/.cache/rotocraft/models``).
    """
    env_dir = os.environ.get("ROTOCRAFT_CACHE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
    else:
        path = Path.home() / ".cache" / "rotocraft" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(preferred: Optional[str] = None) -> str:
    """Detect the optimal hardware computation device (CUDA > MPS > CPU).

    Args:
        preferred: Optional explicit override ('cuda', 'mps', 'cpu', or device index like 'cuda:0').

    Returns:
        String device identifier ('cuda', 'mps', 'cpu', or 'cuda:0').
    """
    if preferred:
        pref = preferred.strip().lower()
        if pref.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return preferred
                logger.warning(
                    "CUDA requested but torch.cuda is not available. Falling back to CPU."
                )
            except ImportError:
                logger.warning("CUDA requested but PyTorch is not installed.")
            return "cpu"
        elif pref == "mps":
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                logger.warning(
                    "Apple MPS requested but not available on this system. Falling back to CPU."
                )
            except ImportError:
                logger.warning("MPS requested but PyTorch is not installed.")
            return "cpu"
        elif pref == "cpu":
            return "cpu"

    # Auto-detect best available device
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Using GPU acceleration: %s (CUDA)", device_name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon acceleration (MPS)")
            return "mps"
    except ImportError:
        pass

    # Check ONNX Runtime execution providers if PyTorch not present
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info("ONNX Runtime CUDA execution provider detected")
            return "cuda"
    except ImportError:
        pass

    logger.info("Hardware acceleration not detected; running on CPU")
    return "cpu"


class ModelManager:
    """Manages downloading, local caching, and registry of SAM 2 model weights."""

    def __init__(self, cache_dir: Optional[Path | str] = None) -> None:
        """Initialize ModelManager with optional custom cache directory.

        Args:
            cache_dir: Directory where model weights are stored. Defaults to
                ``~/.cache/rotocraft/models/``.
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = get_default_cache_dir()

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Retrieve model metadata by name or alias."""
        canonical = resolve_model_name(model_name)
        return AVAILABLE_MODELS[canonical]

    def is_downloaded(self, model_name: str) -> bool:
        """Check whether the checkpoint file for a model exists and is non-empty."""
        info = self.get_model_info(model_name)
        dest_path = self.cache_dir / info.checkpoint_filename
        return dest_path.is_file() and dest_path.stat().st_size > 0

    def get_model_path(self, model_name: str, download: bool = True) -> Path:
        """Return local path to model checkpoint, downloading it if necessary.

        Args:
            model_name: Model identifier or alias.
            download: If True, automatically download weights if missing.

        Returns:
            Path to the downloaded checkpoint file.

        Raises:
            ModelNotFoundError: If model_name is unrecognized.
            FileNotFoundError: If checkpoint is missing and download=False.
            ModelDownloadError: If download fails.
        """
        info = self.get_model_info(model_name)
        dest_path = self.cache_dir / info.checkpoint_filename

        if dest_path.is_file() and dest_path.stat().st_size > 0:
            return dest_path

        if not download:
            raise FileNotFoundError(
                f"Model checkpoint for '{info.name}' not found at {dest_path}. "
                f"Run `rotocraft models --download {info.name}` or set download=True."
            )

        return self.download_model(info.name)

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download model weights from Meta's CDN with atomic file caching and progress bar.

        Args:
            model_name: Canonical model name or alias.
            force: Re-download even if already present locally.

        Returns:
            Path to downloaded checkpoint.

        Raises:
            ModelDownloadError: If download fails or connection drops.
        """
        info = self.get_model_info(model_name)
        dest_path = self.cache_dir / info.checkpoint_filename
        temp_path = self.cache_dir / f"{info.checkpoint_filename}.tmp"

        if dest_path.is_file() and not force:
            logger.info("Model '%s' is already downloaded at %s", info.name, dest_path)
            return dest_path

        url = info.checkpoint_url
        logger.info("Downloading %s (~%.1f MB) from %s", info.display_name, info.size_mb, url)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Rotocraft-FilmCraft/0.1.0"},
            )
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                chunk_size = 1024 * 1024  # 1 MB chunks

                with open(temp_path, "wb") as f_out, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {info.display_name}",
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        pbar.update(len(chunk))

            # Atomic move from temp to final destination
            if temp_path.exists():
                shutil.move(str(temp_path), str(dest_path))

            logger.info("Successfully downloaded %s to %s", info.name, dest_path)
            return dest_path

        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise ModelDownloadError(
                f"Failed to download model '{info.name}' from {url}: {e}"
            ) from e

    def verify_checksum(self, model_name: str) -> bool:
        """Verify sha256 checksum of downloaded model if defined.

        Returns True if matching or no checksum defined, False if mismatch.
        """
        info = self.get_model_info(model_name)
        dest_path = self.cache_dir / info.checkpoint_filename
        if not dest_path.is_file() or not info.sha256:
            return dest_path.is_file()

        sha256 = hashlib.sha256()
        with open(dest_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest().lower() == info.sha256.lower()

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models and their local download status."""
        results = []
        for name, info in AVAILABLE_MODELS.items():
            path = self.cache_dir / info.checkpoint_filename
            downloaded = path.is_file() and path.stat().st_size > 0
            size_bytes = path.stat().st_size if downloaded else 0
            results.append(
                {
                    "name": info.name,
                    "display_name": info.display_name,
                    "aliases": list(info.aliases),
                    "size_mb": info.size_mb,
                    "downloaded": downloaded,
                    "local_size_mb": round(size_bytes / (1024 * 1024), 2) if downloaded else 0.0,
                    "path": str(path) if downloaded else None,
                    "config_filename": info.config_filename,
                    "description": info.description,
                }
            )
        return results

    def delete_model(self, model_name: str) -> bool:
        """Remove a cached model checkpoint from disk.

        Returns True if a file was deleted, False if it was not present.
        """
        info = self.get_model_info(model_name)
        dest_path = self.cache_dir / info.checkpoint_filename
        if dest_path.is_file():
            dest_path.unlink()
            logger.info("Deleted cached model: %s", dest_path)
            return True
        return False

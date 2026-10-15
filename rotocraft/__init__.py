"""Rotocraft: AI-powered automatic rotoscoping and object masking tool for FilmCraft.

Rotocraft brings Meta's SAM 2 (Segment Anything Model 2) state-of-the-art vision
segmentation and video object tracking directly into the FilmCraft open-source
video editor ecosystem.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "FilmCraft Contributors"

from rotocraft.mask_export import (
    MaskExporter,
    MaskPath,
    MaskVertex,
    contours_to_bezier,
    masks_to_contours,
)
from rotocraft.models import (
    AVAILABLE_MODELS,
    ModelInfo,
    ModelManager,
    get_device,
)
from rotocraft.preview import (
    create_comparison,
    draw_box,
    draw_points,
    overlay_mask,
    save_preview,
)
from rotocraft.segmenter import (
    RotocraftSegmenter,
    SegmentationResult,
)
from rotocraft.video_io import (
    VideoReader,
    VideoWriter,
)

__all__ = [
    "__version__",
    # Model management
    "AVAILABLE_MODELS",
    "ModelInfo",
    "ModelManager",
    "get_device",
    # Core segmentation
    "RotocraftSegmenter",
    "SegmentationResult",
    # FilmCraft mask export
    "MaskExporter",
    "MaskPath",
    "MaskVertex",
    "contours_to_bezier",
    "masks_to_contours",
    # Video I/O
    "VideoReader",
    "VideoWriter",
    # Visualization & preview
    "create_comparison",
    "draw_box",
    "draw_points",
    "overlay_mask",
    "save_preview",
]

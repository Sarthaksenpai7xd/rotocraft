"""Unit tests for RotocraftSegmenter."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from rotocraft.segmenter import RotocraftSegmenter, SegmentationResult
from rotocraft.video_io import VideoWriter


def create_sample_image(width: int = 150, height: int = 150) -> np.ndarray:
    """Create a test image with a bright circle in the center."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (75, 75), 35, (220, 220, 220), -1)
    return img


class TestSegmenter:
    """Test suite for RotocraftSegmenter."""

    def test_segmenter_initialization(self) -> None:
        """Verify segmenter initializes properly."""
        segmenter = RotocraftSegmenter(model_name="small")
        assert segmenter.canonical_name == "sam2_hiera_small"
        assert segmenter.active_backend in ("sam2", "onnx", "fallback")

    def test_segment_frame_box_prompt(self) -> None:
        """Verify segment_frame with bounding box prompt."""
        img = create_sample_image()
        segmenter = RotocraftSegmenter(model_name="tiny")

        box = [40, 40, 110, 110]
        result = segmenter.segment_frame(img, boxes=box)

        assert isinstance(result, SegmentationResult)
        assert result.masks.ndim in (2, 3)
        primary = result.primary_mask
        assert primary.shape == (150, 150)
        assert primary.dtype == bool
        # Center of the circle should be masked
        assert primary[75, 75] is True or primary[75, 75] == 1

    def test_segment_frame_point_prompt(self) -> None:
        """Verify segment_frame with foreground and background point prompts."""
        img = create_sample_image()
        segmenter = RotocraftSegmenter(model_name="tiny")

        points = [[75, 75], [10, 10]]
        labels = [1, 0]  # center is fg, corner is bg
        result = segmenter.segment_frame(img, points=points, labels=labels)

        primary = result.primary_mask
        assert primary.shape == (150, 150)
        assert primary[75, 75] is True or primary[75, 75] == 1

    def test_auto_segment(self) -> None:
        """Verify auto_segment detects objects in scene."""
        img = create_sample_image()
        segmenter = RotocraftSegmenter(model_name="tiny")

        objects = segmenter.auto_segment(img, min_area=0.005)
        assert isinstance(objects, list)
        assert len(objects) >= 1
        first = objects[0]
        assert "segmentation" in first
        assert "area" in first
        assert "bbox" in first
        assert first["area"] > 50

    def test_segment_video_end_to_end(self) -> None:
        """Verify segment_video tracking across synthetic video frames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            vid_file = tmp_path / "test_track.mp4"
            out_dir = tmp_path / "masks_out"

            # Create test video with 6 frames
            with VideoWriter(vid_file, fps=24.0, width=120, height=100) as w:
                for i in range(6):
                    f = np.zeros((100, 120, 3), dtype=np.uint8)
                    cv2.circle(f, (40 + i * 4, 50), 20, (255, 255, 255), -1)
                    w.write_frame(f)

            segmenter = RotocraftSegmenter(model_name="tiny")
            prompt = {"points": [[40, 50]], "labels": [1]}

            results = segmenter.segment_video(
                video_path=vid_file,
                prompts=prompt,
                output_dir=out_dir,
                start_frame=0,
                end_frame=4,
                output_format="png",
            )

            assert len(results) == 5
            for f_idx in range(5):
                assert f_idx in results
                assert 1 in results[f_idx]
                mask = results[f_idx][1]
                assert mask.shape == (100, 120)

            # Check exported PNG sequence files
            png_files = list(out_dir.glob("*.png"))
            assert len(png_files) >= 5

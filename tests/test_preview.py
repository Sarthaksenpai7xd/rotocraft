"""Unit tests for Rotocraft preview and visualization functions."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from rotocraft.preview import (
    create_comparison,
    draw_box,
    draw_points,
    overlay_mask,
    save_preview,
)


class TestPreview:
    """Test suite for annotation and visualization utilities."""

    def test_overlay_mask(self) -> None:
        """Verify blending mask onto RGB image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        blended = overlay_mask(img, mask, color=(0, 255, 0), alpha=0.5, border=False)
        assert blended.shape == (100, 100, 3)
        assert blended.dtype == np.uint8

        # Background area outside mask should remain 0
        assert blended[10, 10, 1] == 0
        # Inside masked region, green channel should be blended (alpha * 255 ~= 127)
        assert blended[50, 50, 1] > 100

    def test_draw_points(self) -> None:
        """Verify drawing foreground and background prompt points."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = [[50, 50], [20, 20]]
        labels = [1, 0]  # One foreground, one background

        annotated = draw_points(img, pts, labels=labels, radius=4)
        assert annotated.shape == (100, 100, 3)
        # Point locations should not be all zeros
        assert annotated[50, 50].max() > 0
        assert annotated[20, 20].max() > 0

    def test_draw_box(self) -> None:
        """Verify drawing bounding box."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = [20, 20, 80, 80]
        annotated = draw_box(img, box, color=(255, 255, 0), thickness=2)
        assert annotated.shape == (100, 100, 3)
        assert annotated[20, 50, 0] > 0  # Along top edge

    def test_save_preview(self) -> None:
        """Verify generating and saving preview file to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "preview_test.png"
            img = np.full((120, 120, 3), 50, dtype=np.uint8)
            mask = np.zeros((120, 120), dtype=bool)
            mask[30:90, 30:90] = True

            res = save_preview(
                image=img,
                mask=mask,
                points=[[60, 60]],
                labels=[1],
                boxes=[[30, 30, 90, 90]],
                output_path=out_file,
            )
            assert res.shape == (120, 120, 3)
            assert out_file.is_file() and out_file.stat().st_size > 0

    def test_create_comparison(self) -> None:
        """Verify side-by-side and vertical comparison generation."""
        img1 = np.full((80, 100, 3), 40, dtype=np.uint8)
        img2 = np.full((80, 100, 3), 160, dtype=np.uint8)

        # Horizontal
        comp_h = create_comparison(img1, img2, orientation="horizontal", divider_thickness=4)
        assert comp_h.shape[0] == 80
        assert comp_h.shape[1] == 100 + 4 + 100

        # Vertical
        comp_v = create_comparison(img1, img2, orientation="vertical", divider_thickness=4)
        assert comp_v.shape[0] == 80 + 4 + 80
        assert comp_v.shape[1] == 100

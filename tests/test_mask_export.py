"""Unit tests for Rotocraft mask export and FilmCraft Bézier vector conversion."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from rotocraft.mask_export import (
    MaskExporter,
    MaskPath,
    MaskVertex,
    contours_to_bezier,
    masks_to_contours,
)


def create_test_rect_mask(
    height: int = 200,
    width: int = 200,
    rect: tuple[int, int, int, int] = (40, 40, 160, 160),
) -> np.ndarray:
    """Create a synthetic binary mask with a filled rectangle."""
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = True
    return mask


def create_test_circle_mask(
    height: int = 200,
    width: int = 200,
    center: tuple[int, int] = (100, 100),
    radius: int = 40,
) -> np.ndarray:
    """Create a synthetic binary mask with a filled circle."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask > 127


class TestMaskExport:
    """Test suite for mask extraction and FilmCraft Bézier conversion."""

    def test_contour_extraction_simple_rect(self) -> None:
        """Verify contour extraction on a known rectangular mask."""
        mask = create_test_rect_mask(200, 200, (40, 40, 160, 160))
        contours = masks_to_contours(mask, min_area=20.0, simplify_epsilon=1.0)

        assert len(contours) == 1, "Expected exactly one contour for single rectangle"
        pts = contours[0]
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert len(pts) >= 4, f"A rectangle should have at least 4 vertices, got {len(pts)}"

        # Check bounds
        xs = pts[:, 0]
        ys = pts[:, 1]
        assert np.isclose(xs.min(), 40, atol=2.0)
        assert np.isclose(ys.min(), 40, atol=2.0)
        assert np.isclose(xs.max(), 159, atol=2.0)
        assert np.isclose(ys.max(), 159, atol=2.0)

    def test_contour_extraction_circle(self) -> None:
        """Verify contour extraction on a circular mask."""
        mask = create_test_circle_mask(200, 200, center=(100, 100), radius=50)
        contours = masks_to_contours(mask, min_area=10.0, simplify_epsilon=2.0)

        assert len(contours) == 1
        pts = contours[0]
        # Circle should be approximated by a smooth polygon with multiple vertices
        assert len(pts) >= 8
        # Center of mass should be close to (100, 100)
        mean_x = np.mean(pts[:, 0])
        mean_y = np.mean(pts[:, 1])
        assert abs(mean_x - 100.0) < 3.0
        assert abs(mean_y - 100.0) < 3.0

    def test_contour_extraction_empty_mask(self) -> None:
        """Ensure empty mask yields empty contour list."""
        mask = np.zeros((100, 100), dtype=bool)
        contours = masks_to_contours(mask)
        assert contours == []

    def test_contours_to_bezier_closed(self) -> None:
        """Verify converted Bézier path is marked closed with valid vertex objects."""
        mask = create_test_circle_mask()
        contours = masks_to_contours(mask)
        paths = contours_to_bezier(contours, tension=1.0 / 6.0)

        assert len(paths) == 1
        path = paths[0]
        assert path.closed is True
        assert len(path.vertices) >= 6

        for v in path.vertices:
            assert isinstance(v, MaskVertex)
            assert len(v.p) == 2
            assert len(v.t_in) == 2
            assert len(v.t_out) == 2

    def test_contours_to_bezier_tangents(self) -> None:
        """Verify Catmull-Rom tangent direction: t_in == -t_out for smooth vertices."""
        mask = create_test_circle_mask(200, 200, (100, 100), 40)
        contours = masks_to_contours(mask, simplify_epsilon=2.0)
        paths = contours_to_bezier(contours, tension=1.0 / 6.0)

        path = paths[0]
        smooth_vertex_found = False
        for v in path.vertices:
            # If vertex is smooth (non-zero handles), t_in should equal -t_out
            if math_hypot(v.t_out[0], v.t_out[1]) > 0.01:
                smooth_vertex_found = True
                assert np.isclose(v.t_in[0], -v.t_out[0], atol=1e-3)
                assert np.isclose(v.t_in[1], -v.t_out[1], atol=1e-3)

        assert smooth_vertex_found, "Expected at least one smooth vertex on circular contour"

    def test_to_filmcraft_json_format(self) -> None:
        """Verify serialization matches FilmCraft MaskPath schema."""
        mask = create_test_rect_mask()
        json_data = MaskExporter.to_filmcraft_path(mask)

        assert isinstance(json_data, dict)
        assert "closed" in json_data
        assert json_data["closed"] is True
        assert "vertices" in json_data
        assert isinstance(json_data["vertices"], list)
        assert len(json_data["vertices"]) >= 3

        first_v = json_data["vertices"][0]
        assert "p" in first_v
        assert isinstance(first_v["p"], list) and len(first_v["p"]) == 2
        assert "in" in first_v
        assert isinstance(first_v["in"], list) and len(first_v["in"]) == 2
        assert "out" in first_v
        assert isinstance(first_v["out"], list) and len(first_v["out"]) == 2

    def test_to_filmcraft_command(self) -> None:
        """Verify FilmCraft engine command payload structure."""
        mask = create_test_circle_mask()
        clip_id = "test-clip-uuid-1234"
        cmd = MaskExporter.to_filmcraft_command(
            mask,
            clip_id=clip_id,
            effect_index="opacity",
            shape="bezier",
        )

        assert cmd["clip"] == clip_id
        assert cmd["effect"] == "opacity"
        assert cmd["shape"] == "bezier"
        assert "path" in cmd
        assert cmd["path"]["closed"] is True
        assert len(cmd["path"]["vertices"]) >= 3

    def test_png_sequence_export(self) -> None:
        """Verify exporting mask array to PNG file sequence."""
        mask1 = create_test_rect_mask()
        mask2 = create_test_circle_mask()

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_files = MaskExporter.to_png_sequence(
                [mask1, mask2],
                output_dir=tmp_dir,
                prefix="test_mask_",
            )
            assert len(out_files) == 2
            f1, f2 = out_files
            assert f1.name == "test_mask_000000.png"
            assert f2.name == "test_mask_000001.png"
            assert f1.is_file() and f1.stat().st_size > 0
            assert f2.is_file() and f2.stat().st_size > 0

            # Read back image and verify non-zero content
            read_back = cv2.imread(str(f1), cv2.IMREAD_GRAYSCALE)
            assert read_back is not None
            assert read_back.shape == (200, 200)
            assert (read_back > 128).sum() > 100

    def test_round_trip_raster_to_bezier_and_back(self) -> None:
        """Round trip verification: binary mask -> FilmCraft JSON -> rasterize -> compute IoU."""
        mask = create_test_circle_mask(height=300, width=300, center=(150, 150), radius=70)
        filmcraft_dict = MaskExporter.to_filmcraft_path(mask, simplify_epsilon=1.5)

        # Re-rasterize polygon defined by path vertices
        h, w = mask.shape
        rasterized = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([v["p"] for v in filmcraft_dict["vertices"]], dtype=np.int32)
        cv2.fillPoly(rasterized, [pts], 255)

        # Intersection over Union (IoU) calculation
        orig_bool = mask
        recon_bool = rasterized > 127
        intersection = np.logical_and(orig_bool, recon_bool).sum()
        union = np.logical_or(orig_bool, recon_bool).sum()
        iou = intersection / union

        assert iou > 0.92, f"Expected high IoU between mask and polygon approximation, got {iou:.3f}"


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(x, y))

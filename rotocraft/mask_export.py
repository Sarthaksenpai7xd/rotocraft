"""FilmCraft mask export and Bézier curve vector conversion utilities.

Converts SAM 2 raster segmentation masks into smooth cubic Bézier vector paths
compatible with FilmCraft's native MaskPath schema, engine command protocols
(`masks.add`, `masks.set`), and PNG alpha image sequences.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("rotocraft.mask_export")


@dataclasses.dataclass
class MaskVertex:
    """A Bézier path vertex compatible with FilmCraft's MaskVertex.

    Coordinates are in clip pixel space. Tangents `t_in` and `t_out` are relative
    vectors to anchor `p`. Zero vectors indicate a sharp corner.
    """

    p: tuple[float, float]
    t_in: tuple[float, float] = (0.0, 0.0)
    t_out: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize vertex to FilmCraft JSON vertex format."""
        return {
            "p": [round(self.p[0], 3), round(self.p[1], 3)],
            "in": [round(self.t_in[0], 3), round(self.t_in[1], 3)],
            "out": [round(self.t_out[0], 3), round(self.t_out[1], 3)],
        }


@dataclasses.dataclass
class MaskPath:
    """A closed Bézier mask path matching FilmCraft's MaskPath."""

    vertices: list[MaskVertex] = dataclasses.field(default_factory=list)
    closed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize path to FilmCraft JSON path representation."""
        return {
            "closed": self.closed,
            "vertices": [v.to_dict() for v in self.vertices],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert path to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def masks_to_contours(
    mask: np.ndarray,
    min_area: float = 16.0,
    simplify_epsilon: float = 1.5,
) -> list[np.ndarray]:
    """Extract polygon contour vertices from a binary or float segmentation mask.

    Uses OpenCV's contour tracing followed by Douglas-Peucker polygon approximation
    to produce clean, non-redundant vector boundaries.

    Args:
        mask: 2D numpy array (H, W) of boolean, uint8 (0..255), or float (0.0..1.0).
        min_area: Minimum area in pixels to filter out small noise blobs.
        simplify_epsilon: Douglas-Peucker approximation epsilon in pixels.
            Higher values yield fewer vertices; 0.0 disables simplification.

    Returns:
        List of 2D numpy arrays with shape (N, 2), where each row is an [x, y] vertex.
    """
    if mask is None or mask.size == 0:
        return []

    # Ensure mask is 2D uint8 with values 0 or 255
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask_2d = mask[0]
        elif mask.shape[-1] == 1:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    if mask_2d.dtype == bool:
        binary = (mask_2d.astype(np.uint8)) * 255
    elif np.issubdtype(mask_2d.dtype, np.floating):
        binary = (mask_2d > 0.5).astype(np.uint8) * 255
    else:
        binary = (mask_2d > 127).astype(np.uint8) * 255

    # Find external contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_L1,
    )

    result_contours: list[np.ndarray] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if simplify_epsilon > 0.0:
            approx = cv2.approxPolyDP(c, simplify_epsilon, closed=True)
        else:
            approx = c

        # Reshape from (N, 1, 2) to (N, 2)
        pts = approx.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            result_contours.append(pts)

    # Sort contours by area descending
    result_contours.sort(key=lambda pts: cv2.contourArea(pts.astype(np.int32)), reverse=True)
    return result_contours


def contours_to_bezier(
    contours: Union[np.ndarray, Sequence[np.ndarray]],
    tension: float = 1.0 / 6.0,
    corner_threshold_degrees: float = 145.0,
    max_tangent_ratio: float = 0.45,
) -> list[MaskPath]:
    """Convert polygonal contours to smooth cubic Bézier curves with tangent handles.

    Converts the polygon vertex ring into Catmull-Rom cubic Bézier splines matching
    FilmCraft's native `MaskVertex` anchor + relative handle structure.

    Args:
        contours: A single (N, 2) contour array or sequence of (N, 2) contour arrays.
        tension: Tangent scaling factor. Standard Catmull-Rom to cubic Bézier
            corresponds to 1/6 (~0.1667).
        corner_threshold_degrees: Turning angle in degrees above which a vertex
            is treated as a sharp corner (zero tangent handles).
        max_tangent_ratio: Maximum handle length relative to adjacent edge lengths,
            preventing self-intersecting loops on sharp curves.

    Returns:
        List of `MaskPath` objects.
    """
    if isinstance(contours, np.ndarray) and contours.ndim == 2:
        contour_list = [contours]
    else:
        contour_list = list(contours)

    paths: list[MaskPath] = []
    corner_rad_thresh = math.radians(corner_threshold_degrees)

    for pts in contour_list:
        n = len(pts)
        if n < 3:
            continue

        vertices: list[MaskVertex] = []
        for i in range(n):
            p = pts[i]
            prev_p = pts[(i - 1 + n) % n]
            next_p = pts[(i + 1) % n]

            # Ingoing edge vector: prev_p -> p
            v_in = p - prev_p
            len_in = math.hypot(v_in[0], v_in[1])

            # Outgoing edge vector: p -> next_p
            v_out = next_p - p
            len_out = math.hypot(v_out[0], v_out[1])

            # Check for degenerate vertices
            if len_in < 1e-6 or len_out < 1e-6:
                vertices.append(MaskVertex(p=(float(p[0]), float(p[1]))))
                continue

            # Calculate turn angle between incoming and outgoing directions
            # Unit vectors:
            u_in = v_in / len_in
            u_out = v_out / len_out
            cos_angle = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
            turn_angle = math.acos(cos_angle)

            # If sharp corner, keep zero handles
            if turn_angle > corner_rad_thresh:
                vertices.append(
                    MaskVertex(
                        p=(float(p[0]), float(p[1])),
                        t_in=(0.0, 0.0),
                        t_out=(0.0, 0.0),
                    )
                )
                continue

            # Catmull-Rom tangent direction at P_i is parallel to (next_p - prev_p)
            chord = next_p - prev_p
            t_vec = chord * tension

            # Clamp tangent length so handles do not overshoot adjacent segments
            t_len = math.hypot(t_vec[0], t_vec[1])
            max_len = min(len_in, len_out) * max_tangent_ratio
            if t_len > max_len and t_len > 0:
                t_vec = t_vec * (max_len / t_len)

            # Outgoing handle points forward along curve; incoming handle points backward
            t_out = (float(t_vec[0]), float(t_vec[1]))
            t_in = (-float(t_vec[0]), -float(t_vec[1]))

            vertices.append(
                MaskVertex(
                    p=(float(p[0]), float(p[1])),
                    t_in=t_in,
                    t_out=t_out,
                )
            )

        paths.append(MaskPath(vertices=vertices, closed=True))

    return paths


class MaskExporter:
    """High-level exporter connecting Rotocraft segmentation to FilmCraft and file formats."""

    @staticmethod
    def masks_to_contours(
        mask: np.ndarray,
        min_area: float = 16.0,
        simplify_epsilon: float = 1.5,
    ) -> list[np.ndarray]:
        """Extract polygonal contours from a mask."""
        return masks_to_contours(mask, min_area=min_area, simplify_epsilon=simplify_epsilon)

    @staticmethod
    def contours_to_bezier(
        contours: Union[np.ndarray, Sequence[np.ndarray]],
        tension: float = 1.0 / 6.0,
        corner_threshold_degrees: float = 145.0,
    ) -> list[MaskPath]:
        """Convert contours into Bézier paths."""
        return contours_to_bezier(
            contours,
            tension=tension,
            corner_threshold_degrees=corner_threshold_degrees,
        )

    @classmethod
    def to_filmcraft_path(
        cls,
        mask: np.ndarray,
        simplify_epsilon: float = 1.5,
        min_area: float = 16.0,
        tension: float = 1.0 / 6.0,
    ) -> dict[str, Any]:
        """Convert binary mask into a single primary FilmCraft MaskPath dictionary.

        If multiple contours are detected, uses the largest contour.

        Returns:
            Dictionary matching FilmCraft's native MaskPath schema:
            `{"closed": true, "vertices": [{"p": [x,y], "in": [dx,dy], "out": [dx,dy]}]}`.
        """
        contours = cls.masks_to_contours(mask, min_area=min_area, simplify_epsilon=simplify_epsilon)
        if not contours:
            return {"closed": True, "vertices": []}

        # Take largest contour as main mask path
        paths = cls.contours_to_bezier(contours[:1], tension=tension)
        return paths[0].to_dict() if paths else {"closed": True, "vertices": []}

    @classmethod
    def to_filmcraft_json(
        cls,
        masks: Union[np.ndarray, Sequence[np.ndarray], dict[int, np.ndarray]],
        frame_idx: Optional[int] = None,
        simplify_epsilon: float = 1.5,
        min_area: float = 16.0,
        tension: float = 1.0 / 6.0,
    ) -> dict[str, Any]:
        """Convert mask(s) to FilmCraft JSON mask format with frame metadata.

        Args:
            masks: Single 2D mask, list of masks, or dict of {frame_idx: mask}.
            frame_idx: Optional frame index for single mask export.
            simplify_epsilon: Polygon simplification tolerance.
            min_area: Minimum contour area.
            tension: Bézier tension factor.

        Returns:
            Structured dictionary with MaskPath or frame keyframes.
        """
        if isinstance(masks, dict):
            # Animated sequence of masks keyed by frame index
            frames_data = {}
            for f_idx, m in sorted(masks.items()):
                frames_data[str(f_idx)] = cls.to_filmcraft_path(
                    m,
                    simplify_epsilon=simplify_epsilon,
                    min_area=min_area,
                    tension=tension,
                )
            return {
                "format": "filmcraft_animated_mask",
                "keyframes": frames_data,
            }

        if isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                return {"closed": True, "vertices": []}
            if len(masks) == 1:
                path_dict = cls.to_filmcraft_path(
                    masks[0],
                    simplify_epsilon=simplify_epsilon,
                    min_area=min_area,
                    tension=tension,
                )
                if frame_idx is not None:
                    path_dict["frame"] = frame_idx
                return path_dict

            # Multiple masks in single frame
            all_paths = []
            for m in masks:
                all_paths.append(
                    cls.to_filmcraft_path(
                        m,
                        simplify_epsilon=simplify_epsilon,
                        min_area=min_area,
                        tension=tension,
                    )
                )
            res = {"masks": all_paths}
            if frame_idx is not None:
                res["frame"] = frame_idx
            return res

        # Single mask array
        path_dict = cls.to_filmcraft_path(
            masks,
            simplify_epsilon=simplify_epsilon,
            min_area=min_area,
            tension=tension,
        )
        if frame_idx is not None:
            path_dict["frame"] = frame_idx
        return path_dict

    @classmethod
    def to_filmcraft_command(
        cls,
        masks: Union[np.ndarray, dict[str, Any]],
        clip_id: Optional[str] = None,
        effect_index: Union[int, str] = "opacity",
        shape: str = "bezier",
        simplify_epsilon: float = 1.5,
        min_area: float = 16.0,
    ) -> dict[str, Any]:
        """Generate a FilmCraft engine command payload (`masks.add`).

        The returned dictionary can be sent directly over the FilmCraft MCP or
        WebSocket control server to instantiate an effect mask.

        Schema:
        ```json
        {
            "clip": "<id>",
            "effect": "opacity",
            "shape": "bezier",
            "path": {"closed": true, "vertices": [{"p": [x, y], "in": [0,0], "out": [0,0]}]}
        }
        ```
        """
        if isinstance(masks, dict) and "vertices" in masks:
            path_obj = masks
        else:
            path_obj = cls.to_filmcraft_path(
                masks,
                simplify_epsilon=simplify_epsilon,
                min_area=min_area,
            )

        cmd: dict[str, Any] = {
            "effect": effect_index,
            "shape": shape,
            "path": path_obj,
        }
        if clip_id is not None:
            cmd["clip"] = clip_id
        return cmd

    @staticmethod
    def to_png_sequence(
        masks: Union[np.ndarray, Sequence[np.ndarray], dict[int, np.ndarray]],
        output_dir: Union[str, Path],
        prefix: str = "mask_",
        format: str = "png",
        start_index: int = 0,
    ) -> list[Path]:
        """Export mask(s) as an 8-bit PNG alpha image sequence.

        Args:
            masks: Single 2D mask, 3D array of masks (T, H, W), list of masks,
                or dict of {frame_idx: mask}.
            output_dir: Target directory to write PNG files.
            prefix: Filename prefix (e.g., 'mask_').
            format: Image format extension ('png').
            start_index: Starting frame index if exporting a list/array.

        Returns:
            List of generated file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written_paths: list[Path] = []

        def save_single(mask_arr: np.ndarray, index: int) -> Path:
            if mask_arr.dtype == bool:
                img_data = (mask_arr.astype(np.uint8)) * 255
            elif np.issubdtype(mask_arr.dtype, np.floating):
                img_data = np.clip(mask_arr * 255.0, 0, 255).astype(np.uint8)
            else:
                img_data = mask_arr.astype(np.uint8)

            file_path = out_dir / f"{prefix}{index:06d}.{format}"
            cv2.imwrite(str(file_path), img_data)
            return file_path

        if isinstance(masks, dict):
            for f_idx, m in sorted(masks.items()):
                written_paths.append(save_single(m, f_idx))
        elif isinstance(masks, (list, tuple)):
            for i, m in enumerate(masks):
                written_paths.append(save_single(m, start_index + i))
        elif isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                written_paths.append(save_single(masks, start_index))
            elif masks.ndim == 3:
                for i in range(masks.shape[0]):
                    written_paths.append(save_single(masks[i], start_index + i))
            else:
                raise ValueError(f"Unsupported mask array dimension: {masks.ndim}")

        logger.info("Exported %d PNG mask frames to %s", len(written_paths), out_dir)
        return written_paths

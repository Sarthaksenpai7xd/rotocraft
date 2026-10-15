"""Preview, visualization, and annotation utilities for Rotocraft.

Generates composited mask overlays, prompt point/box annotations, side-by-side
before/after comparisons, and exportable preview images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("rotocraft.preview")


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.5,
    border: bool = True,
    border_color: tuple[int, int, int] = (255, 255, 255),
    border_thickness: int = 2,
) -> np.ndarray:
    """Blend a segmentation mask onto an RGB image with transparency and border outline.

    Args:
        image: Base image array (H, W, 3) in RGB order.
        mask: 2D mask array (H, W) or 3D (1, H, W). Values can be bool, float, or uint8.
        color: RGB color tuple for the mask overlay tint (default: vibrant emerald).
        alpha: Blending transparency (0.0: invisible, 1.0: solid color).
        border: If True, draws an outline around the mask boundary.
        border_color: RGB color tuple for the boundary stroke.
        border_thickness: Stroke thickness in pixels.

    Returns:
        Composited RGB uint8 image array (H, W, 3).
    """
    img = np.array(image, copy=True)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {img.shape}")

    # Standardize mask to 2D boolean
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            m = mask[0]
        elif mask.shape[-1] == 1:
            m = mask[:, :, 0]
        else:
            m = mask[0]
    else:
        m = mask

    if m.dtype == bool:
        bool_mask = m
    elif np.issubdtype(m.dtype, np.floating):
        bool_mask = m > 0.5
    else:
        bool_mask = m > 127

    if not bool_mask.any():
        return img

    h, w = img.shape[:2]
    if bool_mask.shape[:2] != (h, w):
        bool_mask = cv2.resize(
            bool_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Apply color overlay where mask is True
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:] = color
    blended = np.where(
        bool_mask[:, :, None],
        cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0),
        img,
    )

    # Draw border stroke if requested
    if border:
        mask_uint8 = (bool_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(blended, contours, -1, border_color, border_thickness, cv2.LINE_AA)

    return blended


def draw_points(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    labels: Optional[Sequence[int]] = None,
    radius: int = 6,
    fg_color: tuple[int, int, int] = (0, 255, 0),
    bg_color: tuple[int, int, int] = (255, 40, 40),
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw interactive foreground and background prompt points on an RGB image.

    Args:
        image: Base RGB image array (H, W, 3).
        points: Sequence of [x, y] coordinates.
        labels: Sequence of labels (1 = foreground, 0 = background).
            If None, all points are drawn as foreground.
        radius: Dot radius in pixels.
        fg_color: RGB color for positive foreground points (default green).
        bg_color: RGB color for negative background points (default red).
        border_color: RGB outline ring color.

    Returns:
        Annotated RGB image array.
    """
    annotated = np.array(image, copy=True)
    if points is None or len(points) == 0:
        return annotated

    pts = np.asarray(points, dtype=np.float64)
    if labels is None:
        lbls = np.ones(len(pts), dtype=np.int32)
    else:
        lbls = np.asarray(labels, dtype=np.int32)

    for pt, lbl in zip(pts, lbls):
        x, y = int(round(pt[0])), int(round(pt[1]))
        color = fg_color if lbl == 1 else bg_color

        # Outer ring
        cv2.circle(annotated, (x, y), radius + 2, border_color, -1, cv2.LINE_AA)
        # Inner dot
        cv2.circle(annotated, (x, y), radius, color, -1, cv2.LINE_AA)

        if lbl == 0:
            # Draw an 'x' cross for background negative points
            arm = max(2, radius - 2)
            cv2.line(annotated, (x - arm, y - arm), (x + arm, y + arm), border_color, 2, cv2.LINE_AA)
            cv2.line(annotated, (x - arm, y + arm), (x + arm, y - arm), border_color, 2, cv2.LINE_AA)
        else:
            # Center white dot for positive points
            cv2.circle(annotated, (x, y), 2, border_color, -1, cv2.LINE_AA)

    return annotated


def draw_box(
    image: np.ndarray,
    box: Sequence[float],
    color: tuple[int, int, int] = (255, 215, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw a bounding box prompt on an RGB image.

    Args:
        image: Base RGB image array (H, W, 3).
        box: Coordinates [x1, y1, x2, y2].
        color: RGB line color (default gold).
        thickness: Stroke thickness in pixels.
        label: Optional text label displayed above the box.

    Returns:
        Annotated RGB image array.
    """
    annotated = np.array(image, copy=True)
    if box is None or len(box) < 4:
        return annotated

    x1, y1, x2, y2 = [int(round(coord)) for coord in box[:4]]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Corner brackets for cinematic styling
    corner_len = min(16, max(6, (x2 - x1) // 6), max(6, (y2 - y1) // 6))
    bright_white = (255, 255, 255)
    # Top-left
    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), bright_white, thickness + 1, cv2.LINE_AA)
    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), bright_white, thickness + 1, cv2.LINE_AA)
    # Bottom-right
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), bright_white, thickness + 1, cv2.LINE_AA)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), bright_white, thickness + 1, cv2.LINE_AA)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
        tw, th = text_size
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(th, y1 - 4)),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def save_preview(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    points: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence[int]] = None,
    boxes: Optional[Union[Sequence[float], Sequence[Sequence[float]]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    color: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.45,
) -> np.ndarray:
    """Create and optionally save a complete annotated preview visualization.

    Combines mask overlay, prompt points, and prompt boxes.

    Args:
        image: Base RGB image (H, W, 3).
        mask: Optional mask array.
        points: Optional prompt point coordinates.
        labels: Optional prompt point labels (1=fg, 0=bg).
        boxes: Optional box [x1, y1, x2, y2] or sequence of boxes.
        output_path: If specified, writes the preview image to disk.
        color: Mask overlay RGB color.
        alpha: Mask overlay transparency.

    Returns:
        Final composited RGB uint8 array.
    """
    result = np.array(image, copy=True)

    if mask is not None:
        result = overlay_mask(result, mask, color=color, alpha=alpha)

    if boxes is not None:
        b_arr = np.asarray(boxes)
        if b_arr.ndim == 1 and len(b_arr) >= 4:
            result = draw_box(result, b_arr[:4])
        elif b_arr.ndim == 2:
            for b in b_arr:
                if len(b) >= 4:
                    result = draw_box(result, b[:4])

    if points is not None and len(points) > 0:
        result = draw_points(result, points, labels=labels)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pil_img = Image.fromarray(result)
        pil_img.save(str(out_p))
        logger.info("Saved segmentation preview to %s", out_p)

    return result


def create_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    orientation: str = "horizontal",
    divider_thickness: int = 4,
    divider_color: tuple[int, int, int] = (255, 255, 255),
    add_labels: bool = True,
) -> np.ndarray:
    """Generate a side-by-side or stacked split comparison image.

    Args:
        original: Original RGB image.
        masked: Segmented/composited RGB image.
        output_path: If given, writes comparison to disk.
        orientation: 'horizontal' (side-by-side) or 'vertical' (stacked).
        divider_thickness: Thickness of separator bar in pixels.
        divider_color: RGB color of separator bar.
        add_labels: Whether to render "ORIGINAL" and "ROTOCRAFT MASK" label banners.

    Returns:
        Side-by-side comparison RGB image.
    """
    img1 = np.array(original, copy=True)
    img2 = np.array(masked, copy=True)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Harmonize dimensions if needed
    if orientation == "horizontal":
        target_h = max(h1, h2)
        if h1 != target_h:
            img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
        if h2 != target_h:
            img2 = cv2.resize(img2, (int(w2 * target_h / h2), target_h))

        divider = np.zeros((target_h, divider_thickness, 3), dtype=np.uint8)
        divider[:] = divider_color
        comparison = np.hstack([img1, divider, img2])
    else:
        target_w = max(w1, w2)
        if w1 != target_w:
            img1 = cv2.resize(img1, (target_w, int(h1 * target_w / w1)))
        if w2 != target_w:
            img2 = cv2.resize(img2, (target_w, int(h2 * target_w / w2)))

        divider = np.zeros((divider_thickness, target_w, 3), dtype=np.uint8)
        divider[:] = divider_color
        comparison = np.vstack([img1, divider, img2])

    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        bg_col = (20, 20, 20)
        fg_col = (255, 255, 255)

        # Label for original
        cv2.rectangle(comparison, (10, 10), (140, 40), bg_col, -1)
        cv2.putText(comparison, "ORIGINAL", (18, 32), font, font_scale, fg_col, 1, cv2.LINE_AA)

        # Label for rotocraft mask
        offset_x = (img1.shape[1] + divider_thickness + 10) if orientation == "horizontal" else 10
        offset_y = 10 if orientation == "horizontal" else (img1.shape[0] + divider_thickness + 10)
        cv2.rectangle(comparison, (offset_x, offset_y), (offset_x + 190, offset_y + 30), bg_col, -1)
        cv2.putText(comparison, "ROTOCRAFT MASK", (offset_x + 8, offset_y + 22), font, font_scale, (0, 255, 128), 1, cv2.LINE_AA)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(comparison).save(str(out_p))
        logger.info("Saved comparison preview to %s", out_p)

    return comparison

"""Rotocraft Web Studio Application.

A lightweight, local, interactive graphical studio for automatic AI rotoscoping
and object masking. Serves a sleek modern dark-mode single-page application with
zero external web framework dependencies.
"""

from __future__ import annotations

import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import webbrowser

import cv2
import numpy as np
from PIL import Image

from rotocraft import __version__
from rotocraft.filmcraft_bridge import FilmCraftBridge
from rotocraft.mask_export import MaskExporter, contours_to_bezier, masks_to_contours
from rotocraft.models import AVAILABLE_MODELS, ModelManager, get_device
from rotocraft.preview import overlay_mask
from rotocraft.segmenter import RotocraftSegmenter
from rotocraft.video_io import VideoReader, VideoWriter

logger = logging.getLogger("rotocraft.web_app")


class StudioSession:
    """State of the active user session in Rotocraft Studio."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.media_path: Optional[str] = None
        self.is_video: bool = False
        self.video_reader: Optional[VideoReader] = None
        self.current_frame_idx: int = 0
        self.current_frame_rgb: Optional[np.ndarray] = None
        self.total_frames: int = 1
        self.fps: float = 30.0
        self.width: int = 1280
        self.height: int = 720

        # Current segmentation prompt points: list of (x, y, label) where label is 1 (fg) or 0 (bg)
        self.points: List[List[float]] = []
        self.box: Optional[List[float]] = None

        # Mask state for current frame
        self.current_mask: Optional[np.ndarray] = None  # (H, W) bool
        self.current_bezier_json: Optional[Dict[str, Any]] = None

        # Video tracking cache: frame_idx -> mask (H, W)
        self.tracked_masks: Dict[int, np.ndarray] = {}
        self.tracking_active: bool = False
        self.tracking_progress: float = 0.0
        self.tracking_status: str = "Idle"
        self.stop_tracking: bool = False

        # Model and device configuration
        self.model_name: str = "sam2_hiera_small"
        self.device = get_device()
        self.segmenter: Optional[RotocraftSegmenter] = None
        self.bridge = FilmCraftBridge()

    def get_segmenter(self) -> RotocraftSegmenter:
        if self.segmenter is None:
            self.segmenter = RotocraftSegmenter(model_name=self.model_name, device=self.device)
        return self.segmenter

    def load_media(self, path: str) -> Dict[str, Any]:
        with self.lock:
            p = Path(path)
            if not p.exists():
                return {"ok": False, "error": f"File not found: {path}"}

            self.media_path = str(p.resolve())
            self.points = []
            self.box = None
            self.current_mask = None
            self.current_bezier_json = None
            self.tracked_masks = {}

            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
            if p.suffix.lower() in video_exts:
                self.is_video = True
                self.video_reader = VideoReader(self.media_path)
                self.total_frames = self.video_reader.total_frames
                self.fps = self.video_reader.fps
                self.width = self.video_reader.width
                self.height = self.video_reader.height
                self.current_frame_idx = 0
                self.current_frame_rgb = self.video_reader.get_frame(0)
            else:
                self.is_video = False
                self.video_reader = None
                self.total_frames = 1
                self.fps = 30.0
                bgr = cv2.imread(self.media_path)
                if bgr is None:
                    return {"ok": False, "error": f"Failed to load image: {path}"}
                self.current_frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.height, self.width = self.current_frame_rgb.shape[:2]
                self.current_frame_idx = 0

            return {
                "ok": True,
                "is_video": self.is_video,
                "total_frames": self.total_frames,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "filename": p.name,
            }

    def set_frame(self, index: int) -> bool:
        with self.lock:
            if not self.is_video or self.video_reader is None:
                return False
            index = max(0, min(index, self.total_frames - 1))
            self.current_frame_idx = index
            frame = self.video_reader.get_frame(index)
            if frame is not None:
                self.current_frame_rgb = frame
                # If we have tracked mask for this frame, load it
                if index in self.tracked_masks:
                    self.current_mask = self.tracked_masks[index]
                    self.current_bezier_json = MaskExporter.to_filmcraft_json(self.current_mask)
                return True
            return False

    def compute_mask(self) -> Optional[np.ndarray]:
        if self.current_frame_rgb is None:
            return None
        if not self.points and self.box is None:
            self.current_mask = None
            self.current_bezier_json = None
            return None

        seg = self.get_segmenter()
        pts = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32) if self.points else None
        labels = np.array([int(p[2]) for p in self.points], dtype=np.int32) if self.points else None
        box = np.array(self.box, dtype=np.float32) if self.box else None

        result = seg.segment_frame(self.current_frame_rgb, points=pts, labels=labels, boxes=box)
        mask = result.primary_mask if hasattr(result, "primary_mask") else result
        self.current_mask = mask
        self.current_bezier_json = MaskExporter.to_filmcraft_json(mask)
        if self.is_video:
            self.tracked_masks[self.current_frame_idx] = mask
        return mask


SESSION = StudioSession()


class RotocraftRequestHandler(BaseHTTPRequestHandler):
    """Custom request handler implementing REST API and serving UI."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress routine GET logging to keep terminal clean
        if "GET /api/" in format % args or "GET / " in format % args:
            return
        logger.debug(format, *args)

    def send_json(self, data: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html = render_studio_html()
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/status":
            filmcraft_port = SESSION.bridge.probe_connection()
            self.send_json({
                "version": __version__,
                "device": SESSION.device,
                "model": SESSION.model_name,
                "available_models": [m["name"] for m in ModelManager().list_models()],
                "filmcraft_connected": filmcraft_port is not None,
                "filmcraft_port": filmcraft_port,
                "media_loaded": SESSION.media_path is not None,
                "is_video": SESSION.is_video,
                "current_frame": SESSION.current_frame_idx,
                "total_frames": SESSION.total_frames,
                "fps": SESSION.fps,
                "width": SESSION.width,
                "height": SESSION.height,
                "tracking_active": SESSION.tracking_active,
                "tracking_progress": SESSION.tracking_progress,
                "tracking_status": SESSION.tracking_status,
            })
            return

        if path == "/api/frame":
            frame_idx = int(query.get("index", [SESSION.current_frame_idx])[0])
            mode = query.get("mode", ["original"])[0]  # original, overlay, cutout, green, matte
            SESSION.set_frame(frame_idx)

            if SESSION.current_frame_rgb is None:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return

            frame = SESSION.current_frame_rgb.copy()
            mask = SESSION.current_mask

            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                if mode == "overlay":
                    alpha = float(query.get("alpha", [0.5])[0])
                    frame = overlay_mask(frame, mask, color=(0, 255, 230), alpha=alpha)
                elif mode == "cutout":
                    # Transparent cutout on checkerboard
                    h, w = frame.shape[:2]
                    # Checkerboard background
                    sq = 16
                    check = np.zeros((h, w, 3), dtype=np.uint8)
                    for r in range(0, h, sq):
                        for c in range(0, w, sq):
                            val = 40 if ((r // sq) + (c // sq)) % 2 == 0 else 60
                            check[r : r + sq, c : c + sq] = val
                    frame = np.where(mask[:, :, None], frame, check)
                elif mode == "green":
                    # Pure Chroma Green #00FF00 background
                    green_bg = np.zeros_like(frame)
                    green_bg[:, :] = [0, 255, 0]
                    frame = np.where(mask[:, :, None], frame, green_bg)
                elif mode == "matte":
                    # Black and white silhouette
                    frame = np.where(mask[:, :, None], 255, 0).astype(np.uint8)

            # Encode frame to JPEG
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.end_headers()
                return

            img_bytes = buf.tobytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(img_bytes)))
            self.end_headers()
            self.wfile.write(img_bytes)
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"

        try:
            req_data = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req_data = {}

        if path == "/api/load_media":
            media_path = req_data.get("path", "")
            res = SESSION.load_media(media_path)
            self.send_json(res)
            return

        if path == "/api/load_sample":
            # Create a clean synthetic animated demonstration shot
            sample_dir = Path.home() / ".cache" / "rotocraft" / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_video = sample_dir / "demo_shot.mp4"
            if not sample_video.exists():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(sample_video), fourcc, 30.0, (960, 540))
                for i in range(90):
                    f = np.zeros((540, 960, 3), dtype=np.uint8)
                    # Background gradient
                    f[:, :, 0] = np.linspace(20, 50, 540)[:, None]
                    f[:, :, 1] = np.linspace(25, 60, 540)[:, None]
                    f[:, :, 2] = np.linspace(35, 75, 540)[:, None]
                    # Moving subject circle + box
                    cx = int(300 + i * 4)
                    cy = int(270 + 30 * np.sin(i * 0.1))
                    cv2.circle(f, (cx, cy), 90, (230, 180, 70), -1)
                    cv2.putText(f, "ACTOR", (cx - 45, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
                    vw.write(f)
                vw.release()

            res = SESSION.load_media(str(sample_video))
            self.send_json(res)
            return

        if path == "/api/add_point":
            x = float(req_data.get("x", 0))
            y = float(req_data.get("y", 0))
            label = int(req_data.get("label", 1))  # 1 = fg, 0 = bg
            SESSION.points.append([x, y, label])
            SESSION.compute_mask()
            self.send_json({
                "ok": True,
                "points": SESSION.points,
                "has_mask": SESSION.current_mask is not None,
                "bezier": SESSION.current_bezier_json,
            })
            return

        if path == "/api/set_box":
            box = req_data.get("box")  # [x1, y1, x2, y2]
            SESSION.box = box
            SESSION.compute_mask()
            self.send_json({
                "ok": True,
                "box": SESSION.box,
                "has_mask": SESSION.current_mask is not None,
                "bezier": SESSION.current_bezier_json,
            })
            return

        if path == "/api/clear_prompts":
            SESSION.points = []
            SESSION.box = None
            SESSION.current_mask = None
            SESSION.current_bezier_json = None
            self.send_json({"ok": True})
            return

        if path == "/api/undo_point":
            if SESSION.points:
                SESSION.points.pop()
                SESSION.compute_mask()
            self.send_json({
                "ok": True,
                "points": SESSION.points,
                "has_mask": SESSION.current_mask is not None,
                "bezier": SESSION.current_bezier_json,
            })
            return

        if path == "/api/track":
            start_f = int(req_data.get("start", 0))
            end_f = int(req_data.get("end", SESSION.total_frames - 1))

            def run_tracking():
                SESSION.tracking_active = True
                SESSION.stop_tracking = False
                SESSION.tracking_progress = 0.0
                SESSION.tracking_status = f"Tracking frames {start_f} to {end_f}..."
                t0 = time.time()
                total = max(1, end_f - start_f + 1)

                seg = SESSION.get_segmenter()
                for step, idx in enumerate(range(start_f, end_f + 1)):
                    if SESSION.stop_tracking:
                        SESSION.tracking_status = "Stopped by user"
                        break
                    frame = SESSION.video_reader.get_frame(idx)
                    if frame is not None:
                        if idx in SESSION.tracked_masks:
                            mask = SESSION.tracked_masks[idx]
                        elif SESSION.points:
                            pts = np.array([[p[0], p[1]] for p in SESSION.points], dtype=np.float32)
                            lbls = np.array([int(p[2]) for p in SESSION.points], dtype=np.int32)
                            res = seg.segment_frame(frame, points=pts, labels=lbls)
                            mask = res.primary_mask if hasattr(res, "primary_mask") else res
                            SESSION.tracked_masks[idx] = mask
                    SESSION.tracking_progress = (step + 1) / total
                    fps_calc = (step + 1) / max(0.001, time.time() - t0)
                    SESSION.tracking_status = f"Frame {idx}/{end_f} ({fps_calc:.1f} fps)"

                SESSION.tracking_active = False
                SESSION.tracking_status = f"Finished tracking {len(SESSION.tracked_masks)} frames."

            threading.Thread(target=run_tracking, daemon=True).start()
            self.send_json({"ok": True, "message": "Tracking started"})
            return

        if path == "/api/stop_tracking":
            SESSION.stop_tracking = True
            self.send_json({"ok": True})
            return

        if path == "/api/filmcraft/send":
            port = req_data.get("port")
            clip_id = req_data.get("clip_id")
            effect = req_data.get("effect", "opacity")

            if not SESSION.current_bezier_json:
                self.send_json({"ok": False, "error": "No mask available on current frame. Segment an object first."})
                return

            res = SESSION.bridge.apply_mask(
                mask_path_json=SESSION.current_bezier_json["path"],
                clip_id=clip_id,
                effect_index_or_name=effect,
                port=port,
            )
            self.send_json(res)
            return

        if path == "/api/export_json":
            out_path = req_data.get("path")
            if not out_path:
                out_path = str(Path(SESSION.media_path or "mask").with_suffix(".fcmask.json"))

            data = {
                "format": "filmcraft_mask",
                "version": 1,
                "media": SESSION.media_path,
                "current_frame": SESSION.current_frame_idx,
                "mask": SESSION.current_bezier_json,
                "total_tracked_frames": len(SESSION.tracked_masks),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.send_json({"ok": True, "path": out_path})
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def render_studio_html() -> str:
    """Generate the complete single-page application interface HTML/CSS/JS."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rotocraft Studio | AI Rotoscoping for FilmCraft</title>
    <style>
        :root {
            --bg-base: #121417;
            --bg-surface: #1a1e24;
            --bg-elevated: #242a33;
            --accent-cyan: #00e5ff;
            --accent-green: #00e676;
            --accent-red: #ff1744;
            --border-color: #2f3742;
            --text-main: #f0f3f6;
            --text-muted: #9ba7b6;
            --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-stack);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            user-select: none;
        }

        /* Top Navbar */
        header {
            height: 52px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            z-index: 10;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 1.15rem;
            letter-spacing: -0.5px;
        }

        .brand-badge {
            background: linear-gradient(135deg, #00e5ff, #00e676);
            color: #050c14;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 800;
            text-transform: uppercase;
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85rem;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-elevated);
            padding: 4px 10px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
        }

        .status-dot.offline { background: #888; }

        /* Main Workspace */
        main {
            display: flex;
            flex: 1;
            height: calc(100vh - 52px - 80px);
        }

        /* Center Viewport */
        .viewport {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #090b0d;
            position: relative;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .canvas-container {
            position: relative;
            box-shadow: 0 10px 40px rgba(0,0,0,0.7);
            border-radius: 4px;
            overflow: hidden;
            max-width: 95%;
            max-height: 95%;
            border: 1px solid var(--border-color);
            cursor: crosshair;
        }

        #video-canvas {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Floating Viewport Overlay Controls */
        .viewport-controls {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(26, 30, 36, 0.85);
            backdrop-filter: blur(8px);
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            display: flex;
            gap: 8px;
            align-items: center;
            font-size: 0.8rem;
        }

        .btn-group {
            display: flex;
            background: var(--bg-base);
            border-radius: 6px;
            padding: 2px;
        }

        .btn-group button {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.15s ease;
        }

        .btn-group button.active {
            background: var(--bg-elevated);
            color: var(--text-main);
            font-weight: 600;
        }

        /* Side Panel */
        aside {
            width: 340px;
            background: var(--bg-surface);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 16px;
            overflow-y: auto;
        }

        .panel-section {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 14px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .panel-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }

        /* Buttons */
        .btn {
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.15s;
        }

        .btn:hover { background: #323b47; }

        .btn-primary {
            background: linear-gradient(135deg, #00b0ff, #00e5ff);
            color: #04121f;
            border: none;
            font-weight: 700;
        }

        .btn-primary:hover { filter: brightness(1.1); }

        .btn-success {
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #041a0b;
            border: none;
            font-weight: 700;
        }

        .btn-success:hover { filter: brightness(1.1); }

        /* Timeline Footer */
        footer {
            height: 80px;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0 20px;
            gap: 8px;
        }

        .timeline-bar {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .timeline-slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #2a313d;
            outline: none;
            cursor: pointer;
            accent-color: var(--accent-cyan);
        }

        .transport-controls {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .transport-buttons {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .timecode-display {
            font-family: ui-monospace, monospace;
            font-size: 1rem;
            font-weight: 700;
            background: var(--bg-base);
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            color: var(--accent-cyan);
        }

        /* Tracking progress bar */
        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: var(--bg-base);
            border-radius: 3px;
            overflow: hidden;
            display: none;
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
            transition: width 0.1s linear;
        }

        .instruction-box {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
            background: rgba(0, 229, 255, 0.05);
            border-left: 3px solid var(--accent-cyan);
            padding: 8px 10px;
            border-radius: 0 4px 4px 0;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <span>Rotocraft Studio</span>
            <span class="brand-badge">FilmCraft AI</span>
        </div>
        <div class="nav-status">
            <div class="status-pill">
                <span class="status-dot" id="hw-dot"></span>
                <span id="hw-label">Hardware: Autodetecting...</span>
            </div>
            <div class="status-pill">
                <span class="status-dot offline" id="fc-dot"></span>
                <span id="fc-label">FilmCraft: Offline</span>
            </div>
        </div>
    </header>

    <main>
        <div class="viewport">
            <div class="viewport-controls">
                <span>View Mode:</span>
                <div class="btn-group">
                    <button class="active" onclick="setViewMode('overlay')">Overlay</button>
                    <button onclick="setViewMode('cutout')">Cutout</button>
                    <button onclick="setViewMode('green')">Green Screen</button>
                    <button onclick="setViewMode('matte')">Alpha Matte</button>
                    <button onclick="setViewMode('original')">Source</button>
                </div>
            </div>

            <div class="canvas-container" id="canvas-wrap">
                <img id="video-canvas" src="" alt="Rotocraft Viewport">
            </div>
        </div>

        <aside>
            <div class="panel-section">
                <div class="panel-title">Media Selection</div>
                <button class="btn btn-primary" onclick="loadSampleDemo()">⚡ Open Demo Shot</button>
                <div style="display:flex; gap:6px;">
                    <input type="text" id="media-path-input" placeholder="Path to .mp4 or .png..." style="flex:1; background:var(--bg-base); border:1px solid var(--border-color); color:#fff; padding:6px 10px; border-radius:4px; font-size:0.8rem;">
                    <button class="btn" onclick="loadCustomPath()">Load</button>
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-title">Prompt & Masking</div>
                <div class="instruction-box">
                    • <b>Left-Click:</b> Add Foreground (+) point<br>
                    • <b>Right-Click or Alt+Click:</b> Add Background (-) point<br>
                    • Automatic live Bézier contour generation.
                </div>
                <div style="display:flex; gap:8px;">
                    <button class="btn" style="flex:1;" onclick="undoPoint()">Undo Point</button>
                    <button class="btn" style="flex:1;" onclick="clearPrompts()">Clear All</button>
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-title">Video Propagation</div>
                <button class="btn btn-success" id="track-btn" onclick="startTracking()">▶ Auto-Rotoscope Timeline</button>
                <div class="progress-bar-container" id="track-prog-wrap">
                    <div class="progress-bar" id="track-prog-bar"></div>
                </div>
                <div id="track-status" style="font-size:0.8rem; color:var(--text-muted); text-align:center;">Ready to track</div>
            </div>

            <div class="panel-section">
                <div class="panel-title">Export & FilmCraft Bridge</div>
                <button class="btn btn-primary" onclick="sendToFilmCraft()">⚡ Send Mask to FilmCraft</button>
                <button class="btn" onclick="exportMaskJson()">💾 Download FilmCraft Mask (.json)</button>
            </div>
        </aside>
    </main>

    <footer>
        <div class="timeline-bar">
            <span id="current-frame-txt" style="font-size:0.85rem; font-family:monospace; min-width:60px;">0 / 0</span>
            <input type="range" class="timeline-slider" id="timeline" min="0" max="0" value="0" oninput="onTimelineScrub(this.value)">
            <div class="timecode-display" id="timecode">00:00:00:00</div>
        </div>
        <div class="transport-controls">
            <div class="transport-buttons">
                <button class="btn" onclick="stepFrame(-10)">⏮ -10</button>
                <button class="btn" onclick="stepFrame(-1)">◀ -1</button>
                <button class="btn btn-primary" id="play-btn" onclick="togglePlay()">Play</button>
                <button class="btn" onclick="stepFrame(1)">+1 ▶</button>
                <button class="btn" onclick="stepFrame(10)">+10 ⏭</button>
            </div>
            <div style="font-size:0.8rem; color:var(--text-muted);" id="resolution-txt">
                No Media Loaded
            </div>
        </div>
    </footer>

    <script>
        let currentFrame = 0;
        let totalFrames = 1;
        let fps = 30;
        let isPlaying = false;
        let playInterval = null;
        let viewMode = 'overlay';

        const canvasImg = document.getElementById('video-canvas');
        const timeline = document.getElementById('timeline');
        const timecode = document.getElementById('timecode');
        const frameTxt = document.getElementById('current-frame-txt');
        const resTxt = document.getElementById('resolution-txt');

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('hw-label').innerText = 'Hardware: ' + data.device.toUpperCase();
                
                const fcDot = document.getElementById('fc-dot');
                const fcLabel = document.getElementById('fc-label');
                if (data.filmcraft_connected) {
                    fcDot.className = 'status-dot';
                    fcLabel.innerText = 'FilmCraft: Connected (' + data.filmcraft_port + ')';
                } else {
                    fcDot.className = 'status-dot offline';
                    fcLabel.innerText = 'FilmCraft: Offline';
                }

                if (data.tracking_active) {
                    document.getElementById('track-prog-wrap').style.display = 'block';
                    document.getElementById('track-prog-bar').style.width = (data.tracking_progress * 100) + '%';
                    document.getElementById('track-status').innerText = data.tracking_status;
                } else if (document.getElementById('track-prog-wrap').style.display === 'block') {
                    document.getElementById('track-status').innerText = data.tracking_status;
                }
            } catch (e) {
                console.error(e);
            }
        }

        setInterval(updateStatus, 1500);
        updateStatus();

        function refreshFrame() {
            canvasImg.src = '/api/frame?index=' + currentFrame + '&mode=' + viewMode + '&t=' + Date.now();
            timeline.value = currentFrame;
            frameTxt.innerText = currentFrame + ' / ' + (totalFrames - 1);
            
            // Format timecode
            const secs = Math.floor(currentFrame / fps);
            const frames = currentFrame % Math.round(fps);
            const m = Math.floor(secs / 60);
            const s = secs % 60;
            const h = Math.floor(m / 60);
            timecode.innerText = String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0') + ':' + String(frames).padStart(2,'0');
        }

        function setViewMode(mode) {
            viewMode = mode;
            document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            refreshFrame();
        }

        async function loadSampleDemo() {
            const res = await fetch('/api/load_sample', { method: 'POST' });
            const data = await res.json();
            if (data.ok) {
                totalFrames = data.total_frames;
                fps = data.fps;
                timeline.max = totalFrames - 1;
                resTxt.innerText = data.width + ' × ' + data.height + ' (' + data.fps + ' fps)';
                currentFrame = 0;
                refreshFrame();
            }
        }

        async function loadCustomPath() {
            const p = document.getElementById('media-path-input').value.trim();
            if (!p) return;
            const res = await fetch('/api/load_media', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path: p })
            });
            const data = await res.json();
            if (data.ok) {
                totalFrames = data.total_frames;
                fps = data.fps;
                timeline.max = totalFrames - 1;
                resTxt.innerText = data.width + ' × ' + data.height + ' (' + data.fps + ' fps)';
                currentFrame = 0;
                refreshFrame();
            } else {
                alert(data.error);
            }
        }

        function onTimelineScrub(val) {
            currentFrame = parseInt(val);
            refreshFrame();
        }

        function stepFrame(delta) {
            currentFrame = Math.max(0, Math.min(totalFrames - 1, currentFrame + delta));
            refreshFrame();
        }

        function togglePlay() {
            const btn = document.getElementById('play-btn');
            if (isPlaying) {
                clearInterval(playInterval);
                isPlaying = false;
                btn.innerText = 'Play';
            } else {
                isPlaying = true;
                btn.innerText = 'Pause';
                playInterval = setInterval(() => {
                    if (currentFrame >= totalFrames - 1) {
                        currentFrame = 0;
                    } else {
                        currentFrame++;
                    }
                    refreshFrame();
                }, 1000 / fps);
            }
        }

        // Canvas Click for Prompts
        canvasImg.addEventListener('click', async (e) => {
            const rect = canvasImg.getBoundingClientRect();
            const scaleX = canvasImg.naturalWidth / rect.width;
            const scaleY = canvasImg.naturalHeight / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            const label = e.altKey ? 0 : 1; // 1 = Foreground, 0 = Background

            await fetch('/api/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ x, y, label })
            });
            refreshFrame();
        });

        canvasImg.addEventListener('contextmenu', async (e) => {
            e.preventDefault();
            const rect = canvasImg.getBoundingClientRect();
            const scaleX = canvasImg.naturalWidth / rect.width;
            const scaleY = canvasImg.naturalHeight / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            await fetch('/api/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ x, y, label: 0 }) // Background point
            });
            refreshFrame();
        });

        async function clearPrompts() {
            await fetch('/api/clear_prompts', { method: 'POST' });
            refreshFrame();
        }

        async function undoPoint() {
            await fetch('/api/undo_point', { method: 'POST' });
            refreshFrame();
        }

        async function startTracking() {
            document.getElementById('track-prog-wrap').style.display = 'block';
            await fetch('/api/track', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ start: 0, end: totalFrames - 1 })
            });
        }

        async function sendToFilmCraft() {
            const res = await fetch('/api/filmcraft/send', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.ok) {
                alert('Success: Mask sent to FilmCraft timeline!');
            } else {
                alert('FilmCraft Bridge: ' + (data.error || 'Failed to send mask'));
            }
        }

        async function exportMaskJson() {
            const res = await fetch('/api/export_json', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.ok) {
                alert('Exported FilmCraft Mask JSON to: ' + data.path);
            }
        }

        // Keyboard navigation
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                togglePlay();
            } else if (e.code === 'ArrowLeft') {
                stepFrame(-1);
            } else if (e.code === 'ArrowRight') {
                stepFrame(1);
            } else if (e.code === 'KeyZ' && (e.ctrlKey || e.metaKey)) {
                undoPoint();
            }
        });
    </script>
</body>
</html>"""


def start_studio_server(port: int = 8765, open_browser: bool = True) -> ThreadingHTTPServer:
    """Start the Rotocraft Studio local web server."""
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, RotocraftRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"  🎬 Rotocraft Studio: AI Rotoscoping for FilmCraft")
    print(f"  URL: {url}")
    print(f"  Device: {SESSION.device.upper()}")
    print(f"=======================================================\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    return httpd

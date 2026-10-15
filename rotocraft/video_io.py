"""Video I/O utilities for video frame extraction and mask rendering in Rotocraft.

Wraps OpenCV VideoCapture and VideoWriter with Pythonic iterator protocols,
context managers, frame seeking, metadata extraction, and mask overlay rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("rotocraft.video_io")


class VideoIOError(Exception):
    """Raised when video file cannot be opened or read."""


class VideoReader:
    """Fast, memory-efficient video reader with seeking and iterator protocols.

    Yields RGB frames as numpy arrays `(frame_index, frame_rgb_numpy)`.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize VideoReader.

        Args:
            path: Path to the video file.

        Raises:
            FileNotFoundError: If video file does not exist.
            VideoIOError: If video cannot be opened by OpenCV.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoIOError(f"OpenCV failed to open video file: {self.path}")

        # Cache properties
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0.0 or not np.isfinite(self._fps):
            self._fps = 30.0  # Safe default if metadata missing

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._total_frames < 0:
            self._total_frames = 0

        self._current_index = 0

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Video width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Video height in pixels."""
        return self._height

    @property
    def total_frames(self) -> int:
        """Total frame count in the video."""
        return self._total_frames

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        if self._fps > 0 and self._total_frames > 0:
            return float(self._total_frames / self._fps)
        return 0.0

    def seek(self, frame_index: int) -> bool:
        """Seek video playhead to a specific 0-based frame index.

        Args:
            frame_index: Target 0-based frame number.

        Returns:
            True if seek succeeded, False otherwise.
        """
        if self._cap is None or not self._cap.isOpened():
            return False

        clamped = max(0, min(frame_index, max(0, self._total_frames - 1)))
        success = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, clamped))
        if success:
            self._current_index = clamped
        return success

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame in sequence, returning RGB numpy array.

        Returns:
            Tuple of (success, frame_rgb_array or None).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return False, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._current_index += 1
        return True, frame_rgb

    def get_frame(self, frame_index: int) -> np.ndarray:
        """Retrieve a specific frame by index as RGB numpy array.

        Args:
            frame_index: 0-based frame number.

        Returns:
            RGB numpy array with shape (H, W, 3).

        Raises:
            VideoIOError: If seeking or reading fails.
        """
        if not self.seek(frame_index):
            raise VideoIOError(f"Failed to seek to frame {frame_index}")
        ret, frame = self.read_frame()
        if not ret or frame is None:
            raise VideoIOError(f"Failed to read frame at index {frame_index}")
        return frame

    def __getitem__(self, frame_index: int) -> np.ndarray:
        """Enable bracket indexing `reader[index]`."""
        return self.get_frame(frame_index)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all remaining frames from current position."""
        while True:
            idx = self._current_index
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            yield idx, frame

    def read_range(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield frames in range [start_frame, end_frame) with step.

        Args:
            start_frame: Starting 0-based frame index.
            end_frame: Ending 0-based frame index (exclusive). If None, reads to end.
            step: Step size between frames (default 1).
        """
        if not self.seek(start_frame):
            return

        stop = end_frame if end_frame is not None else self._total_frames
        curr = start_frame
        while curr < stop:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            yield curr, frame
            if step > 1:
                curr += step
                if not self.seek(curr):
                    break
            else:
                curr += 1

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self.path.name}', {self.width}x{self.height}, "
            f"{self.fps:.2f}fps, {self.total_frames} frames)"
        )


class VideoWriter:
    """Writes video files and mask overlay visualizations using OpenCV."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float = 30.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fourcc: str = "mp4v",
    ) -> None:
        """Initialize VideoWriter.

        Args:
            output_path: Path where output video is written.
            fps: Frame rate.
            width: Frame width in pixels (inferred on first frame if None).
            height: Frame height in pixels (inferred on first frame if None).
            fourcc: 4-character codec code (default: 'mp4v').
        """
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.width = width
        self.height = height
        self.fourcc_str = fourcc
        self._writer: Optional[cv2.VideoWriter] = None

        if self.width is not None and self.height is not None:
            self._init_writer(self.width, self.height)

    def _init_writer(self, width: int, height: int) -> None:
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc_str)
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc_code,
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise VideoIOError(f"Failed to open VideoWriter for {self.path}")
        self.width = width
        self.height = height

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        """Write an RGB numpy frame to the output video.

        Converts RGB to BGR before passing to OpenCV.
        """
        if self._writer is None:
            h, w = frame_rgb.shape[:2]
            self._init_writer(w, h)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)

    def write_mask_overlay(
        self,
        original_frame_rgb: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5,
    ) -> None:
        """Blend mask onto frame and write to output video.

        Args:
            original_frame_rgb: RGB frame numpy array (H, W, 3).
            mask: Binary mask numpy array (H, W).
            color: RGB color tuple for the mask tint.
            alpha: Mask opacity blend factor (0.0 to 1.0).
        """
        from rotocraft.preview import overlay_mask

        composited = overlay_mask(original_frame_rgb, mask, color=color, alpha=alpha)
        self.write_frame(composited)

    def release(self) -> None:
        """Finalize and close video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"VideoWriter('{self.path.name}', fps={self.fps})"

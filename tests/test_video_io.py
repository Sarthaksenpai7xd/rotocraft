"""Unit tests for Rotocraft video I/O (VideoReader and VideoWriter)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from rotocraft.video_io import VideoIOError, VideoReader, VideoWriter


def create_synthetic_video(
    file_path: Path,
    num_frames: int = 15,
    width: int = 120,
    height: int = 90,
    fps: float = 24.0,
) -> None:
    """Create a small test video file with moving color squares."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Shift a color block across frames
        x = int((i * 6) % (width - 20))
        cv2.rectangle(frame, (x, 20), (x + 20, 60), (0, 255, i * 15), -1)
        writer.write(frame)
    writer.release()


class TestVideoIO:
    """Test suite for VideoReader and VideoWriter."""

    def test_reader_properties(self) -> None:
        """Verify reading video metadata properties."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vid_path = Path(tmp_dir) / "test.mp4"
            create_synthetic_video(vid_path, num_frames=12, width=160, height=120, fps=24.0)

            with VideoReader(vid_path) as reader:
                assert reader.width == 160
                assert reader.height == 120
                assert np.isclose(reader.fps, 24.0, atol=1.0)
                assert reader.total_frames >= 10
                assert reader.duration > 0.0

    def test_reader_iteration_and_seeking(self) -> None:
        """Verify sequential iteration and frame seeking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vid_path = Path(tmp_dir) / "test.mp4"
            create_synthetic_video(vid_path, num_frames=10, width=80, height=60, fps=25.0)

            reader = VideoReader(vid_path)
            frames = list(reader)
            assert len(frames) >= 8

            # Each item is (frame_idx, frame_rgb_numpy)
            idx, frame = frames[0]
            assert idx == 0
            assert frame.shape == (60, 80, 3)

            # Test seeking to frame 4
            seek_ok = reader.seek(4)
            assert seek_ok is True
            ret, frame_4 = reader.read_frame()
            assert ret is True
            assert frame_4 is not None

            reader.release()

    def test_writer_and_mask_overlay(self) -> None:
        """Verify writing RGB frames and mask overlays to video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_vid = Path(tmp_dir) / "out_overlay.mp4"
            width, height = 100, 80
            mask = np.zeros((height, width), dtype=bool)
            mask[20:60, 30:70] = True

            with VideoWriter(out_vid, fps=30.0, width=width, height=height) as writer:
                for _ in range(8):
                    base_frame = np.full((height, width, 3), 100, dtype=np.uint8)
                    writer.write_mask_overlay(base_frame, mask, color=(0, 255, 0), alpha=0.5)

            assert out_vid.is_file()
            assert out_vid.stat().st_size > 0

            # Verify it can be read back
            with VideoReader(out_vid) as reader:
                assert reader.width == width
                assert reader.height == height
                ret, read_frame = reader.read_frame()
                assert ret is True
                assert read_frame.shape == (height, width, 3)

    def test_missing_file_raises_error(self) -> None:
        """Verify proper exception when video file is missing."""
        with pytest.raises(FileNotFoundError):
            VideoReader("non_existent_video_path_12345.mp4")

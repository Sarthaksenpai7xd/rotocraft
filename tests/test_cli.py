"""Unit tests for Rotocraft CLI interface using Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from click.testing import CliRunner
import cv2
import numpy as np
import pytest

from rotocraft.cli import main
from rotocraft.video_io import VideoWriter


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_version(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(main, ["--version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.output


def test_cli_models_list(cli_runner: CliRunner) -> None:
    res = cli_runner.invoke(main, ["models", "--list"])
    assert res.exit_code == 0
    assert "sam2_hiera_tiny" in res.output
    assert "sam2_hiera_small" in res.output
    assert "sam2_hiera_base_plus" in res.output
    assert "sam2_hiera_large" in res.output


def test_cli_segment_command(cli_runner: CliRunner) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        img_p = tmp_p / "input.png"
        out_p = tmp_p / "out"

        # Create synthetic image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 25, (255, 255, 255), -1)
        cv2.imwrite(str(img_p), img)

        res = cli_runner.invoke(
            main,
            [
                "segment",
                "-i",
                str(img_p),
                "-o",
                str(out_p),
                "-p",
                "50,50",
                "-m",
                "tiny",
                "-f",
                "json",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "Segmentation complete" in res.output
        json_file = out_p / "mask.json"
        assert json_file.is_file()
        with open(json_file, "r") as f:
            data = json.load(f)
        assert "closed" in data
        assert "vertices" in data


def test_cli_segment_filmcraft_format(cli_runner: CliRunner) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        img_p = tmp_p / "input.png"
        out_p = tmp_p / "out"

        img = np.full((80, 80, 3), 120, dtype=np.uint8)
        cv2.circle(img, (40, 40), 20, (0, 0, 0), -1)
        cv2.imwrite(str(img_p), img)

        res = cli_runner.invoke(
            main,
            [
                "segment",
                "-i",
                str(img_p),
                "-o",
                str(out_p),
                "-b",
                "20,20,60,60",
                "-m",
                "tiny",
                "-f",
                "filmcraft",
                "--clip-id",
                "clip-99",
            ],
        )
        assert res.exit_code == 0, res.output
        cmd_file = out_p / "filmcraft_cmd.json"
        assert cmd_file.is_file()
        with open(cmd_file, "r") as f:
            cmd = json.load(f)
        assert cmd["clip"] == "clip-99"
        assert cmd["shape"] == "bezier"
        assert cmd["effect"] == "opacity"


def test_cli_track_command(cli_runner: CliRunner) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        vid_p = tmp_p / "clip.mp4"
        out_p = tmp_p / "tracked"

        with VideoWriter(vid_p, fps=24.0, width=60, height=60) as w:
            for _ in range(4):
                f = np.zeros((60, 60, 3), dtype=np.uint8)
                cv2.circle(f, (30, 30), 10, (255, 255, 255), -1)
                w.write_frame(f)

        res = cli_runner.invoke(
            main,
            [
                "track",
                "-i",
                str(vid_p),
                "-o",
                str(out_p),
                "-p",
                "30,30",
                "--start-frame",
                "0",
                "--end-frame",
                "2",
                "-m",
                "tiny",
                "-f",
                "png",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "Successfully tracked" in res.output


def test_cli_auto_command(cli_runner: CliRunner) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        img_p = tmp_p / "scene.png"
        out_p = tmp_p / "auto_out"

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
        cv2.imwrite(str(img_p), img)

        res = cli_runner.invoke(
            main,
            [
                "auto",
                "-i",
                str(img_p),
                "-o",
                str(out_p),
                "-m",
                "tiny",
                "-f",
                "png",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "Discovered" in res.output

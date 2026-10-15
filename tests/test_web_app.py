"""Unit tests for Rotocraft Web Studio server and endpoints."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from rotocraft.web_app import SESSION, start_studio_server


class TestStudioServer:
    @classmethod
    def setup_class(cls) -> None:
        cls.port = 48765
        cls.server = start_studio_server(port=cls.port, open_browser=False)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def teardown_class(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=1.0)

    def test_root_index_html(self) -> None:
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=2.0) as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "Rotocraft Studio" in html
            assert "video-canvas" in html

    def test_api_status(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/status"
        with urllib.request.urlopen(url, timeout=2.0) as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert "version" in data
            assert "device" in data
            assert "available_models" in data

    def test_api_load_sample(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/load_sample"
        req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert data.get("ok") is True
            assert data.get("is_video") is True
            assert data.get("total_frames") > 0

    def test_api_add_point_and_frame_render(self) -> None:
        # Add a point prompt
        url = f"http://127.0.0.1:{self.port}/api/add_point"
        payload = json.dumps({"x": 480, "y": 270, "label": 1}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert data.get("ok") is True
            assert len(data.get("points", [])) == 1

        # Fetch rendered overlay frame
        frame_url = f"http://127.0.0.1:{self.port}/api/frame?index=0&mode=overlay"
        with urllib.request.urlopen(frame_url, timeout=5.0) as res:
            assert res.status == 200
            assert res.headers.get("Content-Type") == "image/jpeg"
            img_data = res.read()
            assert len(img_data) > 1000

        # Clear prompts
        clear_url = f"http://127.0.0.1:{self.port}/api/clear_prompts"
        clear_req = urllib.request.Request(clear_url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(clear_req, timeout=5.0) as res:
            assert res.status == 200
            assert json.loads(res.read().decode("utf-8")).get("ok") is True

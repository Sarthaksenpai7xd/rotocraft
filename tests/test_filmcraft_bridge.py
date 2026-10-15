"""Unit tests for FilmCraft TCP control server bridge."""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Dict

import pytest

from rotocraft.filmcraft_bridge import FilmCraftBridge


def mock_filmcraft_server(port: int, stop_event: threading.Event) -> None:
    """Mock a FilmCraft control server on localhost for testing."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    s.settimeout(0.5)

    while not stop_event.is_set():
        try:
            conn, _ = s.accept()
            data = conn.recv(4096)
            if data:
                line = data.decode("utf-8").strip()
                req = json.loads(line)
                method = req.get("method")
                if method == "version":
                    reply = json.dumps({"ok": True, "version": "0.8.0"}) + "\n"
                elif method == "masks.add":
                    reply = json.dumps({"ok": True, "mask": 0, "name": "Mask (1)"}) + "\n"
                elif method == "masks.set":
                    reply = json.dumps({"ok": True}) + "\n"
                else:
                    reply = json.dumps({"ok": True, "method": method}) + "\n"
                conn.sendall(reply.encode("utf-8"))
            conn.close()
        except socket.timeout:
            continue
        except Exception:
            break
    s.close()


class TestFilmCraftBridge:
    def test_bridge_probe_and_send(self) -> None:
        test_port = 49152
        stop_event = threading.Event()
        srv_thread = threading.Thread(target=mock_filmcraft_server, args=(test_port, stop_event), daemon=True)
        srv_thread.start()
        time.sleep(0.1)

        try:
            bridge = FilmCraftBridge(port=test_port)
            res = bridge.send_command("version")
            assert res.get("ok") is True
            assert res.get("version") == "0.8.0"

            # Test apply_mask
            mask_path = {
                "closed": True,
                "vertices": [{"p": [100.0, 100.0], "in": [0.0, 0.0], "out": [0.0, 0.0]}],
            }
            mask_res = bridge.apply_mask(mask_path, effect_index_or_name="opacity")
            assert mask_res.get("ok") is True
            assert mask_res.get("mask") == 0
        finally:
            stop_event.set()
            srv_thread.join(timeout=1.0)

    def test_bridge_offline_graceful_handling(self) -> None:
        # Port with no listening server
        bridge = FilmCraftBridge(port=65432, timeout=0.2)
        res = bridge.send_command("version")
        assert res.get("ok") is False
        assert "error" in res

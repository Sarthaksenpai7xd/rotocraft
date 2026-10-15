"""FilmCraft Bridge module.

Handles bidirectional communication between Rotocraft and a running FilmCraft instance
using FilmCraft's localhost JSON-lines TCP control server.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rotocraft.filmcraft_bridge")

# Default ports to probe for running FilmCraft control servers
COMMON_PORTS = [48123, 48124, 48125, 48126, 8080, 9000]


class FilmCraftBridge:
    """Communicates with FilmCraft's TCP control server."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, timeout: float = 3.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def probe_connection(self) -> Optional[int]:
        """Probe localhost to find an active FilmCraft control server.

        Returns:
            Active port number if found, otherwise None.
        """
        candidate_ports = [self.port] if self.port else COMMON_PORTS

        for p in candidate_ports:
            if p is None:
                continue
            try:
                with socket.create_connection((self.host, p), timeout=0.04) as s:
                    s.settimeout(0.04)
                    req = json.dumps({"method": "version", "params": {}}) + "\n"
                    s.sendall(req.encode("utf-8"))
                    res = s.recv(4096)
                    if res:
                        self.port = p
                        logger.info("Discovered FilmCraft control server on port %d", p)
                        return p
            except (socket.timeout, ConnectionRefusedError, OSError):
                continue

        return None

    def send_command(self, method: str, params: Optional[Dict[str, Any]] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Send a JSON request to FilmCraft and return the parsed JSON response.

        Args:
            method: The command ID (e.g. 'masks.add', 'masks.list', 'playhead').
            params: Parameters dictionary for the command.
            port: Specific port, or uses discovered/configured port.

        Returns:
            Dictionary with response or error.
        """
        target_port = port or self.port or self.probe_connection()
        if not target_port:
            return {
                "ok": False,
                "error": "FilmCraft control server is not running or not found. Start FilmCraft with --control-port <port>.",
            }

        params = params or {}
        request_obj = {"method": method, "params": params}
        payload = json.dumps(request_obj) + "\n"

        try:
            with socket.create_connection((self.host, target_port), timeout=self.timeout) as s:
                s.sendall(payload.encode("utf-8"))
                response_data = b""
                while True:
                    chunk = s.recv(8192)
                    if not chunk:
                        break
                    response_data += chunk
                    if b"\n" in chunk:
                        break

                if not response_data:
                    return {"ok": False, "error": "Empty response from FilmCraft"}

                line = response_data.decode("utf-8").strip()
                return json.loads(line)
        except ConnectionRefusedError:
            return {"ok": False, "error": f"Connection refused on {self.host}:{target_port}"}
        except socket.timeout:
            return {"ok": False, "error": f"Connection timed out waiting for FilmCraft on port {target_port}"}
        except Exception as e:
            return {"ok": False, "error": f"Failed to communicate with FilmCraft: {e}"}

    def apply_mask(
        self,
        mask_path_json: Dict[str, Any],
        clip_id: Optional[str] = None,
        effect_index_or_name: str = "opacity",
        port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a 'masks.add' command to create a Bézier mask in FilmCraft.

        Args:
            mask_path_json: Dictionary with 'closed' and 'vertices' (FilmCraft MaskPath format).
            clip_id: Target clip ID, or None to use the active/selected clip.
            effect_index_or_name: Target effect ('opacity', 'lumetri_color', etc.).
            port: Specific port.

        Returns:
            Result dictionary from FilmCraft engine.
        """
        params: Dict[str, Any] = {
            "effect": effect_index_or_name,
            "shape": "bezier",
            "path": mask_path_json,
        }
        if clip_id:
            params["clip"] = clip_id

        return self.send_command("masks.add", params, port=port)

    def apply_animated_masks(
        self,
        keyframes: List[Tuple[int, Dict[str, Any]]],
        clip_id: Optional[str] = None,
        effect_index_or_name: str = "opacity",
        port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply an animated sequence of tracked mask keyframes to FilmCraft.

        Args:
            keyframes: List of (frame_index, mask_path_dict) pairs.
            clip_id: Target clip ID.
            effect_index_or_name: Target effect.
            port: Specific port.

        Returns:
            Result dictionary.
        """
        if not keyframes:
            return {"ok": False, "error": "No keyframes provided"}

        # 1. Create the initial mask at frame 0
        _, initial_path = keyframes[0]
        create_res = self.apply_mask(initial_path, clip_id, effect_index_or_name, port)
        if not create_res.get("ok", True) and "error" in create_res:
            return create_res

        mask_idx = create_res.get("mask", 0)

        # 2. Add remaining keyframes via 'masks.set'
        for frame_idx, path in keyframes[1:]:
            params: Dict[str, Any] = {
                "effect": effect_index_or_name,
                "mask": mask_idx,
                "path": path,
                "time": frame_idx,
            }
            if clip_id:
                params["clip"] = clip_id
            self.send_command("masks.set", params, port=port)

        return {"ok": True, "applied_keyframes": len(keyframes), "mask": mask_idx}

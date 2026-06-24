"""CSV and image logging for OpenVLA inference runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class InferenceLogger:
    """Append inference metadata to CSV and periodically save RGB frames."""

    def __init__(
        self,
        logs_dir: Path,
        image_dir: Path,
        csv_name: str,
        save_image_every_n_frames: int,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.image_dir = Path(image_dir)
        self.csv_path = self.logs_dir / csv_name
        self.save_image_every_n_frames = max(0, int(save_image_every_n_frames))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def log(
        self,
        frame_index: int,
        instruction: str,
        action: np.ndarray,
        inference_time_s: float,
        fps: float,
        image: Optional[Image.Image] = None,
    ) -> Optional[Path]:
        """Write one inference row and save an image when the interval matches."""

        timestamp = datetime.now(timezone.utc).isoformat()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        image_path = self._maybe_save_image(frame_index, timestamp, image)

        row = {
            "timestamp_utc": timestamp,
            "frame_index": frame_index,
            "instruction": instruction,
            "action_dx": self._get(action, 0),
            "action_dy": self._get(action, 1),
            "action_dz": self._get(action, 2),
            "action_droll": self._get(action, 3),
            "action_dpitch": self._get(action, 4),
            "action_dyaw": self._get(action, 5),
            "action_gripper": self._get(action, 6),
            "inference_time_s": f"{inference_time_s:.6f}",
            "fps": f"{fps:.4f}",
            "image_path": str(image_path) if image_path else "",
        }

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        return image_path

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return

        fields = [
            "timestamp_utc",
            "frame_index",
            "instruction",
            "action_dx",
            "action_dy",
            "action_dz",
            "action_droll",
            "action_dpitch",
            "action_dyaw",
            "action_gripper",
            "inference_time_s",
            "fps",
            "image_path",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    def _maybe_save_image(
        self,
        frame_index: int,
        timestamp: str,
        image: Optional[Image.Image],
    ) -> Optional[Path]:
        if image is None or self.save_image_every_n_frames <= 0:
            return None
        if frame_index % self.save_image_every_n_frames != 0:
            return None

        safe_ts = timestamp.replace(":", "-").replace("+", "Z")
        path = self.image_dir / f"frame_{frame_index:06d}_{safe_ts}.jpg"
        image.convert("RGB").save(path, quality=92)
        return path

    @staticmethod
    def _get(action: np.ndarray, index: int) -> str:
        return f"{float(action[index]):.8f}" if index < action.size else ""

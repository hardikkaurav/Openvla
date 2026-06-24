"""Live visualization for camera frames and OpenVLA action vectors."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config import ACTION_LABELS_COMPACT


class ActionVisualizer:
    """OpenCV visualization of the RGB frame and a 7-DoF action bar chart."""

    def __init__(self, window_name: str = "OpenVLA Real-Time Inference") -> None:
        self.window_name = window_name

    def update(
        self,
        image: Image.Image,
        action: np.ndarray,
        inference_time_s: float,
        fps: float,
        instruction: str,
        tcp_pose: list[float] | None = None,
        ur5_connected: bool = False,
    ) -> int | None:
        """Render the latest camera image and action vector.

        Press ``q`` or ``Esc`` in the visualization window to exit.
        """

        frame_panel = self._image_panel(image, instruction, inference_time_s, fps, tcp_pose, ur5_connected)
        chart_panel = self._bar_chart(action)
        combined = np.hstack([frame_panel, chart_panel])
        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        return key if key != 255 else None

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    @staticmethod
    def _image_panel(
        image: Image.Image,
        instruction: str,
        inference_time_s: float,
        fps: float,
        tcp_pose: list[float] | None,
        ur5_connected: bool,
    ) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        panel = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (640, 130), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, dst=panel)

        status_text = "Camera Connected | OpenVLA Loaded | "
        status_text += "UR5 Connected" if ur5_connected else "UR5 Disconnected"
        status_color = (100, 255, 100) if ur5_connected else (100, 100, 255)
        cv2.putText(panel, status_text, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        instruction_text = f"Instruction: {instruction}"
        cv2.putText(panel, instruction_text[:86], (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(
            panel,
            f"Inference: {inference_time_s * 1000:.1f} ms | FPS: {fps:.2f}",
            (14, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (130, 220, 255),
            2,
        )

        pose_str = "None"
        if tcp_pose and len(tcp_pose) >= 6:
            pose_str = (
                f"x={tcp_pose[0]:.3f} y={tcp_pose[1]:.3f} z={tcp_pose[2]:.3f} "
                f"rx={tcp_pose[3]:.3f} ry={tcp_pose[4]:.3f} rz={tcp_pose[5]:.3f}"
            )
        cv2.putText(panel, f"Pose: {pose_str}", (14, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        return panel

    @staticmethod
    def _bar_chart(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        panel_w, panel_h = 460, 480
        panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
        cv2.putText(panel, "Predicted Action Vector", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (25, 25, 25), 2)

        chart_left = 122
        chart_right = panel_w - 32
        center_x = (chart_left + chart_right) // 2
        bar_h = 36
        row_gap = 24
        y0 = 72
        scale = max(1.0, float(np.max(np.abs(action))) if action.size else 1.0)

        cv2.line(panel, (center_x, 58), (center_x, panel_h - 30), (80, 80, 80), 1)
        cv2.putText(panel, "-1", (chart_left - 8, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        cv2.putText(panel, "+1", (chart_right - 22, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        for idx, label in enumerate(ACTION_LABELS_COMPACT):
            value = float(action[idx]) if idx < action.size else 0.0
            y = y0 + idx * (bar_h + row_gap)
            normalized = np.clip(value / scale, -1.0, 1.0)
            x_end = int(center_x + normalized * ((chart_right - chart_left) / 2))
            color = (60, 150, 80) if value >= 0 else (70, 80, 210)
            x1, x2 = sorted((center_x, x_end))
            cv2.rectangle(panel, (x1, y), (x2, y + bar_h), color, thickness=-1)
            cv2.rectangle(panel, (chart_left, y), (chart_right, y + bar_h), (185, 185, 185), thickness=1)
            cv2.putText(panel, label, (22, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 2)
            cv2.putText(
                panel,
                f"{value:+.4f}",
                (chart_right - 96, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
            )

        return panel

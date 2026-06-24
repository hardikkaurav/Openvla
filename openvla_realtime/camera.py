"""Intel RealSense D435i RGB camera integration.

This module keeps RealSense-specific code isolated from the OpenVLA policy.
The main pipeline receives PIL RGB images through ``get_rgb_image()``, which is
the format expected by the Hugging Face OpenVLA processor.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the RealSense camera cannot provide a valid RGB frame."""


class RealSenseCamera:
    """RGB-only wrapper for an Intel RealSense D435i camera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.pipeline = None
        self.rs_config = None
        self.profile = None
        self.device = None
        self._rs = None
        self._started = False

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        """Connect to the camera and start the RGB stream."""

        if self._started:
            return

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "pyrealsense2 is not installed. On Ubuntu, install librealsense "
                "and then run `pip install pyrealsense2` inside the virtualenv."
            ) from exc

        self._rs = rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise CameraError(
                "No Intel RealSense device found. Check USB 3 connection, camera power, "
                "udev permissions, and verify with `realsense-viewer`."
            )

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )

        try:
            self.profile = self.pipeline.start(self.rs_config)
        except RuntimeError as exc:
            raise CameraError(
                "Failed to start RealSense RGB stream. Close other programs using the "
                "camera, reconnect the device, and try `realsense-viewer`."
            ) from exc

        self.device = self.profile.get_device()
        self._started = True
        self.print_camera_info()
        self._warm_up()

    def stop(self) -> None:
        """Stop streaming and close preview windows."""

        if self.pipeline is not None and self._started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        self._started = False
        try:
            cv2.destroyWindow(self.config.preview_window_name)
        except cv2.error:
            pass

    def print_camera_info(self) -> None:
        """Print model, serial, firmware, and active stream information."""

        if self.device is None or self._rs is None:
            print("RealSense camera information is unavailable before start().")
            return

        rs = self._rs

        def safe_info(field) -> str:
            try:
                if self.device.supports(field):
                    return self.device.get_info(field)
            except RuntimeError:
                return "unavailable"
            return "unavailable"

        print("Connected RealSense camera:")
        print(f"  Name:     {safe_info(rs.camera_info.name)}")
        print(f"  Serial:   {safe_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {safe_info(rs.camera_info.firmware_version)}")
        print(f"  RGB:      {self.config.width}x{self.config.height} @ {self.config.fps} FPS")
        print(f"  Config:   {asdict(self.config)}")

    def _warm_up(self) -> None:
        """Drop initial auto-exposure frames before inference starts."""

        if self.pipeline is None:
            return
        for _ in range(max(0, self.config.warmup_frames)):
            try:
                self.pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break

    def get_rgb_frame(self, timeout_ms: int = 5000) -> np.ndarray:
        """Return one RGB frame as a NumPy array with shape HxWx3."""

        if not self._started or self.pipeline is None:
            raise CameraError("RealSense camera is not started. Call camera.start() first.")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
        except RuntimeError as exc:
            raise CameraError(
                "RealSense frame capture timed out or the camera disconnected. "
                "Reconnect the D435i, confirm USB bandwidth, and restart the script."
            ) from exc

        if not color_frame:
            raise CameraError("RealSense returned an empty color frame.")

        bgr = np.asanyarray(color_frame.get_data())
        if bgr is None or bgr.size == 0:
            raise CameraError("RealSense returned an invalid image buffer.")

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_rgb_image(self, timeout_ms: int = 5000) -> Image.Image:
        """Return one RGB frame as a PIL image ready for OpenVLA."""

        rgb = self.get_rgb_frame(timeout_ms=timeout_ms)
        return Image.fromarray(rgb)

    def show_preview(self, image: Image.Image | np.ndarray, wait_ms: int = 1) -> Optional[int]:
        """Display a live RGB preview and return the pressed key code, if any."""

        if isinstance(image, Image.Image):
            rgb = np.asarray(image)
        else:
            rgb = image

        if rgb is None or rgb.size == 0:
            raise CameraError("Cannot display an empty image frame.")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.config.preview_window_name, bgr)
        key = cv2.waitKey(wait_ms) & 0xFF
        return key if key != 255 else None

    @staticmethod
    def sleep_for_rate(max_loop_hz: float, started_at: float) -> None:
        """Throttle the loop when a positive max-loop rate is configured."""

        if max_loop_hz <= 0:
            return
        target_dt = 1.0 / max_loop_hz
        elapsed = time.perf_counter() - started_at
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

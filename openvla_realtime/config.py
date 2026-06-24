"""Central configuration for the RealSense -> OpenVLA dry-run pipeline.

The defaults are intentionally conservative: inference is enabled, but robot
execution is dry-run only. Change values here or override them from the command
line in ``main.py`` when running experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ACTION_LABELS = (
    "Delta X",
    "Delta Y",
    "Delta Z",
    "Delta Roll",
    "Delta Pitch",
    "Delta Yaw",
    "Gripper",
)

ACTION_LABELS_COMPACT = ("dX", "dY", "dZ", "dRoll", "dPitch", "dYaw", "Grip")


@dataclass(frozen=True)
class CameraConfig:
    """Intel RealSense D435i color-stream settings."""

    width: int = 640
    height: int = 480
    fps: int = 30
    preview_window_name: str = "RealSense D435i RGB Preview"
    warmup_frames: int = 10


@dataclass(frozen=True)
class OpenVLAConfig:
    """OpenVLA model and inference settings."""

    model_id: str = "openvla/openvla-7b"
    instruction: str = "pick up the red block"
    unnorm_key: str = "bridge_orig"
    torch_dtype: str = "bfloat16"
    require_gpu: bool = True
    trust_remote_code: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime loop, logging, and safety defaults."""

    logs_dir: Path = Path("logs")
    image_dir: Path = Path("logs/images")
    csv_name: str = "openvla_inference_log.csv"
    save_image_every_n_frames: int = 30
    print_every_n_frames: int = 1
    max_loop_hz: float = 0.0
    enable_preview: bool = True
    enable_robot_adapter: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    """Workspace limits applied by all robot adapters.

    Units:
        Position deltas: meters
        Rotation deltas: radians
        Gripper command: normalized OpenVLA value, usually in [-1, 1]
    """

    robot_ip: str = "169.254.76.5"
    translation_scale: float = 1.0
    rotation_scale: float = 0.0
    max_translation_step: float = 0.01
    max_rotation_step: float = 0.017

    # 3x3 Coordinate transform matrix mapping OpenVLA outputs to UR5 base frame.
    #
    # Calibration result (confirmed 2026-06-18):
    #   UR5  +X = toward robot base
    #   UR5  +Y = left
    #   UR5  +Z = up
    #
    # OpenVLA bridge_orig training frame (BridgeV2 / WidowX):
    #   OpenVLA +X = AWAY from base, toward workspace objects  ← opposite of UR5
    #   OpenVLA +Y = left                                      ← same as UR5
    #   OpenVLA +Z = up                                        ← same as UR5
    #
    # Fix: invert X only.
    coordinate_transform: tuple[tuple[float, ...], ...] = (
        (-1.0, 0.0, 0.0),   # Invert X: OpenVLA "forward" → UR5 -X (away from base)
        ( 0.0, 1.0, 0.0),   # Y unchanged
        ( 0.0, 0.0, 1.0),   # Z unchanged
    )

    debug_step_size: float = 0.01  # Step size for terminal debug commands (meters)

    # Dead-zone: if the L2 norm of the clamped translation is below this value,
    # the movel() call is skipped entirely.  OpenVLA always outputs non-zero
    # actions even for "don't move" instructions, so this prevents drift.
    min_translation_magnitude: float = 0.0005  # 0.5 mm minimum meaningful step

    max_delta_xyz_m: float = 0.05
    max_delta_rpy_rad: float = 0.25
    min_gripper: float = -1.0
    max_gripper: float = 1.0

    workspace_x_min_m: float = -0.5
    workspace_x_max_m: float = 0.5
    workspace_y_min_m: float = -0.5
    workspace_y_max_m: float = 0.5
    workspace_z_min_m: float = 0.02
    workspace_z_max_m: float = 0.6


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    openvla: OpenVLAConfig = field(default_factory=OpenVLAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def build_openvla_prompt(instruction: str) -> str:
    """Return the prompt format used by the official OpenVLA examples."""

    cleaned = instruction.strip()
    if not cleaned:
        raise ValueError("Instruction is empty. Provide a task such as 'pick up the red block'.")
    return f"In: What action should the robot take to {cleaned}?\nOut:"

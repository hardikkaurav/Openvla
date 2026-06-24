"""Robot adapters.

Contains the URBasicRobot adapter for the live UR5 pipeline.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

try:
    import URBasic
except ImportError:
    # Depending on how the URBasic folder is set up, it might need to be added to PYTHONPATH
    pass

from config import SafetyConfig


class RobotSafetyError(RuntimeError):
    """Raised when an action violates configured safety limits."""


class URBasicRobot:
    """Real UR5 motion using the URBasic library."""

    def __init__(self, safety: SafetyConfig, enable_hardware: bool = True) -> None:
        self.safety = safety
        self.enable_hardware = enable_hardware
        self.robot = None
        self.robotModel = None
        self.connected = False
        self.paused = False          # Set True via terminal `pause` command to halt motion
        self._idle_count = 0         # Consecutive frames skipped due to dead-zone

    def connect(self) -> None:
        if not self.enable_hardware:
            print("URBasicRobot is disabled. No hardware connection.")
            return

        print(f"Connecting to UR5 at {self.safety.robot_ip} via URBasic...")
        try:
            # We must import URBasic here if it's placed dynamically, but assuming it's available.
            import URBasic
            
            self.robotModel = URBasic.robotModel.RobotModel()
            self.robot = URBasic.urScriptExt.UrScriptExt(
                host=self.safety.robot_ip,
                robotModel=self.robotModel
            )
            self.robot.reset_error()
            self.connected = True
            print("UR5 Connected Successfully.")
        except Exception as e:
            raise RobotSafetyError(f"Failed to connect to UR5: {e}")

    def get_tcp_pose(self) -> list[float] | None:
        """Returns the current [x, y, z, rx, ry, rz] TCP pose."""
        if not self.connected or not self.robotModel:
            return None
        return self.robotModel.ActualTCPPose()

    def move_delta(self, action: Sequence[float]) -> list[float] | None:
        """
        Convert OpenVLA 7-DoF action to a UR5 target pose and execute it.
        Action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if not self.connected or not self.robot:
            return None

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise RobotSafetyError(f"Expected 7 action values, received {action.size}.")

        # 1. Apply OpenVLA output scaling
        scaled_action = action.copy()
        scaled_action[:3] *= self.safety.translation_scale
        scaled_action[3:6] *= self.safety.rotation_scale

        # 1.5 Apply Coordinate Transformation Matrix to translations
        # OpenVLA action -> UR5 base frame mapping
        transform = np.array(self.safety.coordinate_transform)
        transformed_translation = transform @ scaled_action[:3]
        scaled_action[:3] = transformed_translation

        # 2. Apply step clamping for safety
        clipped = scaled_action.copy()
        clipped[:3] = np.clip(
            clipped[:3],
            -self.safety.max_translation_step,
            self.safety.max_translation_step,
        )

        # TEMPORARY DEBUG: Force all rotations and gripper to 0
        clipped[3:6] = 0.0
        gripper_cmd = 0.0

        # 2b. Dead-zone guard — OpenVLA always outputs non-zero actions even
        # for "don't move" instructions.  Skip movel() when the magnitude is
        # below the configured threshold to prevent unwanted drift.
        translation_magnitude = float(np.linalg.norm(clipped[:3]))
        if translation_magnitude < self.safety.min_translation_magnitude:
            self._idle_count += 1
            if self._idle_count % 10 == 1:   # print once every 10 skipped frames
                print(
                    f"[Dead-zone] Action magnitude {translation_magnitude:.5f} m < "
                    f"{self.safety.min_translation_magnitude:.5f} m threshold. "
                    f"Motion skipped. ({self._idle_count} consecutive idle frames)"
                )
            return self.get_tcp_pose()
        self._idle_count = 0

        # 2c. Paused check — operator typed `pause` in the terminal
        if self.paused:
            print("[PAUSED] Motion suppressed. Type `resume` to continue.")
            return self.get_tcp_pose()

        # 3. Read actual TCP pose
        current_pose = self.get_tcp_pose()
        if current_pose is None or len(current_pose) < 6:
            print("Failed to read actual TCP pose. Skipping motion.")
            return None

        # 4. Calculate target pose
        target_pose = [
            current_pose[0] + clipped[0],
            current_pose[1] + clipped[1],
            current_pose[2] + clipped[2],
            current_pose[3] + clipped[3],
            current_pose[4] + clipped[4],
            current_pose[5] + clipped[5],
        ]

        # 5. Check workspace bounds
        try:
            target_pose[0] = self._clip_workspace(target_pose[0], "x")
            target_pose[1] = self._clip_workspace(target_pose[1], "y")
            target_pose[2] = self._clip_workspace(target_pose[2], "z")
        except RobotSafetyError as e:
            print(f"Motion rejected: {e}")
            return current_pose

        print(
            "--- DEBUG MOTION LOG ---\n"
            f"Raw OpenVLA action:    dx={action[0]:+.5f}  dy={action[1]:+.5f}  dz={action[2]:+.5f}\n"
            f"Transformed action:    dx={scaled_action[0]:+.5f}  dy={scaled_action[1]:+.5f}  dz={scaled_action[2]:+.5f}\n"
            f"Clamped delta applied: dx={clipped[0]:+.5f}  dy={clipped[1]:+.5f}  dz={clipped[2]:+.5f}\n"
            f"Translation magnitude: {translation_magnitude:.5f} m\n"
            f"Current TCP Pose:      x={current_pose[0]:+.4f}  y={current_pose[1]:+.4f}  z={current_pose[2]:+.4f}\n"
            f"Final Target Pose:     x={target_pose[0]:+.4f}  y={target_pose[1]:+.4f}  z={target_pose[2]:+.4f}\n"
            "------------------------"
        )

        # 6. Execute motion
        self.robot.movel(target_pose, a=0.01, v=0.01)

        return target_pose

    def _clip_workspace(self, value: float, axis: str) -> float:
        if axis == "x":
            if not (self.safety.workspace_x_min_m <= value <= self.safety.workspace_x_max_m):
                raise RobotSafetyError(f"X position {value:.3f} outside workspace bounds.")
            return value
        if axis == "y":
            if not (self.safety.workspace_y_min_m <= value <= self.safety.workspace_y_max_m):
                raise RobotSafetyError(f"Y position {value:.3f} outside workspace bounds.")
            return value
        if axis == "z":
            if not (self.safety.workspace_z_min_m <= value <= self.safety.workspace_z_max_m):
                raise RobotSafetyError(f"Z position {value:.3f} outside workspace bounds.")
            return value
        raise RobotSafetyError(f"Unknown workspace axis `{axis}`.")

    def close(self) -> None:
        if self.connected and self.robot:
            print("Closing UR5 connection...")
            self.robot.close()
            self.connected = False

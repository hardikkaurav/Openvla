"""Real-time Intel RealSense D435i -> OpenVLA -> UR5 action pipeline."""

from __future__ import annotations

import argparse
import sys
import time
import threading
import queue
from dataclasses import replace
from pathlib import Path

import numpy as np

from camera import CameraError, RealSenseCamera
from config import AppConfig
from openvla_policy import OpenVLAError, OpenVLAPolicy
from utils.logging_utils import InferenceLogger
from utils.robot_adapters import URBasicRobot, RobotSafetyError
from visualizer import ActionVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVLA on live RealSense D435i RGB frames.")
    parser.add_argument("--test", action="store_true", help="Capture one frame, run one inference, print action, and exit.")
    parser.add_argument("--instruction", default=None, help="Language instruction for OpenVLA.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model ID. Defaults to openvla/openvla-7b.")
    parser.add_argument("--unnorm-key", default=None, help="OpenVLA action unnormalization key. Defaults to bridge_orig.")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV visualization windows.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback. This is not recommended for OpenVLA 7B.")
    parser.add_argument("--logs-dir", default=None, help="Directory for CSV logs.")
    parser.add_argument("--image-dir", default=None, help="Directory for periodic image saves.")
    parser.add_argument("--save-image-every", type=int, default=None, help="Save every N frames. Use 0 to disable image saves.")
    parser.add_argument("--max-loop-hz", type=float, default=None, help="Optional loop throttle. 0 means unthrottled.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()

    openvla = base.openvla
    if args.instruction is not None:
        openvla = replace(openvla, instruction=args.instruction)
    if args.model_id is not None:
        openvla = replace(openvla, model_id=args.model_id)
    if args.unnorm_key is not None:
        openvla = replace(openvla, unnorm_key=args.unnorm_key)
    if args.allow_cpu:
        openvla = replace(openvla, require_gpu=False)

    runtime = base.runtime
    if args.no_preview:
        runtime = replace(runtime, enable_preview=False)
    if args.logs_dir is not None:
        runtime = replace(runtime, logs_dir=Path(args.logs_dir))
    if args.image_dir is not None:
        runtime = replace(runtime, image_dir=Path(args.image_dir))
    if args.save_image_every is not None:
        runtime = replace(runtime, save_image_every_n_frames=args.save_image_every)
    if args.max_loop_hz is not None:
        runtime = replace(runtime, max_loop_hz=args.max_loop_hz)

    return replace(base, openvla=openvla, runtime=runtime)


def command_input_thread(cmd_queue: queue.Queue) -> None:
    """Thread function to read commands from the terminal without blocking."""
    while True:
        try:
            new_cmd = input()
            if new_cmd:
                cmd_queue.put(new_cmd)
        except EOFError:
            break


def main() -> int:
    args = parse_args()
    config = build_config(args)

    print("OpenVLA Real-Time D435i UR5 Pipeline")
    print("Mode: test" if args.test else "Mode: real-time")
    print(f"Instruction: {config.openvla.instruction}")
    print("Safety: Warning! Real hardware control enabled.")

    robot = None
    try:
        policy = OpenVLAPolicy(config.openvla)
        policy.load()

        logger = InferenceLogger(
            logs_dir=config.runtime.logs_dir,
            image_dir=config.runtime.image_dir,
            csv_name=config.runtime.csv_name,
            save_image_every_n_frames=config.runtime.save_image_every_n_frames,
        )
        
        if config.runtime.enable_robot_adapter:
            robot = URBasicRobot(config.safety, enable_hardware=True)
            robot.connect()
            
        visualizer = ActionVisualizer() if config.runtime.enable_preview else None

        with RealSenseCamera(config.camera) as camera:
            if args.test:
                return run_test_mode(config, camera, policy, logger, robot, visualizer)
            return run_realtime_mode(config, camera, policy, logger, robot, visualizer)

    except (CameraError, OpenVLAError, RobotSafetyError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nInterrupted by user. Shutting down cleanly.")
            if robot:
                robot.close()
            return 130
        print(f"\nERROR: {exc}", file=sys.stderr)
        print_recovery_help(exc)
        if robot:
            robot.close()
        return 1


def run_test_mode(
    config: AppConfig,
    camera: RealSenseCamera,
    policy: OpenVLAPolicy,
    logger: InferenceLogger,
    robot: URBasicRobot | None,
    visualizer: ActionVisualizer | None,
) -> int:
    image = camera.get_rgb_image()
    result = policy.predict(image, config.openvla.instruction)
    fps = 1.0 / result.inference_time_s if result.inference_time_s > 0 else 0.0

    print_action(result.action, result.inference_time_s, fps)
    logger.log(0, config.openvla.instruction, result.action, result.inference_time_s, fps, image=image)
    
    tcp_pose = None
    if robot is not None:
        robot.move_delta(result.action)
        tcp_pose = robot.get_tcp_pose()
        
    if visualizer is not None:
        visualizer.update(
            image, 
            result.action, 
            result.inference_time_s, 
            fps, 
            config.openvla.instruction,
            tcp_pose=tcp_pose,
            ur5_connected=(robot is not None and robot.connected)
        )
        time.sleep(1.0)
        visualizer.close()
        
    if robot is not None:
        robot.close()
        
    return 0


def run_realtime_mode(
    config: AppConfig,
    camera: RealSenseCamera,
    policy: OpenVLAPolicy,
    logger: InferenceLogger,
    robot: URBasicRobot | None,
    visualizer: ActionVisualizer | None,
) -> int:
    frame_index = 0
    last_frame_time = time.perf_counter()

    cmd_queue = queue.Queue()
    input_thread = threading.Thread(target=command_input_thread, args=(cmd_queue,), daemon=True)
    input_thread.start()

    print("Starting real-time loop. Press q/Esc in the preview window or Ctrl+C to exit.")
    print(
        "Terminal commands: pause | resume | status | help | move +x/-x/+y/-y/+z/-z | q | <instruction text>"
    )

    while True:
        manual_debug_action = None
        try:
            while not cmd_queue.empty():
                new_instruction = cmd_queue.get_nowait()
                cleaned_cmd = new_instruction.strip().lower()

                if cleaned_cmd == 'q':
                    print("Exit requested from terminal.")
                    if visualizer is not None:
                        visualizer.close()
                    if robot is not None:
                        robot.close()
                    return 0

                elif cleaned_cmd == 'pause':
                    if robot is not None:
                        robot.paused = True
                    print("[PAUSED] Robot motion halted. Type `resume` to continue.")

                elif cleaned_cmd == 'resume':
                    if robot is not None:
                        robot.paused = False
                    print("[RESUMED] Robot motion active.")

                elif cleaned_cmd == 'status':
                    tcp = robot.get_tcp_pose() if robot else None
                    paused = robot.paused if robot else False
                    print(
                        f"\n--- STATUS ---\n"
                        f"  Paused:       {paused}\n"
                        f"  Instruction:  {config.openvla.instruction}\n"
                        f"  TCP Pose:     {tcp}\n"
                        f"  Dead-zone:    {config.safety.min_translation_magnitude:.5f} m\n"
                        f"--------------"
                    )

                elif cleaned_cmd == 'help':
                    print(
                        "\n--- AVAILABLE COMMANDS ---\n"
                        "  pause          Stop robot motion immediately\n"
                        "  resume         Resume OpenVLA-driven motion\n"
                        "  status         Print current pose, instruction, and state\n"
                        "  move +x/−x     Debug: step robot exactly once in X axis\n"
                        "  move +y/−y     Debug: step robot exactly once in Y axis\n"
                        "  move +z/−z     Debug: step robot exactly once in Z axis\n"
                        "  q              Quit\n"
                        "  <any text>     Change the active OpenVLA instruction\n"
                        "--------------------------"
                    )

                elif cleaned_cmd.startswith("move "):
                    axis_val = cleaned_cmd.split(" ")[1]
                    step = config.safety.debug_step_size
                    action = np.zeros(7, dtype=np.float32)
                    if axis_val == "+x":   action[0] = step
                    elif axis_val == "-x": action[0] = -step
                    elif axis_val == "+y": action[1] = step
                    elif axis_val == "-y": action[1] = -step
                    elif axis_val == "+z": action[2] = step
                    elif axis_val == "-z": action[2] = -step
                    else:
                        print(f"Unknown debug move command: '{axis_val}'. Try: move +x  move -y  move +z")
                        continue
                    manual_debug_action = action
                    print(f"Manual debug step intercepted: {cleaned_cmd}")

                else:
                    config.openvla = replace(config.openvla, instruction=new_instruction.strip())
                    print(f"Instruction updated to: {config.openvla.instruction}")

        except queue.Empty:
            pass

        loop_started = time.perf_counter()
        image = camera.get_rgb_image()
        
        if manual_debug_action is not None:
            result_action = manual_debug_action
            inference_time_s = 0.0
            print("Bypassing OpenVLA inference for manual debug step.")
        else:
            result = policy.predict(image, config.openvla.instruction)
            result_action = result.action
            inference_time_s = result.inference_time_s

        now = time.perf_counter()
        fps = 1.0 / max(now - last_frame_time, 1e-9)
        last_frame_time = now

        if frame_index % max(1, config.runtime.print_every_n_frames) == 0:
            print_action(result_action, inference_time_s, fps)

        logger.log(
            frame_index=frame_index,
            instruction=config.openvla.instruction,
            action=result_action,
            inference_time_s=inference_time_s,
            fps=fps,
            image=image,
        )

        tcp_pose = None
        if robot is not None:
            robot.move_delta(result_action)
            tcp_pose = robot.get_tcp_pose()

        key = None
        if visualizer is not None:
            key = visualizer.update(
                image, 
                result_action, 
                inference_time_s, 
                fps, 
                config.openvla.instruction,
                tcp_pose=tcp_pose,
                ur5_connected=(robot is not None and robot.connected)
            )
        elif config.runtime.enable_preview:
            key = camera.show_preview(image)

        if key in (ord("q"), 27):
            print("Exit requested from preview window.")
            break

        frame_index += 1
        RealSenseCamera.sleep_for_rate(config.runtime.max_loop_hz, loop_started)

    if visualizer is not None:
        visualizer.close()
    if robot is not None:
        robot.close()
    return 0


def print_action(action: np.ndarray, inference_time_s: float, fps: float) -> None:
    formatted = np.array2string(np.asarray(action, dtype=np.float32), precision=5, separator=", ")
    print(f"Action: {formatted} | inference={inference_time_s * 1000:.1f} ms | FPS={fps:.2f}")


def print_recovery_help(exc: BaseException) -> None:
    """Print focused recovery steps based on the type of failure."""

    if isinstance(exc, CameraError):
        print(
            "Recovery:\n"
            "  1. Confirm the D435i is connected via USB 3.\n"
            "  2. Run `realsense-viewer` to verify the camera outside Python.\n"
            "  3. Check Linux udev permissions for librealsense.\n"
            "  4. Restart this script after reconnecting the camera.",
            file=sys.stderr,
        )
    elif isinstance(exc, OpenVLAError):
        print(
            "Recovery:\n"
            "  1. Verify `nvidia-smi` shows an NVIDIA GPU.\n"
            "  2. Install CUDA-enabled PyTorch, not CPU-only PyTorch.\n"
            "  3. Confirm startup diagnostics show bf16 model and pixel tensors.\n"
            "  4. Make sure Hugging Face can download `openvla/openvla-7b`.",
            file=sys.stderr,
        )
    elif isinstance(exc, RobotSafetyError):
        print(
            "Recovery:\n"
            "  1. Inspect the predicted action values.\n"
            "  2. Tighten or adjust safety limits in config.py before testing again.\n"
            "  3. Verify the UR5 is powered on, e-stop is cleared, and remote control is enabled.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())

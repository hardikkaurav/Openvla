# OpenVLA Real-Time RealSense Pipeline

This project runs an inference-only Vision-Language-Action pipeline on an Ubuntu laboratory workstation:

```text
Intel RealSense D435i
          |
          v
      OpenVLA
          |
          v
   Action Vector
          |
          v
    DryRunRobot
```

The system streams RGB frames from an Intel RealSense D435i, sends each frame plus a language instruction to `openvla/openvla-7b`, prints a 7-DoF action vector, logs results to CSV, saves images periodically, and visualizes the action vector live.

Important: this project does not move a real robot. The default adapter is `DryRunRobot`, which safety-clamps and prints simulated target poses only.

## Target Platform

- Ubuntu 22.04
- Python 3.10+
- NVIDIA GPU
- NVIDIA driver visible through `nvidia-smi`
- CUDA-enabled PyTorch
- Intel RealSense D435i connected over USB 3

## Files

```text
openvla_realtime/
|
├── main.py
├── camera.py
├── openvla_policy.py
├── visualizer.py
├── config.py
├── requirements.txt
├── README.md
└── utils/
    ├── __init__.py
    ├── logging_utils.py
    └── robot_adapters.py
```

## System Setup

Install NVIDIA drivers and confirm the GPU:

```bash
nvidia-smi
```

Install Intel RealSense support. Follow the official librealsense Ubuntu instructions for Ubuntu 22.04, then verify the camera:

```bash
realsense-viewer
```

If the camera is not visible, check USB 3 cabling, udev rules, and whether another program is already using the device.

## Python Environment

From the repository root:

```bash
cd openvla_realtime
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch for your CUDA version. Example for current PyTorch CUDA wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you already installed CUDA-enabled `torch` and `torchvision`, `pip` may report them as satisfied.

## Configure the Instruction

The default instruction is in `config.py`:

```python
instruction = "pick up the red block"
```

You can also override it at runtime:

```bash
python main.py --instruction "pick up the red block"
```

The prompt is generated automatically:

```text
In: What action should the robot take to pick up the red block?
Out:
```

## Test Mode

Capture one frame, run one OpenVLA prediction, print the action, log it, show the visualization briefly, and exit:

```bash
python main.py --test
```

## Real-Time Mode

Run continuously until `q`, `Esc`, or `Ctrl+C`:

```bash
python main.py
```

Useful options:

```bash
python main.py --no-preview
python main.py --instruction "move the blue cup to the left"
python main.py --save-image-every 10
python main.py --max-loop-hz 2
```

## OpenVLA Loading

The model defaults to:

```python
MODEL_ID = "openvla/openvla-7b"
```

The loader uses:

- `AutoProcessor`
- `AutoModelForVision2Seq`
- `trust_remote_code=True`
- automatic CUDA GPU detection
- full-weight `torch.bfloat16` inference by default on RTX 5090
- automatic 4-bit `bitsandbytes` fallback only if full bf16 memory exceeds 20 GB or OOMs
- CUDA memory reporting before and after model load
- startup diagnostics for GPU model, CUDA version, model dtype, `pixel_values` dtype, and parameter dtype summary

OpenVLA 7B is large. The first run downloads model weights from Hugging Face and can take several minutes.

## Logs

CSV logs are written to:

```text
logs/openvla_inference_log.csv
```

Each row contains:

- UTC timestamp
- frame index
- instruction
- action vector values
- inference time
- FPS
- saved image path when an image is saved

Images are periodically saved to:

```text
logs/images/
```

Change the image interval with:

```bash
python main.py --save-image-every 30
```

Use `0` to disable image saving.

## Action Vector

OpenVLA returns a 7-DoF action vector:

```text
Delta X, Delta Y, Delta Z, Delta Roll, Delta Pitch, Delta Yaw, Gripper
```

The visualizer displays the live camera frame and a continuously updated bar chart for:

```text
Delta X
Delta Y
Delta Z
Delta Roll
Delta Pitch
Delta Yaw
Gripper
```

## Safety

The current robot adapter is:

```python
DryRunRobot
```

It never sends commands to physical hardware. It:

- clips action deltas to conservative limits
- updates a simulated end-effector pose
- clips simulated position to workspace limits
- prints the target pose

## Future UR5 RTDE Integration

`utils/robot_adapters.py` includes a disabled `UR5RTDERobot` placeholder for future work with `ur_rtde`.

Do not enable UR5 motion until all of the following exist and are tested:

- physical emergency stop
- reduced-speed first-motion procedure
- workspace limits
- joint limits
- TCP and payload validation
- collision risk assessment
- operator line-of-sight procedure
- audited RTDE motion code

The placeholder intentionally raises `NotImplementedError` before any hardware command can be sent.

## Troubleshooting

Missing GPU:

```text
CUDA GPU was not detected.
```

Run `nvidia-smi`, install the NVIDIA driver, and install CUDA-enabled PyTorch. CPU fallback is available with `--allow-cpu`, but it is not practical for real-time OpenVLA 7B inference.

Camera not found:

```text
No Intel RealSense device found.
```

Reconnect the D435i over USB 3, verify with `realsense-viewer`, and check librealsense udev permissions.

Camera disconnected:

```text
RealSense frame capture timed out or the camera disconnected.
```

Reconnect the camera and restart the script.

OpenVLA loading failure:

Check internet access, Hugging Face cache permissions, CUDA PyTorch, and available GPU memory.

CUDA out of memory:

Close other GPU jobs and restart Python. On the RTX 5090 path, the loader tries full bf16 first and falls back to 4-bit only if full bf16 exceeds 20 GB or OOMs.

Invalid image frame:

Confirm the RealSense RGB stream is active in `realsense-viewer` and no other program is consuming the stream.

## Notes on Robot Transfer

The default `unnorm_key` is `bridge_orig`, matching the commonly shown OpenVLA BridgeV2 example. For a UR5, the model should be fine-tuned or calibrated for the target robot and workspace before actions are considered meaningful for real execution.

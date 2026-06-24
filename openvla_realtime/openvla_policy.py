"""OpenVLA model loading and action prediction for an RTX 5090 workstation.

This loader is intentionally optimized for a modern 32 GB NVIDIA workstation
GPU rather than a memory-constrained Colab/T4 setup:

- Prefer full-weight bfloat16 inference on CUDA.
- Keep model weights and floating inference tensors in one consistent dtype.
- Load without 4-bit quantization first.
- Only fall back to 4-bit quantization if full bf16 exceeds 20 GB or OOMs.
- Print startup dtype diagnostics that catch float/bfloat16 mismatches early.
"""

from __future__ import annotations

import gc
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from config import OpenVLAConfig, build_openvla_prompt


FULL_PRECISION_MEMORY_LIMIT_GB = 20.0


class OpenVLAError(RuntimeError):
    """Raised for model loading or inference failures."""


@dataclass(frozen=True)
class PredictionResult:
    """One policy inference result."""

    action: np.ndarray
    inference_time_s: float
    prompt: str


class OpenVLAPolicy:
    """Hugging Face OpenVLA wrapper for RTX 5090 bfloat16 inference."""

    def __init__(self, config: OpenVLAConfig) -> None:
        self.config = config
        self.processor = None
        self.model = None
        self.torch = None
        self.device = None
        self.compute_dtype = None
        self.is_quantized = False
        self.model_memory_gb = 0.0

    def load(self) -> None:
        """Load OpenVLA with bf16 full weights, falling back to 4-bit only when needed."""

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise OpenVLAError(
                "Missing OpenVLA dependencies. Install the OpenVLA-recommended versions: "
                "`transformers==4.40.1` and `tokenizers==0.19.1`, then run "
                "`pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.device = self._select_device(torch)
        self.compute_dtype = self._select_compute_dtype(torch)

        self._print_startup_header()

        print(f"Loading OpenVLA processor: {self.config.model_id}")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as exc:
            raise OpenVLAError(
                "Failed to load the OpenVLA processor from Hugging Face. Check internet "
                "access, Hugging Face cache permissions, and the model ID."
            ) from exc

        self._report_cuda_memory("before model load")

        try:
            self._load_full_precision_model(AutoModelForVision2Seq)
            self.model_memory_gb = self._allocated_cuda_gb()
            print(f"OpenVLA full-weight memory after load: {self.model_memory_gb:.2f} GB")

            if self.model_memory_gb > FULL_PRECISION_MEMORY_LIMIT_GB:
                print(
                    f"Full bf16 model uses {self.model_memory_gb:.2f} GB, which exceeds "
                    f"{FULL_PRECISION_MEMORY_LIMIT_GB:.1f} GB. Reloading with 4-bit quantization."
                )
                self._unload_model()
                self._load_quantized_model(AutoModelForVision2Seq, BitsAndBytesConfig)
                self.model_memory_gb = self._allocated_cuda_gb()
        except torch.cuda.OutOfMemoryError:
            self._cleanup_cuda()
            print(
                "CUDA out of memory while loading full bf16 OpenVLA. "
                "Falling back to 4-bit quantization."
            )
            try:
                self._load_quantized_model(AutoModelForVision2Seq, BitsAndBytesConfig)
                self.model_memory_gb = self._allocated_cuda_gb()
            except Exception as exc:
                self._cleanup_cuda()
                raise OpenVLAError(
                    "CUDA out of memory while loading OpenVLA, even with the quantized fallback. "
                    "Close other GPU jobs, verify available VRAM with `nvidia-smi`, and restart Python."
                ) from exc
        except Exception as exc:
            raise OpenVLAError(
                "Failed to load OpenVLA. Confirm CUDA PyTorch is installed, the GPU driver "
                "works (`nvidia-smi`), and Hugging Face can download `openvla/openvla-7b`."
            ) from exc

        self._report_cuda_memory("after model load")
        self._print_parameter_dtype_summary()
        self._print_model_dtype()

    def predict(self, image: Image.Image, instruction: str | None = None) -> PredictionResult:
        """Run one OpenVLA action prediction for a PIL RGB image."""

        if self.model is None or self.processor is None or self.torch is None or self.device is None:
            raise OpenVLAError("OpenVLA model is not loaded. Call policy.load() first.")

        if not isinstance(image, Image.Image):
            raise OpenVLAError("OpenVLA expected a PIL.Image.Image input.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = build_openvla_prompt(instruction or self.config.instruction)

        try:
            inputs = self.processor(prompt, image)
            inputs = self._move_inputs_to_device_and_dtype(inputs)
            self._print_pixel_dtype_once(inputs)
        except Exception as exc:
            raise OpenVLAError(
                "Failed to preprocess the camera image. Verify the frame is a valid RGB image."
            ) from exc

        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key=self.config.unnorm_key,
                    do_sample=False,
                )
        except self.torch.cuda.OutOfMemoryError as exc:
            self._cleanup_cuda()
            raise OpenVLAError(
                "CUDA out of memory during inference. Close other GPU jobs, restart Python, "
                "or allow the loader to use the 4-bit fallback if full bf16 exceeds 20 GB."
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "mat1 and mat2" in message and "dtype" in message:
                raise OpenVLAError(
                    "OpenVLA dtype mismatch during inference. This loader casts all floating "
                    f"processor tensors to {self.compute_dtype} and loads model weights in the "
                    "same dtype. Check the startup diagnostics for `pixel_values dtype` and "
                    "`parameter dtype summary`."
                ) from exc
            raise OpenVLAError(
                "OpenVLA inference failed. If this is a new robot/domain, check whether "
                f"the unnormalization key `{self.config.unnorm_key}` is available."
            ) from exc
        except Exception as exc:
            raise OpenVLAError(
                "OpenVLA inference failed. Check image validity, CUDA health, and model cache integrity."
            ) from exc

        elapsed = time.perf_counter() - start
        return PredictionResult(action=self._to_numpy_action(action), inference_time_s=elapsed, prompt=prompt)

    def _load_full_precision_model(self, AutoModelForVision2Seq) -> None:
        """Load full OpenVLA weights in the single selected inference dtype."""

        print(
            "Loading OpenVLA full weights without 4-bit quantization "
            f"(dtype={self.compute_dtype})."
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()
        self.is_quantized = False

    def _load_quantized_model(self, AutoModelForVision2Seq, BitsAndBytesConfig) -> None:
        """Load OpenVLA with 4-bit weights using the same bf16 compute dtype."""

        if self.device.type != "cuda":
            raise OpenVLAError("4-bit quantization requires a CUDA GPU.")

        print(
            "Loading OpenVLA with 4-bit quantization because full bf16 exceeded the "
            f"{FULL_PRECISION_MEMORY_LIMIT_GB:.1f} GB limit or OOMed."
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=quantization_config,
            device_map={"": self.device.index or 0},
        )
        self.model.eval()
        self.is_quantized = True

    def _move_inputs_to_device_and_dtype(self, inputs):
        """Move tensors to CUDA and cast only floating tensors to the model dtype.

        Token IDs and masks must remain integer/bool tensors. The important fix for
        `mat1 float != bfloat16` is explicitly casting `pixel_values` and any other
        floating processor output to the exact same dtype as the model.
        """

        converted = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(dtype=self.compute_dtype)
            converted[key] = value
        return converted

    def _select_device(self, torch):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            name = torch.cuda.get_device_name(device)
            capability = torch.cuda.get_device_capability(device)
            print(f"Using CUDA GPU: {name} (compute capability {capability[0]}.{capability[1]})")
            return device

        message = (
            "CUDA GPU was not detected. This project targets the RTX 5090 workstation. "
            "Install the NVIDIA driver, CUDA-enabled PyTorch, and verify with `nvidia-smi`."
        )
        if self.config.require_gpu:
            raise OpenVLAError(message)

        print(f"WARNING: {message} Falling back to CPU; OpenVLA will be extremely slow.")
        return torch.device("cpu")

    def _select_compute_dtype(self, torch):
        """Prefer bf16 on RTX 5090 and avoid mixed float32/bf16 inference paths."""

        requested_dtype = self._dtype_from_name(torch, self.config.torch_dtype)
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                if requested_dtype is not torch.bfloat16:
                    print("RTX workstation supports bfloat16; using torch.bfloat16 for OpenVLA.")
                return torch.bfloat16
            print("WARNING: CUDA device does not report bf16 support; using torch.float16.")
            return torch.float16

        if requested_dtype is torch.float32:
            return torch.float32
        print("WARNING: CPU fallback uses torch.float32 for compatibility.")
        return torch.float32

    @staticmethod
    def _dtype_from_name(torch, name: str):
        normalized = name.lower().strip()
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise OpenVLAError(f"Unsupported torch dtype `{name}`.")

    @staticmethod
    def _to_numpy_action(action) -> np.ndarray:
        if hasattr(action, "detach"):
            action = action.detach().cpu().float().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise OpenVLAError(f"Expected a 7-DoF action vector, got shape {action.shape}.")
        return action

    def _print_startup_header(self) -> None:
        torch = self.torch
        cuda_version = getattr(torch.version, "cuda", "unavailable")
        print("OpenVLA startup diagnostics:")
        print(f"  GPU model:    {torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'}")
        print(f"  CUDA version: {cuda_version}")
        print(f"  model dtype:  {self.compute_dtype}")
        print(f"  quantization: disabled by default; fallback only above {FULL_PRECISION_MEMORY_LIMIT_GB:.1f} GB")

    def _print_model_dtype(self) -> None:
        dtype = self._first_parameter_dtype()
        print(f"  active model dtype: {dtype}")
        print(f"  quantized fallback: {self.is_quantized}")

    def _print_pixel_dtype_once(self, inputs) -> None:
        if getattr(self, "_printed_pixel_dtype", False):
            return
        pixel_values = inputs.get("pixel_values")
        dtype = getattr(pixel_values, "dtype", "missing")
        print(f"  pixel_values dtype: {dtype}")
        self._printed_pixel_dtype = True

    def _print_parameter_dtype_summary(self) -> None:
        summary = self._parameter_dtype_summary()
        formatted = ", ".join(f"{dtype}: {count}" for dtype, count in summary.items())
        print(f"  parameter dtype summary: {formatted or 'no parameters found'}")

    def _parameter_dtype_summary(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        if self.model is None:
            return counter
        for parameter in self.model.parameters():
            counter[str(parameter.dtype)] += int(parameter.numel())
        return counter

    def _first_parameter_dtype(self):
        if self.model is None:
            return "unloaded"
        for parameter in self.model.parameters():
            return parameter.dtype
        return "no parameters"

    def _allocated_cuda_gb(self) -> float:
        if self.torch is None or self.device is None or self.device.type != "cuda":
            return 0.0
        self.torch.cuda.synchronize(self.device)
        return self.torch.cuda.memory_allocated(self.device) / (1024**3)

    def _report_cuda_memory(self, stage: str) -> None:
        if self.torch is None or self.device is None or self.device.type != "cuda":
            return
        torch = self.torch
        allocated_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(self.device) / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        print(
            f"CUDA memory {stage}: allocated={allocated_gb:.2f} GB, "
            f"reserved={reserved_gb:.2f} GB, free={free_bytes / (1024**3):.2f} GB, "
            f"total={total_bytes / (1024**3):.2f} GB"
        )

    def _unload_model(self) -> None:
        self.model = None
        self.is_quantized = False
        self._cleanup_cuda()

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

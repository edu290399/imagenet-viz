"""Diagnóstico de dispositivo e memória para a bancada de inferência."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import torch


def is_cuda_oom(exc: BaseException) -> bool:
    """Reconhece estouro de memória da GPU sem depender só do tipo da exceção."""
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def explain_failure(exc: BaseException) -> str:
    if is_cuda_oom(exc):
        return (
            "A GPU ficou sem memória (CUDA out of memory). "
            "Use Liberar VRAM, feche outros programas que estejam usando a placa "
            "ou escolha um modelo menor, como ResNet-18 ou MobileNetV3-Large. "
            f"Detalhe técnico: {exc}"
        )
    return str(exc)


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_runtime() -> None:
    """TF32 e cuDNN benchmark deixam a medição estável em GPUs Ampere ou mais novas."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        return f"cuda:{index}"
    return device.type


def detect_device() -> torch.device:
    """Escolhe cuda:0, depois MPS e, por último, CPU.

    `IMAGENET_VIZ_DEVICE` força um dispositivo (`cuda:0`, `cpu` ou `mps`).
    """
    override = os.environ.get("IMAGENET_VIZ_DEVICE", "").strip()
    if override:
        alias = {"cuda": "cuda:0", "gpu": "cuda:0"}
        device = torch.device(alias.get(override.lower(), override))
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "IMAGENET_VIZ_DEVICE pede CUDA, mas este PyTorch não enxerga uma GPU. "
                "Confira o driver NVIDIA e se o wheel instalado não é o de CPU."
            )
        if device.type == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("IMAGENET_VIZ_DEVICE=mps, mas o backend MPS não está disponível.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


@dataclass(frozen=True)
class HardwareSnapshot:
    device: torch.device
    backend: str
    gpu_name: str
    capability: str | None
    torch_version: str
    cuda_build: str | None
    total_bytes: int | None
    free_bytes: int | None
    allocated_bytes: int | None
    reserved_bytes: int | None
    tf32: bool
    cudnn_benchmark: bool


def snapshot(device: torch.device | None = None) -> HardwareSnapshot:
    device = device or detect_device()
    total = free = allocated = reserved = None
    capability = None
    name = gpu_name(device)
    tf32 = False
    cudnn_benchmark = False

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        name = props.name
        capability = f"{props.major}.{props.minor} (sm_{props.major}{props.minor})"
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    elif device.type == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            allocated = int(current())

    backend = {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}.get(device.type, device.type)
    return HardwareSnapshot(
        device=device,
        backend=backend,
        gpu_name=name,
        capability=capability,
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
        total_bytes=total,
        free_bytes=free,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        tf32=tf32,
        cudnn_benchmark=cudnn_benchmark,
    )


def format_bytes(num: int | None) -> str:
    if num is None:
        return "—"
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"

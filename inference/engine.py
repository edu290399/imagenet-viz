"""Forward em modo de inferência, com latência medida na GPU e Top-5."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from inference.cam import CamResult, compute_grad_cam
from models.zoo import CPU_FALLBACK_NOTE, LoadedModel, ModelZoo
from utils.hardware import device_label, gpu_name, is_cuda_oom, release_cuda_cache

logger = logging.getLogger("imagenet_viz.engine")

WARMUP_ITERS = 10
MEASURE_ITERS = 30

_TELEMETRY: dict[tuple[str, str], "Telemetry"] = {}


@dataclass(frozen=True)
class ClassScore:
    index: int
    name: str
    probability: float


@dataclass(frozen=True)
class Telemetry:
    latency_ms: float
    latency_std_ms: float
    samples: int
    warmup: int
    timing_method: str
    activation_peak_bytes: int
    allocated_bytes: int
    free_bytes: int | None
    total_bytes: int | None
    device: str
    gpu_name: str


@dataclass
class Analysis:
    model_name: str
    scores: list[ClassScore]
    telemetry: Telemetry
    cam: CamResult
    warnings: list[str] = field(default_factory=list)


def clear_telemetry() -> None:
    _TELEMETRY.clear()


def coerce_image(image) -> Image.Image:
    if isinstance(image, dict):
        image = image.get("path") or image.get("name") or image.get("image")
    if image is None:
        raise ValueError("Envie uma imagem ou escolha um dos exemplos.")
    try:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, np.ndarray):
            array = image
            if array.dtype != np.uint8:
                floats = array.astype(np.float32)
                if floats.size and float(np.nanmax(floats)) <= 1.0:
                    floats *= 255.0
                array = np.clip(floats, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            return ImageOps.exif_transpose(Image.fromarray(array)).convert("RGB")
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                rgb = ImageOps.exif_transpose(handle).convert("RGB")
                rgb.load()
                return rgb
    except UnidentifiedImageError as exc:
        raise ValueError("Não consegui abrir essa imagem. Use JPG ou PNG.") from exc
    except OSError as exc:
        raise ValueError(f"Falha ao ler a imagem: {exc}") from exc
    raise TypeError(f"Tipo de imagem não suportado: {type(image).__name__}.")


def preprocess(image: Image.Image, loaded: LoadedModel) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    try:
        tensor = loaded.transform(image)
    except Exception as exc:
        raise ValueError(
            f"O pré-processamento de {loaded.spec.name} rejeitou a imagem: {exc}"
        ) from exc
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float().div(255.0)
    return tensor.to(loaded.device)


def benchmark_model(
    zoo: ModelZoo,
    name: str,
    preferred: torch.device,
    *,
    refresh: bool = False,
) -> tuple[LoadedModel, Telemetry]:
    """Latência de batch 1 com um tensor sintético do tamanho que a rede espera."""
    with zoo.guard():
        loaded = zoo.load(name, preferred)
        key = (loaded.spec.name, device_label(loaded.device))
        if not refresh and key in _TELEMETRY:
            return loaded, _TELEMETRY[key]
        try:
            telemetry = measure_module(loaded)
        except Exception as exc:
            if not (is_cuda_oom(exc) and loaded.device.type == "cuda"):
                raise
            release_cuda_cache()
            loaded = zoo.relocate_to_cpu(loaded, CPU_FALLBACK_NOTE)
            telemetry = measure_module(loaded)
            key = (loaded.spec.name, device_label(loaded.device))
        _TELEMETRY[key] = telemetry
        return loaded, telemetry


def analyze_image(zoo: ModelZoo, name: str, image, preferred: torch.device) -> Analysis:
    """Top-5, latência da foto e Grad-CAM. CUDA OOM cai para CPU em vez de derrubar a UI."""
    photo = coerce_image(image)
    with zoo.guard():
        loaded = zoo.load(name, preferred)
        try:
            scores, telemetry = predict(loaded, photo)
        except Exception as exc:
            if not (is_cuda_oom(exc) and loaded.device.type == "cuda"):
                raise
            release_cuda_cache()
            loaded = zoo.relocate_to_cpu(loaded, CPU_FALLBACK_NOTE)
            scores, telemetry = predict(loaded, photo)

        batch = preprocess(photo, loaded)
        cam = _cam_with_fallback(zoo, loaded, batch, scores[0])
        warnings = list(loaded.notes)
        for note in cam.warnings:
            if note not in warnings:
                warnings.append(note)
        return Analysis(
            model_name=loaded.spec.name,
            scores=scores,
            telemetry=telemetry,
            cam=cam,
            warnings=warnings,
        )


def measure_module(loaded: LoadedModel) -> Telemetry:
    spatial = loaded.input_size
    dummy = torch.zeros(1, 3, spatial, spatial, device=loaded.device)
    return _measure_batch(loaded, dummy)


def predict(loaded: LoadedModel, image: Image.Image) -> tuple[list[ClassScore], Telemetry]:
    batch = preprocess(image, loaded)
    telemetry = _measure_batch(loaded, batch)
    scores = topk(loaded, batch, k=5)
    return scores, telemetry


def topk(loaded: LoadedModel, batch: torch.Tensor, k: int = 5) -> list[ClassScore]:
    module = loaded.module
    module.eval()
    with torch.inference_mode():
        logits = module(batch)
        probabilities = torch.softmax(logits, dim=-1)[0]
        count = min(k, int(probabilities.shape[-1]))
        values, indices = torch.topk(probabilities, count)
        values = values.float().cpu().tolist()
        indices = indices.cpu().tolist()
    scores = []
    for index, probability in zip(indices, values, strict=True):
        scores.append(ClassScore(index=int(index), name=_class_name(loaded, int(index)), probability=float(probability)))
    return scores


def _measure_batch(loaded: LoadedModel, batch: torch.Tensor) -> Telemetry:
    module = loaded.module
    device = batch.device
    module.eval()
    mean_ms, std_ms, method = _latency(module, batch)
    activation = _activation_peak(module, batch)
    allocated = free = total = None
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        free, total = int(free_b), int(total_b)
        allocated = int(torch.cuda.memory_allocated(device))
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        allocated = int(torch.mps.current_allocated_memory())
    return Telemetry(
        latency_ms=mean_ms,
        latency_std_ms=std_ms,
        samples=MEASURE_ITERS,
        warmup=WARMUP_ITERS,
        timing_method=method,
        activation_peak_bytes=activation,
        allocated_bytes=0 if allocated is None else allocated,
        free_bytes=free,
        total_bytes=total,
        device=device_label(device),
        gpu_name=gpu_name(device),
    )


def _latency(module: torch.nn.Module, batch: torch.Tensor) -> tuple[float, float, str]:
    device = batch.device
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            module(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples: list[float] = []
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(MEASURE_ITERS):
                starter.record()
                module(batch)
                ender.record()
                ender.synchronize()
                samples.append(float(starter.elapsed_time(ender)))
            method = "torch.cuda.Event"
        else:
            synchronize = torch.mps.synchronize if device.type == "mps" and hasattr(torch, "mps") else None
            for _ in range(MEASURE_ITERS):
                if synchronize is not None:
                    synchronize()
                started = time.perf_counter()
                module(batch)
                if synchronize is not None:
                    synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)
            method = "time.perf_counter"
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev, method


def _activation_peak(module: torch.nn.Module, batch: torch.Tensor) -> int:
    """Pico de memória alocada menos o patamar antes do forward.

    `empty_cache` roda fora do cronômetro. O número é o custo extra de ativação
    de uma imagem, não a latência.
    """
    device = batch.device
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        module(batch)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(0, int(peak - baseline))


def _cam_with_fallback(zoo: ModelZoo, loaded: LoadedModel, batch: torch.Tensor, top: ClassScore) -> CamResult:
    kwargs = dict(
        class_index=top.index,
        class_name=top.name,
        probability=top.probability,
        mean=loaded.transform_info.mean,
        std=loaded.transform_info.std,
    )
    try:
        return compute_grad_cam(loaded.module, batch, **kwargs)
    except Exception as exc:
        if not (is_cuda_oom(exc) and loaded.device.type == "cuda"):
            raise
        logger.warning("CUDA OOM no Grad-CAM de %s; tentando de novo", loaded.spec.name)
        release_cuda_cache()

    try:
        return compute_grad_cam(loaded.module, batch, **kwargs)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        logger.warning("Grad-CAM de %s vai para uma cópia em CPU", loaded.spec.name)
        release_cuda_cache()
        cpu_module = zoo.cpu_copy(loaded)
        try:
            result = compute_grad_cam(cpu_module, batch.detach().cpu(), **kwargs)
        finally:
            del cpu_module
            release_cuda_cache()
        result.warnings.append(
            "O Grad-CAM estourou a VRAM e foi calculado numa cópia em CPU. "
            "O Top-5 e a latência continuam sendo da GPU."
        )
        return result


def _class_name(loaded: LoadedModel, index: int) -> str:
    if 0 <= index < len(loaded.categories):
        return loaded.categories[index]
    return f"classe {index}"

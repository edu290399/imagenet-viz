"""Fábrica de CNNs do torchvision com pesos oficiais do ImageNet-1k.

O estado do modelo só é baixado na primeira chamada de `ModelZoo.load`.
Chamadas seguintes devolvem o módulo já residente no dispositivo pedido.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet18,
    resnet50,
    vgg16,
)

from utils.hardware import device_label, is_cuda_oom, release_cuda_cache

logger = logging.getLogger("imagenet_viz.zoo")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CPU_FALLBACK_NOTE = (
    "CUDA out of memory ao carregar ou executar este modelo. "
    "A inferência passou para a CPU: a latência não é comparável com medições feitas na GPU. "
    "Libere a VRAM para tentar a GPU de novo."
)


class UnknownModelError(KeyError):
    def __init__(self, name: str, options: list[str]) -> None:
        self.name = name
        super().__init__(f"Modelo desconhecido: {name}. Opções: {', '.join(options)}")


@dataclass(frozen=True)
class TransformInfo:
    resize: int | None
    crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def spatial(self) -> int:
        return self.crop


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    blocks: str
    summary: str
    reference: str
    builder: Callable[..., nn.Module]
    weights: Any


@dataclass
class LoadedModel:
    spec: ModelSpec
    module: nn.Module
    device: torch.device
    categories: list[str]
    transform: Callable
    transform_info: TransformInfo
    parameters: int
    parameter_bytes: int
    buffer_bytes: int
    macs: int | None
    conv_layers: int
    checkpoint: str
    published_top1: float | None
    published_top5: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def input_size(self) -> int:
        return self.transform_info.spatial


def _specs() -> dict[str, ModelSpec]:
    return {
        "ResNet-18": ModelSpec(
            name="ResNet-18",
            family="ResNet",
            blocks=(
                "Residual Blocks (BasicBlock): duas convoluções 3×3 cuja saída é somada "
                "a um atalho. O atalho é o que permite treinar a rede sem o gradiente sumir."
            ),
            summary=(
                "Baseline leve. Serve como primeiro degrau: pouca latência, acurácia publicada "
                "modesta e um Grad-CAM fácil de comparar com a ResNet-50."
            ),
            reference="[He et al., 2016](https://arxiv.org/abs/1512.03385)",
            builder=resnet18,
            weights=ResNet18_Weights.DEFAULT,
        ),
        "ResNet-50": ModelSpec(
            name="ResNet-50",
            family="ResNet",
            blocks=(
                "Residual Blocks (Bottleneck): 1×1 reduz os canais, 3×3 filtra e outra 1×1 "
                "restaura a largura. O atalho soma a identidade (ou uma projeção) à saída do bloco."
            ),
            summary=(
                "A referência clássica do ImageNet. É o modelo contra o qual vale a pena "
                "comparar custo, latência e o mapa de atenção."
            ),
            reference="[He et al., 2016](https://arxiv.org/abs/1512.03385)",
            builder=resnet50,
            weights=ResNet50_Weights.DEFAULT,
        ),
        "MobileNetV3-Large": ModelSpec(
            name="MobileNetV3-Large",
            family="MobileNetV3",
            blocks=(
                "Inverted Residual Blocks e convoluções depthwise separable. "
                "O bloco expande com uma 1×1, filtra cada canal com uma depthwise e projeta "
                "de volta com outra 1×1, com squeeze-and-excitation e hard-swish."
            ),
            summary=(
                "Desenhada para latência baixa. A depthwise separa a filtragem espacial da "
                "mistura de canais, então o custo cai muito em relação a uma convolução densa."
            ),
            reference="[Howard et al., 2019](https://arxiv.org/abs/1905.02244)",
            builder=mobilenet_v3_large,
            weights=MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "EfficientNet-B0": ModelSpec(
            name="EfficientNet-B0",
            family="EfficientNet",
            blocks=(
                "MBConv (inverted bottleneck móvel) com squeeze-and-excitation. "
                "A família escala profundidade, largura e resolução juntas; o B0 é a base."
            ),
            summary=(
                "Um ponto de equilíbrio entre a acurácia publicada e o custo. "
                "Útil para ver uma rede compacta ultrapassar, no ImageNet, arquiteturas bem mais pesadas."
            ),
            reference="[Tan e Le, 2019](https://arxiv.org/abs/1905.11946)",
            builder=efficientnet_b0,
            weights=EfficientNet_B0_Weights.DEFAULT,
        ),
        "ConvNeXt-Tiny": ModelSpec(
            name="ConvNeXt-Tiny",
            family="ConvNeXt",
            blocks=(
                "Blocos ConvNeXt: convolução depthwise 7×7, LayerNorm e uma expansão 1×1 com GELU, "
                "organizados em estágios no estilo de um Transformer."
            ),
            summary=(
                "ConvNet moderna que absorve ideias dos Vision Transformers sem usar atenção. "
                "O contraste com a ResNet mostra o que mudou no desenho entre 2015 e 2022."
            ),
            reference="[Liu et al., 2022](https://arxiv.org/abs/2201.03545)",
            builder=convnext_tiny,
            weights=ConvNeXt_Tiny_Weights.DEFAULT,
        ),
        "VGG-16": ModelSpec(
            name="VGG-16",
            family="VGG",
            blocks=(
                "Pilha homogênea de convoluções 3×3 e max-pooling. "
                "A maior parte dos parâmetros está no classificador denso do final."
            ),
            summary=(
                "Marco de 2014. Entra na bancada como contraste histórico: muita memória, "
                "muita latência e um mapa de atenção em geral mais grosso."
            ),
            reference="[Simonyan e Zisserman, 2015](https://arxiv.org/abs/1409.1556)",
            builder=vgg16,
            weights=VGG16_Weights.DEFAULT,
        ),
    }


SPECS: dict[str, ModelSpec] = _specs()


def list_model_names() -> list[str]:
    return list(SPECS)


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise UnknownModelError(name, list_model_names()) from exc


def checkpoint_name(weights: Any) -> str:
    text = str(weights)
    return text.split(".")[-1]


def _as_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def published_accuracy(weights: Any) -> tuple[float | None, float | None]:
    meta = getattr(weights, "meta", None) or {}
    metrics = meta.get("_metrics") or {}
    if not isinstance(metrics, dict):
        return None, None
    for key, value in metrics.items():
        if "imagenet" not in str(key).lower() or not isinstance(value, dict):
            continue
        return _as_percent(value.get("acc@1")), _as_percent(value.get("acc@5"))
    return None, None


def preview_parameters(weights: Any) -> int | None:
    meta = getattr(weights, "meta", None) or {}
    value = meta.get("num_params")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def categories_of(weights: Any) -> list[str]:
    meta = getattr(weights, "meta", None) or {}
    raw = meta.get("categories") or []
    return [str(item) for item in raw]


def _edge(size: Any) -> int:
    if isinstance(size, int):
        return size
    if isinstance(size, (list, tuple)):
        if not size:
            return 224
        return _edge(size[0])
    return int(size)


def _triple(values: Any) -> tuple[float, float, float]:
    seq = [float(item) for item in list(values)]
    if len(seq) == 1:
        return (seq[0], seq[0], seq[0])
    if len(seq) < 3:
        raise ValueError(f"Normalização incompleta: {values!r}")
    return (seq[0], seq[1], seq[2])


def _walk(transform: Any):
    children = getattr(transform, "transforms", None)
    if children:
        for child in children:
            yield from _walk(child)
        return
    yield transform


def inspect_transform(transform: Any) -> TransformInfo:
    """Lê crop, média e desvio tanto do preset novo quanto de um Compose clássico."""
    if hasattr(transform, "mean") and hasattr(transform, "std"):
        crop = getattr(transform, "crop_size", None)
        resize = getattr(transform, "resize_size", None)
        spatial_source = crop if crop is not None else resize if resize is not None else 224
        return TransformInfo(
            resize=_edge(resize) if resize is not None else None,
            crop=_edge(spatial_source),
            mean=_triple(transform.mean),
            std=_triple(transform.std),
        )

    mean, std, crop, resize = IMAGENET_MEAN, IMAGENET_STD, 224, None
    for child in _walk(transform):
        name = child.__class__.__name__
        if name == "Normalize":
            mean = _triple(child.mean)
            std = _triple(child.std)
        elif name == "CenterCrop":
            crop = _edge(child.size)
        elif name == "Resize" and resize is None:
            resize = _edge(child.size)
    return TransformInfo(resize=resize, crop=crop, mean=mean, std=std)


def _call_builder(builder: Callable[..., nn.Module], weights: Any) -> nn.Module:
    params = inspect.signature(builder).parameters
    kwargs: dict[str, Any] = {"weights": weights}
    if "progress" in params and weights is not None:
        kwargs["progress"] = True
    return builder(**kwargs)


def _disable_inplace(module: nn.Module) -> int:
    """ReLU inplace quebra o backward do Grad-CAM ao reaproveitar o tensor da convolução."""
    changed = 0
    for child in module.modules():
        if getattr(child, "inplace", False):
            child.inplace = False
            changed += 1
    return changed


def _parameter_stats(module: nn.Module) -> tuple[int, int, int, int]:
    parameters = 0
    parameter_bytes = 0
    for tensor in module.parameters():
        parameters += tensor.numel()
        parameter_bytes += tensor.numel() * tensor.element_size()
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for tensor in module.buffers())
    conv_layers = sum(isinstance(child, nn.Conv2d) for child in module.modules())
    return parameters, parameter_bytes, buffer_bytes, conv_layers


def estimate_macs(module: nn.Module, spatial: int, device: torch.device) -> int:
    """Conta MACs de Conv2d e Linear num forward de batch 1.

    GFLOPs ≈ 2 × GMACs (uma multiplicação e uma adição). Batch norm, ativação
    e pooling ficam de fora: perto das convoluções, o custo deles é pequeno.
    """
    macs = 0
    handles: list[Any] = []

    def conv_hook(layer: nn.Conv2d, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if not torch.is_tensor(output) or output.ndim != 4:
            return
        batch, _, height, width = output.shape
        kernel_h, kernel_w = layer.kernel_size
        macs += (
            batch
            * layer.out_channels
            * height
            * width
            * (layer.in_channels // layer.groups)
            * kernel_h
            * kernel_w
        )

    def linear_hook(layer: nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        incoming = inputs[0]
        rows = incoming.numel() // incoming.shape[-1]
        macs += rows * layer.in_features * layer.out_features

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            handles.append(child.register_forward_hook(conv_hook))
        elif isinstance(child, nn.Linear):
            handles.append(child.register_forward_hook(linear_hook))

    try:
        dummy = torch.zeros(1, 3, spatial, spatial, device=device)
        with torch.inference_mode():
            module(dummy)
    finally:
        for handle in handles:
            handle.remove()
    return int(macs)


class ModelZoo:
    """Cache de módulos por (nome, dispositivo). O download do checkpoint é preguiçoso."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], LoadedModel] = {}
        self._cpu_fallback: set[str] = set()
        self._lock = threading.RLock()

    def guard(self):
        return self._lock

    def clear(self) -> None:
        with self._lock:
            for loaded in self._cache.values():
                loaded.module.to("cpu")
            self._cache.clear()
            self._cpu_fallback.clear()
            release_cuda_cache()
            logger.info("cache de modelos esvaziado")

    def load(self, name: str, preferred: torch.device) -> LoadedModel:
        with self._lock:
            spec = get_spec(name)
            if preferred.type == "cuda" and name in self._cpu_fallback:
                preferred = torch.device("cpu")
            key = (name, device_label(preferred))
            cached = self._cache.get(key)
            if cached is not None:
                logger.info("reusando %s em %s", name, key[1])
                return cached

            loaded, notes = self._load_with_fallback(spec, preferred)
            loaded.notes.extend(note for note in notes if note not in loaded.notes)
            self._cache[(name, device_label(loaded.device))] = loaded
            if loaded.device.type == "cpu" and preferred.type == "cuda":
                self._cpu_fallback.add(name)
            logger.info(
                "%s pronto em %s · %.2f M parâmetros · checkpoint %s",
                name,
                device_label(loaded.device),
                loaded.parameters / 1e6,
                loaded.checkpoint,
            )
            return loaded

    def relocate_to_cpu(self, loaded: LoadedModel, note: str) -> LoadedModel:
        with self._lock:
            self._cache.pop((loaded.spec.name, device_label(loaded.device)), None)
            loaded.module.to("cpu")
            loaded.device = torch.device("cpu")
            if note not in loaded.notes:
                loaded.notes.append(note)
            self._cpu_fallback.add(loaded.spec.name)
            self._cache[(loaded.spec.name, "cpu")] = loaded
            release_cuda_cache()
            logger.warning("%s movido para CPU após CUDA OOM", loaded.spec.name)
            return loaded

    def cpu_copy(self, loaded: LoadedModel) -> nn.Module:
        """Segunda instância em RAM, para um Grad-CAM que não cabe na GPU.

        O módulo cacheado permanece na GPU. `weights=None` não baixa outro checkpoint:
        os tensores são copiados do modelo já carregado.
        """
        with self._lock:
            module = _call_builder(loaded.spec.builder, None)
            _disable_inplace(module)
            module.eval()
            state = {key: value.detach().cpu() for key, value in loaded.module.state_dict().items()}
            module.load_state_dict(state)
            return module

    def _load_with_fallback(
        self, spec: ModelSpec, preferred: torch.device
    ) -> tuple[LoadedModel, list[str]]:
        if preferred.type != "cuda":
            return self._materialize(spec, preferred), []

        try:
            return self._materialize(spec, preferred), []
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            logger.warning("CUDA OOM ao carregar %s; esvaziando o cache e tentando de novo", spec.name)
            release_cuda_cache()

        try:
            return self._materialize(spec, preferred), []
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            logger.warning("CUDA OOM persiste em %s; liberando os outros modelos", spec.name)
            self._drop_cached_modules()
            release_cuda_cache()

        try:
            return self._materialize(spec, preferred), []
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            logger.warning("CUDA OOM definitivo em %s; fallback para CPU", spec.name)
            return self._materialize(spec, torch.device("cpu")), [CPU_FALLBACK_NOTE]

    def _drop_cached_modules(self) -> None:
        for loaded in self._cache.values():
            loaded.module.to("cpu")
        self._cache.clear()

    def _materialize(self, spec: ModelSpec, device: torch.device) -> LoadedModel:
        logger.info("carregando pesos de %s para %s", spec.name, device_label(device))
        module = _call_builder(spec.builder, spec.weights)
        disabled = _disable_inplace(module)
        if disabled:
            logger.info("%s: %d ativações inplace desligadas para o Grad-CAM", spec.name, disabled)
        module.eval()
        try:
            module.to(device)
        except Exception:
            try:
                module.to("cpu")
            except Exception:
                logger.warning("não foi possível devolver %s à CPU após falha no .to()", spec.name)
            del module
            release_cuda_cache()
            raise

        transform = spec.weights.transforms()
        info = inspect_transform(transform)
        parameters, parameter_bytes, buffer_bytes, conv_layers = _parameter_stats(module)
        notes: list[str] = []
        try:
            macs = estimate_macs(module, info.spatial, device)
            if macs <= 0:
                macs = None
                notes.append("Não foi possível estimar os MACs desta arquitetura.")
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            release_cuda_cache()
            macs = None
            notes.append("Os FLOPs não foram estimados porque o forward de perfil estourou a VRAM.")

        top1, top5 = published_accuracy(spec.weights)
        return LoadedModel(
            spec=spec,
            module=module,
            device=device,
            categories=categories_of(spec.weights),
            transform=transform,
            transform_info=info,
            parameters=parameters,
            parameter_bytes=parameter_bytes,
            buffer_bytes=buffer_bytes,
            macs=macs,
            conv_layers=conv_layers,
            checkpoint=checkpoint_name(spec.weights),
            published_top1=top1,
            published_top5=top5,
            notes=notes,
        )

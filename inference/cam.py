"""Grad-CAM em PyTorch puro, no último mapa que ainda tem altura e largura."""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch import nn

_CMAP = colormaps["jet"]
_PANEL = 320


@dataclass
class CamResult:
    image: Image.Image
    layer_name: str
    map_height: int
    map_width: int
    class_index: int
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)


def placeholder_image(message: str, width: int = 960, height: int = 360) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "#eef2ff")
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    margin = 28
    lines = textwrap.wrap(message.strip() or "Sem mapa.", width=62) or [""]
    y = margin
    for line in lines[:8]:
        draw.text((margin, y), line, fill="#312e81", font=font)
        y += 28
    return canvas


def compute_grad_cam(
    module: nn.Module,
    batch: torch.Tensor,
    *,
    class_index: int | None,
    class_name: str,
    probability: float | None,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> CamResult:
    """Gera o tríptico recorte / heatmap / sobreposição para uma classe.

    O mapa é a combinação linear do último tensor 4D com altura e largura
    maiores que 1 — a saída do último estágio, já com a ativação aplicada.
    Uma convolução crua, antes da ReLU, desloca o mapa (a VGG-16 faz isso).
    Os pesos dos canais são a média espacial do gradiente do logit da classe.
    """
    if batch.ndim != 4 or batch.shape[0] != 1 or batch.shape[1] != 3:
        raise ValueError(f"O Grad-CAM espera um tensor 1×3×H×W, recebeu {tuple(batch.shape)}.")

    module.eval()
    device = next(module.parameters()).device
    batch = batch.detach().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    warnings: list[str] = []

    layer_name, layer = _select_layer(module, batch)
    bag: dict[str, torch.Tensor] = {}

    def save_activation(_layer: nn.Module, _inputs, output: torch.Tensor) -> None:
        if torch.is_tensor(output):
            bag["act"] = output.detach().float().clone()

    def save_gradient(_layer: nn.Module, _grad_in, grad_out) -> None:
        if grad_out and torch.is_tensor(grad_out[0]):
            bag["grad"] = grad_out[0].detach().float().clone()

    forward_handle = layer.register_forward_hook(save_activation)
    backward_handle = layer.register_full_backward_hook(save_gradient)
    try:
        class_index = _backward_from_class(module, batch, class_index)
    finally:
        forward_handle.remove()
        backward_handle.remove()
        module.zero_grad(set_to_none=True)

    if "act" not in bag or "grad" not in bag:
        raise RuntimeError(
            f"A camada `{layer_name}` não devolveu ativação e gradiente. "
            "O Grad-CAM precisa enxergar o forward dessa convolução."
        )

    activation = bag["act"]
    gradient = bag["grad"]
    map_h, map_w = int(activation.shape[-2]), int(activation.shape[-1])
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))

    image = _denormalize(batch, mean, std)
    cam_up = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().float().cpu().numpy()
    cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=0.0, neginf=0.0)
    span = float(cam_np.max() - cam_np.min()) if cam_np.size else 0.0
    if span < 1e-8:
        normalized = np.zeros_like(cam_np)
        warnings.append(
            "O Grad-CAM ficou vazio: a ReLU zerou o mapa. "
            "Essa classe pode não ter suporte espacial claro na última convolução."
        )
    else:
        normalized = (cam_np - float(cam_np.min())) / span

    heat = _colorize(normalized)
    overlay = Image.fromarray((0.55 * np.asarray(image) + 0.45 * heat).clip(0, 255).astype(np.uint8))
    probability_text = f"{probability:.1%}" if probability is not None else "—"
    detail = (
        f"Classe explicada: {class_name} ({probability_text}). "
        f"Camada {layer_name}, mapa {map_h}×{map_w} ampliado ao recorte."
    )
    panel = _triptych(
        image,
        Image.fromarray(heat),
        overlay,
        [*textwrap.wrap(detail, width=96), "Azul = pouca contribuição para a classe; vermelho = muita."],
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return CamResult(
        image=panel,
        layer_name=layer_name,
        map_height=map_h,
        map_width=map_w,
        class_index=class_index,
        elapsed_ms=elapsed_ms,
        warnings=warnings,
    )


# Convolução crua e normalização ainda não são o mapa que segue para o classificador.
# Só o Sequential "puro" é ignorado: blocos que herdam dele, como Conv2dNormActivation, ficam.
_SKIP_LAYER_TYPES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.Dropout,
    nn.Dropout2d,
    nn.Identity,
)


def _is_plain_container(layer: nn.Module) -> bool:
    return type(layer) in {nn.Sequential, nn.ModuleList}


def _select_layer(module: nn.Module, batch: torch.Tensor) -> tuple[str, nn.Module]:
    """Último tensor espacial antes do vetor de classes.

    A sondagem percorre o forward inteiro. Prefere o bloco ou a ativação
    finais; se a rede só expuser convoluções, cai na última `Conv2d`.
    """
    spatial: list[tuple[str, nn.Module]] = []
    handles = []
    for name, layer in module.named_modules():
        if not name:
            continue

        def remember(_layer, _inputs, output, name=name, layer=layer):
            if (
                torch.is_tensor(output)
                and output.ndim == 4
                and int(output.shape[-2]) > 1
                and int(output.shape[-1]) > 1
            ):
                spatial.append((name, layer))

        handles.append(layer.register_forward_hook(remember))
    try:
        with torch.inference_mode():
            module(batch)
    finally:
        for handle in handles:
            handle.remove()

    if not spatial:
        raise RuntimeError("Nenhum mapa espacial encontrado; o Grad-CAM não se aplica a este modelo.")

    for name, layer in reversed(spatial):
        if _is_plain_container(layer) or isinstance(layer, _SKIP_LAYER_TYPES):
            continue
        return name, layer
    for name, layer in reversed(spatial):
        if isinstance(layer, nn.Conv2d):
            return name, layer
    return spatial[-1]


def _backward_from_class(module: nn.Module, batch: torch.Tensor, class_index: int | None) -> int:
    with torch.enable_grad():
        logits = module(batch.detach())
        chosen = _resolve_index(logits, class_index)
        score = logits[0, chosen]
        if not score.requires_grad:
            logits = module(batch.detach().requires_grad_(True))
            chosen = _resolve_index(logits, chosen)
            score = logits[0, chosen]
        if not score.requires_grad:
            raise RuntimeError(
                "O logit da classe não participa do grafo de gradiente. "
                "O Grad-CAM não consegue atribuir importância espacial."
            )
        module.zero_grad(set_to_none=True)
        score.backward()
    return chosen


def _resolve_index(logits: torch.Tensor, class_index: int | None) -> int:
    count = int(logits.shape[-1])
    if class_index is None:
        return int(torch.argmax(logits, dim=-1)[0].item())
    if class_index < 0 or class_index >= count:
        raise ValueError(f"Índice de classe {class_index} fora do intervalo 0..{count - 1}.")
    return int(class_index)


def _denormalize(batch: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    image = batch[0].detach().float().cpu().contiguous()
    mean_t = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    image = (image * std_t + mean_t).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _colorize(cam_hw: np.ndarray) -> np.ndarray:
    rgba = _CMAP(np.clip(cam_hw, 0.0, 1.0))
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _triptych(
    original: Image.Image,
    heat: Image.Image,
    overlay: Image.Image,
    footer_lines: list[str],
) -> Image.Image:
    tiles = [
        ("Recorte visto pela CNN", original),
        ("Grad-CAM", heat),
        ("Sobreposição", overlay),
    ]
    font = _font(16)
    small = _font(14)
    footer_lines = footer_lines or [""]
    margin = 16
    gap = 12
    caption_h = 28
    footer_h = 8 + 20 * len(footer_lines)
    width = margin * 2 + _PANEL * 3 + gap * 2
    height = margin + caption_h + _PANEL + footer_h + margin
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    x = margin
    top = margin + caption_h
    for title, tile in tiles:
        resized = tile.convert("RGB").resize((_PANEL, _PANEL), Image.Resampling.BICUBIC)
        label_w = _text_width(draw, title, font)
        draw.text((x + (_PANEL - label_w) / 2, margin), title, fill="#1e1b4b", font=font)
        canvas.paste(resized, (x, top))
        draw.rectangle((x, top, x + _PANEL - 1, top + _PANEL - 1), outline="#e2e8f0")
        x += _PANEL + gap
    y = top + _PANEL + 10
    for line in footer_lines:
        draw.text((margin, y), line, fill="#334155", font=small)
        y += 20
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]

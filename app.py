"""Bancada Gradio para comparar CNNs pré-treinadas no ImageNet-1k."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import torch

from inference.cam import placeholder_image
from inference.engine import (
    MEASURE_ITERS,
    WARMUP_ITERS,
    Analysis,
    ClassScore,
    Telemetry,
    analyze_image,
    benchmark_model,
    clear_telemetry,
)
from models.zoo import (
    LoadedModel,
    ModelZoo,
    categories_of,
    checkpoint_name,
    get_spec,
    inspect_transform,
    list_model_names,
    preview_parameters,
    published_accuracy,
)
from utils.hardware import (
    configure_runtime,
    detect_device,
    device_label,
    explain_failure,
    format_bytes,
    is_cuda_oom,
    release_cuda_cache,
    snapshot,
)

logger = logging.getLogger("imagenet_viz.app")

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = ROOT / "assets" / "examples"
EXAMPLE_FILES = ("cachorro.jpg", "aviao.jpg", "cafe.jpg")

COLOR_A = "#4f46e5"
COLOR_B = "#059669"

ZOO = ModelZoo()
DEVICE = torch.device("cpu")
STARTUP_WARNING: str | None = None

BENCH_HEADERS = [
    "Modelo",
    "Família",
    "Params (M)",
    "Pesos (MiB)",
    "GMACs",
    "GFLOPs",
    "Latência (ms)",
    "Desvio (ms)",
    "img/s (batch 1)",
    "Ativação pico (MiB)",
    "Dispositivo",
    "Top-1 publicado",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def activate_device() -> None:
    global DEVICE, STARTUP_WARNING
    configure_runtime()
    try:
        DEVICE = detect_device()
        STARTUP_WARNING = None
    except Exception as exc:
        DEVICE = torch.device("cpu")
        STARTUP_WARNING = str(exc)
        logger.warning("fallback de dispositivo: %s", exc)


def example_rows() -> list[list[str]]:
    rows = []
    for filename in EXAMPLE_FILES:
        path = EXAMPLE_DIR / filename
        if path.is_file():
            rows.append([str(path.resolve())])
    return rows


def render_hardware() -> str:
    snap = snapshot(DEVICE)
    lines = [
        "### Hardware",
        f"**Dispositivo ativo:** `{device_label(snap.device)}` · {snap.gpu_name}",
    ]
    if STARTUP_WARNING:
        lines.append(f"> {STARTUP_WARNING}")
    rows = [
        ("Backend", snap.backend),
        ("PyTorch", snap.torch_version),
        ("CUDA do wheel", snap.cuda_build or "—"),
    ]
    if snap.capability:
        rows.append(("Compute capability", snap.capability))
    if snap.total_bytes is not None:
        rows.append(("VRAM livre agora", f"{format_bytes(snap.free_bytes)} / {format_bytes(snap.total_bytes)}"))
    if snap.allocated_bytes is not None:
        rows.append(("Alocado neste processo", format_bytes(snap.allocated_bytes)))
    if snap.reserved_bytes is not None:
        rows.append(("Reservado pelo allocator", format_bytes(snap.reserved_bytes)))
    if snap.device.type == "cuda":
        rows.append(("TF32", "ativo" if snap.tf32 else "desligado"))
        rows.append(("cuDNN benchmark", "ativo" if snap.cudnn_benchmark else "desligado"))
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    lines.append("| | |\n|---|---|\n" + table)
    lines.append(
        "Protocolo da bancada: **FP32, NCHW, batch 1**, sem autocast e sem TensorRT. "
        f"Cada latência descarta {WARMUP_ITERS} forwards de aquecimento e tira a média "
        f"de {MEASURE_ITERS} medições."
    )
    return "\n\n".join(lines)


def format_static(name: str) -> str:
    spec = get_spec(name)
    top1, top5 = published_accuracy(spec.weights)
    info = inspect_transform(spec.weights.transforms())
    preview = preview_parameters(spec.weights)
    geometry = _geometry(info.resize, info.crop)
    lines = [
        f"### {spec.name}",
        f"**Família:** {spec.family} · **Checkpoint:** `{checkpoint_name(spec.weights)}`",
        f"**Blocos:** {spec.blocks}",
        spec.summary,
        f"Entrada oficial: {geometry}. Referência: {spec.reference}.",
    ]
    if preview:
        lines.append(
            f"Parâmetros informados pelo torchvision: **{preview / 1e6:.2f} M** "
            f"(cerca de **{preview * 4 / (1024 ** 2):.1f} MiB** se os pesos estiverem em fp32)."
        )
    if top1 is not None:
        top5_text = f" · Top-5 {top5:.2f}%" if top5 is not None else ""
        lines.append(f"Acurácia ImageNet-1k publicada: **Top-1 {top1:.2f}%**{top5_text}.")
    lines.append("A latência e a VRAM entram no card quando o carregamento dos pesos termina.")
    return "\n\n".join(lines)


def format_model_card(loaded: LoadedModel, telemetry: Telemetry) -> str:
    spec = loaded.spec
    info = loaded.transform_info
    lines = [
        f"### {spec.name}",
        f"**Família:** {spec.family} · **Checkpoint:** `{loaded.checkpoint}`",
        f"**Blocos:** {spec.blocks}",
        spec.summary,
        "",
        "| | |",
        "|---|---|",
        f"| Parâmetros | **{loaded.parameters / 1e6:.2f} M** |",
        f"| Memória dos parâmetros | {format_bytes(loaded.parameter_bytes)} (fp32) |",
        f"| Buffers (batch norm etc.) | {format_bytes(loaded.buffer_bytes)} |",
        f"| Camadas Conv2d | {loaded.conv_layers} |",
        f"| GMACs / GFLOPs | {_giga(loaded.macs)} / {_giga(loaded.macs, flops=True)} |",
        f"| Entrada | {_geometry(info.resize, info.crop)} |",
        f"| Normalização | média { _fmt_triple(info.mean) } · desvio { _fmt_triple(info.std) } |",
        f"| Top-1 / Top-5 publicados | {_published(loaded.published_top1, loaded.published_top5)} |",
        f"| Latência média | **{_ms(telemetry)}** |",
        f"| Relógio | `{telemetry.timing_method}` · {telemetry.warmup} warmup + {telemetry.samples} medições |",
        f"| Throughput (batch 1) | {_throughput(telemetry.latency_ms)} img/s |",
        f"| Pico de ativação | {format_bytes(telemetry.activation_peak_bytes)} |",
        f"| VRAM do processo após a medição | {format_bytes(telemetry.allocated_bytes)} |",
        f"| VRAM livre na GPU | {_vram_free(telemetry)} |",
        f"| Dispositivo | `{telemetry.device}` · {telemetry.gpu_name} |",
        "",
        (
            f"A latência deste card usa um tensor sintético 1×3×{loaded.input_size}×{loaded.input_size}. "
            "A classificação de uma foto mede de novo, com o mesmo protocolo, o tensor já pré-processado."
        ),
        "GFLOPs ≈ 2 × GMACs. A conta inclui `Conv2d` e `Linear`; batch norm, ativação e pooling ficam de fora.",
        f"Referência: {spec.reference}.",
        "",
        "**Pré-processamento (`weights.transforms()`):**",
        _fence(str(loaded.transform)),
    ]
    if loaded.notes:
        lines.extend(["", *[f"> {note}" for note in loaded.notes]])
    return "\n".join(lines)


def format_result(analysis: Analysis) -> str:
    telemetry = analysis.telemetry
    top = analysis.scores[0]
    lines = [f"### Resultado · {analysis.model_name}"]
    if analysis.warnings:
        lines.extend(f"> {note}" for note in analysis.warnings)
        lines.append("")
    lines.extend(
        [
            f"Top-1: **{top.name}** ({top.probability:.1%}, índice {top.index}).",
            "",
            "| | |",
            "|---|---|",
            f"| Latência desta imagem | **{_ms(telemetry)}** |",
            f"| Relógio | `{telemetry.timing_method}` em `{telemetry.device}` |",
            f"| Throughput (batch 1) | {_throughput(telemetry.latency_ms)} img/s |",
            f"| Pico de ativação | {format_bytes(telemetry.activation_peak_bytes)} |",
            f"| VRAM do processo | {format_bytes(telemetry.allocated_bytes)} |",
            f"| VRAM livre | {_vram_free(telemetry)} |",
            f"| Grad-CAM | `{analysis.cam.layer_name}` · mapa {analysis.cam.map_height}×{analysis.cam.map_width} |",
            f"| Tempo até o mapa | {analysis.cam.elapsed_ms:.1f} ms (sondagem + backward + desenho) |",
            "",
            "**Top-5**",
        ]
    )
    for rank, score in enumerate(analysis.scores, start=1):
        lines.append(f"{rank}. {score.name} — {score.probability:.1%} (índice {score.index})")
    lines.append("")
    lines.append(
        "O heatmap usa o recorte central que a rede viu, não a foto inteira. "
        "Vermelho pesa mais no logit da classe explicada; azul pesa menos."
    )
    return "\n".join(lines)


def comparison_summary(left: Analysis, right: Analysis) -> str:
    lines = [f"### {left.model_name} vs {right.model_name}"]
    if left.model_name == right.model_name:
        lines.append(
            "Os dois lados usam o mesmo checkpoint. Qualquer diferença de tempo é a variação da medição."
        )
    top_left, top_right = left.scores[0], right.scores[0]
    if top_left.name == top_right.name:
        lines.append(f"As redes concordam no Top-1: **{top_left.name}**.")
    else:
        lines.append(
            f"As redes discordam no Top-1: **{left.model_name}** diz **{top_left.name}** "
            f"({top_left.probability:.1%}) e **{right.model_name}** diz **{top_right.name}** "
            f"({top_right.probability:.1%})."
        )
    if left.telemetry.device == right.telemetry.device and min(left.telemetry.latency_ms, right.telemetry.latency_ms) > 0:
        fast, slow = (left, right) if left.telemetry.latency_ms <= right.telemetry.latency_ms else (right, left)
        ratio = slow.telemetry.latency_ms / fast.telemetry.latency_ms
        lines.append(
            f"Em `{left.telemetry.device}`, **{fast.model_name}** respondeu em "
            f"{fast.telemetry.latency_ms:.2f} ms e **{slow.model_name}** em "
            f"{slow.telemetry.latency_ms:.2f} ms — razão de **{ratio:.2f}×**. "
            "Os gráficos usam a escala 0–100% nos dois lados."
        )
    else:
        lines.append(
            f"Os dispositivos foram `{left.telemetry.device}` e `{right.telemetry.device}`. "
            "Essas latências não estão na mesma condição de medida."
        )
    return "\n\n".join(lines)


def top5_figure(title: str, scores: list[ClassScore] | None, color: str) -> go.Figure:
    figure = go.Figure()
    if not scores:
        figure.update_layout(
            title=title,
            template="plotly_white",
            height=380,
            margin=dict(t=48, r=16, b=16),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        figure.add_annotation(text="Aguardando uma imagem", showarrow=False, font=dict(size=14, color="#64748b"))
        return figure
    figure.add_bar(
        x=[score.probability for score in scores],
        y=[score.name for score in scores],
        orientation="h",
        marker_color=color,
        text=[f"{score.probability:.1%}" for score in scores],
        textposition="auto",
        customdata=[score.index for score in scores],
        hovertemplate="índice %{customdata}<br>%{y}<br>probabilidade %{x:.2%}<extra></extra>",
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        margin=dict(t=48, r=16, b=8),
        xaxis=dict(title="Probabilidade (softmax)", range=[0, 1], tickformat=".0%"),
        yaxis=dict(autorange="reversed", title=""),
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=13),
        bargap=0.28,
    )
    return figure


def profile_card(name: str, progress: gr.Progress | None = None, *, refresh: bool = False) -> str:
    if progress is not None:
        progress(0.15, desc=f"Carregando {name}…")
    loaded, telemetry = benchmark_model(ZOO, name, DEVICE, refresh=refresh)
    if progress is not None:
        progress(0.85, desc="Montando o card…")
    return format_model_card(loaded, telemetry)


def safe_profile(name: str, progress: gr.Progress | None = None, *, refresh: bool = False) -> str:
    try:
        return profile_card(name, progress, refresh=refresh)
    except Exception as exc:
        logger.exception("falha ao perfilar %s", name)
        return format_static(name) + f"\n\n> **Falha ao carregar:** {explain_failure(exc)}"


def on_profile(name: str, progress=gr.Progress()) -> str:
    return safe_profile(name, progress, refresh=True)


def boot(infer_name: str, name_a: str, name_b: str, progress=gr.Progress()):
    progress(0.05, desc="Lendo o hardware…")
    hardware = render_hardware()
    card = safe_profile(infer_name, progress, refresh=False)
    progress(0.55, desc=f"Medindo {name_a}…")
    card_a = safe_profile(name_a, refresh=False)
    progress(0.8, desc=f"Medindo {name_b}…")
    card_b = safe_profile(name_b, refresh=False)
    progress(1.0, desc="Bancada pronta")
    return hardware, card, card_a, card_b


def on_classify(image, model_name: str, progress=gr.Progress()):
    try:
        if image is None:
            raise ValueError("Envie uma imagem ou escolha um dos exemplos.")
        progress(0.2, desc=f"Classificando com {model_name}…")
        analysis = analyze_image(ZOO, model_name, image, DEVICE)
        progress(1.0, desc="Pronto")
        return (
            top5_figure(f"Top-5 · {analysis.model_name}", analysis.scores, COLOR_A),
            analysis.cam.image,
            format_result(analysis),
        )
    except Exception as exc:
        logger.exception("classificação falhou")
        message = explain_failure(exc)
        return top5_figure(f"{model_name} · erro", None, COLOR_A), placeholder_image(message), f"### Não foi possível classificar\n\n{message}"


def on_compare(image, name_a: str, name_b: str, progress=gr.Progress()):
    empty = top5_figure("Sem predição", None, COLOR_A)
    try:
        if image is None:
            raise ValueError("Envie uma imagem ou escolha um dos exemplos.")
        progress(0.1, desc=f"Rodando {name_a}…")
        left = analyze_image(ZOO, name_a, image, DEVICE)
        progress(0.55, desc=f"Rodando {name_b}…")
        right = analyze_image(ZOO, name_b, image, DEVICE)
        progress(1.0, desc="Pronto")
        return (
            comparison_summary(left, right),
            top5_figure(f"Top-5 · {left.model_name}", left.scores, COLOR_A),
            left.cam.image,
            format_result(left),
            top5_figure(f"Top-5 · {right.model_name}", right.scores, COLOR_B),
            right.cam.image,
            format_result(right),
        )
    except Exception as exc:
        logger.exception("comparação falhou")
        message = explain_failure(exc)
        blank = placeholder_image(message)
        error = f"### Não foi possível comparar\n\n{message}"
        return error, empty, blank, error, top5_figure("Sem predição", None, COLOR_B), blank, error


def on_benchmark(selected: list[str] | None, progress=gr.Progress()):
    names = _ordered(selected or [])
    if not names:
        return [], "Selecione ao menos um modelo para medir."
    collected: list[tuple[LoadedModel, Telemetry]] = []
    failures: list[str] = []
    for index, name in enumerate(names):
        progress(index / len(names), desc=f"Medindo {name}…")
        try:
            collected.append(benchmark_model(ZOO, name, DEVICE, refresh=True))
        except Exception as exc:
            logger.exception("benchmark falhou em %s", name)
            failures.append(f"- **{name}:** {explain_failure(exc)}")
    collected.sort(key=lambda item: item[1].latency_ms)
    rows = [_bench_row(loaded, telemetry) for loaded, telemetry in collected]
    lines = [
        f"Protocolo: FP32, NCHW, batch 1, {WARMUP_ITERS} warmup + {MEASURE_ITERS} medições. "
        "O throughput é `1000 / latência` e não é o pico com batch maior."
    ]
    if collected:
        loaded, telemetry = collected[0]
        lines.append(
            f"**Menor latência neste lote:** {loaded.spec.name} em {telemetry.latency_ms:.2f} ms "
            f"(`{telemetry.device}`)."
        )
    if any(item[1].device == "cpu" for item in collected) and any(item[1].device.startswith("cuda") for item in collected):
        lines.append("Há modelos em CPU e em CUDA no mesmo lote. Não compare essas linhas como se o hardware fosse o mesmo.")
    if failures:
        lines.append("Falhas:\n" + "\n".join(failures))
    progress(1.0, desc="Benchmark concluído")
    return rows, "\n\n".join(lines)


def on_free_vram() -> str:
    ZOO.clear()
    clear_telemetry()
    release_cuda_cache()
    return (
        render_hardware()
        + "\n\nModelos retirados da memória. Os arquivos de pesos continuam no cache em disco do PyTorch."
    )


def build_demo() -> gr.Blocks:
    choices = list_model_names()
    waiting = placeholder_image("O mapa Grad-CAM aparece depois da classificação.")
    rows = example_rows()

    with gr.Blocks(title="Laboratório ImageNet · CNNs", analytics_enabled=False) as demo:
        gr.Markdown(
            """
# Laboratório ImageNet
Compare CNNs pré-treinadas no ImageNet-1k, meça a latência na GPU e veja onde a rede olhou com Grad-CAM.

Não há treinamento e o dataset ImageNet não é baixado. Os pesos são os checkpoints oficiais do `torchvision` (`weights=…DEFAULT`), pedidos na primeira seleção de cada arquitetura e reaproveitados em disco.
"""
        )
        hw_md = gr.Markdown(render_hardware())
        with gr.Row():
            refresh_btn = gr.Button("Atualizar hardware")
            free_btn = gr.Button("Liberar VRAM")

        with gr.Tabs():
            with gr.Tab("Inferência"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dd = gr.Dropdown(
                            choices=choices,
                            value="ResNet-50",
                            label="Modelo",
                            info="Um modelo, com card de arquitetura, Top-5 e Grad-CAM.",
                        )
                        arch_card = gr.Markdown(format_static("ResNet-50"))
                        with gr.Accordion("Como o Grad-CAM é calculado", open=False):
                            gr.Markdown(
                                """
1. Um forward de sondagem escolhe o **último tensor que ainda tem altura e largura** — a saída do último bloco ou do pooling, já depois da ativação. Uma convolução crua, antes da ReLU, empurra o mapa para a borda.
2. Outro forward guarda essas ativações. O backward parte do **logit da classe Top-1**.
3. O peso de cada canal é a **média espacial do gradiente**. O mapa é `ReLU(soma dos canais ponderados)`.
4. O mapa sobe, por interpolação bilinear, até o **recorte** que `weights.transforms()` entregou à rede.

ReLUs inplace são desligadas no carregamento. Sem isso, a ativação seria reescrita antes do backward e o mapa sairia errado ou quebraria.

Referência: [Selvaraju et al., Grad-CAM (2017)](https://arxiv.org/abs/1610.02391).
"""
                            )
                    with gr.Column(scale=1):
                        image = gr.Image(
                            label="Imagem",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=360,
                            buttons=["download", "fullscreen"],
                        )
                        if rows:
                            gr.Examples(
                                examples=rows,
                                inputs=[image],
                                label="Exemplos: cachorro, avião e café",
                                cache_examples=False,
                            )
                        classify_btn = gr.Button("Classificar imagem", variant="primary")
                result_md = gr.Markdown()
                with gr.Row():
                    plot = gr.Plot(label="Top-5", value=top5_figure("Top-5", None, COLOR_A))
                    cam = gr.Image(label="Grad-CAM", value=waiting, type="pil", buttons=["download", "fullscreen"])

            with gr.Tab("Comparação"):
                gr.Markdown(
                    "Dois modelos, a mesma foto. Predição, mapa e latência ficam lado a lado. "
                    "O par inicial é ResNet-50 contra MobileNetV3-Large."
                )
                with gr.Row():
                    model_a = gr.Dropdown(choices=choices, value="ResNet-50", label="Modelo A")
                    model_b = gr.Dropdown(choices=choices, value="MobileNetV3-Large", label="Modelo B")
                with gr.Row():
                    card_a = gr.Markdown(format_static("ResNet-50"))
                    card_b = gr.Markdown(format_static("MobileNetV3-Large"))
                image_cmp = gr.Image(
                    label="Imagem compartilhada",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=320,
                    buttons=["download", "fullscreen"],
                )
                if rows:
                    gr.Examples(
                        examples=rows,
                        inputs=[image_cmp],
                        label="Exemplos: cachorro, avião e café",
                        cache_examples=False,
                    )
                compare_btn = gr.Button("Comparar modelos", variant="primary")
                summary_md = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        result_a = gr.Markdown()
                        plot_a = gr.Plot(label="Top-5 · A", value=top5_figure("Top-5 · A", None, COLOR_A))
                        cam_a = gr.Image(label="Grad-CAM · A", value=waiting, type="pil", buttons=["download", "fullscreen"])
                    with gr.Column():
                        result_b = gr.Markdown()
                        plot_b = gr.Plot(label="Top-5 · B", value=top5_figure("Top-5 · B", None, COLOR_B))
                        cam_b = gr.Image(label="Grad-CAM · B", value=waiting, type="pil", buttons=["download", "fullscreen"])

            with gr.Tab("Benchmark"):
                gr.Markdown(
                    "Marque várias arquiteturas e meça cada uma em sequência, com o mesmo tensor sintético. "
                    "A ordem da tabela é a da latência, da mais rápida para a mais lenta."
                )
                bench_models = gr.CheckboxGroup(
                    choices=choices,
                    value=["ResNet-18", "ResNet-50", "MobileNetV3-Large"],
                    label="Modelos",
                )
                bench_btn = gr.Button("Rodar benchmark", variant="primary")
                bench_md = gr.Markdown()
                bench_table = gr.Dataframe(
                    headers=BENCH_HEADERS,
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="Telemetria",
                )
                with gr.Accordion("Por que a latência usa CUDA Events", open=False):
                    gr.Markdown(
                        """
`torch.cuda.Event` marca o início e o fim **na fila da GPU**. `elapsed_time` só é lido depois de sincronizar, então o número não mistura o tempo em que a CPU apenas enfileirou o trabalho.

`time.perf_counter` entra apenas em CPU ou MPS, onde não existe esse relógio.

O aquecimento deixa o cuDNN escolher o algoritmo (`cudnn.benchmark`) fora da média. TF32 fica ligado em GPU Ampere ou mais nova e vale para todos os modelos, então a comparação entre arquiteturas continua justa.
"""
                    )

        gr.Markdown(
            "Pesos oficiais do torchvision para o ImageNet-1k. "
            "A aplicação não treina e não distribui o dataset."
        )

        refresh_btn.click(render_hardware, outputs=hw_md, concurrency_limit=1, api_name="hardware")
        free_btn.click(on_free_vram, outputs=hw_md, concurrency_limit=1, api_name="liberar_vram")
        model_dd.change(on_profile, inputs=model_dd, outputs=arch_card, concurrency_limit=1, api_name="perfil_inferencia")
        model_a.change(on_profile, inputs=model_a, outputs=card_a, concurrency_limit=1)
        model_b.change(on_profile, inputs=model_b, outputs=card_b, concurrency_limit=1)
        classify_btn.click(
            on_classify,
            inputs=[image, model_dd],
            outputs=[plot, cam, result_md],
            concurrency_limit=1,
            api_name="classificar",
        )
        compare_btn.click(
            on_compare,
            inputs=[image_cmp, model_a, model_b],
            outputs=[summary_md, plot_a, cam_a, result_a, plot_b, cam_b, result_b],
            concurrency_limit=1,
            api_name="comparar",
        )
        bench_btn.click(
            on_benchmark,
            inputs=bench_models,
            outputs=[bench_table, bench_md],
            concurrency_limit=1,
            api_name="benchmark",
        )
        demo.load(boot, inputs=[model_dd, model_a, model_b], outputs=[hw_md, arch_card, card_a, card_b])

    return demo


def run_check() -> int:
    activate_device()
    snap = snapshot(DEVICE)
    print(f"dispositivo: {device_label(snap.device)} · {snap.gpu_name}")
    print(f"pytorch: {snap.torch_version} · cuda do wheel: {snap.cuda_build}")
    if not is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")):
        print("falha: o detector de CUDA OOM não reconheceu a mensagem.")
        return 1
    if is_cuda_oom(RuntimeError("arquivo ausente")):
        print("falha: o detector de CUDA OOM deu falso positivo.")
        return 1
    for name in list_model_names():
        spec = get_spec(name)
        info = inspect_transform(spec.weights.transforms())
        count = len(categories_of(spec.weights))
        print(f"  {name}: {checkpoint_name(spec.weights)} · {count} classes · crop {info.crop}")
        if count != 1000:
            print("falha: este catálogo deveria ser ImageNet-1k (1000 classes).")
            return 1
    loaded, telemetry = benchmark_model(ZOO, "ResNet-18", DEVICE, refresh=True)
    gmacs = None if loaded.macs is None else loaded.macs / 1e9
    print(
        f"ResNet-18: {loaded.parameters / 1e6:.2f} M parâmetros · "
        f"{gmacs if gmacs is not None else 'sem'} GMACs · "
        f"{telemetry.latency_ms:.2f} ± {telemetry.latency_std_ms:.2f} ms via {telemetry.timing_method}"
    )
    if loaded.macs is None or not 1.4e9 <= loaded.macs <= 2.6e9:
        print(f"falha: MACs de ResNet-18 fora da faixa esperada ({loaded.macs}).")
        return 1
    dog = EXAMPLE_DIR / "cachorro.jpg"
    if not dog.is_file():
        print("aviso: assets/examples/cachorro.jpg ausente; Grad-CAM não foi exercitado.")
        return 0
    analysis = analyze_image(ZOO, "ResNet-18", dog, DEVICE)
    print("top-5 do cachorro:")
    for score in analysis.scores:
        print(f"  {score.probability:6.1%}  {score.index:4d}  {score.name}")
    print(
        f"grad-cam: {analysis.cam.layer_name} "
        f"{analysis.cam.map_height}x{analysis.cam.map_width} em {analysis.cam.elapsed_ms:.1f} ms"
    )
    if analysis.cam.map_height <= 1 or analysis.cam.map_width <= 1:
        print("falha: o mapa Grad-CAM não tem extensão espacial.")
        return 1
    return 0


def main() -> None:
    configure_logging()
    args = _parse_args()
    if args.check:
        raise SystemExit(run_check())
    activate_device()
    demo = build_demo()
    snap = snapshot(DEVICE)
    print(f"Laboratório ImageNet · {snap.gpu_name} · {device_label(snap.device)}")
    print(f"Abra http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_theme(),
        css=".gradio-container {max-width: 1120px !important; margin-inline: auto !important;}",
        footer_links=["api", "gradio"],
        inbrowser=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laboratório didático de CNNs no ImageNet-1k")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Cria um link público temporário do Gradio.")
    parser.add_argument("--check", action="store_true", help="Valida o dispositivo, a ResNet-18 e o Grad-CAM, e encerra.")
    return parser.parse_args()


def _theme():
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
    )


def _ordered(names: list[str]) -> list[str]:
    order = {name: index for index, name in enumerate(list_model_names())}
    unique = []
    for name in names:
        if name not in unique:
            get_spec(name)
            unique.append(name)
    return sorted(unique, key=lambda name: order.get(name, 10_000))


def _bench_row(loaded: LoadedModel, telemetry: Telemetry) -> list[str]:
    return [
        loaded.spec.name,
        loaded.spec.family,
        f"{loaded.parameters / 1e6:.2f}",
        f"{loaded.parameter_bytes / (1024 ** 2):.1f}",
        _giga(loaded.macs),
        _giga(loaded.macs, flops=True),
        f"{telemetry.latency_ms:.2f}",
        f"{telemetry.latency_std_ms:.2f}",
        f"{_throughput_value(telemetry.latency_ms):.1f}",
        f"{telemetry.activation_peak_bytes / (1024 ** 2):.1f}",
        telemetry.device,
        "—" if loaded.published_top1 is None else f"{loaded.published_top1:.2f}%",
    ]


def _geometry(resize: int | None, crop: int) -> str:
    if resize and resize != crop:
        return f"resize {resize} → crop {crop}×{crop}"
    return f"{crop}×{crop}"


def _fmt_triple(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{item:.3f}" for item in values) + ")"


def _published(top1: float | None, top5: float | None) -> str:
    if top1 is None and top5 is None:
        return "—"
    left = "—" if top1 is None else f"{top1:.2f}%"
    right = "—" if top5 is None else f"{top5:.2f}%"
    return f"{left} / {right}"


def _giga(macs: int | None, *, flops: bool = False) -> str:
    if macs is None:
        return "—"
    scale = 2.0 if flops else 1.0
    return f"{macs * scale / 1e9:.2f}"


def _ms(telemetry: Telemetry) -> str:
    return f"{telemetry.latency_ms:.2f} ± {telemetry.latency_std_ms:.2f} ms"


def _throughput_value(latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return 1000.0 / latency_ms


def _throughput(latency_ms: float) -> str:
    return f"{_throughput_value(latency_ms):.1f}"


def _vram_free(telemetry: Telemetry) -> str:
    if telemetry.free_bytes is None or telemetry.total_bytes is None:
        return "não se aplica a este dispositivo"
    return f"{format_bytes(telemetry.free_bytes)} / {format_bytes(telemetry.total_bytes)}"


def _fence(text: str) -> str:
    return "```text\n" + text.replace("```", "'''") + "\n```"


if __name__ == "__main__":
    main()

# generar_pdf.py
# =====================================================
# LEVISTRO — Exportador de diagramas conceptuales a PDF
# =====================================================

import json
import os
import textwrap
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx


COLORES = {
    "central":        "#FFD166",
    "autor":          "#C9B1FF",
    "concepto":       "#FF8C69",
    "caracteristica": "#87CEEB",
    "ejemplo":        "#90EE90",
    "tension":        "#FFB3BA",
    "pregunta":       "#FFDAC1",
}
COLOR_DEFAULT = "#E0E0E0"

NIVELES = {
    "central":        0,
    "autor":          1,
    "concepto":       2,
    "tension":        2,
    "pregunta":       2,
    "caracteristica": 3,
    "ejemplo":        3,
}

# Ancho máx de texto por contexto
WRAP_LABEL = 16
WRAP_DESC  = 26
WRAP_CITA  = 30

# Espaciado entre niveles (en unidades de figura)
ESPACIO_NIVEL = 0.22

# Ancho y alto base de nodo
NODE_W = 0.13
NODE_H = 0.06


def wrap(texto, w):
    if not texto:
        return ""
    return "\n".join(textwrap.wrap(str(texto), w))


def altura_texto(texto, fontsize, fig_h_in):
    """Altura aproximada en unidades de figura para N líneas de texto."""
    n_lines = texto.count("\n") + 1 if texto else 0
    pts_por_linea = fontsize * 1.35
    return n_lines * pts_por_linea / (fig_h_in * 72)


def layout(nodos, aristas):
    """Layout jerárquico con separación dinámica horizontal."""
    filas = defaultdict(list)
    for n in nodos:
        nivel = NIVELES.get(n.get("tipo", "concepto"), 2)
        filas[nivel].append(n["id"])

    n_filas = max(filas.keys()) + 1 if filas else 1
    max_por_fila = max(len(ids) for ids in filas.values())

    # Figura más ancha si hay muchos nodos por fila
    fig_w = max(18, max_por_fila * 3.2)
    fig_h = max(14, n_filas * 4.5)

    pos = {}
    for nivel, ids in sorted(filas.items()):
        # Y: empieza cerca del techo, baja por nivel
        y = 0.88 - nivel * ESPACIO_NIVEL * (10 / n_filas)
        n = len(ids)
        for i, nid in enumerate(ids):
            # Distribuir uniformemente con margen lateral
            x = 0.08 + (i / max(n - 1, 1)) * 0.84 if n > 1 else 0.5
            pos[nid] = (x, y)

    return pos, fig_w, fig_h


def dibujar_grafico(datos, output_path):
    nodos    = datos.get("nodos", [])
    aristas  = datos.get("aristas", [])
    titulo   = datos.get("titulo", "Diagrama Conceptual")
    subtitulo = datos.get("subtitulo", "")

    nodo_map = {n["id"]: n for n in nodos}
    pos, fig_w, fig_h = layout(nodos, aristas)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#F5F5F0")
    fig.patch.set_facecolor("#F5F5F0")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Título en la cabecera de la figura (fuera del área de nodos) ──
    fig.text(
        0.5, 0.97, titulo,
        ha="center", va="top",
        fontsize=18, fontweight="bold", color="#111111",
    )
    if subtitulo:
        fig.text(
            0.5, 0.94, subtitulo,
            ha="center", va="top",
            fontsize=11, color="#555555", style="italic",
        )

    # ── Pre-calcular bounding boxes de nodos ──────────────────────────
    boxes = {}
    for nid, (x, y) in pos.items():
        nodo  = nodo_map.get(nid, {})
        tipo  = nodo.get("tipo", "concepto")
        label = wrap(nodo.get("label", nid), WRAP_LABEL)
        es_c  = tipo == "central"

        n_lines = label.count("\n") + 1
        w = NODE_W * (1.4 if es_c else 1.0)
        h = NODE_H + 0.012 * (n_lines - 1)
        boxes[nid] = (x, y, w, h)

    # ── Aristas ───────────────────────────────────────────────────────
    for a in aristas:
        src, dst = a.get("desde"), a.get("hacia")
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        _, _, _, h1 = boxes[src]
        _, _, _, h2 = boxes[dst]

        # Salir del borde inferior del nodo origen y entrar por el superior del destino
        y1_exit = y1 - h1 / 2
        y2_enter = y2 + h2 / 2

        ax.annotate(
            "", xy=(x2, y2_enter), xytext=(x1, y1_exit),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#777777", lw=1.4,
                mutation_scale=14,
                connectionstyle="arc3,rad=0.05",
            ),
            zorder=2,
        )

        etiqueta = a.get("label", "")
        if etiqueta:
            mx = (x1 + x2) / 2
            my = (y1_exit + y2_enter) / 2
            ax.text(
                mx, my, etiqueta,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=7, style="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.12",
                          facecolor="white", edgecolor="#BBBBBB", alpha=0.9),
                zorder=5,
            )

    # ── Nodos ─────────────────────────────────────────────────────────
    for nid, (x, y, w, h) in boxes.items():
        nodo  = nodo_map.get(nid, {})
        tipo  = nodo.get("tipo", "concepto")
        color = COLORES.get(tipo, COLOR_DEFAULT)
        label = wrap(nodo.get("label", nid), WRAP_LABEL)
        desc  = nodo.get("descripcion", "")
        cita  = nodo.get("cita", "")
        es_c  = tipo == "central"

        # Caja
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.01",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="#111111" if es_c else "#555555",
            linewidth=2.5 if es_c else 1.4,
            zorder=3,
        )
        ax.add_patch(box)

        # Label dentro de la caja
        ax.text(
            x, y, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10 if es_c else 8.5,
            fontweight="bold",
            color="#111111",
            zorder=4,
        )

        # Texto debajo de la caja — descripción
        y_bajo = y - h/2 - 0.008
        if desc:
            desc_w = wrap(desc, WRAP_DESC)
            ax.text(
                x, y_bajo, desc_w,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, color="#333333",
                zorder=4,
            )
            y_bajo -= altura_texto(desc_w, 7, fig_h) + 0.005

        # Cita en itálica
        if cita:
            cita_w = wrap(f'"{cita}"', WRAP_CITA)
            ax.text(
                x, y_bajo, cita_w,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=6.5, color="#555555", style="italic",
                zorder=4,
            )

    # ── Leyenda ───────────────────────────────────────────────────────
    presentes = {n.get("tipo", "concepto") for n in nodos}
    leyenda = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in COLORES.items() if t in presentes
    ]
    if leyenda:
        ax.legend(
            handles=leyenda,
            loc="lower right",
            fontsize=8, framealpha=0.9,
        )

    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  PDF generado: {output_path}")


def main():
    ruta = Path("ultimo_grafico.json")
    if not ruta.exists():
        print("No hay ningún gráfico generado todavía.")
        print("Pedile al agente: 'haceme un gráfico de [tema]'")
        return

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    carpeta = Path("diagramas")
    carpeta.mkdir(exist_ok=True)

    nombre = datos.get("titulo", "diagrama").replace(" ", "_")[:50]
    output = carpeta / f"{nombre}.pdf"

    print(f"\nGenerando PDF: {output}")
    dibujar_grafico(datos, str(output))

    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    abrir = messagebox.askyesno(
        "Diagrama listo",
        f"PDF generado:\n{output.name}\n\n¿Abrirlo ahora?"
    )
    root.destroy()
    if abrir:
        os.startfile(str(output))


if __name__ == "__main__":
    main()

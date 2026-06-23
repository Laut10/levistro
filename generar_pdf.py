# generar_pdf.py
# =====================================================
# LEVISTRO — Exportador de diagramas conceptuales a PDF
#
# Lee ultimo_grafico.json generado por el agente y
# produce un PDF con jerarquía visual, descripciones
# y citas opcionales bajo cada nodo.
#
# Uso: python generar_pdf.py
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

# Nivel jerárquico por tipo — determina posición vertical
NIVELES = {
    "central":        0,
    "autor":          1,
    "concepto":       2,
    "tension":        2,
    "pregunta":       2,
    "caracteristica": 3,
    "ejemplo":        3,
}


def envolver(texto, max_chars=20):
    if not texto:
        return ""
    return "\n".join(textwrap.wrap(str(texto), max_chars))


def layout_jerarquico(nodos, aristas):
    """
    Posiciona nodos en filas según su tipo/nivel.
    Dentro de cada fila los distribuye horizontalmente.
    """
    niveles = defaultdict(list)
    for n in nodos:
        nivel = NIVELES.get(n.get("tipo", "concepto"), 2)
        niveles[nivel].append(n["id"])

    pos = {}
    max_nivel = max(niveles.keys()) if niveles else 0

    for nivel, ids in sorted(niveles.items()):
        y = 1.0 - nivel / (max_nivel + 1)
        n_ids = len(ids)
        for i, nid in enumerate(ids):
            x = (i + 1) / (n_ids + 1)
            pos[nid] = (x, y)

    return pos


def dibujar_grafico(datos, output_path="diagrama_conceptual.pdf"):
    nodos  = datos.get("nodos", [])
    aristas = datos.get("aristas", [])
    titulo  = datos.get("titulo", "Diagrama Conceptual")
    subtitulo = datos.get("subtitulo", "")

    nodo_map = {n["id"]: n for n in nodos}
    pos = layout_jerarquico(nodos, aristas)

    # Calcular alto de figura según contenido
    fig_h = max(14, len(nodos) * 1.2)
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.set_facecolor("#F5F5F0")
    fig.patch.set_facecolor("#F5F5F0")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    # ── Aristas ──────────────────────────────────────────
    for a in aristas:
        if a["desde"] not in pos or a["hacia"] not in pos:
            continue
        x1, y1 = pos[a["desde"]]
        x2, y2 = pos[a["hacia"]]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            xycoords="data", textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#666666", lw=1.6,
                mutation_scale=16,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=2,
        )
        etiqueta = a.get("label", "")
        if etiqueta:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            ax.text(
                mx, my, etiqueta,
                ha="center", va="center",
                fontsize=7, style="italic", color="#444444",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="#CCCCCC", alpha=0.85),
                zorder=5,
            )

    # ── Nodos ────────────────────────────────────────────
    for nid, (x, y) in pos.items():
        nodo   = nodo_map.get(nid, {})
        tipo   = nodo.get("tipo", "concepto")
        color  = COLORES.get(tipo, COLOR_DEFAULT)
        label  = nodo.get("label", nid)
        desc   = nodo.get("descripcion", "")
        cita   = nodo.get("cita", "")
        central = tipo == "central"

        label_wrap = envolver(label, 18 if central else 20)
        n_label_lines = label_wrap.count("\n") + 1

        BOX_W = 0.14 if central else 0.12
        BOX_H = 0.045 + 0.018 * n_label_lines

        # Caja del nodo
        box = FancyBboxPatch(
            (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
            boxstyle="round,pad=0.012",
            facecolor=color,
            edgecolor="#222222" if central else "#555555",
            linewidth=2.5 if central else 1.5,
            zorder=3,
        )
        ax.add_patch(box)

        # Texto del label dentro del nodo
        ax.text(
            x, y, label_wrap,
            ha="center", va="center",
            fontsize=10 if central else 8.5,
            fontweight="bold" if central else "semibold",
            color="#111111",
            zorder=4,
        )

        # Descripción bajo el nodo
        y_offset = y - BOX_H/2 - 0.012
        if desc:
            desc_wrap = envolver(desc, 30)
            ax.text(
                x, y_offset, desc_wrap,
                ha="center", va="top",
                fontsize=7, color="#333333",
                zorder=4,
            )
            y_offset -= 0.012 * (desc_wrap.count("\n") + 2)

        # Cita bajo la descripción
        if cita:
            cita_wrap = envolver(f'"{cita}"', 34)
            ax.text(
                x, y_offset, cita_wrap,
                ha="center", va="top",
                fontsize=6.5, color="#666666", style="italic",
                zorder=4,
            )

    # ── Título ───────────────────────────────────────────
    ax.text(
        0.5, 1.03, titulo,
        ha="center", va="bottom",
        fontsize=17, fontweight="bold", color="#111111",
        transform=ax.transData,
    )
    if subtitulo:
        ax.text(
            0.5, 1.005, subtitulo,
            ha="center", va="bottom",
            fontsize=10, color="#555555", style="italic",
            transform=ax.transData,
        )

    # ── Leyenda ──────────────────────────────────────────
    presentes = {n.get("tipo", "concepto") for n in nodos}
    leyenda = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in COLORES.items() if t in presentes
    ]
    if leyenda:
        ax.legend(
            handles=leyenda, loc="lower right",
            fontsize=8, framealpha=0.9,
        )

    plt.tight_layout()
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

    # Popup con opción de abrir
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

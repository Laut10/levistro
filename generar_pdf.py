# generar_pdf.py
# =====================================================
# LEVISTRO — Exportador de diagramas conceptuales a PDF
#
# Lee ultimo_grafico.json generado por el agente y
# produce un PDF con el diagrama de nodos y relaciones.
#
# Uso: python generar_pdf.py
# =====================================================

import json
import sys
import os
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx


COLORES = {
    "central":       "#FFD166",
    "concepto":      "#FF8C69",
    "caracteristica":"#87CEEB",
    "ejemplo":       "#90EE90",
    "autor":         "#C9B1FF",
    "tension":       "#FFB3BA",
    "pregunta":      "#FFDAC1",
}
COLOR_DEFAULT = "#E0E0E0"


def envolver_texto(texto, max_chars=22):
    return "\n".join(textwrap.wrap(texto, max_chars))


def dibujar_grafico(datos, output_path="diagrama_conceptual.pdf"):
    nodos = datos.get("nodos", [])
    aristas = datos.get("aristas", [])
    titulo = datos.get("titulo", "Diagrama Conceptual")
    subtitulo = datos.get("subtitulo", "")

    G = nx.DiGraph()
    nodo_map = {}
    for n in nodos:
        G.add_node(n["id"])
        nodo_map[n["id"]] = n

    for a in aristas:
        G.add_edge(a["desde"], a["hacia"], label=a.get("label", ""))

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("#F8F8F8")
    ax.axis("off")

    # Layout — jerarquico si hay nodo central, spring si no
    central = next((n["id"] for n in nodos if n.get("tipo") == "central"), None)
    try:
        if central and len(G.nodes) > 1:
            pos = nx.spring_layout(G, seed=42, k=2.5, center=(0.5, 0.5))
        else:
            pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = {n: (i * 0.3, 0) for i, n in enumerate(G.nodes())}

    # Escalar posiciones al espacio del eje
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin = 0.15

    def escalar(x, y):
        rx = (x - x_min) / (x_max - x_min + 1e-9)
        ry = (y - y_min) / (y_max - y_min + 1e-9)
        return margin + rx * (1 - 2*margin), margin + ry * (1 - 2*margin)

    pos_scaled = {n: escalar(*p) for n, p in pos.items()}

    # Dibujar aristas primero
    for u, v, data in G.edges(data=True):
        x1, y1 = pos_scaled[u]
        x2, y2 = pos_scaled[v]
        dx, dy = x2 - x1, y2 - y1
        dist = math.sqrt(dx**2 + dy**2)

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                lw=1.8,
                mutation_scale=18,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=2,
        )
        # Etiqueta de arista
        etiqueta = data.get("label", "")
        if etiqueta:
            mx = (x1 + x2) / 2 + dy * 0.04
            my = (y1 + y2) / 2 - dx * 0.04
            ax.text(
                mx, my, etiqueta,
                transform=ax.transAxes,
                fontsize=7.5, ha="center", va="center",
                style="italic", color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9),
                zorder=5,
            )

    # Dibujar nodos
    BOX_W = 0.13
    BOX_H = 0.08

    for nodo_id, (x, y) in pos_scaled.items():
        nodo = nodo_map.get(nodo_id, {})
        tipo = nodo.get("tipo", "concepto")
        color = COLORES.get(tipo, COLOR_DEFAULT)
        label = nodo.get("label", nodo_id)
        es_central = tipo == "central"

        w = BOX_W * (1.5 if es_central else 1.0)
        h = BOX_H * (1.3 if es_central else 1.0)

        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.015",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="#222222" if es_central else "#555555",
            linewidth=2.5 if es_central else 1.5,
            zorder=3,
        )
        ax.add_patch(box)

        texto = envolver_texto(label, max_chars=18 if es_central else 20)
        ax.text(
            x, y, texto,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9 if es_central else 8,
            fontweight="bold" if es_central else "normal",
            color="#111111",
            zorder=4,
        )

        # Descripción opcional debajo del nodo
        desc = nodo.get("descripcion", "")
        if desc:
            desc_wrap = envolver_texto(desc, max_chars=28)
            ax.text(
                x, y - h/2 - 0.025, desc_wrap,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=6.5, color="#444444",
                zorder=4,
            )

    # Título
    y_titulo = 0.97
    ax.text(
        0.5, y_titulo, titulo,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=16, fontweight="bold", color="#111111",
    )
    if subtitulo:
        ax.text(
            0.5, y_titulo - 0.04, subtitulo,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, color="#555555", style="italic",
        )

    # Leyenda de colores
    leyenda = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in COLORES.items()
        if any(n.get("tipo") == t for n in nodos)
    ]
    if leyenda:
        ax.legend(
            handles=leyenda,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            fontsize=8,
            framealpha=0.9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    nombre = datos.get("titulo", "diagrama").replace(" ", "_")[:40]
    output = f"{nombre}.pdf"

    print(f"\nGenerando PDF: {output}")
    dibujar_grafico(datos, output)

    # Abrir el PDF automáticamente
    os.startfile(output)


if __name__ == "__main__":
    main()

# graph_rag.py
# =====================================================
# LEVISTRO — GraphRAG con LightRAG
#
# Construye un grafo de conocimiento a partir de los
# resúmenes estructurados de la biblioteca (resumenes.json).
#
# Cada resumen ya tiene: tesis, conceptos operativos,
# posición epistemológica, punto de vista decantado y
# diálogo con otros autores — materia prima ideal para
# extraer entidades y relaciones entre autores/conceptos.
#
# El grafo se guarda en lightrag_storage/ y puede
# consultarse desde el agente con modos:
#   - local:  relaciones directas entre entidades
#   - global: patrones y temas que cruzan toda la biblioteca
#   - hybrid: combina ambos
# =====================================================

import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc


WORKING_DIR = "./lightrag_storage"
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


async def inicializar_rag():
    Path(WORKING_DIR).mkdir(exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        llm_model_kwargs={
            "host": OLLAMA_BASE,
            "options": {"temperature": 0},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host=OLLAMA_BASE,
            ),
        ),
    )
    return rag


async def construir_grafo():
    """
    Ingesta los resúmenes estructurados en LightRAG.
    Cada resumen incluye tesis, conceptos, posición epistemológica
    y diálogo con otros autores — LightRAG extrae las relaciones.
    """
    ruta = Path("resumenes.json")
    if not ruta.exists():
        print("No se encontró resumenes.json. Corré summarizer.py primero.")
        return

    resumenes = json.loads(ruta.read_text(encoding="utf-8"))
    print(f"\n{len(resumenes)} textos para construir el grafo.\n")

    rag = await inicializar_rag()

    # Verificar si ya hay datos en el grafo
    grafo_existente = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
    if grafo_existente.exists():
        print("Grafo existente encontrado.")
        print("Para reconstruir desde cero, borrá la carpeta lightrag_storage/\n")
        return rag

    documentos = []
    for fuente, resumen in resumenes.items():
        # Prefijamos con el nombre del archivo para que LightRAG
        # pueda anclar las entidades al texto correcto
        documentos.append(f"TEXTO: {fuente}\n\n{resumen}")

    print("Construyendo grafo de conocimiento...")
    print("(qwen extrae entidades y relaciones de cada resumen)\n")

    await rag.ainsert(documentos)

    print("\nGrafo construido. Guardado en lightrag_storage/")
    return rag


async def consultar(pregunta: str, modo: str = "hybrid"):
    """
    Consulta el grafo de conocimiento.
    Modos:
      local  — relaciones directas entre entidades mencionadas
      global — patrones que cruzan toda la biblioteca
      hybrid — combina local + global (recomendado)
    """
    rag = await inicializar_rag()
    respuesta = await rag.aquery(
        pregunta,
        param=QueryParam(mode=modo)
    )
    return respuesta


async def modo_consulta_interactivo():
    """Loop interactivo para consultar el grafo."""
    print("\n" + "="*55)
    print("  LEVISTRO — Consulta GraphRAG")
    print("  Modos: /local | /global | /hybrid (default)")
    print("  'salir' para terminar")
    print("="*55 + "\n")

    rag = await inicializar_rag()
    modo = "hybrid"

    while True:
        pregunta = input("Vos: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ["salir", "exit"]:
            break
        if pregunta.lower() in ["/local", "/global", "/hybrid"]:
            modo = pregunta[1:].lower()
            print(f"  Modo: {modo}\n")
            continue

        print(f"\n  [Consultando grafo — modo {modo}]\n")
        print("Grafo: ", end="", flush=True)
        respuesta = await rag.aquery(pregunta, param=QueryParam(mode=modo))
        print(respuesta)
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "construir":
        asyncio.run(construir_grafo())
    else:
        # Verificar que el grafo existe antes de consultar
        grafo = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
        if not grafo.exists():
            print("El grafo no existe todavía. Corré primero:")
            print("  python graph_rag.py construir")
        else:
            asyncio.run(modo_consulta_interactivo())

# summarizer.py
# =====================================================
# OBJETIVO: Generar un resumen conceptual de cada libro
# usando los chunks ya guardados en Qdrant.
#
# Se ejecuta UNA SOLA VEZ. Los resúmenes quedan en
# resumenes.json y el chatbot los incluye en cada
# conversación como "conocimiento de fondo" del agente.
#
# Esto simula que el modelo "leyó" todos los libros —
# siempre tiene los conceptos clave disponibles sin
# necesidad de recuperarlos con cada pregunta.
# =====================================================

import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient


def obtener_chunks_por_fuente(cliente, coleccion, fuente):
    """
    Recupera todos los chunks de un archivo específico desde Qdrant.
    Los ordena para reconstruir el texto en orden aproximado.
    """
    resultados = cliente.scroll(
        collection_name=coleccion,
        scroll_filter={
            "must": [{
                "key": "metadata.fuente",
                "match": {"value": fuente}
            }]
        },
        limit=2000,
        with_payload=True
    )
    return [p.payload.get("page_content", "") for p in resultados[0] if p.payload]


def generar_resumen(fuente, chunks, llm):
    """
    Genera un resumen académico estructurado usando método comparativo clásico
    de la antropología: ficha temática + preguntas problematizadoras +
    posicionamiento epistemológico + punto de vista decantado.
    """
    texto_muestra = "\n\n---\n\n".join(chunks[:30])

    prompt = f"""Analizá los siguientes fragmentos del texto "{fuente}" y generá un resumen académico estructurado.
Usá el método comparativo clásico de la antropología: no solo describí el texto sino que decantá el punto de vista específico que representa este autor en el campo.

FRAGMENTOS:
{texto_muestra}

Generá el resumen con este formato exacto:

AUTOR Y OBRA: [nombre del autor y título inferido]

TESIS CENTRAL: [la afirmación principal que el texto defiende, en 2-3 oraciones]

CONCEPTOS OPERATIVOS: [5-8 términos técnicos que el autor construye o resignifica, con una línea de explicación cada uno]

ARGUMENTO: [cómo desarrolla y sostiene la tesis paso a paso, en 3-4 oraciones]

POSICIÓN EPISTEMOLÓGICA: [desde qué lugar teórico habla — empirismo, estructuralismo, fenomenología, etc. — y qué supuestos asume sobre el conocimiento]

PREGUNTAS QUE ABRE: [2-4 tensiones, problemas o interrogantes que el texto deja abiertos o sin resolver]

PUNTO DE VISTA DECANTADO: [qué perspectiva específica representa este autor en el debate del campo — qué puede ver desde su posición que otros no ven, y qué queda fuera de su mirada]

DIÁLOGO CON OTROS AUTORES: [con quién conversa explícita o implícitamente, dónde coincide y dónde diverge]

Respondé solo con el resumen, sin comentarios adicionales."""

    respuesta = llm.invoke(prompt)
    return respuesta.content


def main():
    print("\n" + "="*55)
    print("  LEVISTRO — Generador de resúmenes por autor")
    print("  Usa qwen2.5:14b para analizar cada texto")
    print("="*55 + "\n")

    # Verificar si ya existe resumenes.json
    ruta_resumenes = Path("resumenes.json")
    if ruta_resumenes.exists():
        resumenes_existentes = json.loads(ruta_resumenes.read_text(encoding="utf-8"))
        print(f"resumenes.json encontrado con {len(resumenes_existentes)} resúmenes existentes.")
        print("Solo procesará los textos que falten.\n")
    else:
        resumenes_existentes = {}

    # Conectar a Qdrant
    cliente = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    coleccion = os.getenv("QDRANT_COLLECTION", "mi-biblioteca")

    # Obtener lista de todos los archivos en la biblioteca
    todos = cliente.scroll(collection_name=coleccion, limit=20000, with_payload=True)
    fuentes = sorted(set(
        p.payload.get("metadata", {}).get("fuente", "")
        for p in todos[0]
        if p.payload
    ))
    fuentes = [f for f in fuentes if f]  # quitar vacíos

    print(f"Archivos en la biblioteca: {len(fuentes)}")
    pendientes = [f for f in fuentes if f not in resumenes_existentes]
    print(f"Pendientes de resumir: {len(pendientes)}\n")

    if not pendientes:
        print("Todos los textos ya tienen resumen. resumenes.json está al día.")
        return

    # qwen2.5:14b para generar los resúmenes
    # Este modelo entiende bien el vocabulario académico en español
    llm = ChatOllama(
        model="qwen2.5:14b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    for i, fuente in enumerate(pendientes, 1):
        print(f"[{i}/{len(pendientes)}] Resumiendo: {fuente}")

        chunks = obtener_chunks_por_fuente(cliente, coleccion, fuente)
        if not chunks:
            print(f"  Sin chunks — saltando\n")
            continue

        print(f"  {len(chunks)} chunks disponibles...", end=" ", flush=True)

        try:
            resumen = generar_resumen(fuente, chunks, llm)
            resumenes_existentes[fuente] = resumen

            # Guardar después de cada resumen — si se interrumpe no se pierde nada
            ruta_resumenes.write_text(
                json.dumps(resumenes_existentes, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print("listo\n")

        except Exception as e:
            print(f"Error: {e}\n")
            continue

    print(f"\nResúmenes completados: {len(resumenes_existentes)} textos en resumenes.json")
    print("Reiniciá el chatbot para que los use automáticamente.")


if __name__ == "__main__":
    main()

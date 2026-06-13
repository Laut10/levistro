# chatbot.py
# =====================================================
# LEVISTRO — Agente de conocimiento académico
# Fase 2: Recuperación profunda + Resúmenes de fondo
#
# Cada respuesta usa DOS capas de conocimiento:
#
#   CAPA 1 — Resúmenes (espectro de fondo)
#   resumenes.json contiene la tesis, conceptos clave y
#   posición teórica de cada texto. El agente siempre sabe
#   de qué tratan los autores relevantes, incluso antes
#   de leer los chunks específicos.
#
#   CAPA 2 — Chunks (evidencia específica)
#   MultiQuery genera 4 versiones de tu pregunta,
#   Qdrant busca con cada una y recupera hasta 160 chunks
#   únicos. Son los fragmentos textuales concretos.
#
# El agente combina ambas capas: entiende el marco general
# del autor (resumen) y tiene evidencia textual específica
# (chunks) para responder con precisión.
# =====================================================

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText


def cargar_resumenes():
    """
    Carga resumenes.json al iniciar el chatbot.
    Estos resúmenes son el conocimiento de fondo del agente —
    generados por qwen2.5 a partir de todos los chunks de cada libro.
    Si el archivo no existe, el agente funciona igual pero sin ese contexto.
    """
    ruta = Path("resumenes.json")
    if ruta.exists():
        resumenes = json.loads(ruta.read_text(encoding="utf-8"))
        print(f"  {len(resumenes)} resúmenes de autores cargados como conocimiento de fondo.")
        return resumenes
    print("  resumenes.json no encontrado — el agente funciona sin conocimiento de fondo.")
    return {}


def seleccionar_resumenes_relevantes(resumenes, fuentes_recuperadas):
    """
    De todos los resúmenes disponibles, selecciona solo los que
    corresponden a los autores encontrados en los chunks recuperados.
    Así el prompt no se satura con información irrelevante.
    """
    resumenes_relevantes = {}
    for fuente in fuentes_recuperadas:
        if fuente in resumenes:
            resumenes_relevantes[fuente] = resumenes[fuente]
    return resumenes_relevantes


def formatear_resumenes(resumenes_relevantes):
    """
    Formatea los resúmenes seleccionados para incluirlos en el prompt
    como sección de conocimiento de fondo.
    """
    if not resumenes_relevantes:
        return ""
    secciones = []
    for fuente, resumen in resumenes_relevantes.items():
        secciones.append(f"[Conocimiento de fondo — {fuente}]\n{resumen}")
    return "\n\n".join(secciones)


def formatear_docs(docs):
    """
    Convierte los chunks recuperados en texto numerado.
    Cada fragmento muestra su fuente para que qwen pueda referenciarla.
    """
    fragmentos = []
    for i, doc in enumerate(docs, 1):
        fuente = doc.metadata.get("fuente", "fuente desconocida")
        fragmentos.append(f"[Fragmento {i} — {fuente}]\n{doc.page_content}")
    return "\n\n".join(fragmentos)


def crear_componentes():
    """
    Inicializa todos los servicios:
    - Qdrant: 16.478 chunks de 33 textos académicos
    - nomic-embed-text: convierte preguntas en vectores
    - qwen2.5:14b: 14B parámetros, razonamiento profundo
    """
    print("Conectando a Qdrant...")
    cliente = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    vector_store = QdrantVectorStore(
        client=cliente,
        collection_name=os.getenv("QDRANT_COLLECTION", "mi-biblioteca"),
        embedding=embeddings,
    )

    # Modelo híbrido — dos modelos para dos tareas distintas:
    #
    # llm_rapido (mistral 7B): genera las versiones de búsqueda
    # Tarea simple y repetitiva → modelo pequeño = más velocidad
    #
    # llm_profundo (qwen2.5 14B): sintetiza la respuesta final
    # Tarea compleja con mucho contexto → modelo grande = mejor calidad
    #
    # Resultado: la búsqueda es 2x más rápida sin sacrificar
    # la calidad de la respuesta final
    llm_rapido = ChatOllama(
        model="mistral",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    llm_profundo = ChatOllama(
        model="qwen2.5:14b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    # k=40 — hasta 40 chunks por versión de búsqueda
    retriever_base = vector_store.as_retriever(search_kwargs={"k": 40})

    # Prompt con DOS secciones de contexto:
    # {resumenes} = conocimiento de fondo (qué trata cada autor)
    # {context}   = chunks específicos recuperados para esta pregunta
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un agente de conocimiento académico especializado en ciencias sociales, antropología y filosofía.
Trabajás con una biblioteca personal de textos teóricos.

Tenés acceso a DOS tipos de información:
1. CONOCIMIENTO DE FONDO: resúmenes de los autores relevantes para esta pregunta
2. FRAGMENTOS ESPECÍFICOS: pasajes textuales recuperados directamente de los libros

CÓMO USARLOS:
- El conocimiento de fondo te da el marco general del autor y su posición teórica
- Los fragmentos te dan la evidencia textual específica para responder
- Combiná ambos para dar respuestas precisas y contextualizadas
- Si una idea viene de los fragmentos, citá el número: (Fragmento 3)
- Si algo no aparece en ninguna de las dos fuentes, lo decís claramente
- Nunca agregues información de tu entrenamiento que no esté en estas fuentes

CONOCIMIENTO DE FONDO DE LOS AUTORES RELEVANTES:
{resumenes}

FRAGMENTOS RECUPERADOS:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # La cadena final usa llm_profundo (qwen) para la respuesta
    cadena = prompt | llm_profundo | StrOutputParser()

    return vector_store, retriever_base, llm_rapido, cadena


def generar_versiones(pregunta, llm):
    """
    qwen genera 4 reformulaciones de la pregunta con vocabulario distinto.
    Cada versión busca el mismo tema desde un ángulo diferente,
    recuperando chunks que una sola búsqueda perdería.
    """
    prompt_versiones = f"""Generá 4 versiones distintas de esta pregunta académica para buscar en una biblioteca de ciencias sociales.
Cada versión debe usar vocabulario diferente pero buscar el mismo tema.
Respondé SIEMPRE en español, sin importar el idioma de la pregunta original.
Devolvé SOLO las 4 preguntas, una por línea, sin numeración ni explicaciones.

Pregunta original: {pregunta}"""

    respuesta = llm.invoke(prompt_versiones).content
    versiones = [l.strip() for l in respuesta.strip().split("\n") if l.strip()]
    return [pregunta] + versiones[:4]


def buscar_multiquery(retriever, pregunta, llm):
    """
    5 búsquedas en Qdrant (pregunta original + 4 versiones).
    Los resultados se deduplicann por contenido.
    Resultado posible: hasta 200 chunks únicos.
    """
    versiones = generar_versiones(pregunta, llm)
    todos_los_docs = {}

    for version in versiones:
        docs = retriever.invoke(version)
        for doc in docs:
            clave = doc.page_content[:100]
            if clave not in todos_los_docs:
                todos_los_docs[clave] = doc

    return list(todos_los_docs.values())


def buscar_con_filtro(vector_store, pregunta, k=40, filtro_fuente=None):
    """Búsqueda directa filtrando por nombre de archivo."""
    return vector_store.similarity_search(
        pregunta,
        k=k,
        filter=Filter(must=[
            FieldCondition(
                key="metadata.fuente",
                match=MatchText(text=filtro_fuente)
            )
        ])
    )


def main():
    print("\n" + "="*55)
    print("  LEVISTRO — Agente de conocimiento académico")
    print("  Modelo: qwen2.5:14b | 'salir' para terminar")
    print("="*55 + "\n")

    # Cargar resúmenes de fondo al iniciar
    resumenes = cargar_resumenes()

    vector_store, retriever_base, llm_rapido, cadena = crear_componentes()
    print("Listo. Modelo híbrido: mistral para búsqueda, qwen2.5:14b para respuesta.\n")

    print("Comandos:")
    print("  /fuente <texto>  — filtrar por autor o archivo")
    print("  /fuente off      — buscar en toda la biblioteca")
    print("  /fuentes         — ver todos los archivos disponibles\n")

    historial = []
    filtro_fuente = None

    while True:
        pregunta = input("Vos: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Hasta luego.")
            break

        if pregunta.lower() == "/fuentes":
            c = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            resultados = c.scroll(collection_name=os.getenv("QDRANT_COLLECTION", "mi-biblioteca"), limit=20000, with_payload=True)
            fuentes = sorted(set(
                p.payload.get("metadata", {}).get("fuente", "?")
                for p in resultados[0]
            ))
            print("\nArchivos en la biblioteca:")
            for f in fuentes:
                print(f"  · {f}")
            print()
            continue

        if pregunta.lower().startswith("/fuente"):
            partes = pregunta.split(maxsplit=1)
            if len(partes) > 1 and partes[1].lower() != "off":
                filtro_fuente = partes[1].lower()
                print(f"  Filtro activo: '{filtro_fuente}'\n")
            else:
                filtro_fuente = None
                print("  Filtro desactivado\n")
            continue

        print("  [Buscando en la biblioteca...]\n")
        t_inicio = time.time()

        if filtro_fuente:
            docs = buscar_con_filtro(vector_store, pregunta, filtro_fuente=filtro_fuente)
        else:
            # mistral genera las versiones (rápido), qwen responde (profundo)
            docs = buscar_multiquery(retriever_base, pregunta, llm_rapido)

        t_busqueda = time.time() - t_inicio

        if not docs:
            print("  No encontré fragmentos relevantes.\n")
            continue

        # Seleccionar resúmenes solo de los autores encontrados
        fuentes = sorted(set(doc.metadata.get("fuente", "?") for doc in docs))
        resumenes_relevantes = seleccionar_resumenes_relevantes(resumenes, fuentes)

        contexto_chunks = formatear_docs(docs)
        contexto_resumenes = formatear_resumenes(resumenes_relevantes)

        print(f"  [{len(docs)} fragmentos de {len(fuentes)} archivo(s) — {len(resumenes_relevantes)} resúmenes de fondo — búsqueda: {t_busqueda:.1f}s]\n")
        print("Levistro: ", end="", flush=True)

        # Manejo de errores de conexión — si Ollama se reinicia o cae
        # el chatbot muestra el error y sigue funcionando en vez de crashear
        respuesta_completa = ""
        try:
            for chunk in cadena.stream({
                "resumenes": contexto_resumenes if contexto_resumenes else "No hay resúmenes disponibles para estos autores.",
                "context": contexto_chunks,
                "input": pregunta,
                "chat_history": historial
            }):
                print(chunk, end="", flush=True)
                respuesta_completa += chunk
        except Exception as e:
            print(f"\n  [Error de conexión con Ollama: {type(e).__name__}. Verificá que Ollama esté corriendo.]\n")
            continue

        t_total = time.time() - t_inicio
        print(f"\n  [Fuentes: {', '.join(fuentes)}]")
        print(f"  [Tiempos — búsqueda: {t_busqueda:.1f}s | respuesta: {t_total - t_busqueda:.1f}s | total: {t_total:.1f}s]\n")

        historial.append(HumanMessage(content=pregunta))
        historial.append(AIMessage(content=respuesta_completa))


if __name__ == "__main__":
    main()

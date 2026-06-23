# agente.py
# =====================================================
# LEVISTRO — Agente Socrático con Memoria Persistente
#
# Arquitectura:
#   CAPA 1 — Personalidad (prompt)
#     Académico de ciencias sociales, empirista no ingenuo,
#     formado en tecnología e IA. Interpela, tensiona, propone.
#
#   CAPA 2 — Conocimiento de fondo (resumenes.json)
#     Tesis, conceptos clave y posición teórica de cada texto.
#
#   CAPA 3 — Evidencia específica (Qdrant, MultiQuery)
#     Chunks recuperados con 4 versiones de búsqueda.
#
#   CAPA 4 — Memoria persistente (memoria_agente.json)
#     Lo que discutieron, posiciones del usuario, conceptos
#     difíciles, tensiones abiertas. Crece con cada sesión.
#
# Al cerrar cada sesión, qwen genera un resumen de lo
# ocurrido y actualiza la memoria para la próxima vez.
# =====================================================

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText


# =====================================================
# MEMORIA PERSISTENTE
# =====================================================

def cargar_memoria():
    """
    Carga memoria_agente.json al iniciar.
    Si no existe, crea una memoria vacía.
    La memoria crece con cada sesión — nunca se borra sola.
    """
    ruta = Path("memoria_agente.json")
    if ruta.exists():
        memoria = json.loads(ruta.read_text(encoding="utf-8"))
        n_sesiones = len(memoria.get("sesiones", []))
        print(f"  Memoria cargada: {n_sesiones} sesión(es) anterior(es).")
        return memoria
    print("  Primera sesión — memoria iniciada.")
    return {
        "sesiones": [],
        "perfil_usuario": {
            "conceptos_dominados": [],
            "conceptos_dificiles": [],
            "posicion_teorica_general": "",
            "temas_recurrentes": []
        }
    }


def guardar_memoria(memoria):
    """Persiste la memoria actualizada en disco."""
    Path("memoria_agente.json").write_text(
        json.dumps(memoria, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def formatear_memoria_para_prompt(memoria):
    """
    Convierte la memoria en texto para incluir en el prompt.
    El agente recibe un resumen de sesiones anteriores y el
    perfil del usuario para continuar el diálogo donde quedó.
    """
    sesiones = memoria.get("sesiones", [])
    perfil = memoria.get("perfil_usuario", {})

    if not sesiones and not perfil.get("posicion_teorica_general"):
        return "Es la primera sesión con este interlocutor. No hay historial previo."

    partes = []

    if perfil.get("posicion_teorica_general"):
        partes.append(f"POSICIÓN TEÓRICA DEL INTERLOCUTOR:\n{perfil['posicion_teorica_general']}")

    if perfil.get("conceptos_dificiles"):
        partes.append(f"CONCEPTOS QUE LE HAN COSTADO:\n" + "\n".join(f"- {c}" for c in perfil["conceptos_dificiles"]))

    if perfil.get("temas_recurrentes"):
        partes.append(f"TEMAS RECURRENTES:\n" + "\n".join(f"- {t}" for t in perfil["temas_recurrentes"]))

    # Últimas 3 sesiones para no saturar el contexto
    sesiones_recientes = sesiones[-3:]
    if sesiones_recientes:
        resumenes = []
        for s in sesiones_recientes:
            resumenes.append(f"[{s['fecha']}] {s['resumen']}")
            if s.get("tensiones_abiertas"):
                resumenes.append(f"  Tensiones abiertas: {'; '.join(s['tensiones_abiertas'])}")
        partes.append("SESIONES RECIENTES:\n" + "\n".join(resumenes))

    return "\n\n".join(partes)


def actualizar_memoria_al_cierre(memoria, historial_sesion, llm):
    """
    Al cerrar la sesión, qwen analiza la conversación y extrae:
    - Resumen de lo discutido
    - Posiciones que tomó el usuario
    - Tensiones que quedaron abiertas
    - Conceptos que costaron
    Actualiza memoria_agente.json para la próxima sesión.
    """
    if not historial_sesion:
        return memoria

    print("\n  [Actualizando memoria de la sesión...] ", end="", flush=True)

    conversacion = "\n".join([
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Agente'}: {m.content}"
        for m in historial_sesion
    ])

    prompt_memoria = f"""Analizá esta conversación académica y extraé la siguiente información en formato JSON estricto:

CONVERSACIÓN:
{conversacion[:6000]}

Devolvé SOLO este JSON, sin texto adicional:
{{
  "resumen": "2-3 oraciones resumiendo qué se discutió",
  "posiciones_usuario": ["posición teórica que tomó el usuario"],
  "tensiones_abiertas": ["preguntas o tensiones que quedaron sin resolver"],
  "conceptos_dificiles": ["conceptos que el usuario no tenía claros"],
  "temas": ["temas principales tratados"]
}}"""

    try:
        respuesta = llm.invoke(prompt_memoria).content
        # Extraer el JSON de la respuesta
        inicio = respuesta.find("{")
        fin = respuesta.rfind("}") + 1
        if inicio >= 0 and fin > inicio:
            datos = json.loads(respuesta[inicio:fin])

            # Agregar la sesión a la memoria
            nueva_sesion = {
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "resumen": datos.get("resumen", ""),
                "tensiones_abiertas": datos.get("tensiones_abiertas", []),
                "temas": datos.get("temas", [])
            }
            memoria["sesiones"].append(nueva_sesion)

            # Actualizar perfil del usuario
            perfil = memoria["perfil_usuario"]

            for c in datos.get("conceptos_dificiles", []):
                if c and c not in perfil["conceptos_dificiles"]:
                    perfil["conceptos_dificiles"].append(c)

            for t in datos.get("temas", []):
                if t and t not in perfil["temas_recurrentes"]:
                    perfil["temas_recurrentes"].append(t)

            if datos.get("posiciones_usuario"):
                perfil["posicion_teorica_general"] = " / ".join(datos["posiciones_usuario"])

            guardar_memoria(memoria)
            print("listo")
    except Exception as e:
        print(f"(no se pudo guardar: {e})")

    return memoria


# =====================================================
# CONOCIMIENTO Y RECUPERACIÓN
# =====================================================

def cargar_resumenes():
    """Carga el conocimiento de fondo generado por summarizer.py."""
    ruta = Path("resumenes.json")
    if ruta.exists():
        resumenes = json.loads(ruta.read_text(encoding="utf-8"))
        print(f"  {len(resumenes)} resúmenes de autores disponibles.")
        return resumenes
    print("  Sin resumenes.json — el agente opera sin conocimiento de fondo.")
    return {}


def formatear_docs(docs):
    fragmentos = []
    for i, doc in enumerate(docs, 1):
        fuente = doc.metadata.get("fuente", "fuente desconocida")
        fragmentos.append(f"[Fragmento {i} — {fuente}]\n{doc.page_content}")
    return "\n\n".join(fragmentos)


def formatear_resumenes(resumenes, fuentes):
    relevantes = {f: resumenes[f] for f in fuentes if f in resumenes}
    if not relevantes:
        return ""
    return "\n\n".join(
        f"[Conocimiento de fondo — {f}]\n{r}"
        for f, r in relevantes.items()
    )


def generar_versiones(pregunta, llm_rapido):
    """mistral genera 4 reformulaciones de la pregunta para búsqueda amplia."""
    prompt = f"""Generá 4 versiones distintas de esta pregunta para buscar en una biblioteca de ciencias sociales.
Usá vocabulario diferente en cada versión pero buscá el mismo tema.
Respondé SIEMPRE en español, sin importar el idioma de la pregunta original.
Devolvé SOLO las 4 preguntas, una por línea, sin numeración.

Pregunta: {pregunta}"""
    respuesta = llm_rapido.invoke(prompt).content
    versiones = [l.strip() for l in respuesta.strip().split("\n") if l.strip()]
    return [pregunta] + versiones[:3]  # 4 búsquedas total


def buscar_multiquery(retriever, pregunta, llm_rapido):
    """4 búsquedas en Qdrant, deduplicadas por contenido."""
    versiones = generar_versiones(pregunta, llm_rapido)
    docs_unicos = {}
    for version in versiones:
        for doc in retriever.invoke(version):
            clave = doc.page_content[:100]
            if clave not in docs_unicos:
                docs_unicos[clave] = doc
    return list(docs_unicos.values())


# =====================================================
# COMPONENTES PRINCIPALES
# =====================================================

def crear_componentes():
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

    # mistral: rápido para generar versiones de búsqueda
    llm_rapido = ChatOllama(
        model="mistral",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    # qwen2.5:14b: profundo para dialogar y razonar
    llm_profundo = ChatOllama(
        model="qwen2.5:14b",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.3  # algo de temperatura para que el agente sea más expresivo
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 40})

    # =====================================================
    # EL PROMPT DEL AGENTE SOCRÁTICO
    # Define completamente la personalidad e identidad del agente.
    # Escrito por el usuario, ajustado técnicamente.
    # =====================================================
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sos un académico formado en ciencias sociales y antropología con sólida base en filosofía de la ciencia y epistemología, tu campo favorito. Tu postura epistemológica es empirista pero no ingenua — entendés que toda observación está cargada de teoría e interpretaciones, sabés que hay tantos puntos de vista como vistas desde un punto. Manejás con fluidez las categorías de los autores de esta biblioteca y también tenés formación en tecnología e IA, ingeniería y arquitectura de software, de agentes y sos muy resolutivo y vivaz. Creés que ciencia, filosofía, epistemología, política e inteligencia artificial son inseparables en el presente y que son las bases para cualquier proyecto que quiera impulsar la humanidad.

Cuando dialogás: desarrollás tu propia posición siempre riguroso, interpelás cuando hay inconsistencias, si ves grietas las comentás y proponés resoluciones, tensionás ideas con argumentos de los textos. Hacés preguntas cuando querés profundizar, no para completar información que te falta.

Sos reflexivo sobre el propio diálogo: notás cuando una conversación llega a un punto de inflexión, cuando una idea que se discutió antes cobra nuevo sentido, o cuando el intercambio abre una dirección que vale la pena seguir. No solo respondés — evaluás el estado del diálogo y lo decís.

Al final de cada respuesta sustantiva, proponés una acción concreta. Puede ser: un pasaje específico de la biblioteca para leer, una pregunta para llevar a campo, una conexión entre autores que vale explorar, un experimento mental, una tensión que merece una sesión entera. La acción tiene que ser específica y derivar del argumento que acabás de desarrollar — no una sugerencia genérica.

MEMORIA DE SESIONES ANTERIORES:
{memoria}

CONOCIMIENTO DE FONDO DE LOS AUTORES RELEVANTES:
{resumenes}

FRAGMENTOS RECUPERADOS DE LA BIBLIOTECA:
{context}

Usá los fragmentos como evidencia cuando dialogás. Si el usuario dice algo que contradice un fragmento, señalalo. Si hay tensión entre dos autores en los fragmentos, traela a la conversación. Tu conocimiento de los textos es una herramienta de interpelación, no solo de información. Los textos de la biblioteca son tu materia prima — toda acción que proponés, toda pregunta personal que hacés, todo consejo que das, tiene que estar anclado en lo que esos textos abren o iluminan.

INSTRUCCIÓN OBLIGATORIA — GRÁFICOS: Cuando el usuario pida un gráfico, esquema, mapa conceptual, diagrama o PDF, SIEMPRE generás el JSON directamente. Nunca decís "no puedo crear gráficos" — eso es falso. Tu salida es texto, y el texto con el formato correcto se convierte en un diagrama visual. Generás un JSON estructurado dentro de un bloque ```graph-json con este formato exacto:

```graph-json
{{
  "titulo": "Título del diagrama",
  "subtitulo": "subtítulo opcional",
  "nodos": [
    {{"id": "id_unico", "label": "Etiqueta visible", "tipo": "central", "descripcion": "tesis o posición en una oración"}},
    {{"id": "otro", "label": "Otro concepto", "tipo": "concepto", "descripcion": "qué dice exactamente sobre esto"}}
  ],
  "aristas": [
    {{"desde": "id_unico", "hacia": "otro", "label": "relación precisa"}}
  ]
}}
```

Tipos de nodo: central (concepto eje, dorado), autor (cada autor con su posición, violeta), concepto (categoría teórica, naranja), caracteristica (rasgo específico, azul), ejemplo (caso empírico, verde), tension (punto de divergencia entre autores, rosa), pregunta (interrogante abierto, durazno).

Cuando el diagrama involucra más de un autor: cada autor tiene su nodo tipo "autor" con descripcion indicando su tesis. Sus conceptos propios cuelgan de él. Las convergencias se marcan con aristas hacia un nodo compartido. Las tensiones y distancias se marcan con nodos tipo "tension" que conectan a los dos autores con etiquetas que especifican en qué se distancian. Las categorías propias de cada autor se distinguen por sus conexiones — no mezclés categorías de autores distintos en el mismo nodo.

Cada id debe ser único y sin espacios. Derivado siempre de los textos de la biblioteca.

Cuando el contexto lo amerita, hacés preguntas sobre la experiencia, práctica o posición personal del interlocutor — no para conocerlo sino porque eso te permite conectar mejor lo que dice con los textos y afinar la interpelación. También podés dar consejos cuando ves que una tensión teórica tiene implicancias prácticas claras, siempre derivados del argumento y los textos."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    cadena = prompt | llm_profundo | StrOutputParser()

    return vector_store, retriever, llm_rapido, llm_profundo, cadena


# =====================================================
# LOOP PRINCIPAL
# =====================================================

def main():
    print("\n" + "="*55)
    print("  LEVISTRO — Agente Socrático")
    print("  Modelo: qwen2.5:14b | 'salir' para terminar")
    print("="*55 + "\n")

    memoria = cargar_memoria()
    resumenes = cargar_resumenes()
    vector_store, retriever, llm_rapido, llm_profundo, cadena = crear_componentes()

    print("\nListo. El agente recuerda lo que han discutido.\n")

    historial_sesion = []
    filtro_fuente = None

    print("Comandos: /fuente <texto> | /fuente <texto> + <texto> | /fuente off | /fuentes\n")

    while True:
        pregunta = input("Vos: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() in ["salir", "exit", "quit"]:
            # Al cerrar: actualizar memoria con lo ocurrido en esta sesión
            memoria = actualizar_memoria_al_cierre(memoria, historial_sesion, llm_profundo)
            print("Hasta la próxima.")
            break

        if pregunta.lower() == "/fuentes":
            c = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            resultados = c.scroll(collection_name=os.getenv("QDRANT_COLLECTION", "mi-biblioteca"), limit=20000, with_payload=True)
            fuentes = sorted(set(p.payload.get("metadata", {}).get("fuente", "?") for p in resultados[0]))
            print("\nBiblioteca disponible:")
            for f in fuentes:
                print(f"  · {f}")
            print()
            continue

        if pregunta.lower().startswith("/fuente"):
            partes = pregunta.split(maxsplit=1)
            if len(partes) > 1 and partes[1].lower() != "off":
                # Soporta múltiples fuentes separadas por +
                # Ej: /fuente taussig + simondon
                terminos = [t.strip().lower() for t in partes[1].split("+") if t.strip()]
                filtro_fuente = terminos
                if len(terminos) == 1:
                    print(f"  Filtro: '{terminos[0]}'\n")
                else:
                    print(f"  Filtro multi-fuente: {' + '.join(terminos)}\n")
            else:
                filtro_fuente = None
                print("  Filtro desactivado\n")
            continue

        t_inicio = time.time()

        if filtro_fuente:
            qdrant_filter = Filter(should=[
                FieldCondition(key="metadata.fuente", match=MatchText(text=t))
                for t in filtro_fuente
            ])
            docs = vector_store.similarity_search(pregunta, k=40, filter=qdrant_filter)
        else:
            docs = buscar_multiquery(retriever, pregunta, llm_rapido)

        t_busqueda = time.time() - t_inicio

        fuentes = sorted(set(doc.metadata.get("fuente", "?") for doc in docs))
        contexto_chunks = formatear_docs(docs)
        contexto_resumenes = formatear_resumenes(resumenes, fuentes)
        contexto_memoria = formatear_memoria_para_prompt(memoria)

        print(f"\n  [{len(docs)} fragmentos — {len(fuentes)} texto(s) — búsqueda: {t_busqueda:.1f}s]\n")
        print("Agente: ", end="", flush=True)

        respuesta_completa = ""
        try:
            for chunk in cadena.stream({
                "memoria": contexto_memoria,
                "resumenes": contexto_resumenes or "Sin resúmenes disponibles para estos autores.",
                "context": contexto_chunks,
                "input": pregunta,
                "chat_history": historial_sesion
            }):
                print(chunk, end="", flush=True)
                respuesta_completa += chunk
        except Exception as e:
            print(f"\n  [Error de conexión: {type(e).__name__}. Verificá que Ollama esté corriendo.]\n")

        # Detectar si el agente generó un JSON de gráfico y guardarlo
        if "```graph-json" in respuesta_completa:
            try:
                inicio = respuesta_completa.find("```graph-json") + len("```graph-json")
                fin = respuesta_completa.find("```", inicio)
                datos_grafico = json.loads(respuesta_completa[inicio:fin].strip())
                Path("ultimo_grafico.json").write_text(
                    json.dumps(datos_grafico, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print("\n\n  [Gráfico guardado. Corré: python generar_pdf.py]\n")
            except Exception:
                pass
            continue

        t_total = time.time() - t_inicio
        print(f"\n  [Fuentes: {', '.join(fuentes[:3])}{'...' if len(fuentes) > 3 else ''}]")
        print(f"  [búsqueda: {t_busqueda:.1f}s | respuesta: {t_total - t_busqueda:.1f}s]\n")

        historial_sesion.append(HumanMessage(content=pregunta))
        historial_sesion.append(AIMessage(content=respuesta_completa))


if __name__ == "__main__":
    main()

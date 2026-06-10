# Levistro — Agente de conocimiento académico

**Proyecto personal.** Lo que comenzó como un lector de PDFs con búsqueda semántica está evolucionando hacia un agente intelectual capaz de dialogar, conectar autores y pensar con una biblioteca completa de textos académicos en antropología, filosofía y ciencias sociales.

Todo corre localmente — sin APIs de pago, sin datos enviados a terceros (excepto LangSmith para monitoreo opcional).

---

## Fases del proyecto

**Fase 1 — Lector RAG** ✅ *completada*
Sistema de consulta sobre documentos: ingesta de PDFs (OCR incluido), vectorización, búsqueda semántica y respuesta fundamentada en fuentes.

**Fase 2 — Recuperación profunda + Modelo híbrido** ✅ *completada*
MultiQuery: el agente genera 4 versiones de cada pregunta y busca con cada una (hasta 160 chunks únicos vs 6 del RAG básico).

Modelo híbrido para reducir latencia: mistral 7B genera las versiones de búsqueda (tarea simple, modelo rápido) y qwen2.5 14B sintetiza la respuesta final (tarea compleja, modelo profundo). La búsqueda es 2x más rápida sin sacrificar calidad.

Resúmenes de fondo: summarizer.py genera un resumen estructurado de cada texto (tesis, conceptos clave, posición teórica) usando qwen2.5. El chatbot los carga al iniciar y los incluye como conocimiento de fondo junto con los chunks específicos recuperados.

**Fase 3 — Agente socrático con memoria** 📋 *planificada*
El agente lleva el diálogo, recuerda conversaciones anteriores, propone conexiones entre autores y usa el vocabulario conceptual de los textos. Deja de ser reactivo y se convierte en interlocutor.

**Fase 4 — GraphRAG** 📋 *planificada*
Construcción de un mapa de relaciones conceptuales entre todos los textos. El agente entiende que Bourdieu y Geertz abordan el mismo problema desde ángulos distintos sin que el usuario se lo indique.

---

## Distinción clave: Transformer vs RAG

**Transformer** es la arquitectura matemática interna de los modelos de IA (qwen, mistral, llama, GPT, Claude). Es el "cerebro" — ya viene entrenado con miles de millones de textos. Este proyecto no lo modifica.

**RAG (Retrieval Augmented Generation)** es lo que construye este proyecto: una estrategia para darle información externa a ese cerebro. Los chunks, los embeddings, Qdrant, el pipeline de ingesta — todo eso es RAG.

```
Transformer  →  el cerebro de una persona (ya existe, no se toca)
RAG          →  la biblioteca + el sistema para que el cerebro acceda a ella
```

Lo que hace este proyecto: convierte documentos en vectores (embeddings), los organiza en una base de datos vectorial (Qdrant), y construye un pipeline que recupera los fragmentos relevantes y se los pasa al modelo cuando necesita responder.

---

## Qué hace

1. Procesa PDFs (digitales y escaneados), TXT y DOCX
2. Extrae el texto, lo limpia y lo divide en fragmentos de ~800 caracteres
3. Convierte cada fragmento en un vector numérico (embedding) usando un modelo local
4. Guarda los vectores en una base de datos vectorial (Qdrant)
5. Cuando hacés una pregunta, busca los fragmentos más relevantes y genera una respuesta usando un modelo de lenguaje local

---

## Arquitectura

```
Tu pregunta
    → nomic-embed-text la convierte en vector
    → Qdrant busca los 4 fragmentos más cercanos semánticamente
    → llama3.2 lee esos fragmentos y genera la respuesta
    → LangSmith registra toda la traza (opcional)
```

### Componentes

| Componente | Rol | Dónde corre |
|---|---|---|
| `nomic-embed-text` | Convierte texto en vectores | Ollama (local) |
| `llama3.2` | Genera las respuestas | Ollama (local) |
| Qdrant | Base de datos vectorial | Docker (local) |
| LangSmith | Monitoreo y trazas | Nube (opcional) |

---

## Requisitos del sistema

- Python 3.10+
- Docker Desktop
- Ollama
- Tesseract OCR (para PDFs escaneados)
- Poppler (para convertir páginas PDF a imágenes)

### Instalaciones del sistema (no Python)

**Tesseract OCR**
- Windows: descargar de https://github.com/UB-Mannheim/tesseract/wiki
- Instalar en `C:\Program Files\Tesseract-OCR\`
- Durante la instalación, seleccionar idioma Spanish

**Poppler**
- Windows: descargar de https://github.com/oschwartz10612/poppler-windows/releases
- Extraer en la carpeta del proyecto en `poppler/`

**Docker Desktop**
- Descargar de https://www.docker.com/products/docker-desktop/

**Ollama**
- Descargar de https://ollama.com
- Luego descargar los modelos:
```
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install langchain langchain-community langchain-openai
pip install langsmith
pip install qdrant-client langchain-qdrant langchain-ollama
pip install pypdf pdfplumber
pip install pytesseract pillow pdf2image
pip install python-dotenv python-docx
pip install langchain-text-splitters
```

---

## Configuración

Crear un archivo `.env` en la raíz del proyecto:

```env
# LangSmith (monitoreo — opcional)
LANGCHAIN_API_KEY=tu_clave_aqui
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=levistro

# Qdrant
QDRANT_URL=http://localhost:6333

# Tesseract (ruta al ejecutable en Windows)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Poppler (ruta a los binarios en Windows)
POPPLER_PATH=C:\Users\tu_usuario\Desktop\levistro\poppler\poppler-24.08.0\Library\bin

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Levantar Qdrant con Docker

```bash
# Primera vez — con volumen para persistir los datos
docker run -d --name qdrant -p 6333:6333 \
  -v C:\Users\tu_usuario\Desktop\levistro\qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Las veces siguientes (después de reiniciar la PC)
docker start qdrant
```

El dashboard de Qdrant queda disponible en: http://localhost:6333/dashboard

---

## Pipeline de ingesta

El script `ingest.py` procesa todos los archivos en la carpeta `documentos/`:

### Tipos de archivos soportados
- **PDF digital**: texto extraído directamente con `pdfplumber`
- **PDF escaneado**: cada página se convierte a imagen con `pdf2image` + Poppler, luego Tesseract hace OCR
- **TXT**: lectura directa
- **DOCX**: extracción de párrafos con `python-docx`

### Detección automática PDF digital vs escaneado
El script abre las primeras 3 páginas con `pdfplumber`. Si extrae más de 50 caracteres, asume que es digital. Si no, aplica OCR.

### Chunking
`RecursiveCharacterTextSplitter` divide el texto en fragmentos de:
- **chunk_size**: 800 caracteres (~150 palabras)
- **chunk_overlap**: 150 caracteres de solapamiento entre fragmentos consecutivos

El solapamiento evita perder contexto en los bordes entre fragmentos.

### Embeddings
`nomic-embed-text` convierte cada chunk en un vector de **768 dimensiones**. Qdrant almacena estos vectores usando similitud coseno.

### Registro de archivos procesados
`procesados.json` guarda los nombres de archivos ya ingestados. Si la ingesta se interrumpe o agregás nuevos documentos, solo procesa lo pendiente sin duplicar.

```bash
# Correr la ingesta
python ingest.py
```

---

## Uso del chatbot

```bash
python chatbot.py
```

El chatbot busca los 4 fragmentos más relevantes en Qdrant para cada pregunta y los usa como contexto para que `llama3.2` genere la respuesta. Mantiene historial de conversación para preguntas de seguimiento.

Muestra la fuente (nombre del archivo) de donde provino la información.

---

## Estructura del proyecto

```
levistro/
├── .env                 # Configuración y claves (no subir a GitHub)
├── documentos/          # PDFs, TXTs y DOCXs a ingestar
├── qdrant_storage/      # Datos persistentes de Qdrant (no subir a GitHub)
├── poppler/             # Binarios de Poppler para Windows
├── ingest.py            # Pipeline de ingesta de documentos
├── chatbot.py           # Interfaz de conversación
├── procesados.json      # Registro de archivos ya ingestados
├── venv/                # Entorno virtual Python
└── README.md            # Este archivo
```

---

## Monitoreo con LangSmith

Con `LANGCHAIN_TRACING_V2=true` y la clave configurada, cada conversación queda registrada en https://smith.langchain.com bajo el proyecto `levistro`.

Cada traza muestra:
- La pregunta realizada
- Los fragmentos recuperados de Qdrant con sus scores de similitud
- El prompt completo enviado a llama3.2
- La respuesta generada
- Tiempo de cada paso

---

## Notas

- Los modelos de Ollama se guardan en `C:\Users\tu_usuario\.ollama\models\` — son del sistema, no del proyecto
- La primera respuesta del chatbot puede tardar 20-60 segundos (llama3.2 en CPU)
- Con GPU el tiempo de respuesta baja considerablemente
- Para agregar más documentos: copiar a `documentos/` y volver a correr `ingest.py`

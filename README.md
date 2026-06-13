# Levistro — Agente de conocimiento académico

**Proyecto personal.** Lo que comenzó como un lector de PDFs con búsqueda semántica está evolucionando hacia un agente intelectual capaz de dialogar, conectar autores y pensar con una biblioteca completa de textos académicos en antropología, filosofía y ciencias sociales.

Todo corre localmente — sin APIs de pago, sin datos enviados a terceros (excepto LangSmith para monitoreo opcional).

---

## Fases del proyecto

**Fase 1 — Lector RAG** 
Sistema de consulta sobre documentos: ingesta de PDFs (OCR incluido), vectorización, búsqueda semántica y respuesta fundamentada en fuentes.

**Fase 2 — Recuperación profunda + Modelo híbrido** 
MultiQuery: el agente genera 4 versiones de cada pregunta y busca con cada una (hasta 160 chunks únicos vs 6 del RAG básico).

Modelo híbrido para reducir latencia: mistral 7B genera las versiones de búsqueda (tarea simple, modelo rápido) y qwen2.5 14B sintetiza la respuesta final (tarea compleja, modelo profundo). La búsqueda es 2x más rápida sin sacrificar calidad.

Resúmenes de fondo: summarizer.py genera un resumen estructurado de cada texto (tesis, conceptos clave, posición teórica) usando qwen2.5. El chatbot los carga al iniciar y los incluye como conocimiento de fondo junto con los chunks específicos recuperados.

**Fase 3 — Agente socrático con memoria** ✅ *completada*
El agente lleva el diálogo, recuerda conversaciones anteriores, propone conexiones entre autores y usa el vocabulario conceptual de los textos. Deja de ser reactivo y se convierte en interlocutor. Implementado en `agente.py`.

**Fase 4 — GraphRAG** (en siguiente repsitorio)
Construcción de un mapa de relaciones conceptuales entre todos los textos. El agente entiende que Bourdieu y Geertz abordan el mismo problema desde ángulos distintos sin que el usuario se lo indique.

---

## Metodología de interpretación de textos

La fiabilidad epistemológica del agente depende de cómo se estructuran los resúmenes de cada texto. No es un resumen descriptivo — sigue el método comparativo clásico de la antropología, que no solo captura lo que dice un autor sino que decanta el punto de vista específico desde el que lo dice.

Cada texto de la biblioteca es resumido con esta estructura:

**TESIS CENTRAL** — la afirmación principal que el texto defiende.

**CONCEPTOS OPERATIVOS** — los términos técnicos que el autor construye o resignifica, no los que toma prestados. Lo que es específico de su aparato conceptual.

**ARGUMENTO** — cómo desarrolla y sostiene la tesis paso a paso.

**POSICIÓN EPISTEMOLÓGICA** — desde qué lugar teórico habla (empirismo, estructuralismo, fenomenología, etc.) y qué supuestos asume sobre el conocimiento. Esto es lo que permite al agente detectar inconsistencias cuando el usuario mezcla marcos incompatibles.

**PREGUNTAS QUE ABRE** — tensiones, problemas o interrogantes que el texto deja sin resolver. Son el punto de entrada para la interpelación.

**PUNTO DE VISTA DECANTADO** — qué puede ver este autor desde su posición que otros no pueden ver, y qué queda fuera de su mirada. Este campo es el núcleo del método comparativo: cada autor no solo dice algo, lo dice *desde un lugar* que habilita ciertas visibilidades y obtura otras.

**DIÁLOGO CON OTROS AUTORES** — con quién conversa explícita o implícitamente, dónde coincide y dónde diverge.

Esta estructura permite al agente no solo recuperar información sino posicionar autores en el campo de debates, detectar cuando dos autores abordan el mismo problema desde ángulos incompatibles, y tensionar lo que el usuario dice contra lo que los textos efectivamente sostienen.

---

## Distinción clave para entender el proyecto: Transformer vs RAG

**Transformer** es la arquitectura matemática interna de los modelos de IA (qwen, mistral, llama, GPT, Claude). Es el "cerebro" — ya viene entrenado con miles de millones de textos. Este proyecto no lo modifica.

**RAG (Retrieval Augmented Generation)** es lo que construye este proyecto: una estrategia para darle información externa a ese cerebro. Los chunks, los embeddings, Qdrant, el pipeline de ingesta — todo eso es RAG.

```
Transformer  →  el cerebro de una persona (ya existe, no se toca)
RAG          →  la biblioteca + el sistema para que el cerebro acceda a ella
```

Lo que hace este proyecto es conviertir documentos en vectores (embeddings), los organiza en una base de datos vectorial (Qdrant), y construye un pipeline que recupera los fragmentos relevantes y se los pasa al modelo cuando necesita responder.

---

## Qué puede hacer... 

1. Procesa PDFs (digitales y escaneados), TXT y DOCX
2. Extrae el texto, lo limpia y lo divide en fragmentos de ~800 caracteres
3. Convierte cada fragmento en un vector numérico (embedding) usando un modelo local
4. Guarda los vectores en una base de datos vectorial (Qdrant)
5. Cuando hacés una pregunta, busca los fragmentos más relevantes y genera una respuesta usando un modelo de lenguaje local

---

## Arquitectura

```
    → nomic-embed-text la convierte en vector
    → Qdrant busca los 4 fragmentos más cercanos semánticamente
    → llama3.2 lee esos fragmentos y genera la respuesta
    → LangSmith registra toda la traza (opcional)
```

### Componentes

| Componente | Rol | Dónde corre |
|---|---|---|
| `nomic-embed-text` | Convierte texto en vectores | Ollama (local) |
| `mistral` | Genera versiones de búsqueda (MultiQuery) | Ollama (local) |
| `qwen2.5:14b` | Genera las respuestas y dialoga | Ollama (local) |
| Qdrant | Base de datos vectorial | Docker (local) |
| LangSmith | Monitoreo y trazas | Nube (opcional) |

---

## Requisitos para correrlo

- Python 3.10+
- Docker Desktop
- Ollama
- Tesseract OCR (para PDFs escaneados)
- Poppler (para convertir páginas PDF a imágenes)

### Instalaciones del sistema dejo links! (no Python)

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
ollama pull mistral
ollama pull qwen2.5:14b
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

Usa MultiQuery (4 versiones de cada pregunta) para recuperar hasta 160 fragmentos únicos y los sintetiza con `qwen2.5:14b`. Mantiene historial de conversación para preguntas de seguimiento.

**`agente.py`** es la versión principal: agente socrático con memoria persistente entre sesiones. Escribe `salir` para cerrar — al hacerlo, el agente guarda un resumen de lo discutido para retomarlo en la próxima sesión.

### Comandos del agente

| Comando | Efecto |
|---|---|
| `/fuentes` | Lista todos los textos disponibles en la biblioteca |
| `/fuente taussig` | Filtra la búsqueda solo a textos que contengan "taussig" en el nombre |
| `/fuente taussig + simondon` | Filtra a múltiples textos simultáneamente (OR) |
| `/fuente off` | Desactiva el filtro, vuelve a buscar en toda la biblioteca |
| `salir` | Cierra la sesión y guarda la memoria |

**Consejo de uso:** `/fuente` activa el pipeline corto (una búsqueda directa, sin MultiQuery), por lo que es considerablemente más rápido. Usarlo cuando querés explorar un texto específico. Sin filtro, el agente recupera de toda la biblioteca y puede tensionar ideas entre autores.

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

- Esto es importante en investigación ya que en estas metricas se aloja una interpretacion implicita de comportamientoo

---

## Rendimiento y hardware

**Este es el punto más importante antes de usar el proyecto.**

El agente corre modelos de 7B y 14B de parámetros completamente en local. La velocidad depende casi enteramente del hardware disponible.

### Tiempos esperados por hardware

| Hardware | Tiempo por respuesta | Usabilidad |
|---|---|---|
| CPU puro (sin GPU) | 3-8 minutos | Inviable para conversación |
| iGPU integrada (Intel Arc, AMD Radeon integrada) | 35-60 segundos | Usable para consultas puntuales |
| GPU dedicada gama media (RTX 3060, RX 6700) | 8-15 segundos | Fluido |
| GPU dedicada gama alta (RTX 4070+, RX 7900) | 3-6 segundos | Muy fluido |
| Apple Silicon M2/M3 Pro o Max | 5-12 segundos | Muy fluido |

### Cómo activar la GPU

El código ya está preparado para usar GPU. Solo hay que configurar Ollama:

**NVIDIA (CUDA)** — Ollama lo detecta automáticamente. No requiere configuración adicional.

**AMD (ROCm)** — Ollama lo detecta automáticamente en Linux. En Windows puede requerir drivers actualizados.

**Intel Arc (Vulkan)** — Ollama detecta la GPU pero por defecto descarta las iGPUs. Para activarla:
```powershell
# Ejecutar en PowerShell como usuario normal (no admin)
[Environment]::SetEnvironmentVariable("OLLAMA_IGPU_ENABLE", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
# Reiniciar la PC para que Ollama tome los cambios al arrancar
```

Para verificar que la GPU está siendo usada, revisar el log de Ollama:
```
C:\Users\tu_usuario\AppData\Local\Ollama\server.log
```
Debe aparecer `msg="inference compute"` con el nombre de tu GPU, sin la palabra `dropping`.

### Por qué la velocidad es central

Este proyecto no es un buscador — es un interlocutor académico. La calidad del diálogo depende de respuestas que lleguen en segundos, no en minutos. Con hardware sin GPU dedicada, el proyecto funciona técnicamente pero la experiencia conversacional se fragmenta.

---

## Notas

- Los modelos de Ollama se guardan en `C:\Users\tu_usuario\.ollama\models\` — son del sistema, no del proyecto
- Para agregar más documentos: copiar a `documentos/` y volver a correr `ingest.py`
- `memoria_agente.json` y `resumenes.json` son personales y están excluidos del repositorio — cada usuario genera los suyos

# ingest.py
# =====================================================
# OBJETIVO: Tomar los PDFs de la carpeta "documentos/",
# extraer su texto, dividirlo en fragmentos y guardarlo
# en Qdrant como vectores para que el chatbot pueda buscarlo
# =====================================================

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# Cargamos las variables del archivo .env
# (la ruta a Tesseract, la URL de Qdrant, etc.)
load_dotenv()

# --- EXTRACCIÓN DE TEXTO ---
import pdfplumber                          # Lee texto de PDFs digitales
from pdf2image import convert_from_path   # Convierte páginas de PDF a imágenes (para OCR)
import pytesseract                         # Lee texto desde imágenes (OCR)
from PIL import Image
from docx import Document as DocxDocument  # Lee archivos Word (.docx)

# Le decimos a pytesseract exactamente dónde está Tesseract en Windows
# Sin esto, pytesseract no sabe dónde buscar el programa
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --- LANGCHAIN ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OllamaEmbeddings reemplaza a OpenAIEmbeddings — usa nuestro modelo local
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# --- QDRANT ---
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# =====================================================
# FUNCIÓN 1: Detectar si el PDF tiene texto real o es un escaneo
# =====================================================
def tiene_texto_digital(ruta_pdf: str) -> bool:
    """
    Abre las primeras páginas del PDF e intenta extraer texto.
    Si encuentra poco o nada, el PDF es una imagen (escaneado).
    Esto determina si usamos extracción directa u OCR.
    """
    with pdfplumber.open(ruta_pdf) as pdf:
        texto_total = ""
        for pagina in pdf.pages[:3]:  # Solo revisamos las primeras 3 páginas
            texto = pagina.extract_text()
            if texto:
                texto_total += texto

    # Si tiene más de 50 caracteres, consideramos que es digital
    return len(texto_total.strip()) > 50


# =====================================================
# FUNCIÓN 2: Extraer texto de un PDF digital
# =====================================================
def extraer_texto_digital(ruta_pdf: str) -> str:
    """
    Lee el texto directamente del PDF sin OCR.
    Aplica a PDFs creados por computadora (Word, Google Docs, etc.)
    """
    texto_completo = []

    with pdfplumber.open(ruta_pdf) as pdf:
        print(f"  PDF digital: {len(pdf.pages)} páginas")
        for num_pagina, pagina in enumerate(pdf.pages, 1):
            texto = pagina.extract_text()
            if texto:
                texto_completo.append(texto)
                print(f"  Página {num_pagina}: {len(texto)} caracteres extraídos")

    return "\n\n".join(texto_completo)


# =====================================================
# FUNCIÓN 3: Extraer texto de un PDF escaneado usando OCR
# =====================================================
def extraer_texto_ocr(ruta_pdf: str) -> str:
    """
    Convierte cada página del PDF en una imagen y le aplica OCR.
    Aplica a PDFs que son fotografías de documentos físicos.
    300 DPI da suficiente resolución para que Tesseract lea bien.
    """
    texto_completo = []

    print(f"  PDF escaneado — aplicando OCR con Tesseract...")

    # Convertimos cada página del PDF a una imagen de alta resolución
    # poppler_path apunta a los binarios que pdf2image necesita en Windows
    poppler_path = os.getenv("POPPLER_PATH")
    imagenes = convert_from_path(ruta_pdf, dpi=300, poppler_path=poppler_path)
    print(f"  {len(imagenes)} páginas a procesar")

    for num_pagina, imagen in enumerate(imagenes, 1):
        # pytesseract lee la imagen y devuelve el texto
        # spa+eng permite reconocer español e inglés mezclados
        texto = pytesseract.image_to_string(imagen, lang='spa+eng')
        if texto.strip():
            texto_completo.append(texto)
            print(f"  Página {num_pagina}: {len(texto)} caracteres (OCR)")

    return "\n\n".join(texto_completo)


# =====================================================
# FUNCIÓN 4: Limpiar el texto extraído
# =====================================================
def limpiar_texto(texto: str) -> str:
    """
    Elimina el "ruido" que introducen la extracción y el OCR:
    caracteres raros, espacios dobles, saltos de línea excesivos.
    Un texto más limpio produce mejores vectores y mejores respuestas.
    """
    # Eliminar caracteres no imprimibles (basura típica del OCR)
    texto = re.sub(r'[^\x20-\x7E\xA0-\xFF\n]', ' ', texto)

    # Reemplazar múltiples espacios consecutivos por uno solo
    texto = re.sub(r' {2,}', ' ', texto)

    # Reducir más de 2 saltos de línea a exactamente 2
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # Quitar líneas muy cortas que suelen ser basura del OCR (headers, números de página)
    lineas = texto.split('\n')
    lineas_limpias = [l for l in lineas if len(l.strip()) > 2 or l.strip() == '']
    texto = '\n'.join(lineas_limpias)

    return texto.strip()


# =====================================================
# FUNCIÓN 5: Procesar un PDF completo
# =====================================================
def procesar_pdf(ruta_pdf: str) -> str:
    """
    Función orquestadora: detecta el tipo de PDF y
    usa la función de extracción correcta automáticamente.
    """
    print(f"\nProcesando: {ruta_pdf}")

    if tiene_texto_digital(ruta_pdf):
        texto = extraer_texto_digital(ruta_pdf)
    else:
        texto = extraer_texto_ocr(ruta_pdf)

    texto_limpio = limpiar_texto(texto)
    print(f"  Texto final: {len(texto_limpio)} caracteres")

    return texto_limpio


# =====================================================
# FUNCIÓN 6: Dividir el texto en chunks (fragmentos)
# =====================================================
def dividir_en_chunks(texto: str, nombre_archivo: str) -> list:
    """
    Divide el texto en trozos pequeños con solapamiento.

    Por qué dividimos:
    - El modelo no puede procesar documentos enteros de una vez
    - Los chunks pequeños permiten búsquedas más precisas en Qdrant

    chunk_size=800: cada fragmento tiene ~800 caracteres (~150 palabras)
    chunk_overlap=150: los fragmentos comparten 150 caracteres con el anterior
                       para no perder el contexto entre dos fragmentos
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.create_documents(
        texts=[texto],
        metadatas=[{"fuente": nombre_archivo}]  # guardamos qué PDF originó cada chunk
    )

    print(f"  Dividido en {len(chunks)} fragmentos")
    return chunks


# =====================================================
# FUNCIÓN PRINCIPAL: Ejecuta el pipeline completo
# =====================================================
def main():
    # PASO 1: Conectar a Qdrant (base de datos vectorial corriendo en Docker)
    print("\nConectando a Qdrant...")
    cliente_qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    nombre_coleccion = os.getenv("QDRANT_COLLECTION", "mi-biblioteca")

    # Crear la colección solo si no existe todavía
    # Una colección es como una "tabla" en Qdrant donde guardamos los vectores
    colecciones_existentes = [c.name for c in cliente_qdrant.get_collections().collections]
    if nombre_coleccion not in colecciones_existentes:
        cliente_qdrant.create_collection(
            collection_name=nombre_coleccion,
            vectors_config=VectorParams(
                size=768,             # nomic-embed-text produce vectores de 768 dimensiones
                distance=Distance.COSINE  # medimos similitud por ángulo entre vectores
            )
        )
        print(f"  Colección '{nombre_coleccion}' creada en Qdrant")
    else:
        print(f"  Colección '{nombre_coleccion}' ya existe — agregando documentos")

    # PASO 2: Configurar el modelo de embeddings local (Ollama)
    # nomic-embed-text convierte cada chunk de texto en un vector de 768 números
    print("\nCargando modelo de embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # PASO 3: Cargar el registro de archivos ya procesados
    # procesados.json guarda los nombres de archivos ya ingestados
    # así si se interrumpe o agregás nuevos, no se duplica nada
    registro_path = Path("procesados.json")
    if registro_path.exists():
        procesados = json.loads(registro_path.read_text())
    else:
        procesados = []

    # PASO 4: Buscar todos los archivos soportados en "documentos/"
    carpeta = Path("documentos")
    pdfs = list(carpeta.glob("*.pdf"))
    txts = list(carpeta.glob("*.txt"))
    docxs = list(carpeta.glob("*.docx"))
    archivos_total = pdfs + txts + docxs

    if not archivos_total:
        print(f"\nNo encontré archivos en '{carpeta}/'")
        print("Coloca PDFs o TXTs en esa carpeta y volvé a ejecutar.")
        return

    # Filtrar los que ya fueron procesados
    pendientes = [f for f in archivos_total if f.name not in procesados]

    if not pendientes:
        print("\nTodos los archivos ya fueron procesados. Nada nuevo que ingestar.")
        return

    print(f"\nTotal encontrados: {len(pdfs)} PDF(s), {len(txts)} TXT(s), {len(docxs)} DOCX(s) — Pendientes: {len(pendientes)}")

    # PASO 5: Conectar el vector store una vez, reutilizar para cada archivo
    vector_store = QdrantVectorStore(
        client=cliente_qdrant,
        collection_name=nombre_coleccion,
        embedding=embeddings,
    )

    total_chunks = 0

    for archivo in pendientes:
        # Extraer texto según el tipo de archivo
        if archivo.suffix.lower() == ".pdf":
            texto = procesar_pdf(str(archivo))
        elif archivo.suffix.lower() == ".txt":
            # TXT no necesita OCR — se lee directo
            print(f"\nProcesando: {archivo}")
            texto = archivo.read_text(encoding="utf-8", errors="ignore")
            texto = limpiar_texto(texto)
            print(f"  Texto: {len(texto)} caracteres")
        elif archivo.suffix.lower() == ".docx":
            # DOCX: extrae texto de cada párrafo del documento Word
            print(f"\nProcesando: {archivo}")
            doc = DocxDocument(str(archivo))
            parrafos = [p.text for p in doc.paragraphs if p.text.strip()]
            texto = "\n\n".join(parrafos)
            texto = limpiar_texto(texto)
            print(f"  Texto: {len(texto)} caracteres ({len(parrafos)} párrafos)")
        else:
            continue

        # Dividir en chunks y guardar en Qdrant inmediatamente
        # Así cada archivo queda disponible para el chatbot al instante
        # y si la ingesta se interrumpe, los archivos ya guardados no se pierden
        chunks = dividir_en_chunks(texto, archivo.name)
        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"  Guardado en Qdrant ({len(chunks)} fragmentos)")

        # Registrar como procesado después de guardarlo exitosamente
        procesados.append(archivo.name)
        registro_path.write_text(json.dumps(procesados, ensure_ascii=False, indent=2))

    print(f"\nIngesta completada. {total_chunks} fragmentos nuevos listos para el chatbot.")


if __name__ == "__main__":
    main()

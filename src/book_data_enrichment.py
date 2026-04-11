"""
Script de Enriquecimiento del Catálogo de Libros
=================================================
Enriquece y consolida la base de datos de libros mediante fases escalonadas:
  Fase 1: Detecta/corrige el idioma usando la librería lingua (offline/llm fallback).
  Fase 2: Rellena descripciones faltantes usando Google Books API + LM Studio local.
  Fase 2.5: Convierte las descripciones guardadas en texto plano a formato HTML rico.
  Fase 3: Normaliza autores y rellena NULLs tratando separadores estrictamente estructurados.

Uso:
  python -m src.book_data_enrichment                           # Ejecutar todas las fases secuenciales
  python -m src.book_data_enrichment --phase language           # Solo detección de idiomas
  python -m src.book_data_enrichment --phase description        # Solo generación de descripciones
  python -m src.book_data_enrichment --phase format_html        # Solo conversión a HTML
  python -m src.book_data_enrichment --phase author             # Solo normalización de autores
  python -m src.book_data_enrichment --dry-run                  # Preview de cambios sin escribir BD
  python -m src.book_data_enrichment --limit 20 --dry-run       # Preview rapida con primeros 20 libros
"""

import argparse
import json
import re
import sys
import time

# Fix para consolas de Windows que no soportan emoji/Unicode por defecto
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from lingua import Language, LanguageDetectorBuilder
from sqlalchemy import create_engine, text
from tqdm import tqdm

from config.settings import (
    DB_CONFIG,
    GOOGLE_BOOKS_API_KEY,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
)

# =====================================================
# Conexión a BD (mismo patrón que book_recommender.py)
# =====================================================
DB_URL = (
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)
engine = create_engine(DB_URL)

# =====================================================
# Configuración de lingua
# =====================================================
# Solo los idiomas relevantes para el catálogo
# Esto mejora la precisión y reduce memoria (~20MB vs ~100MB con todos)
IDIOMAS_SOPORTADOS = [Language.SPANISH, Language.ENGLISH, Language.PORTUGUESE, Language.FRENCH]

detector = LanguageDetectorBuilder.from_languages(*IDIOMAS_SOPORTADOS) \
    .with_minimum_relative_distance(0.25) \
    .build()

# Mapeo de lingua Language enum → nombre legible para la BD
LINGUA_A_BD = {
    Language.SPANISH: "Spanish",
    Language.ENGLISH: "English",
    Language.PORTUGUESE: "Portuguese",
    Language.FRENCH: "French",
}

# Idiomas que se consideran "incorrectos" o genéricos y deben re-evaluarse
IDIOMAS_INVALIDOS = {"", "Desconocido", "Unknown", "N/A", "null", "None"}

# =====================================================
# Mapeo de idioma → instrucciones de idioma para el LLM
# =====================================================
IDIOMA_INSTRUCCION = {
    "Spanish": "en español",
    "English": "in English",
    "Portuguese": "em português",
    "French": "en français",
}


# =====================================================
# Utilidades
# =====================================================
def limpiar_html(texto):
    """Elimina etiquetas HTML de un texto."""
    return re.sub(r'<[^>]+>', ' ', str(texto)).strip()


def convertir_a_html_legible(texto):
    """Convierte texto plano en un formato HTML legible con párrafos y primera oración en negrita.
    Si ya contiene HTML (e.g., <p>), devuelve el texto intacto.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""
        
    # Validar si ya es HTML (contiene etiquetas estructurales)
    if '<p' in texto.lower() or '<b' in texto.lower() or '<strong' in texto.lower():
        return texto.strip()

    texto = re.sub(r'\s+', ' ', texto.strip())
    oraciones = re.split(r'(?<=[.!?]) +', texto)
    oraciones = [o.strip() for o in oraciones if o.strip()]
    if not oraciones:
        return ""
    if len(oraciones) == 1:
        return f"<p>{oraciones[0]}</p>"
        
    html_final = f"<p><strong>{oraciones[0]}</strong></p>\n"
    parrafo_actual = []
    for oracion in oraciones[1:]:
        parrafo_actual.append(oracion)
        if len(parrafo_actual) == 3:
            html_final += f"<p>{' '.join(parrafo_actual)}</p>\n"
            parrafo_actual = []
    if parrafo_actual:
        html_final += f"<p>{' '.join(parrafo_actual)}</p>"
    return html_final


def es_idioma_valido(idioma):
    """Retorna True si el idioma es un valor válido y no necesita corrección."""
    if not idioma:
        return False
    idioma_str = str(idioma).strip()
    return idioma_str not in IDIOMAS_INVALIDOS


# =====================================================
# FASE 1: Detección de idioma (lingua + LM Studio fallback)
# =====================================================

# Valores aceptados para el campo language en la BD
IDIOMAS_ACEPTADOS = {"Spanish", "English", "Portuguese", "French"}


def detectar_idioma(titulo, autor=""):
    """Detecta el idioma de un libro usando lingua (offline).
    
    Retorna 'Spanish', 'English', etc. o None si no se puede detectar.
    """
    texto = f"{titulo} {autor}".strip()
    if not texto or len(texto) < 2:
        return None

    resultado = detector.detect_language_of(texto)
    if resultado and resultado in LINGUA_A_BD:
        return LINGUA_A_BD[resultado]
    return None


def extraer_respuesta_lm(data):
    """Extrae el texto de respuesta del JSON de LM Studio.
    
    Modelos con thinking mode (Qwen 3.5, DeepSeek R1) pueden devolver:
    - content: vacío, reasoning_content: con la respuesta
    - content: '<think>...</think>\nrespuesta'
    - content: 'respuesta' (modelos normales como Qwen 2.5)
    """
    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()

    # Si content tiene bloques <think>, eliminarlos
    content_limpio = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Prioridad: content limpio > reasoning_content > content raw
    if content_limpio:
        return content_limpio
    if reasoning:
        return reasoning
    return content


def detectar_idioma_lm_studio(titulo, autor=""):
    """Fallback: detecta el idioma de un libro usando LM Studio (para títulos
    ambiguos, bilingües o demasiado cortos que lingua no pudo resolver).
    
    Retorna 'Spanish', 'English', etc. o None si falla.
    """
    system_msg = (
        "You are a language detection assistant. You determine the PRIMARY language "
        "of book titles. Respond with ONLY ONE of these exact words: "
        "Spanish, English, Portuguese, French. "
        "Do NOT explain your reasoning. Output ONLY the language name."
    )
    user_msg = (
        f"What is the primary language of this book?\n"
        f"Title: {titulo}\n"
        f"Author: {autor or 'Unknown'}"
    )

    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 150,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        respuesta = extraer_respuesta_lm(data).strip(".")

        # Validar contra whitelist — match exacto primero
        for idioma in IDIOMAS_ACEPTADOS:
            if idioma.lower() == respuesta.lower().strip():
                return idioma

        # Fallback: match parcial
        for idioma in IDIOMAS_ACEPTADOS:
            if idioma.lower() in respuesta.lower():
                return idioma

        # Debug: mostrar qué respondió el modelo cuando no se pudo parsear
        tqdm.write(f"   🔍 Debug: respuesta='{respuesta[:80]}'")

    except requests.RequestException as e:
        tqdm.write(f"   ❌ Error LM Studio API: {e}")
    except (KeyError, IndexError) as e:
        tqdm.write(f"   ❌ Respuesta inesperada: {e}")

    return None


def fase_idioma(dry_run=False, limit=None):
    """Fase 1: Detecta y corrige el idioma de libros con language NULL/inválido.
    
    Nivel 1: lingua (offline, instantáneo) — cubre ~89% de los títulos.
    Nivel 2: LM Studio (modelo local) — para títulos bilingües, cortos o ambiguos.
    """
    print("\n" + "=" * 60)
    print("📖 FASE 1: Detección de idioma")
    print("=" * 60)

    # Consultar libros que necesitan idioma
    query = """
        SELECT id, sku, title, author, language
        FROM books
        WHERE language IS NULL OR TRIM(language) = ''
           OR language IN ('Desconocido', 'Unknown', 'N/A', 'null', 'None')
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        print("   ✅ No hay libros con idioma faltante/inválido.")
        return {"total": 0, "lingua": 0, "lm_studio": 0, "no_detectados": 0}

    print(f"   → {len(rows)} libros necesitan detección de idioma")

    # Verificar disponibilidad de LM Studio para fallback
    lm_studio_disponible = False
    try:
        resp = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            modelos = [m["id"] for m in resp.json().get("data", [])]
            lm_studio_disponible = len(modelos) > 0
            if lm_studio_disponible:
                print(f"   ✅ LM Studio disponible como fallback para títulos ambiguos")
            else:
                print(f"   ⚠️ LM Studio activo pero sin modelos cargados.")
    except requests.RequestException:
        print("   ℹ️ LM Studio no disponible. Solo se usará lingua (offline).")

    count_lingua = 0
    count_lm_studio = 0
    no_detectados = 0
    pendientes_lm = []  # Libros que lingua no pudo detectar
    cambios = []

    # --- Paso 1: lingua (offline, instantáneo) ---
    print("\n   Nivel 1: lingua (offline)...")
    for row in tqdm(rows, desc="lingua"):
        book_id, sku, titulo, autor, idioma_actual = row
        idioma_detectado = detectar_idioma(titulo or "", autor or "")

        if idioma_detectado:
            cambios.append({
                "id": book_id, "sku": sku, "titulo": titulo,
                "idioma_anterior": idioma_actual or "NULL",
                "idioma_nuevo": idioma_detectado, "fuente": "lingua",
            })
            count_lingua += 1
        else:
            pendientes_lm.append(row)

    print(f"   → lingua detectó {count_lingua}/{len(rows)} idiomas")

    # --- Paso 2: LM Studio fallback (para los que lingua no pudo) ---
    if pendientes_lm and lm_studio_disponible:
        print(f"\n   Nivel 2: LM Studio fallback ({len(pendientes_lm)} pendientes)...")
        for row in tqdm(pendientes_lm, desc="LM Studio"):
            book_id, sku, titulo, autor, idioma_actual = row
            idioma_detectado = detectar_idioma_lm_studio(titulo or "", autor or "")

            if idioma_detectado:
                cambios.append({
                    "id": book_id, "sku": sku, "titulo": titulo,
                    "idioma_anterior": idioma_actual or "NULL",
                    "idioma_nuevo": idioma_detectado, "fuente": "LM Studio",
                })
                count_lm_studio += 1
            else:
                no_detectados += 1
                tqdm.write(f"   ⚠️ No detectado: [{sku}] {titulo}")
    else:
        no_detectados = len(pendientes_lm)
        if pendientes_lm and not lm_studio_disponible:
            print(f"\n   ⚠️ {len(pendientes_lm)} libros sin detectar (LM Studio no disponible para fallback)")
            for row in pendientes_lm[:10]:
                print(f"      [{row[1]}] {row[2]}")
            if len(pendientes_lm) > 10:
                print(f"      ... y {len(pendientes_lm) - 10} más")

    # Aplicar cambios a BD (o solo mostrar en dry-run)
    if cambios:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se actualizarían {len(cambios)} libros:")
            for c in cambios[:15]:
                print(f"   [{c['sku']}] \"{c['titulo'][:50]}\" → {c['idioma_nuevo']} ({c['fuente']})")
            if len(cambios) > 15:
                print(f"   ... y {len(cambios) - 15} más")
        else:
            print(f"\n💾 Guardando {len(cambios)} idiomas en la BD...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET language = :lang WHERE id = :id")
                for c in cambios:
                    conn.execute(sql, {"lang": c["idioma_nuevo"], "id": c["id"]})
            print("   ✅ Idiomas actualizados correctamente.")

    stats = {
        "total": len(rows), "lingua": count_lingua,
        "lm_studio": count_lm_studio, "no_detectados": no_detectados,
    }
    total_ok = count_lingua + count_lm_studio
    print(f"\n📊 Fase 1: {count_lingua} lingua + {count_lm_studio} LM Studio = {total_ok} detectados, {no_detectados} sin detectar")
    return stats


# =====================================================
# FASE 2: Enriquecimiento de descripciones
# =====================================================

# --- Nivel 1: Google Books API ---
def buscar_google_books(isbn, titulo=""):
    """Busca la descripción de un libro en Google Books API por ISBN.
    
    Retorna la descripción (str) o None si no se encuentra.
    """
    if not isbn or str(isbn).strip() == "":
        return None

    isbn_limpio = re.sub(r'[^0-9X]', '', str(isbn).upper())
    if not isbn_limpio:
        return None

    try:
        params = {"q": f"isbn:{isbn_limpio}", "maxResults": 1}
        if GOOGLE_BOOKS_API_KEY:
            params["key"] = GOOGLE_BOOKS_API_KEY

        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("totalItems", 0) > 0:
            volume_info = data["items"][0].get("volumeInfo", {})
            descripcion = volume_info.get("description", "")
            if descripcion and len(descripcion.strip()) > 20:
                return limpiar_html(descripcion.strip())
    except requests.RequestException:
        pass  # Silenciar errores de red, se intenta con el siguiente nivel

    # Fallback: buscar por título si ISBN no dio resultado
    if titulo and len(titulo) > 3:
        try:
            params = {"q": f"intitle:{titulo}", "maxResults": 1}
            if GOOGLE_BOOKS_API_KEY:
                params["key"] = GOOGLE_BOOKS_API_KEY

            resp = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("totalItems", 0) > 0:
                volume_info = data["items"][0].get("volumeInfo", {})
                descripcion = volume_info.get("description", "")
                if descripcion and len(descripcion.strip()) > 20:
                    return limpiar_html(descripcion.strip())
        except requests.RequestException:
            pass

    return None


# --- Nivel 2: LM Studio (modelo local, API OpenAI-compatible) ---
def generar_descripcion_lm_studio(titulo, autor, categoria, idioma):
    """Genera una descripción de libro usando LM Studio (Qwen 3.5 9B).
    
    LM Studio expone una API compatible con OpenAI en /v1/chat/completions.
    La descripción se genera en el idioma original del libro.
    Retorna la descripción (str) o None si falla.
    """
    instruccion_idioma = IDIOMA_INSTRUCCION.get(idioma, "en español")

    system_msg = (
        f"Eres un redactor de descripciones de libros. Escribe descripciones comerciales "
        f"breves y atractivas {instruccion_idioma}. Responde SOLO con la descripción, "
        f"sin explicaciones, sin comillas, sin prefijos."
    )
    user_msg = (
        f"Genera una descripción comercial breve (2-3 oraciones, máximo 80 palabras) para:\n"
        f"Título: {titulo}\n"
        f"Autor: {autor or 'Desconocido'}\n"
        f"Categoría: {categoria or 'General'}\n"
        f"NO incluyas el título ni el autor en la descripción."
    )

    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        respuesta = extraer_respuesta_lm(data)

        # Limpiar posibles artefactos del modelo
        respuesta = re.sub(r'^["\']+|["\']+$', '', respuesta)
        respuesta = re.sub(r'^(Descripción|Description|Respuesta):\s*', '', respuesta, flags=re.IGNORECASE)
        respuesta = respuesta.strip()

        if respuesta and len(respuesta) > 20:
            return respuesta
    except requests.RequestException as e:
        tqdm.write(f"   ❌ Error LM Studio: {e}")
    except (KeyError, IndexError) as e:
        tqdm.write(f"   ❌ Respuesta inesperada de LM Studio: {e}")

    return None


def fase_descripcion(dry_run=False, limit=None):
    """Fase 2: Rellena descripciones faltantes con Google Books API + LM Studio."""
    print("\n" + "=" * 60)
    print("📝 FASE 2: Enriquecimiento de descripciones")
    print("=" * 60)

    # Consultar libros sin descripción
    query = """
        SELECT id, sku, title, author, isbn, category, language
        FROM books
        WHERE description IS NULL OR TRIM(description) = ''
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        print("   ✅ Todos los libros ya tienen descripción.")
        return {"total": 0, "google_books": 0, "lm_studio": 0, "sin_descripcion": 0}

    print(f"   → {len(rows)} libros sin descripción")

    # Verificar disponibilidad de LM Studio
    lm_studio_disponible = False
    try:
        resp = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            modelos = [m["id"] for m in resp.json().get("data", [])]
            lm_studio_disponible = len(modelos) > 0
            if lm_studio_disponible:
                print(f"   ✅ LM Studio disponible. Modelos: {', '.join(modelos[:3])}")
            else:
                print(f"   ⚠️ LM Studio activo pero sin modelos cargados.")
    except requests.RequestException:
        print("   ⚠️ LM Studio no disponible. Solo se usará Google Books API.")

    count_google = 0
    count_lm_studio = 0
    count_sin = 0
    cambios = []

    for row in tqdm(rows, desc="Buscando descripciones"):
        book_id, sku, titulo, autor, isbn, categoria, idioma = row

        descripcion = None
        fuente = None

        # Nivel 1: Google Books API
        descripcion = buscar_google_books(isbn, titulo)
        if descripcion:
            fuente = "Google Books"
            count_google += 1
        elif lm_studio_disponible:
            # Nivel 2: LM Studio local (Qwen 3.5 9B)
            idioma_libro = idioma if es_idioma_valido(idioma) else "Spanish"
            descripcion = generar_descripcion_lm_studio(titulo, autor, categoria, idioma_libro)
            if descripcion:
                fuente = "LM Studio"
                count_lm_studio += 1

        if descripcion:
            # Aplicamos formato HTML antes de guardar si la fase formatea automático
            descripcion_html = convertir_a_html_legible(descripcion)
            cambios.append({
                "id": book_id,
                "sku": sku,
                "titulo": titulo,
                "fuente": fuente,
                "descripcion": descripcion[:200] + "..." if len(descripcion) > 200 else descripcion,
                "descripcion_completa": descripcion_html,
            })
        else:
            count_sin += 1

        # Rate limiting para Google Books API (~1 req/seg para no exceder cuota)
        time.sleep(0.3)

    # Aplicar cambios
    if cambios:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se actualizarían {len(cambios)} descripciones:")
            for c in cambios[:10]:
                print(f"   [{c['sku']}] ({c['fuente']}) \"{c['titulo'][:40]}\"")
                print(f"      → {c['descripcion'][:100]}...")
            if len(cambios) > 10:
                print(f"   ... y {len(cambios) - 10} más")
        else:
            print(f"\n💾 Guardando {len(cambios)} descripciones en la BD...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET description = :desc WHERE id = :id")
                lote = []
                for i, c in enumerate(cambios):
                    lote.append({"desc": c["descripcion_completa"], "id": c["id"]})
                    if (i + 1) % 50 == 0:
                        conn.execute(sql, lote)
                        lote.clear()
                if lote:
                    conn.execute(sql, lote)
            print("   ✅ Descripciones actualizadas correctamente.")

    stats = {
        "total": len(rows),
        "google_books": count_google,
        "lm_studio": count_lm_studio,
        "sin_descripcion": count_sin,
    }
    print(f"\n📊 Fase 2: {count_google} Google Books + {count_lm_studio} LM Studio = {count_google + count_lm_studio} enriquecidos")
    if count_sin > 0:
        print(f"   ⚠️ {count_sin} libros quedaron sin descripción")
    return stats


# =====================================================
# FASE 2.5: Formato HTML
# =====================================================
def fase_formato_html(dry_run=False, limit=None):
    """Fase intermidia: Convierte a HTML todas las descripciones en BD que sean texto plano."""
    print("\n" + "=" * 60)
    print("🎨 FASE 2.5: Conversión a HTML de descripciones")
    print("=" * 60)

    # Consultar libros con descripción pero sin formato HTML
    query = """
        SELECT id, sku, title, description
        FROM books
        WHERE description IS NOT NULL 
          AND TRIM(description) != '' 
          AND description NOT LIKE '%<p>%'
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        print("   ✅ Todas las descripciones ya tienen formato HTML o no hay descripciones para procesar.")
        return {"total": 0, "formateados": 0}

    print(f"   → {len(rows)} descripciones necesitan formato HTML")

    cambios = []
    for row in tqdm(rows, desc="Convirtiendo a HTML"):
        book_id, sku, titulo, desc_actual = row
        desc_html = convertir_a_html_legible(desc_actual)
        if desc_html and desc_html != desc_actual:
            cambios.append({"id": book_id, "sku": sku, "titulo": titulo, "html": desc_html})

    if cambios:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se formatearían {len(cambios)} descripciones:")
            for c in cambios[:10]:
                print(f"   [{c['sku']}] \"{c['titulo'][:40]}\"")
                print(f"      → {c['html'][:80]}...")
            if len(cambios) > 10:
                print(f"   ... y {len(cambios) - 10} más")
        else:
            print(f"\n💾 Guardando {len(cambios)} descripciones en formato HTML...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET description = :html WHERE id = :id")
                lote = []
                for i, c in enumerate(cambios):
                    lote.append({"html": c["html"], "id": c["id"]})
                    if (i + 1) % 50 == 0:
                        conn.execute(sql, lote)
                        lote.clear()
                if lote:
                    conn.execute(sql, lote)
            print("   ✅ Descripciones formateadas correctamente.")
            
    stats = {"total": len(rows), "formateados": len(cambios)}
    print(f"\n📊 Fase 2.5: {len(cambios)}/{len(rows)} descripciones convertidas a HTML")
    return stats


# =====================================================
# FASE 3: Enriquecimiento de autores
# =====================================================

# Sufijos que deben ir al final del nombre (no confundir con apellidos)
SUFIJOS_NOMBRE = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

# Credenciales/títulos académicos a eliminar del nombre del autor
CREDENCIALES = {
    "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
    "phd", "ph.d", "ph.d.", "md", "m.d", "m.d.",
    "rd", "rdn", "ld", "ibclc", "cns", "cssd",
    "esq", "esq.", "rn", "lpn", "np", "pa",
    "dds", "dmd", "od", "do", "dvm",
    "cpa", "mba", "jd", "llm", "edd", "ed.d",
    "ma", "mfa", "msw", "lcsw", "lmft",
    "rev", "rev.", "fr", "fr.", "prof", "prof.",
    "sir", "dame", "capt", "col", "gen", "sgt",
}

# Nombres o entidades que son ruido y deben eliminarse por completo
JUNK_AUTHORS = {"unknown", "anonymous", "varios", "various", "n/a", "none", "various artists", "not available"}
CORP_WORDS = {"ltd", "ltd.", "inc", "co", "publishing", "press", "books", "company", "publications", "llc", "s.a."}


def normalizar_autor_individual(autor_str):
    """Normaliza un solo autor (sin delimitadores de múltiples autores).

    Aplica:
    - Detección y reubicación de sufijos (Jr., Sr., III)
    - Eliminación de credenciales (Dr., PhD, Ms, Rd, Ibclc, etc.)
    - Eliminación de apodos entre comillas
    - Conversión Last, First → First Last
    - Title Case y limpieza de espacios

    Retorna el nombre normalizado o cadena vacía si no queda nada útil.
    """
    texto = str(autor_str).strip()
    if not texto:
        return ""

    if texto.lower() in JUNK_AUTHORS:
        return ""

    # Eliminar apodos entre comillas (ej. "Goose", 'Ace')
    texto = re.sub(r'["\']([^"\']+)["\']', '', texto)

    # Ya no intentamos invertir "Last, First" aquí porque al no ser un LLM
    # no tenemos contexto semántico para distinguir "Apellido, Nombre" de "Autor, Editorial".
    # Las comas se manejarán exclusivamente como separadores de autores en normalizar_autor().

    # Separar en tokens para filtrar credenciales y detectar sufijos sueltos
    tokens = texto.split()
    nombre_tokens = []
    sufijos = []

    for token in tokens:
        token_limpio = token.strip(",")  # Solo quitar comas, NO puntos (J.K. / Jr.)
        token_lower = token_limpio.lower()
        token_sin_punto = token_lower.rstrip(".")

        # ¿Es una credencial? → descartar
        if token_sin_punto in CREDENCIALES or token_lower in CREDENCIALES:
            continue
        # ¿Es un sufijo? → guardar aparte para poner al final
        if token_sin_punto in {s.rstrip(".") for s in SUFIJOS_NOMBRE}:
            sufijos.append(token_limpio)
            continue
        # Token normal → mantener
        if len(token_limpio) > 0:
            nombre_tokens.append(token_limpio)

    if not nombre_tokens:
        return ""

    # Reconstruir: nombre + sufijos al final
    resultado = " ".join(nombre_tokens + sufijos)

    # Aplicar Title Case (respetando iniciales como "J.K.")
    resultado = " ".join(
        palabra if re.match(r'^[A-Z]\.', palabra) else palabra.title()
        for palabra in resultado.split()
    )

    # Limpiar espacios múltiples
    resultado = re.sub(r'\s+', ' ', resultado).strip()
    return resultado


def normalizar_autor(autor_raw):
    """Normaliza un string de autores (puede contener múltiples separados por | o comas).

    Formato estándar resultante:
    - Orden: Nombre Apellido (no Apellido, Nombre)
    - Title Case
    - Sin credenciales ni apodos
    - Múltiples autores separados por '; '
    - Sin duplicados

    Retorna el string normalizado o None si no queda nada útil.
    """
    if not autor_raw or str(autor_raw).strip() == "":
        return None

    texto_raw = str(autor_raw).strip()
    
    # Unificamos separadores: reemplazamos pipes por comas para tener un único delimitador
    texto_raw = texto_raw.replace("|", ",")
    
    # Separamos asumiendo que TODA coma delimita entidades independientes
    # (WooCommerce exporta así las taxonomías múltiples).
    autores_raw = [a.strip() for a in texto_raw.split(",") if a.strip()]

    autores_normalizados = []
    vistos = set()  # Para deduplicar (case-insensitive)

    for autor in autores_raw:
        normalizado = normalizar_autor_individual(autor)
        if normalizado and normalizado.lower().strip() not in JUNK_AUTHORS:
            clave = normalizado.lower()
            if clave not in vistos:
                vistos.add(clave)
                autores_normalizados.append(normalizado)

    if not autores_normalizados:
        return None

    # Volvemos a usar la coma como piden los specs del backend (el backend lo parsea como array)
    return ", ".join(autores_normalizados)


def buscar_autor_google_books(isbn, titulo=""):
    """Busca el autor de un libro en Google Books API por ISBN (o título como fallback).

    Retorna el string de autores normalizado o None si no se encuentra.
    """
    def _buscar(params):
        try:
            if GOOGLE_BOOKS_API_KEY:
                params["key"] = GOOGLE_BOOKS_API_KEY
            resp = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("totalItems", 0) > 0:
                volume_info = data["items"][0].get("volumeInfo", {})
                autores = volume_info.get("authors", [])
                if autores:
                    # Google Books ya devuelve "First Last", solo unir y normalizar
                    return ", ".join(autores)
        except requests.RequestException:
            pass
        return None

    # Intento 1: por ISBN
    if isbn and str(isbn).strip():
        isbn_limpio = re.sub(r'[^0-9X]', '', str(isbn).upper())
        if isbn_limpio:
            resultado = _buscar({"q": f"isbn:{isbn_limpio}", "maxResults": 1})
            if resultado:
                return resultado

    # Intento 2: por título
    if titulo and len(titulo) > 3:
        resultado = _buscar({"q": f"intitle:{titulo}", "maxResults": 1})
        if resultado:
            return resultado

    return None


def buscar_autor_lm_studio(titulo, isbn="", categoria=""):
    """Fallback: identifica el autor de un libro usando LM Studio.

    Retorna el nombre del autor o None si falla.
    """
    system_msg = (
        "You are a book metadata assistant. You identify book authors. "
        "Respond with ONLY the author name(s) in 'First Last' format. "
        "If there are multiple authors, separate them with a comma. "
        "Do NOT explain your reasoning. Output ONLY the author name(s). "
        "If you don't know the author, respond with exactly: UNKNOWN"
    )
    user_msg = f"Who is the author of this book?\nTitle: {titulo}"
    if isbn:
        user_msg += f"\nISBN: {isbn}"
    if categoria:
        user_msg += f"\nCategory: {categoria}"

    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 150,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        respuesta = extraer_respuesta_lm(data).strip()

        # Validar que no sea "UNKNOWN" o basura
        if respuesta and respuesta.upper() not in {"UNKNOWN", "N/A", "NONE", "?"}:
            # Limpiar artefactos del modelo
            respuesta = re.sub(r'^["\']|["\']$', '', respuesta)
            respuesta = re.sub(
                r'^(Author|Autor|Written by|By)[:\s]*',
                '', respuesta, flags=re.IGNORECASE
            )
            respuesta = respuesta.strip()
            if respuesta and len(respuesta) > 1:
                return respuesta

    except requests.RequestException as e:
        tqdm.write(f"   ❌ Error LM Studio API: {e}")
    except (KeyError, IndexError) as e:
        tqdm.write(f"   ❌ Respuesta inesperada: {e}")

    return None


def fase_autor(dry_run=False, limit=None):
    """Fase 3: Normaliza autores existentes y rellena autores NULL.

    Paso 1: Normaliza autores existentes (Last,First → First Last, limpieza).
    Paso 2: Rellena autores NULL con Google Books API + LM Studio.
    """
    print("\n" + "=" * 60)
    print("✍️  FASE 3: Enriquecimiento de autores")
    print("=" * 60)

    # --- Paso 1: Normalizar autores existentes ---
    print("\n   Paso 1: Normalización de autores existentes...")
    query_existentes = """
        SELECT id, sku, title, author
        FROM books
        WHERE author IS NOT NULL AND TRIM(author) != ''
    """
    if limit:
        query_existentes += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows_existentes = conn.execute(text(query_existentes)).fetchall()

    cambios_norm = []
    if rows_existentes:
        for row in tqdm(rows_existentes, desc="Normalizando autores"):
            book_id, sku, titulo, autor_actual = row
            autor_normalizado = normalizar_autor(autor_actual)

            # Solo guardar si realmente cambió
            if autor_normalizado and autor_normalizado != str(autor_actual).strip():
                cambios_norm.append({
                    "id": book_id,
                    "sku": sku,
                    "titulo": titulo,
                    "autor_anterior": str(autor_actual)[:80],
                    "autor_nuevo": autor_normalizado,
                })

        print(f"   → {len(cambios_norm)}/{len(rows_existentes)} autores necesitan normalización")
    else:
        print("   → No hay libros con autor existente.")

    # Aplicar normalizaciones
    if cambios_norm:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se normalizarían {len(cambios_norm)} autores:")
            for c in cambios_norm[:15]:
                print(f"   [{c['sku']}] \"{c['titulo'][:40]}\"")
                print(f"      ❌ {c['autor_anterior']}")
                print(f"      ✅ {c['autor_nuevo']}")
            if len(cambios_norm) > 15:
                print(f"   ... y {len(cambios_norm) - 15} más")
        else:
            print(f"\n💾 Guardando {len(cambios_norm)} autores normalizados...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET author = :autor WHERE id = :id")
                lote = []
                for i, c in enumerate(cambios_norm):
                    lote.append({"autor": c["autor_nuevo"], "id": c["id"]})
                    if (i + 1) % 50 == 0:
                        conn.execute(sql, lote)
                        lote.clear()
                if lote:
                    conn.execute(sql, lote)
            print("   ✅ Autores normalizados correctamente.")

    # --- Paso 2: Rellenar autores NULL ---
    print("\n   Paso 2: Buscando autores para libros sin autor...")
    query_nulls = """
        SELECT id, sku, title, isbn, category
        FROM books
        WHERE author IS NULL OR TRIM(author) = ''
    """
    if limit:
        query_nulls += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows_nulls = conn.execute(text(query_nulls)).fetchall()

    if not rows_nulls:
        print("   ✅ No hay libros sin autor.")
    else:
        print(f"   → {len(rows_nulls)} libros sin autor")

    # Verificar disponibilidad de LM Studio
    lm_studio_disponible = False
    if rows_nulls:
        try:
            resp = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                modelos = [m["id"] for m in resp.json().get("data", [])]
                lm_studio_disponible = len(modelos) > 0
                if lm_studio_disponible:
                    print(f"   ✅ LM Studio disponible como fallback")
                else:
                    print(f"   ⚠️ LM Studio activo pero sin modelos cargados.")
        except requests.RequestException:
            print("   ℹ️ LM Studio no disponible. Solo se usará Google Books API.")

    count_google = 0
    count_lm = 0
    count_sin = 0
    cambios_nulls = []

    for row in tqdm(rows_nulls, desc="Buscando autores"):
        book_id, sku, titulo, isbn, categoria = row
        autor = None
        fuente = None

        # Nivel 1: Google Books API
        autor = buscar_autor_google_books(isbn, titulo)
        if autor:
            fuente = "Google Books"
            count_google += 1
        elif lm_studio_disponible:
            # Nivel 2: LM Studio
            autor = buscar_autor_lm_studio(titulo or "", isbn or "", categoria or "")
            if autor:
                fuente = "LM Studio"
                count_lm += 1

        if autor:
            cambios_nulls.append({
                "id": book_id,
                "sku": sku,
                "titulo": titulo,
                "autor_nuevo": autor,
                "fuente": fuente,
            })
        else:
            count_sin += 1

        # Rate limiting para Google Books API
        time.sleep(0.3)

    # Aplicar cambios
    if cambios_nulls:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se rellenarían {len(cambios_nulls)} autores:")
            for c in cambios_nulls[:15]:
                print(f"   [{c['sku']}] ({c['fuente']}) \"{c['titulo'][:40]}\" → {c['autor_nuevo']}")
            if len(cambios_nulls) > 15:
                print(f"   ... y {len(cambios_nulls) - 15} más")
        else:
            print(f"\n💾 Guardando {len(cambios_nulls)} autores nuevos...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET author = :autor WHERE id = :id")
                lote = []
                for i, c in enumerate(cambios_nulls):
                    lote.append({"autor": c["autor_nuevo"], "id": c["id"]})
                    if (i + 1) % 50 == 0:
                        conn.execute(sql, lote)
                        lote.clear()
                if lote:
                    conn.execute(sql, lote)
            print("   ✅ Autores nuevos guardados correctamente.")

    # Estadísticas
    stats = {
        "total_normalizados": len(cambios_norm),
        "total_existentes": len(rows_existentes) if rows_existentes else 0,
        "total_nulls": len(rows_nulls),
        "google_books": count_google,
        "lm_studio": count_lm,
        "sin_autor": count_sin,
    }
    print(f"\n📊 Fase 3: {len(cambios_norm)} normalizados | "
          f"{count_google} Google Books + {count_lm} LM Studio = "
          f"{count_google + count_lm} autores nuevos | "
          f"{count_sin} sin resolver")
    return stats


# =====================================================
# FASE 4: Enriquecimiento de Tags (Subjects)
# =====================================================

TAGS_PERMITIDOS_ES = [
    "Ficción General", "Ficción Histórica", "Ciencia Ficción", "Fantasía", "Terror", 
    "Romance", "Thriller y Misterio", "Cómics y Manga", "Poesía y Teatro", "Humor",
    "Historia", "Biografías y Memorias", "Música", "Filosofía", "Religión y Espiritualidad", 
    "Ciencia", "Tecnología e Informática", "Arte y Fotografía", "Diseño", "Ensayo y Crítica", 
    "Desarrollo Personal", "Negocios y Economía", "Salud y Bienestar", "Belleza y Cuidado Personal",
    "Cocina y Gastronomía", "Viajes y Turismo", "Ocio y Hobbies", "Animales y Naturaleza",
    "Infantil", "Cuentos y Fábulas", "Juvenil (Young Adult)", "Material Didáctico", "Educación", 
    "Derecho y Política", "Militar y Guerra", "Hogar y Jardinería", "Manualidades",
]

TAGS_PERMITIDOS_EN = [
    "General Fiction", "Historical Fiction", "Science Fiction", "Fantasy", "Horror", 
    "Romance", "Thriller & Mystery", "Comics & Manga", "Poetry & Drama", "Humor",
    "History", "Biographies & Memoirs", "Music", "Philosophy", "Religion & Spirituality", 
    "Science", "Technology & Computing", "Art & Photography", "Design", "Essays & Criticism", 
    "Self-Help", "Business & Economics", "Health & Wellness", "Beauty & Grooming",
    "Cooking & Gastronomy", "Travel & Tourism", "Leisure & Hobbies", "Animals & Nature",
    "Children's", "Fables & Fairy Tales", "Young Adult", "Educational Material", "Education", 
    "Law & Politics", "Home & Garden", "Military", "Crafts",
]

def obtener_prompt_tags(idioma):
    """Devuelve el system prompt, user prompt pre-formateado y lista de tags."""
    if idioma == "English":
        tags = TAGS_PERMITIDOS_EN
        sys_msg = (
            "You are an expert librarian and an automated catalog enrichment system.\n"
            "Your task is to analyze the description of a book and assign between 1 and 3 precise thematic tags.\n\n"
            "<reglas_criticas>\n"
            "1. ONLY select tags from the permitted list. DO NOT use your own knowledge to invent tags.\n"
            "2. If no tag in the list is a perfect match, select the closest related tag(s) or 'General Fiction'.\n"
            "3. NO explanations, NO intro, NO conversational text. The output MUST be ONLY a valid JSON array.\n"
            "4. Do NOT use markdown code blocks (```json).\n"
            "</reglas_criticas>\n\n"
            "<lista_permitida>\n"
            "{lista_tags}\n"
            "</lista_permitida>\n\n"
            "Correct output example:\n"
            '["Science Fiction", "Technology & Computing"]\n'
        )
        user_prompt = "Analyze the following book description and generate the JSON array with the corresponding tags:\n\n{descripcion}"
    else:
        tags = TAGS_PERMITIDOS_ES
        sys_msg = (
            "Eres un bibliotecario experto y un sistema de enriquecimiento de catálogo automatizado.\n"
            "Tu tarea es analizar la descripción de un libro y asignarle entre 1 y 3 etiquetas temáticas precisas.\n\n"
            "<reglas_criticas>\n"
            "1. SOLO puedes seleccionar etiquetas de la lista permitida. NO inventes categorías nuevas aunque creas que son mejores.\n"
            "2. Si ninguna etiqueta de la lista parece encajar perfectamente, elegí la más cercana o 'Ficción General'.\n"
            "3. NO des explicaciones, NO hables, NO agregues introducciones. La salida debe ser ÚNICAMENTE un arreglo JSON válido.\n"
            "4. No utilices bloques de código markdown (```json).\n"
            "</reglas_criticas>\n\n"
            "<lista_permitida>\n"
            "{lista_tags}\n"
            "</lista_permitida>\n\n"
            "Ejemplo de salida correcta:\n"
            '["Ciencia Ficción", "Tecnología e Informática"]\n'
        )
        user_prompt = "Analiza la siguiente descripción del libro y genera el arreglo JSON con las etiquetas correspondientes:\n\n{descripcion}"
        
    lista_tags_formateada = "\n".join(f"- {tag}" for tag in tags)
    sys_msg = sys_msg.replace("{lista_tags}", lista_tags_formateada)
    
    return sys_msg, user_prompt, tags


def generar_tags_lm_studio(descripcion, idioma):
    """Genera tags temáticos para una descripción usando LM Studio.
    
    Retorna una lista de strings con los tags oficiales, o None si falla.
    """
    if not descripcion or len(str(descripcion).strip()) < 20:
        return None
        
    if not isinstance(idioma, str):
        idioma = "Spanish"
        
    sys_msg, user_prompt_template, tags_oficiales = obtener_prompt_tags(idioma)
    user_msg = user_prompt_template.replace("{descripcion}", str(descripcion).strip())
    
    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,  # Muy baja temperatura para asegurar determinismo
                "max_tokens": 150,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        respuesta_raw = extraer_respuesta_lm(data).strip()
        
        # Limpieza robusta: buscar el primer [ y el último ] para extraer solo el JSON
        match = re.search(r'\[.*\]', respuesta_raw, re.DOTALL)
        if not match:
            # tqdm.write(f"   ⚠️ El modelo no devolvió un formato de lista válido.")
            return None
            
        respuesta_json = match.group(0)
        
        # Intentar parsear como lista JSON
        try:
            tags_devueltos = json.loads(respuesta_json)
            if isinstance(tags_devueltos, list):
                # Filtrar estrictamente contra la lista oficial
                tags_filtrados = [tag.strip() for tag in tags_devueltos if tag.strip() in tags_oficiales]
                
                # Reportar tags alucinados (que no están en la lista oficial)
                tags_alucinados = [tag.strip() for tag in tags_devueltos if tag.strip() not in tags_oficiales]
                if tags_alucinados and not tags_filtrados:
                    # Si todos los tags eran alucinados, avisar
                    # tqdm.write(f"   ⚠️ Tags ignorados (fuera de lista): {', '.join(tags_alucinados)}")
                    pass
                
                if tags_filtrados:
                    # Eliminar duplicados manteniendo orden
                    return list(dict.fromkeys(tags_filtrados))
        except json.JSONDecodeError:
            tqdm.write(f"   ❌ Error parseando JSON de LM Studio. Respuesta raw: {respuesta_raw[:100]}...")
            
    except requests.RequestException as e:
        tqdm.write(f"   ❌ Error API LM Studio en tags: {e}")
    except Exception as e:
        tqdm.write(f"   ❌ Error inesperado: {e}")
        
    return None


def fase_tags(dry_run=False, limit=None):
    """Fase 4: Asigna tags (subjects) a libros usando el modelo Qwen (LM Studio).
    
    Solo procesa libros que tengan descripción y no tengan subjects.
    """
    print("\n" + "=" * 60)
    print("🏷️  FASE 4: Enriquecimiento de Tags (Subjects)")
    print("=" * 60)
    
    # Consultar libros sin tags pero CON descripción
    query = """
        SELECT id, sku, title, description, language
        FROM books
        WHERE (subjects IS NULL OR TRIM(subjects) = '')
          AND description IS NOT NULL AND TRIM(description) != ''
    """
    if limit:
        query += f" LIMIT {int(limit)}"
        
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()
        
    if not rows:
        print("   ✅ No hay libros con descripción que necesiten tags.")
        return {"total": 0, "lm_studio": 0, "sin_tags": 0}
        
    print(f"   → {len(rows)} libros necesitan tags")
    
    # Verificar disponibilidad de LM Studio
    lm_studio_disponible = False
    try:
        resp = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            lm_studio_disponible = len(resp.json().get("data", [])) > 0
    except requests.RequestException:
        pass
        
    if not lm_studio_disponible:
        print("   ❌ LM Studio no está disponible. Esta fase requiere un LLM local.")
        return {"total": len(rows), "lm_studio": 0, "sin_tags": len(rows)}
        
    print("   ✅ LM Studio disponible. Comenzando asignación de tags...")
    
    count_tags = 0
    count_sin = 0
    cambios = []
    
    for row in tqdm(rows, desc="Procesando tags"):
        book_id, sku, titulo, descripcion, idioma = row
        
        # Extraer texto limpio de la descripción
        desc_limpia = limpiar_html(descripcion)
        
        tags_generados = generar_tags_lm_studio(desc_limpia, idioma)
        
        if tags_generados:
            # En MySQL almacenamos separado por comas
            tags_str = ", ".join(tags_generados)
            cambios.append({
                "id": book_id,
                "sku": sku,
                "titulo": titulo,
                "tags": tags_str
            })
            count_tags += 1
        else:
            count_sin += 1
            
    # Aplicar cambios
    if cambios:
        if dry_run:
            print(f"\n🔍 DRY-RUN: Se generarían tags para {len(cambios)} libros:")
            for c in cambios[:15]:
                print(f"   [{c['sku']}] \"{c['titulo'][:40]}\"")
                print(f"      🏷️  {c['tags']}")
            if len(cambios) > 15:
                print(f"   ... y {len(cambios) - 15} más")
        else:
            print(f"\n💾 Guardando {len(cambios)} sets de tags...")
            with engine.begin() as conn:
                sql = text("UPDATE books SET subjects = :tags WHERE id = :id")
                lote = []
                for i, c in enumerate(cambios):
                    lote.append({"tags": c["tags"], "id": c["id"]})
                    if (i + 1) % 50 == 0:
                        conn.execute(sql, lote)
                        lote.clear()
                if lote:
                    conn.execute(sql, lote)
            print("   ✅ Tags guardados correctamente.")
            
    stats = {
        "total": len(rows),
        "lm_studio": count_tags,
        "sin_tags": count_sin,
    }
    print(f"\n📊 Fase 4: {count_tags} enriquecidos | {count_sin} sin resolver")
    return stats


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enriquecimiento del catálogo de libros (idioma + descripciones + html + autores + tags)"
    )
    parser.add_argument(
        "--phase",
        choices=["language", "description", "format_html", "author", "tags", "all"],
        default="all",
        help="Fase a ejecutar: 'language', 'description', 'format_html', 'author', 'tags' o 'all' (por defecto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo preview: muestra cambios sin modificar la BD",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a N libros (útil para pruebas)",
    )
    args = parser.parse_args()

    print("🚀 Iniciando enriquecimiento del catálogo de libros")
    print(f"   Modo: {'DRY-RUN (sin cambios en BD)' if args.dry_run else 'PRODUCCIÓN (cambios reales)'}")
    print(f"   Fase: {args.phase}")
    if args.limit:
        print(f"   Límite: {args.limit} libros")

    stats = {}

    if args.phase in ("language", "all"):
        stats["idioma"] = fase_idioma(dry_run=args.dry_run, limit=args.limit)

    if args.phase in ("description", "all"):
        stats["descripcion"] = fase_descripcion(dry_run=args.dry_run, limit=args.limit)
        
    if args.phase in ("format_html", "all"):
        stats["format_html"] = fase_formato_html(dry_run=args.dry_run, limit=args.limit)

    if args.phase in ("author", "all"):
        stats["autor"] = fase_autor(dry_run=args.dry_run, limit=args.limit)

    if args.phase in ("tags", "all"):
        stats["tags"] = fase_tags(dry_run=args.dry_run, limit=args.limit)

    # Resumen final
    print("\n" + "=" * 60)
    print("✅ RESUMEN FINAL")
    print("=" * 60)
    if "idioma" in stats:
        s = stats["idioma"]
        total_ok = s['lingua'] + s['lm_studio']
        print(f"   Idiomas:       {s['lingua']} lingua + {s['lm_studio']} LM Studio = {total_ok}/{s['total']} ({s['no_detectados']} sin detectar)")
    if "descripcion" in stats:
        s = stats["descripcion"]
        print(f"   Descripciones: {s['google_books']} Google Books + {s['lm_studio']} LM Studio / {s['total']} total")
        print(f"                  {s['sin_descripcion']} quedaron sin descripción")
    if "format_html" in stats:
        s = stats["format_html"]
        print(f"   Formato HTML:  {s['formateados']}/{s['total']} descripciones convertidas a HTML")
    if "autor" in stats:
        s = stats["autor"]
        print(f"   Autores:       {s['total_normalizados']}/{s['total_existentes']} normalizados")
        print(f"                  {s['google_books']} Google Books + {s['lm_studio']} LM Studio / {s['total_nulls']} NULLs")
        if s['sin_autor'] > 0:
            print(f"                  {s['sin_autor']} quedaron sin autor")
    if "tags" in stats:
        s = stats["tags"]
        print(f"   Tags:          {s['lm_studio']}/{s['total']} libros enriquecidos con tags")
        if s['sin_tags'] > 0:
            print(f"                  {s['sin_tags']} quedaron sin tags resolubles")


if __name__ == "__main__":
    main()
import pandas as pd
import hashlib
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
from tqdm import tqdm
import re
import os
from config.settings import DB_CONFIG, CANTIDAD_RECOMENDACIONES

# =====================================================
# CLASIFICADOR DE CATEGORÍAS: Modelo entrenado (.pkl)
# Entrenado en Colab con SGDClassifier sobre raw_category + description.
# El pipeline encapsula TF-IDF + SGDClassifier — recibe texto crudo.
# =====================================================
MODELO_CLASIFICADOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "clasificador_categorias_v3.pkl")
UMBRAL_CONFIANZA_CATEGORIA = 0.60  # Solo aceptar predicciones con ≥60% de confianza

clf_categorias = None
if os.path.exists(MODELO_CLASIFICADOR_PATH):
    clf_categorias = joblib.load(MODELO_CLASIFICADOR_PATH)
    print(f"✅ Clasificador de categorías cargado: {MODELO_CLASIFICADOR_PATH}")
else:
    print(f"⚠️  No se encontró {MODELO_CLASIFICADOR_PATH}. Sin bonus de categoría en scoring.")

# =====================================================
# VECTORIZADOR TF-IDF: Vocabulario congelado (.pkl)
# Entrenado en Colab para inferencia directa.
# =====================================================
MODELO_VECTORIZADOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "recommender_tfidf_v1.pkl")

vectorizer_preentrenado = None
if os.path.exists(MODELO_VECTORIZADOR_PATH):
    vectorizer_preentrenado = joblib.load(MODELO_VECTORIZADOR_PATH)
    print(f"✅ Vectorizador TF-IDF cargado: {MODELO_VECTORIZADOR_PATH}")
else:
    print(f"❌ Error crítico: No se encontró {MODELO_VECTORIZADOR_PATH}. El script fallará en la fase 4.")

# Construimos la URL usando el diccionario de settings.py
DB_URL = (
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(DB_URL)

# =====================================================
# Configuración del modelo
# =====================================================
# Score mínimo para aceptar una recomendación (debajo de esto se descarta).
# Un valor de 0.10 filtra recomendaciones relleno que solo comparten idioma
# sin afinidad real por contenido ni categoría.
UMBRAL_SIMILITUD_MINIMA = 0.10

# Pesos del score compuesto. Deben sumar 1.0.
# La categoría tiene peso alto (0.45) porque es la señal más confiable
# para evitar recomendaciones absurdas (ej. novela → cocina).
# El contenido (TF-IDF) captura similitud semántica entre descripciones.
# El autor da un bonus cuando el mismo escritor tiene múltiples obras.
PESO_CONTENIDO = 0.45
PESO_CATEGORIA = 0.45
PESO_AUTOR = 0.10

# Cantidad de vecinos cercanos a recuperar por libro desde NearestNeighbors.
# Se usa un margen amplio (×15) porque muchos candidatos se filtran después
# por idioma, elegibilidad (stock/WC), y deduplicación de títulos.
CANDIDATOS_POR_LIBRO = CANTIDAD_RECOMENDACIONES * 15



# Regex para separar strings con múltiples autores (ej. "García, Borges | Paz")
SEPARADORES_AUTORES = r'[,|;/&]'

# =====================================================
# Funciones utilitarias
# =====================================================
def limpiar_html(texto):
    """Elimina etiquetas HTML de las descripciones (ej. <p>, <br>, <strong>)."""
    return re.sub(r'<[^>]+>', ' ', str(texto))


def limpiar_marketing(texto):
    """Vuela firmas corporativas y sellos literarios ('New York Times Bestseller')."""
    t = str(texto)
    # Volar "New York Times Bestselling Author" o "#1 Bestseller"
    t = re.sub(r'(?i)(new\s*york\s*times|#1|national)?\s*bestsell(er|ing)\s*(author|book)?', ' ', t)
    # Volar derechos de autor genéricos y fechas sueltas de copyright
    t = re.sub(r'(?i)copyright.*?\d{4}', ' ', t)
    return t


def remover_autor_de_descripcion(desc, author_raw):
    """Remueve dinámicamente el nombre del autor de la descripción para evitar overfitting."""
    if not author_raw or str(author_raw).strip() == '':
        return desc
    
    desc_limpia = str(desc)
    # Separar autores si hay múltiples ("Collins, Clifton", etc)
    partes = re.split(SEPARADORES_AUTORES, str(author_raw))
    for a in partes:
        # Limpiar el nombre de puntuaciones
        a_limpio = re.sub(r'[^\w\s]', '', a).strip()
        if not a_limpio:
            continue
        # Borrar cada parte significativa del nombre de forma dinámica
        for token in a_limpio.split():
            if len(token) > 2: # Evitar borrar iniciales que coincidan con otras palabras sueltas
                desc_limpia = re.sub(r'(?i)\b' + re.escape(token) + r'\b', ' ', desc_limpia)
                
    return desc_limpia


def extraer_titulo_base(titulo):
    """Extrae el título base de un libro para detectar duplicados (distintas ediciones)."""
    t = str(titulo).lower().strip()
    # Normalizar entidades HTML que vienen de WooCommerce (ej. &amp; → &)
    t = t.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
    t = re.split(r':| - | — |!|\?|\(|\bpor\b|\bby\b', t)[0]
    t = re.sub(r'^(the|a|an|el|la|los|las|un|una)\s+', '', t.strip())
    t = re.sub(r',\s*(the|a|an|el|la|los|las|un|una)$', '', t.strip())
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def score_categoria(grupo_a, grupo_b):
    """1.0 si ambos pertenecen al mismo grupo, 0.0 si no."""
    if not grupo_a or not grupo_b:
        return 0.0
    return 1.0 if grupo_a == grupo_b else 0.0


def extraer_set_autores(autor_raw):
    """Separa un string de múltiples autores en un set normalizado."""
    if not autor_raw or str(autor_raw).strip() == '':
        return set()
    partes = re.split(SEPARADORES_AUTORES, str(autor_raw))
    return {a.strip().lower() for a in partes if a.strip()}


def score_autor(autores_a, autores_b):
    """1.0 si comparten al menos un autor en común, 0.0 si no."""
    if not autores_a or not autores_b:
        return 0.0
    return 1.0 if autores_a & autores_b else 0.0


# =====================================================
# 1. Extraer catálogo completo (para entrenar TF-IDF)
# =====================================================
print("1. Conectando y extrayendo catálogo con SQLAlchemy...")
try:
    query = """
        SELECT sku, title, author, category, description, language 
        FROM books 
        WHERE description IS NOT NULL AND description != ''
    """
    df = pd.read_sql(query, engine)
except Exception as e:
    print(f"Error de conexión: {e}")
    exit()

# =====================================================
# 2. Cargar set de SKUs elegibles para ser recomendados
#    Criterios: stock > 0 Y existente en WooCommerce
# =====================================================
print("2. Cargando SKUs elegibles (stock > 0 y existentes en WC)...")
try:
    query_elegibles = """
        SELECT DISTINCT b.sku
        FROM books b
        INNER JOIN book_stock_locations bsl ON b.id = bsl.book_id
        INNER JOIN external_product_integration epi ON b.sku = epi.sku
        WHERE bsl.stock > 0
          AND epi.external_id IS NOT NULL
    """
    df_elegibles = pd.read_sql(query_elegibles, engine)
    elegibles = set(df_elegibles['sku'].astype(str).values)
    print(f"   → {len(elegibles)} libros elegibles de {len(df)} totales en catálogo")
except Exception as e:
    print(f"Error al cargar elegibles: {e}")
    exit()

# =====================================================
# 3. Procesamiento de textos
# =====================================================
print("3. Procesando textos y normalizando...")

# Rellenar valores nulos para evitar errores en concatenación y comparación
df['title'] = df['title'].fillna('')
df['author'] = df['author'].fillna('')
df['category'] = df['category'].fillna('')
df['description'] = df['description'].fillna('')
df['language'] = df['language'].fillna('Desconocido')

# Limpiar etiquetas HTML y ruido de marketing en las descripciones
df['Texto_IA_Limpio'] = df['description'].apply(limpiar_html).apply(limpiar_marketing)

# Remover el nombre del propio autor de la descripción ("Efecto Stormie")
df['Texto_IA_Limpio'] = df.apply(lambda row: remover_autor_de_descripcion(row['Texto_IA_Limpio'], row['author']), axis=1)

# Construir el texto de entrada para TF-IDF.
# IMPORTANTE: La categoría NO se incluye aquí.
# El autor explícito TAMPOCO se incluye aquí, porque penaliza la semántica y ya le damos 
# un 10% de peso directo al autor en la fórmula de similitud (score_autor).
df['Texto_IA'] = df['title'] + " " + df['Texto_IA_Limpio']

# Predecir categoría normalizada usando el clasificador entrenado.
# El modelo recibe "raw_category | description" y devuelve la categoría canónica.
if clf_categorias is not None:
    features_clf = df['category'].fillna('') + " | " + df['description'].fillna('')
    df['grupo_categoria'] = clf_categorias.predict(features_clf)

    # Aplicar umbral de confianza: predicciones dudosas → None (sin bonus)
    probas = clf_categorias.predict_proba(features_clf)
    confianzas = probas.max(axis=1)
    df.loc[confianzas < UMBRAL_CONFIANZA_CATEGORIA, 'grupo_categoria'] = None

    clasificados = df['grupo_categoria'].notna().sum()
    descartados = (confianzas < UMBRAL_CONFIANZA_CATEGORIA).sum()
    print(f"   → {clasificados} libros clasificados con confianza ≥ {UMBRAL_CONFIANZA_CATEGORIA}")
    print(f"   → {descartados} libros sin bonus de categoría (confianza insuficiente)")
else:
    df['grupo_categoria'] = None
    print("   ⚠️ Sin clasificador — categorías no disponibles para scoring")

# Precalcular sets de autores separando por delimitadores (coma, pipe, etc.)
# para que la comparación por intersección funcione con múltiples autores.
df['autores_set'] = df['author'].apply(extraer_set_autores)

# Convertir columnas a arrays NumPy para acceso rápido en el loop principal
lista_ids = df['sku'].astype(str).values
lista_grupos_cat = df['grupo_categoria'].values
lista_autores_set = df['autores_set'].values
lista_idiomas = df['language'].astype(str).values
lista_textos_limpios = df['Texto_IA_Limpio'].str.strip().values
titulos_base = [extraer_titulo_base(t) for t in df['title'].values]

# Diagnóstico: verificar cuántos libros se mapearon a grupos y cuántos quedaron sin grupo
grupos_encontrados = df['grupo_categoria'].dropna().nunique()
sin_grupo = df['grupo_categoria'].isna().sum()
print(f"   → {grupos_encontrados} grupos de categoría detectados, {sin_grupo} libros con categoría de ruido/sin categoría")

# =====================================================
# 4. Inferencia con modelo TF-IDF
#
# A diferencia del entrenamiento, ahora cargamos nuestro
# conocimiento congelado (vocabulario y pesos) desde el notebook
# y aplicamos una transformación instantánea al catálogo.
# =====================================================
print("4. Aplicando modelo TF-IDF pre-entrenado desde el notebook...")
if vectorizer_preentrenado is None:
    print("❌ Error crítico: No se encontró el vectorizador TF-IDF ('recommender_tfidf_v1.pkl').")
    print("   Asegurate de exportarlo desde tu notebook en la carpeta '/models/'.")
    exit(1)

# Usamos .transform() para aplicar el vocabulario estático a los libros, 
# conservando la integridad de las distancias evaluadas.
tfidf_matrix = vectorizer_preentrenado.transform(df['Texto_IA'])

# =====================================================
# 5. Entrenar modelo NearestNeighbors
#
# En vez de calcular la similitud coseno de CADA libro contra TODOS los
# demás (O(n²), crece cuadráticamente con el catálogo),
# NearestNeighbors pre-indexa los vectores y busca solo los K vecinos
# más cercanos de forma optimizada, reduciendo drásticamente el tiempo.
# =====================================================
print("5. Entrenando modelo de vecinos cercanos (NearestNeighbors)...")
n_neighbors = min(CANDIDATOS_POR_LIBRO, len(df) - 1)
nn_model = NearestNeighbors(
    n_neighbors=n_neighbors,
    metric='cosine',     # Distancia coseno: 0 = idénticos, 2 = opuestos
    algorithm='brute',   # Búsqueda exacta (más precisa que approximate NN)
    n_jobs=-1            # Usa todos los cores del CPU en paralelo
)
nn_model.fit(tfidf_matrix)

# Obtener todos los vecinos de una sola vez (batch) en lugar de uno por uno
distances, indices = nn_model.kneighbors(tfidf_matrix)
# Convertir distancia coseno → similitud coseno (distancia = 1 - similitud)
similarities = 1 - distances

# =====================================================
# 6. Generar recomendaciones con score compuesto
#
# Para cada libro, se recorren sus K vecinos más cercanos (del paso 5)
# y se calcula un score compuesto que combina:
#   - Similitud de contenido (TF-IDF coseno): ¿las descripciones son similares?
#   - Match de categoría (grupo canónico): ¿pertenecen al mismo género?
#   - Match de autor (intersección de sets): ¿comparten algún autor?
# Los candidatos se filtran por idioma, elegibilidad y deduplicación,
# se ordenan por score descendente, y se guardan los top N en la BD.
# =====================================================
print("6. Generando recomendaciones (score compuesto) e insertando por lotes...")
try:
    # engine.begin() abre una transacción: commit automático si éxito, rollback si excepción
    with engine.begin() as conn:
        # Limpiar todas las recomendaciones anteriores antes de regenerar
        conn.execute(text("TRUNCATE TABLE product_recommendations"))

        sql_insert = text("""
            INSERT INTO product_recommendations (source_sku, recommended_sku, rank_order, similarity_score) 
            VALUES (:s, :r, :o, :score)
        """)

        lote_datos = []
        libros_sin_recs = 0
        total_recs = 0

        for idx in tqdm(range(len(df)), desc="Procesando Catálogo"):
            sku_actual = lista_ids[idx]
            grupo_cat_actual = lista_grupos_cat[idx]
            autores_actual = lista_autores_set[idx]
            idioma_actual = lista_idiomas[idx]
            titulo_base_actual = titulos_base[idx]

            candidatos = []

            # Recorrer los K vecinos más cercanos pre-calculados por NearestNeighbors
            for j_pos in range(len(indices[idx])):
                i = indices[idx][j_pos]
                sim_contenido = float(similarities[idx][j_pos])

                # --- Filtros de exclusión (se aplican antes del scoring para eficiencia) ---
                if i == idx:                                    # No recomendarse a sí mismo
                    continue
                if not lista_textos_limpios[i]:                  # Descartar libros sin descripción útil
                    continue
                if titulos_base[i] == titulo_base_actual:        # Evitar recomendar la misma obra en otra edición
                    continue
                if lista_idiomas[i] != idioma_actual:            # Solo recomendar libros en el mismo idioma
                    continue
                if lista_ids[i] not in elegibles:                # Solo libros con stock > 0 y publicados en WC
                    continue

                # --- Cálculo del score compuesto ---
                # Cada componente retorna 0.0 o 1.0 (binario) excepto sim_contenido (0.0 a 1.0 continuo)
                s_cat = score_categoria(grupo_cat_actual, lista_grupos_cat[i])
                s_autor = score_autor(autores_actual, lista_autores_set[i])

                score_final = (
                    PESO_CONTENIDO * sim_contenido
                    + PESO_CATEGORIA * s_cat
                    + PESO_AUTOR * s_autor
                )

                # Descartar candidatos con score demasiado bajo (irrelevantes)
                if score_final < UMBRAL_SIMILITUD_MINIMA:
                    continue

                candidatos.append((lista_ids[i], score_final, i))

            # Ordenar por score compuesto descendente y seleccionar los mejores
            candidatos.sort(key=lambda x: x[1], reverse=True)
            recomendados = []
            titulos_vistos = set()
            
            for rec_sku, score, rec_idx in candidatos:
                t_base = titulos_base[rec_idx]
                if t_base not in titulos_vistos:
                    titulos_vistos.add(t_base)
                    recomendados.append((rec_sku, score))
                    if len(recomendados) == CANTIDAD_RECOMENDACIONES:
                        break

            if len(recomendados) == 0:
                libros_sin_recs += 1

            total_recs += len(recomendados)

            # Preparar lote de datos como diccionarios para inserción parametrizada
            for rank, (rec_sku, score) in enumerate(recomendados, start=1):
                lote_datos.append({"s": sku_actual, "r": rec_sku, "o": rank, "score": round(score, 4)})

            # Inserción por lotes cada 100 libros para reducir latencia de red con MySQL
            if (idx + 1) % 100 == 0 and lote_datos:
                conn.execute(sql_insert, lote_datos)
                lote_datos.clear()

        if lote_datos:
            conn.execute(sql_insert, lote_datos)

    promedio_recs = total_recs / len(df) if len(df) > 0 else 0
    print(f"¡Éxito! Proceso finalizado y base de datos actualizada.")
    print(f"📊 Estadísticas: {total_recs} recomendaciones generadas (promedio: {promedio_recs:.1f} por libro)")
    if libros_sin_recs > 0:
        print(f"⚠️ {libros_sin_recs} libros no obtuvieron recomendaciones (sin candidatos elegibles)")

except Exception as e:
    print(f"Error durante el procesamiento/guardado: {e}")

# =====================================================
# 7. Exportación del Dataset de Entrada
# =====================================================
print("7. Generando dataset de entrada anonimizado...")

def hashear_sku(sku, salt="=]&Roy%!?vK8"):
    """
    Convierte un SKU real en un hash SHA-256 irreversible.
    Asegurarse de cambiar el 'salt' por una cadena única del entorno.
    """
    texto_a_hashear = f"{str(sku).strip()}_{salt}".encode('utf-8')
    hash_completo = hashlib.sha256(texto_a_hashear).hexdigest()
    return hash_completo[:16] 

def normalizar_idioma(lang):
    lang = str(lang).strip().lower()
    mapa = {'es': 'ES', 'español': 'ES', 'spanish': 'ES', 'spa': 'ES',
            'en': 'EN', 'english': 'EN', 'inglés': 'EN', 'eng': 'EN'}
    return mapa.get(lang, 'OTRO' if lang else 'DES')

df_kaggle = df.copy()
df_kaggle['sku_anonimo'] = df_kaggle['sku'].apply(hashear_sku)

# Normalización estricta
df_kaggle['grupo_categoria'] = df_kaggle['grupo_categoria'].fillna('sin_clasificar')
df_kaggle['language'] = df_kaggle['language'].apply(normalizar_idioma)

# Filtro de calidad: solo libros con texto real y título válido
df_export = df_kaggle[['sku_anonimo', 'title', 'author', 'grupo_categoria', 'language', 'Texto_IA_Limpio']].copy()
df_export = df_export[
    (df_export['title'].str.strip() != '') & 
    (df_export['Texto_IA_Limpio'].str.strip().str.len() > 20)
]

# Exportación limpia
nombre_archivo = 'book_catalog_clean_kaggle.csv'
df_export.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')
print(f"✅ Dataset limpio generado: {nombre_archivo}")
print(f"📊 {len(df_export)} filas válidas exportadas. Categorías únicas: {df_export['grupo_categoria'].nunique()}")
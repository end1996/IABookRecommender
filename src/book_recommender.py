import pandas as pd
import hashlib
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import re
from config.settings import DB_CONFIG, CANTIDAD_RECOMENDACIONES

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
# Un valor muy bajo (0.01) permite cubrir libros con pocos candidatos elegibles.
UMBRAL_SIMILITUD_MINIMA = 0.01

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

# =====================================================
# Stopwords bilingües (español + inglés)
# Se incluyen ambos idiomas porque el TF-IDF se entrena sobre TODO
# el catálogo (ES + EN) junto. Sin stopwords en inglés, palabras como
# "the", "and", "book" crearían ruido en la matriz de features.
# Filtrarlas mejora la calidad de todos los vectores sin afectar
# negativamente las recomendaciones por idioma (esas palabras no
# aportan valor discriminativo en ningún idioma).
# =====================================================
STOPWORDS_ESPANOL = [
    # Artículos, preposiciones, pronombres y conjunciones del español
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'una', 'por', 'con', 'para',
    'su', 'se', 'del', 'las', 'los', 'al', 'lo', 'como', 'más', 'pero', 'sus',
    'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy',
    'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde',
    'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros',
    'ese', 'eso', 'ante', 'ellos', 'esto', 'mí', 'antes', 'algunos', 'qué',
    'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos',
    'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar',
    'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti',
    'tu', 'tus', 'ellas', 'nosotras', 'es', 'soy', 'eres', 'somos', 'son',
    'fui', 'fue', 'fueron', 'ha', 'han', 'ser', 'sido', 'tiene', 'tienen',
    'había', 'haber', 'hacer', 'puede', 'siendo', 'así', 'después',
    # Palabras del dominio literario/editorial que aparecen en casi todas
    # las descripciones y no aportan señal discriminativa entre géneros
    'historia', 'vida', 'mundo', 'años', 'días', 'hombres',
    'libro', 'libros', 'edición', 'editorial', 'tomo', 'colección', 'serie',
    'autor', 'leer', 'lectura', 'página', 'páginas', 'obra', 'obras',
    'texto', 'textos', 'capítulo', 'capítulos', 'volumen',
    'nuevo', 'nueva', 'nuevos', 'nuevas', 'gran', 'grande', 'grandes',
    'primer', 'primera', 'mejor', 'manera', 'forma', 'parte', 'tiempo',
    'lugar', 'cada', 'través', 'cuenta', 'hombre', 'mujer',
    'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez',
]

STOPWORDS_INGLES = [
    # Equivalentes en inglés: artículos, preposiciones, pronombres
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'as', 'was', 'are', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very',
    'just', 'about', 'above', 'after', 'again', 'all', 'also', 'am',
    'any', 'because', 'before', 'between', 'both', 'each', 'few',
    'further', 'get', 'got', 'he', 'her', 'here', 'him', 'his', 'how',
    'into', 'its', 'let', 'me', 'more', 'most', 'my', 'now', 'off',
    'only', 'other', 'our', 'out', 'over', 'own', 'same', 'she', 'some',
    'still', 'such', 'that', 'their', 'them', 'there', 'these', 'they',
    'this', 'those', 'through', 'under', 'until', 'up', 'us', 'we',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'you', 'your', 'yours', 'yourself',
    # Palabras del dominio editorial en inglés
    'book', 'books', 'story', 'stories', 'novel', 'edition', 'chapter',
    'author', 'read', 'reading', 'page', 'pages', 'volume', 'series',
    'new', 'great', 'first', 'one', 'two', 'three', 'way', 'time',
    'world', 'life', 'year', 'years', 'man', 'woman', 'day', 'days',
    'many', 'much', 'well', 'back', 'even', 'also', 'made', 'make',
    'like', 'set', 'part', 'long', 'find', 'work', 'come', 'take',
]

# Se combinan ambas listas para alimentar el TfidfVectorizer
STOPWORDS_BILINGUE = STOPWORDS_ESPANOL + STOPWORDS_INGLES

# Regex para separar strings con múltiples autores (ej. "García, Borges | Paz")
SEPARADORES_AUTORES = r'[,|;/&]'

# =====================================================
# Mapeo de sinónimos de categorías
#
# Las categorías en WooCommerce no están normalizadas: existen variantes
# en español e inglés, con distintas mayúsculas y separadores.
# Ejemplo: "Children's Fiction" (1001), "Libros Infantiles" (607),
#          "Children Fiction" (190), "Children Book" (120)
#          → Todos representan el mismo género: 'infantil'
#
# Este diccionario mapea cada variante conocida a un grupo canónico.
# La normalización ocurre SOLO en el script (no modifica la BD).
# =====================================================
CATEGORIA_SINONIMOS = {
    'infantil': [
        "Children's Fiction", "Children Fiction", "Children Book",
        "Children's Non-Fiction", "Children Non-Fiction",
        "Libros Infantiles", "Libros Infantiles en Español",
        "Cuentos para niños", "Cuentos Infantiles", "Board books",
        "Kids & Children", "Kids (12 & Under)",
        "Libros para bebés", "Libros de actividades", "Activity Books",
        "I CAN READ",
    ],
    'juvenil': [
        "Juvenile Fiction", "Juvenile Nonfiction",
        "Novelas Juveniles", "Young Adult Fiction", "Young Adult",
        "Tweens Fiction", "Kids: Middle Grade", "Kids Middle Grade",
    ],
    'ficcion': [
        "Fiction", "General Fiction", "Novels", "Novelas",
    ],
    'accion_aventura': [
        "Action & Adventure", "Action", "Adventure",
    ],
    'fantasia_scifi': [
        "Fantasy", "Science Fiction",
    ],
    'cocina': [
        "Cook Book", "Cocina",
    ],
    'religion': [
        "Religion", "Religión y Espiritualidad", "Religion & Spirituality",
    ],
    'comics': [
        "Cómics de Colección", "Comics & Graphic Novels",
    ],
    'arte_fotografia': [
        "Arte", "Fotografía y diseño", "Art & Photography",
    ],
    'historia_cultura': [
        "Historia y Cultura",
    ],
    'cuentos': [
        "Cuentos",
    ],
    'aprendizaje': [
        "Aprendizaje", "Learning Books",
    ],
    'animales': [
        "Libros de animales", "Animals",
    ],
    'bienestar_salud': [
        "Bienestar y Salud",
    ],
    'colorear': [
        "Libros para colorear",
    ],
    'navidad': [
        "Libros de Navidad", "Christmas Books",
    ],
    'halloween': [
        "Libros de Halloween para Niños",
    ],
    'conejos': [
        "Libros de Conejos",
    ],
}

# Categorías que son ruido: no describen temática/género sino promociones,
# temporadas o rangos de precio. Si dos libros comparten solo estas categorías,
# NO se les debe dar bonus de categoría (retornan None en get_grupo_categoria).
CATEGORIAS_RUIDO = [
    "Novedades", "2025", "Easter",
    "Día de la Madre", "Día de la mujer",
    "Libros hasta 9.90", "Libros en Español",
]


# =====================================================
# Funciones utilitarias
# =====================================================
def limpiar_html(texto):
    """Elimina etiquetas HTML de las descripciones (ej. <p>, <br>, <strong>)."""
    return re.sub(r'<[^>]+>', ' ', str(texto))


def extraer_titulo_base(titulo):
    """Extrae el título base de un libro para detectar duplicados (distintas ediciones)."""
    t = str(titulo).lower().strip()
    t = re.split(r'[:\-—]|\bpor\b|\bby\b', t)[0]
    t = re.sub(r'^(the|a|an|el|la|los|las|un|una)\s+', '', t.strip())
    t = re.sub(r',\s*(the|a|an|el|la|los|las|un|una)$', '', t.strip())
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def normalizar_categoria(cat):
    """Normaliza una categoría para comparación flexible (sin modificar la BD)."""
    cat = str(cat).strip().lower()
    cat = re.sub(r'[>\-/|,.:;()[\]{}\'\"&]', ' ', cat)
    cat = re.sub(r'\s+', ' ', cat).strip()
    return cat


# Pre-construir diccionario invertido al cargar el módulo:
# {"children s fiction": "infantil", "libros infantiles": "infantil", ...}
# Esto permite búsquedas O(1) en vez de iterar el diccionario original.
_MAPA_CAT_A_GRUPO = {}
for _grupo, _variantes in CATEGORIA_SINONIMOS.items():
    for _variante in _variantes:
        _MAPA_CAT_A_GRUPO[normalizar_categoria(_variante)] = _grupo

_CATEGORIAS_RUIDO_NORM = {normalizar_categoria(c) for c in CATEGORIAS_RUIDO}


def get_grupo_categoria(cat_normalizada):
    """Mapea una categoría normalizada a su grupo canónico.
    Retorna None si es una categoría de ruido (sin señal temática).
    Retorna la categoría misma como fallback si no está en el mapeo."""
    if not cat_normalizada or cat_normalizada in _CATEGORIAS_RUIDO_NORM:
        return None
    # Búsqueda exacta en el mapeo
    if cat_normalizada in _MAPA_CAT_A_GRUPO:
        return _MAPA_CAT_A_GRUPO[cat_normalizada]
    # Búsqueda parcial: si la categoría contiene una variante conocida
    for variante_norm, grupo in _MAPA_CAT_A_GRUPO.items():
        if variante_norm in cat_normalizada:
            return grupo
    # Fallback: la categoría normalizada actúa como su propio grupo
    return cat_normalizada


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
        WHERE bsl.stock > 0
          AND b.wc_product_id IS NOT NULL
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

# Limpiar etiquetas HTML que vienen de WooCommerce en las descripciones
df['Texto_IA_Limpio'] = df['description'].apply(limpiar_html)

# Construir el texto de entrada para TF-IDF.
# IMPORTANTE: La categoría NO se incluye aquí (a diferencia de versiones anteriores).
# La categoría se evalúa como score separado para evitar contaminar el TF-IDF
# con tokens de categoría que distorsionan las estadísticas de frecuencia.
df['Texto_IA'] = df['author'] + " " + df['title'] + " " + df['Texto_IA_Limpio']

# Precalcular el grupo canónico de categoría para cada libro.
# Esto mapea variantes como "Children's Fiction" y "Libros Infantiles" al mismo grupo.
df['categoria_normalizada'] = df['category'].apply(normalizar_categoria)
df['grupo_categoria'] = df['categoria_normalizada'].apply(get_grupo_categoria)

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
# 4. Entrenar modelo TF-IDF (bigrams + sublinear_tf)
#
# TF-IDF convierte cada texto en un vector numérico donde cada dimensión
# representa la importancia de un término (o bigrama) para ese documento.
# =====================================================
print("4. Entrenando el modelo TF-IDF (bigrams + sublinear_tf + stopwords bilingues)...")
vectorizer = TfidfVectorizer(
    max_df=0.80,         # Ignora términos en >80% de documentos (demasiado comunes)
    min_df=2,            # Ignora términos en <2 documentos (demasiado raros)
    stop_words=STOPWORDS_BILINGUE,
    ngram_range=(1, 2),  # Captura unigramas y bigramas (ej. "guerra civil", "amor prohibido")
    sublinear_tf=True,   # Aplica log(1 + tf), reduce impacto de palabras muy repetidas
    max_features=50000,  # Limita el vocabulario para controlar uso de memoria
    # Exige que cada token EMPIECE con una letra, descartando tokens numéricos
    # o con prefijo numérico ('000', '100th', '1000stickers') que vienen de
    # marketing y no aportan señal discriminativa entre géneros.
    token_pattern=r'(?u)\b[a-zA-ZáéíóúñüÀ-ÿ]\w+\b'

)
tfidf_matrix = vectorizer.fit_transform(df['Texto_IA'])

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

                # Fórmula: score = 0.45×contenido + 0.45×categoría + 0.10×autor
                score_final = (
                    PESO_CONTENIDO * sim_contenido
                    + PESO_CATEGORIA * s_cat
                    + PESO_AUTOR * s_autor
                )

                # Descartar candidatos con score demasiado bajo (irrelevantes)
                if score_final < UMBRAL_SIMILITUD_MINIMA:
                    continue

                candidatos.append((lista_ids[i], score_final))

            # Ordenar por score compuesto descendente y seleccionar los mejores
            candidatos.sort(key=lambda x: x[1], reverse=True)
            recomendados = candidatos[:CANTIDAD_RECOMENDACIONES]

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

# Usamos una copia del DataFrame principal (df) de los textos procesados
df_kaggle = df.copy()

# Generación la nueva columna con los SKUs ofuscados
df_kaggle['sku_anonimo'] = df_kaggle['sku'].apply(hashear_sku)

# Filtrar estrictamente las columnas que son públicas y útiles para NLP
columnas_seguras = [
    'sku_anonimo', 
    'title', 
    'author', 
    'grupo_categoria', 
    'language', 
    'Texto_IA_Limpio' 
]

df_export = df_kaggle[columnas_seguras]

# Exportación del archivo CSV final
nombre_archivo = 'book_catalog_features_kaggle.csv'
df_export.to_csv(nombre_archivo, index=False)

print(f"¡Dataset generado con éxito: {nombre_archivo}!")
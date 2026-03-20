import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

CANTIDAD_RECOMENDACIONES = 20

STOPWORDS_ESPANOL = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'una', 'por', 'con', 'para', 'su', 'se', 'del', 'las', 'los', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'es', 'soy', 'eres', 'somos', 'son', 'fui', 'fue', 'fueron', 'ha', 'han', 'historia', 'vida', 'mundo', 'años', 'días', 'hombres']

def limpiar_html(texto):
    return re.sub(r'<[^>]+>', ' ', str(texto))

def extraer_titulo_base(titulo):
    t = str(titulo).lower().strip()
    t = re.split(r'[:\-—]|\\bpor\\b|\\bby\\b', t)[0]
    t = re.sub(r'^(the|a|an|el|la|los|las|un|una)\\s+', '', t.strip())
    t = re.sub(r',\\s*(the|a|an|el|la|los|las|un|una)$', '', t.strip())
    t = re.sub(r'[^\\w\\s]', '', t)
    t = re.sub(r'\\s+', ' ', t).strip()
    return t

# =====================================================
# 1. Extraer catálogo completo (para entrenar TF-IDF)
# =====================================================
print("1. Conectando y extrayendo catálogo con SQLAlchemy...")
try:
    query = """
        SELECT sku, title, category, description, language 
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
df['title'] = df['title'].fillna('')
df['category'] = df['category'].fillna('')
df['description'] = df['description'].fillna('')
df['language'] = df['language'].fillna('Desconocido')

df['Texto_IA_Limpio'] = df['description'].apply(limpiar_html)
df['Texto_IA'] = (df['category'] + " ") * 5 + df['title'] + " " + df['Texto_IA_Limpio']

lista_ids = df['sku'].astype(str).values
lista_categorias = df['category'].values
lista_idiomas = df['language'].astype(str).values
lista_textos_limpios = df['Texto_IA_Limpio'].str.strip().values
titulos_base = [extraer_titulo_base(t) for t in df['title'].values]

# =====================================================
# 4. Entrenar modelo TF-IDF
# =====================================================
print("4. Entrenando el modelo TF-IDF (Sin calcular la matriz completa de similitud)...")
vectorizer = TfidfVectorizer(max_df=0.80, min_df=2, stop_words=STOPWORDS_ESPANOL)
tfidf_matrix = vectorizer.fit_transform(df['Texto_IA'])

# =====================================================
# 5. Generar recomendaciones filtradas e insertar
# =====================================================
print("5. Generando recomendaciones (Fase 1 y 2) e insertando por lotes...")
try:
    # engine.begin() abre la conexión, inicia la transacción y cierra todo al finalizar el bloque
    with engine.begin() as conn:
        # 1. Limpiamos la tabla
        conn.execute(text("TRUNCATE TABLE product_recommendations"))
        
        # 2. Definimos la consulta con parámetros nombrados (:nombre)
        sql_insert = text("""
            INSERT INTO product_recommendations (source_sku, recommended_sku, rank_order) 
            VALUES (:s, :r, :o)
        """)
        
        lote_datos = []
        libros_sin_recs = 0
        
        for idx in tqdm(range(len(df)), desc="Procesando Catálogo"):
            sku_actual = lista_ids[idx]
            
            # Cálculo de similitud
            sim_vector = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            sim_scores_indices = np.argsort(-sim_vector)
            
            recomendados = []
            vistos = set()

            # FASE 1: Estricta (misma categoría + elegible)
            for i in sim_scores_indices:
                if i == idx or not lista_textos_limpios[i] or titulos_base[i] == titulos_base[idx] or lista_idiomas[i] != lista_idiomas[idx]:
                    continue
                if lista_ids[i] not in elegibles:
                    continue
                if lista_categorias[i] == lista_categorias[idx]:
                    recomendados.append(lista_ids[i])
                    vistos.add(lista_ids[i])
                if len(recomendados) >= CANTIDAD_RECOMENDACIONES: break

            # FASE 2: Relajada (cualquier categoría + elegible)
            if len(recomendados) < CANTIDAD_RECOMENDACIONES:
                for i in sim_scores_indices:
                    if i == idx or not lista_textos_limpios[i] or titulos_base[i] == titulos_base[idx] or lista_idiomas[i] != lista_idiomas[idx]:
                        continue
                    if lista_ids[i] not in elegibles:
                        continue
                    if lista_ids[i] not in vistos:
                        recomendados.append(lista_ids[i])
                        vistos.add(lista_ids[i])
                    if len(recomendados) >= CANTIDAD_RECOMENDACIONES: break

            if len(recomendados) == 0:
                libros_sin_recs += 1

            # PREPARACIÓN DEL LOTE (Como Diccionarios)
            for rank, rec_sku in enumerate(recomendados, start=1):
                lote_datos.append({"s": sku_actual, "r": rec_sku, "o": rank})
            
            # Inserción cada 100 libros
            if (idx + 1) % 100 == 0 and lote_datos:
                conn.execute(sql_insert, lote_datos)
                lote_datos.clear()

        # Insertar los últimos registros restantes
        if lote_datos:
            conn.execute(sql_insert, lote_datos)
        
    print(f"¡Éxito! Proceso finalizado y base de datos actualizada.")
    if libros_sin_recs > 0:
        print(f"⚠️ {libros_sin_recs} libros no obtuvieron recomendaciones (sin candidatos elegibles)")

except Exception as e:
    # No hace falta conn.rollback(), engine.begin() lo hace solo si detecta una excepción
    print(f"Error durante el procesamiento/guardado: {e}")
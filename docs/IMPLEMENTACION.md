# Implementación del Software — Módulo de Recomendaciones con IA

## 1. Descripción del Script de Python

El módulo `book_recommender.py` es un script de procesamiento por lotes que se conecta directamente a la base de datos MySQL del sistema de inventario, extrae el catálogo completo de libros (títulos, categorías, descripciones e idiomas), y aplica técnicas de **Procesamiento de Lenguaje Natural (NLP)** mediante el modelo **TF-IDF (Term Frequency–Inverse Document Frequency)** combinado con **Similitud del Coseno** para calcular la afinidad semántica entre cada par de libros. A partir de estas métricas, genera automáticamente un ranking de hasta 20 recomendaciones personalizadas por producto, filtrando duplicados y respetando restricciones de idioma y categoría. Finalmente, escribe los resultados directamente en la tabla `product_recommendations` de la misma base de datos, permitiendo que el backend y frontend consuman estas recomendaciones sin intervención manual.

---

## 2. Tecnologías y Librerías Utilizadas

| Librería | Versión | Propósito |
|---|---|---|
| `pandas` | — | Manipulación y análisis de datos tabulares (DataFrames) |
| `sqlalchemy` | — | ORM y motor de conexión a bases de datos relacionales |
| `mysql-connector-python` | — | Driver nativo para la conexión MySQL desde Python |
| `scikit-learn` | 1.8.0 | Modelos de Machine Learning: `TfidfVectorizer` y `cosine_similarity` |
| `numpy` | — | Operaciones matemáticas vectorizadas (ordenamiento de índices) |
| `scipy` | — | Dependencia de `scikit-learn` para cálculos científicos |
| `tqdm` | — | Barras de progreso en terminal para monitorizar el procesamiento |
| `python-dotenv` | — | Carga de variables de entorno desde archivo `.env` |
| `colorama` | 0.4.6 | Colores en la salida de terminal (compatibilidad Windows) |

> **Motor de IA utilizado:** No se emplea un LLM externo (como GPT o Gemini). El modelo inteligente es **TF-IDF + Similitud del Coseno** de `scikit-learn`, un enfoque clásico de NLP que vectoriza los textos del catálogo y calcula la distancia semántica entre ellos de forma local, sin llamadas a APIs externas.

---

## 3. Arquitectura del Flujo de Datos

```
┌─────────────────┐       ┌─────────────────────────┐       ┌──────────────────────────┐
│  BASE DE DATOS  │──────▶│  SCRIPT PYTHON (IA)     │──────▶│  BASE DE DATOS           │
│  MySQL          │ Lee   │  book_recommender.py     │Escribe│  MySQL                   │
│                 │       │                          │       │                          │
│  Tabla: books   │       │  1. Extrae catálogo      │       │  Tabla:                  │
│  (sku, title,   │       │  2. Limpia HTML          │       │  product_recommendations │
│   category,     │       │  3. Vectoriza con TF-IDF │       │  (source_sku,            │
│   description,  │       │  4. Calcula similitud    │       │   recommended_sku,       │
│   language)     │       │  5. Filtra por fases     │       │   rank_order)            │
└─────────────────┘       └─────────────────────────┘       └──────────────────────────┘
```

**Flujo detallado:**

1. **Lectura (Entrada):** El script lee la tabla `books` de MySQL, filtrando solo registros que tengan una descripción no vacía (`WHERE description IS NOT NULL AND description != ''`).
2. **Procesamiento (Modelo de IA):** Los textos se limpian de HTML, se concatenan los campos relevantes (categoría ×5 + título + descripción) y se vectorizan con TF-IDF. Luego se calcula la similitud del coseno entre cada libro y el resto del catálogo.
3. **Escritura (Salida):** Las recomendaciones calculadas se insertan por lotes en la tabla `product_recommendations`, reemplazando completamente las recomendaciones anteriores (`TRUNCATE TABLE`).

---

## 4. Configuración del Entorno

### 4.1 Crear el entorno virtual e instalar dependencias

```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual (Windows)
venv\Scripts\activate

# Instalar las dependencias del proyecto
pip install -r requirements.txt
```

### 4.2 Archivo `requirements.txt`

```txt
colorama==0.4.6
greenlet==3.3.2
joblib==1.5.3
mysql-connector-python
numpy
pandas
python-dotenv
scikit-learn==1.8.0
scipy
sqlalchemy
tqdm
tzdata==2025.3
```

[CAPTURA: Terminal mostrando la instalación exitosa de dependencias con `pip install -r requirements.txt`]

---

## 5. Configuración de la Base de Datos

### 5.1 Variables de entorno (`.env`)

El script utiliza `python-dotenv` para cargar las credenciales de conexión desde un archivo `.env` en la raíz del proyecto. El archivo `.env.example` proporciona la plantilla:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=tu_contraseña
DB_NAME=sistema_inventario
DB_PORT=3306
CANTIDAD_RECOMENDACIONES=20
```

### 5.2 Módulo de configuración (`config/settings.py`)

Este módulo centraliza la carga de variables de entorno y las expone como un diccionario para el script principal:

```python
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT", "3306")
}

CANTIDAD_RECOMENDACIONES = int(os.getenv("CANTIDAD_RECOMENDACIONES", 20))
```

### 5.3 Construcción de la URL de conexión (SQLAlchemy)

En el script principal, la URL de conexión se construye dinámicamente a partir del diccionario `DB_CONFIG`:

```python
from config.settings import DB_CONFIG, CANTIDAD_RECOMENDACIONES

DB_URL = (
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(DB_URL)
```

> Se utiliza el dialecto `mysql+mysqlconnector` de SQLAlchemy, que emplea el driver puro de Python `mysql-connector-python` para la comunicación con MySQL.

[CAPTURA: Archivo `.env` configurado con las credenciales de la base de datos (con contraseñas censuradas)]

---

## 6. Implementación del Módulo Principal

### 6.1 Paso 1 — Extracción del catálogo desde la base de datos

El script inicia conectándose a MySQL y extrayendo el catálogo de libros con los campos necesarios para el análisis:

```python
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
```

[CAPTURA: Terminal mostrando el log "1. Conectando y extrayendo catálogo con SQLAlchemy..." seguido del conteo de registros extraídos]

### 6.2 Paso 2 — Preprocesamiento y limpieza de textos

Los datos en bruto se normalizan: se limpian etiquetas HTML de las descripciones y se construye un texto enriquecido que **pondera la categoría 5 veces** para darle mayor relevancia semántica frente al título y la descripción:

```python
print("2. Procesando textos y normalizando...")
df['title'] = df['title'].fillna('')
df['category'] = df['category'].fillna('')
# ... lógica de procesamiento ...

df['Texto_IA_Limpio'] = df['description'].apply(limpiar_html)
df['Texto_IA'] = (df['category'] + " ") * 5 + df['title'] + " " + df['Texto_IA_Limpio']
```

> **Decisión de diseño:** Multiplicar la categoría ×5 en el texto de entrada asegura que libros de la misma categoría tengan mayor similitud semántica, priorizando la relevancia temática sobre coincidencias casuales de palabras en las descripciones.

### 6.3 Paso 3 — Entrenamiento del modelo TF-IDF (Núcleo de IA)

Esta es la **función central del motor de inteligencia artificial**. Se configura el vectorizador TF-IDF con *stopwords* en español personalizadas y umbrales de frecuencia para filtrar ruido, y se transforma todo el corpus en una matriz de vectores numéricos:

```python
print("3. Entrenando el modelo TF-IDF...")

STOPWORDS_ESPANOL = ['el', 'la', 'de', 'que', 'y', 'a', 'en',
                     'un', 'una', 'por', 'con', 'para', 'su',
                     # ... lista completa de stopwords ...
                     'historia', 'vida', 'mundo', 'años']

vectorizer = TfidfVectorizer(
    max_df=0.80,      # Ignora términos en >80% de documentos
    min_df=2,          # Ignora términos en <2 documentos
    stop_words=STOPWORDS_ESPANOL
)
tfidf_matrix = vectorizer.fit_transform(df['Texto_IA'])
```

> **Parámetros del modelo:**
> - `max_df=0.80`: Descarta palabras que aparecen en más del 80% de los libros (demasiado comunes para ser discriminativas).
> - `min_df=2`: Descarta palabras que aparecen en menos de 2 libros (demasiado raras para generar conexiones).
> - `stop_words`: Lista personalizada de 100+ palabras vacías en español, incluyendo términos genéricos del dominio literario como "historia", "vida", "mundo".

### 6.4 Paso 4 — Cálculo de similitud y generación de recomendaciones

Para cada libro del catálogo, se calcula su vector de similitud contra todos los demás libros y se seleccionan las mejores recomendaciones en **dos fases**:

```python
for idx in tqdm(range(len(df)), desc="Procesando Catálogo"):
    sku_actual = lista_ids[idx]
    
    # Núcleo de IA: Cálculo de similitud del coseno
    sim_vector = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()
    sim_scores_indices = np.argsort(-sim_vector)
    
    recomendados = []
    vistos = set()

    # FASE 1: Misma categoría + mismo idioma
    for i in sim_scores_indices:
        if i == idx: continue
        # ... filtros de duplicados y validaciones ...
        if lista_categorias[i] == lista_categorias[idx]:
            recomendados.append(lista_ids[i])
        if len(recomendados) >= CANTIDAD_RECOMENDACIONES: break
```

> **Estrategia de dos fases:**
> - **Fase 1 (Estricta):** Solo recomienda libros de la *misma categoría* y *mismo idioma*, excluyendo duplicados por título.
> - **Fase 2 (Relajada):** Si no se alcanzan las 20 recomendaciones, se amplía la búsqueda a *cualquier categoría* manteniendo el filtro de idioma.

### 6.5 Paso 5 — Escritura por lotes en la base de datos

Los resultados se insertan en la tabla `product_recommendations` usando transacciones de SQLAlchemy con inserción por lotes cada 100 libros procesados:

```python
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE product_recommendations"))
    
    sql_insert = text("""
        INSERT INTO product_recommendations 
        (source_sku, recommended_sku, rank_order) 
        VALUES (:s, :r, :o)
    """)
    
    lote_datos = []
    # ... lógica de procesamiento por cada libro ...
    
    # Inserción por lotes cada 100 libros
    if (idx + 1) % 100 == 0 and lote_datos:
        conn.execute(sql_insert, lote_datos)
        lote_datos.clear()
```

> **Optimizaciones de escritura:**
> - `TRUNCATE TABLE`: Limpia completamente las recomendaciones anteriores antes de regenerar, garantizando consistencia.
> - `engine.begin()`: Maneja la transacción automáticamente (commit si éxito, rollback si excepción).
> - **Inserción por lotes (batch):** Cada 100 libros se ejecuta un `INSERT` masivo en lugar de uno por registro, reduciendo significativamente la latencia de red y la carga sobre MySQL.

[CAPTURA: Terminal mostrando la barra de progreso de `tqdm` durante el procesamiento del catálogo completo]

[CAPTURA: Tabla `product_recommendations` en MySQL mostrando registros generados con `source_sku`, `recommended_sku` y `rank_order`]

---

## 7. Integración del Sistema

### 7.1 Impacto en la base de datos compartida

El script `book_recommender.py` opera como un **proceso batch independiente** que se ejecuta bajo demanda (manualmente o mediante una tarea programada). Su único punto de integración con el resto del sistema es la **base de datos MySQL compartida**:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Script Python   │     │  Base de Datos   │     │  Backend         │
│  (IA/ML)         │────▶│  MySQL           │◀────│  Spring Boot     │
│                  │     │                  │     │                  │
│  Escribe en:     │     │  Tablas:         │     │  Lee de:         │
│  product_        │     │  - books         │     │  product_        │
│  recommendations │     │  - product_      │     │  recommendations │
└──────────────────┘     │    recommendations│     └────────┬─────────┘
                         └──────────────────┘              │
                                                           │ REST API
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Frontend        │
                                                  │  React           │
                                                  │                  │
                                                  │  Muestra:        │
                                                  │  "Libros         │
                                                  │   Recomendados"  │
                                                  └──────────────────┘
```

### 7.2 Flujo de integración

1. **El script de IA** lee la tabla `books` y escribe en `product_recommendations`.
2. **El backend Spring Boot** expone un endpoint REST que consulta `product_recommendations` para un SKU dado.
3. **El frontend React** consume ese endpoint y renderiza la sección de "Libros Recomendados" en la vista de detalle de cada producto.

> **Desacoplamiento:** El script de Python no tiene dependencia directa del backend ni del frontend. Se comunica exclusivamente a través de la base de datos, lo que permite ejecutarlo de forma independiente sin afectar la disponibilidad del sistema web.

### 7.3 Ejecución del script

```bash
# Desde la raíz del proyecto, con el entorno virtual activado
python -m src.book_recommender
```

[CAPTURA: Terminal mostrando la ejecución completa del script con los mensajes de cada paso y el mensaje final "¡Éxito! Proceso finalizado y base de datos actualizada."]

[CAPTURA: Vista del frontend mostrando la sección de "Libros Recomendados" para un producto, alimentada por los datos generados por este script]

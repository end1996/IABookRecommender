# Implementación del Software — Módulo de Recomendaciones con IA

## 1. Descripción General de la Arquitectura IA

El proyecto se consolida como un conjunto de procesos *batch* modulares y desacoplados que interactúan directamente con la base de datos MySQL del sistema de inventario. Su funcionalidad se divide en tres pilares:

1. **Motor de Recomendaciones (`book_recommender.py`)**: Extrae el catálogo filtrado y utiliza una arquitectura MLOps cargando modelos predictivos exportados. Genera un ranking altamente optimizado (hasta 20 recomendaciones por producto) evaluando una fórmula de afinidad compuesta: similitud semántica pura (TF-IDF Vectorizer + `NearestNeighbors`), clasificación inteligente de categorías (`SGDClassifier`) y validación autoral, respetando siempre las restricciones de stock y el idioma de la obra.
2. **Enriquecimiento Resiliente (`book_data_enrichment.py`)**: Completa la metadata faltante (idiomas, descripciones) conectándose dinámicamente a APIs (Google Books, Open Library) y LLMs locales (Qwen vía LM Studio), asegurando alta disponibilidad a través de estrategias de *Multi-Layer Caching* y *Fuzzy Matching*.
3. **Higiene de Datos (`normalize_authors.py`)**: Módulo utilitario que sanitiza y estructura proactivamente los metadatos internos existentes (Title Case, inversión de nombres, eliminación de credenciales), maximizando el éxito del NLP.

El resultado de estas fases se escribe transaccionalmente en la base de datos (e.g. `product_recommendations`), sirviendo la data pre-computada y curada para que el backend y frontend la consuman al instante.

---

## 2. Tecnologías y Librerías Utilizadas

| Librería | Versión | Propósito |
|---|---|---|
| `pandas` | 3.0.1 | Manipulación y análisis de datos tabulares (DataFrames) |
| `sqlalchemy` | 2.0.48 | ORM y motor de conexión a bases de datos relacionales |
| `mysql-connector-python` | 9.6.0 | Driver nativo para la conexión MySQL desde Python |
| `scikit-learn` | 1.8.0 | Modelos de Machine Learning: `TfidfVectorizer`, `NearestNeighbors` y `SGDClassifier` |
| `joblib` | 1.5.3 | Carga persistente de modelos de IA exportados (Flujo MLOps) |
| `numpy` | 2.4.2 | Operaciones matemáticas vectorizadas (ordenamiento de índices) |
| `lingua-language-detector` | 2.1.1 | Detección NLP offline de alta velocidad para fases de enriquecimiento |
| `requests` | 2.32.5 | Peticiones HTTP REST para APIs externas y modelo local LM Studio |
| `tqdm` | 4.67.3 | Barras de progreso en terminal para monitorizar el procesamiento |
| `colorama` | 0.4.6 | Colores en la salida de terminal (compatibilidad Windows) |
| `torch` / `sentence-transformers` | 2.4.1 / 5.4.0 | Aceleración matemática (DirectML) y NLP avanzado, exclusivos para entorno de investigación (I+D) |

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

Los datos en bruto se normalizan aplicando múltiples capas de limpieza para evitar *overfitting* (sobreajuste): se limpian etiquetas HTML, se usan expresiones regulares (regex) para remover jerga corporativa y promocional, y se **eliminan dinámicamente los nombres de los autores** del texto para evitar que el motor solo recomiende libros del mismo escritor. Luego, se construye el texto base para la vectorización:

```python
print("2. Procesando textos y normalizando...")
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

# ... lógica de limpieza regex y eliminación dinámica de autores ...
df['Texto_IA_Limpio'] = df['description'].apply(limpiar_html)

# IMPORTANTE: La categoría NO se incluye en el corpus textual del TF-IDF.
# El autor explícito TAMPOCO, ya que ambos tienen su propio peso en la fórmula del score compuesto.
df['Texto_IA'] = df['title'] + " " + df['Texto_IA_Limpio']
```

> **Decisión de diseño:** El vectorizador se alimenta exclusivamente de la "narrativa" pura (título y descripción). Excluir explícitamente la categoría y el autor del corpus TF-IDF evita el emparejamiento erróneo causado por compartir la misma etiqueta estática de género o el mismo nombre de autor repetido (ruido sintáctico), ya que estas dimensiones operan algorítmicamente en su propio componente de la fórmula de afinidad.

### 6.3 Paso 3 — Carga del Pipeline MLOps y Vectorización (Núcleo de IA)

A diferencia de modelos dinámicos que se entrenan al vuelo, el sistema en producción emplea un flujo de trabajo **MLOps**. Esto significa que los modelos pesados se entrena en un entorno dedicado (como Jupyter/Colab) y se exportan para ser cargados en producción vía `joblib`. Se cargan dos componentes principales:
- **SGDClassifier:** Un modelo entrenado para asignar automáticamente categorías faltantes, eliminando la dependencia de un archivo JSON estático.
- **TfidfVectorizer:** El vectorizador configurado con *stopwords* bilingües. 

En lugar de usar `fit_transform()`, el script utiliza explícitamente `.transform()` para aplicar la transformación instantánea:

```python
print("3. Cargando modelos pre-entrenados y vectorizando...")

# Carga de modelos exportados
clf_categorias = joblib.load(MODELO_CLASIFICADOR_PATH)
vectorizer_preentrenado = joblib.load(MODELO_VECTORIZADOR_PATH)

# Aplicar modelo SGDClassifier (clasificación de categorías automatizada)
# ... lógica predictiva ...

# Vectorización con vocabulario estático
tfidf_matrix = vectorizer_preentrenado.transform(df['Texto_IA'])
```

> **Parámetros del vectorizador exportado:**
> El modelo original fue configurado con `max_df=0.80`, `min_df=2`, e inyección de *stopwords* bilingües personalizadas para filtrar términos narrativos comunes ("historia", "mundo"). Al usar `.transform()`, se asegura una consistencia algorítmica total entre las pruebas y el servidor de producción.

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

---

## 8. Pipeline de Enriquecimiento (Data Enrichment)

Junto al motor de recomendaciones, el archivo `book_data_enrichment.py` maneja la optimización del catálogo implementando prácticas de alta resiliencia y mitigación de errores.

### 8.1 Mitigación de Throttling y Resiliencia de API
Debido a las estrictas cuotas de servicios como la API de Google Books (Errores 503), el sistema usa una doble estrategia de redundancia:
- **Fallback a Open Library:** Si Google Books no responde o agota su cuota, se realiza una búsqueda secundaria asíncrona hacia Open Library (fuente gratuita sin límites duros).
- **Control de Calidad (Fuzzy Matching):** En lugar de confiar a ciegas en el ISBN, se verifica la similitud del título usando algoritmos *fuzzy*. Si hay "contaminación de ISBN" (ej: ISBN pertenece a otro libro o idioma), la API es descartada sin detener el flujo global (*batch process*).

### 8.2 Multi-layer Caching (Caché Estratégica en `output/`)
Para minimizar las peticiones de red redundantes, existen dos capas persistentes guardadas dinámicamente en el directorio de `output/`:
- **Caché a nivel de Campo (Field-level Caching):** Archivos como `google_books_no_description.txt` y `google_books_no_author.txt` que previenen consultas repetidas sobre metadatos que sabemos que ya fallaron anteriormente en ser provistos por la API.
- **Caché Negativa (Negative Caching):** Si un ISBN es inválido o se comprueba que no existe en las APIs externas, el script guarda este "estado de inexistencia" en `google_books_not_found.txt`.
- **Mismatches:** Las colisiones y discrepancias algorítmicas entre el catálogo base y las APIs externas (por la validación de *Fuzzy Matching*) se exportan a un CSV de auditoría (`isbn_mismatches_pendientes.csv`).

---

## 9. Pipeline Utilitario: Normalización de Autores (`normalize_authors.py`)

Aparte del ciclo principal de recomendaciones y enriquecimiento, existe un flujo aislado destinado exclusivamente a la **higiene de datos**: el script `normalize_authors.py`. 
Este módulo recorre la base de datos buscando autores ya existentes y aplica una serie de limpiezas algorítmicas sin realizar llamadas a APIs externas:
- Inversiones `Apellido, Nombre` → `Nombre Apellido`.
- Remoción de credenciales académicas inyectadas como basura (`PhD`, `Dr.`, `MD`).
- Estandarización a *Title Case*.
- Detección de entidades nulas o "solo basura" (las exporta a un CSV para auditoría, pero no las borra automáticamente por seguridad).

Su ejecución está desacoplada porque es una tarea de mantenimiento puntual, permitiendo a los administradores generar un reporte `--dry-run --export-csv` antes de impactar los datos en la base principal.

---

## 10. Arquitectura de I+D y Estructura MLOps

El repositorio está fuertemente modularizado para asegurar que las pruebas de experimentación y desarrollo (I+D) no afecten al script de producción liviano.

- **`models/`**: Aloja los objetos serializados con `joblib` (`clasificador_categorias_v3.pkl`, `recommender_tfidf_v1.pkl`). El script de recomendación solo carga la inferencia y no re-entrena los pesos.
- **`research/`**: Espacio de *sandboxing*. Aquí se encuentra, por ejemplo, `clasificador_libros.py`, el script pesado utilizado originalmente para entrenar y re-clasificar datos en crudo apoyándose en el ecosistema PyTorch, aceleración GPU (AMD DirectML), y el modelo de vanguardia **BAAI/bge-m3** (con fusión de *Weighted Embeddings* divididos entre 65% Categoría y 35% Contenido). Todo esto ocurre en *offline research* y **no infla las dependencias** de la aplicación principal en `src/`.

# 📚 IABookRecommender

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img alt="MySQL" src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img alt="SQLAlchemy" src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white"/>
</div>

<br/>

**IABookRecommender** es el núcleo de Inteligencia Artificial y de datos para el sistema de inventario de libros. Este proyecto en Python opera bajo una arquitectura MLOps para generar recomendaciones automatizadas de alta precisión mediante un *score* compuesto algorítmicamente (TF-IDF + SGDClassifier + Intersección de Autores). Además, implementa un robusto pipeline de enriquecimiento e higiene del catálogo, integrándose con APIs externas (Google Books, Open Library) y LLMs locales (LM Studio), fuertemente respaldado por estrategias de *Multi-Layer Caching*.

Funciona como un conjunto de **procesos batch independientes** que se comunican exclusivamente a través de la base de datos MySQL compartida, logrando un desacoplamiento total y seguro respecto al backend (Spring Boot) y al frontend (React).

---

## ✨ Características Principales

### 🧠 Motor de Recomendaciones (`book_recommender.py`)
Genera automáticamente un ranking de hasta 20 recomendaciones personalizadas por libro basado en afinidad semántica.
- **Flujo MLOps y Modelos Pre-entrenados:** En producción, carga modelos pre-entrenados exportados vía `joblib` desde la carpeta `/models/` (TF-IDF Vectorizer y SGDClassifier), utilizando `.transform()` para asegurar consistencia 1:1 con el entorno de entrenamiento, mientras mantiene el modelo `NearestNeighbors` de forma dinámica para adaptarse al nuevo stock en tiempo real.
- **Clasificador SGDClassifier:** Emplea una asignación automatizada de categorías basada en un pipeline de aprendizaje automático (SGDClassifier) con *fallback strategies*, dejando atrás el mapeo estático de JSON.
- **Vectorización Pura:** Se extraen dinámicamente los nombres de autores y etiquetas corporativas del texto. El vectorizador **excluye explícitamente** la categoría y autor del corpus TF-IDF, alimentándose solo del título y descripción limpia para evaluar exclusivamente afinidad narrativa.
- **Score Compuesto Ponderado:** Tras vectorizar, el modelo combina las distancias: contenido semántico TF-IDF (45%), afinidad binaria de categoría predicha por el SGDClassifier (45%), y coincidencias de autores (10%).

### 🌐 Pipeline de Enriquecimiento (`book_data_enrichment.py`)
Mejora y normaliza la metadata del catálogo en la base de datos dividiéndose en 3 fases:
- **Detección de Idioma:** Offline ultrarrápido usando la librería `lingua` (identifica ~89% de libros instantáneamente), completando con Qwen MLLM (vía LM Studio localmente) para títulos ambiguos o bilingües.
- **Enriquecimiento Multinivel y Resiliencia:** Para descripciones y autores faltantes, consulta la **API de Google Books** apoyándose en un fallback gratuito hacia **Open Library** para no agotar cuotas. Además, implementa **Title Fuzzy Matching** para evitar contaminación de datos por ISBN erróneos.
- **Multi-Layer Caching:** Incorpora caché negativo (para ISBNs inexistentes) y caché a nivel de campo para reducir drásticamente la latencia, la cuota de red y proteger al sistema ante bloqueos (Throttling 503).
- **Normalización de Autores:** Estandariza atributos (Title Case, limpia "Dr.", etc.) y resuelve autores invertidos (`"Collins, Clifton"` -> `"Clifton Collins"`), utilizando IA Generativa (Qwen) como último recurso.

---

## 🏗️ Arquitectura del Sistema

```mermaid
graph LR
    A[IABookRecommender] -->|Extrae catalogos y escribe| DB[(MySQL)]
    DB -->|Lee de product_recommendations| C(Backend Spring Boot)
    C -->|Sincroniza Tienda| D(WooCommerce)
    C -->|API REST| E[Frontend React]
    style A fill:#3776AB,stroke:#fff,stroke-width:2px,color:#fff
```

### Integración
El script opera exclusivamente en nivel de base de datos sin levantar puertos:
1. **Lectura:** Extrae listados del inventario validando stock, y atributos nulos.
2. **Procesamiento:** Limpia el HTML a nivel texto e inicia vectorización (en el motor de recomendaciones).
3. **Escritura Transaccional:** `TRUNCATE` de datos anteriores seguidos de *batch inserts* para salvaguardar latencia y reducir carga de BD.

---

## 🛠️ Stack Tecnológico

| Librería / Dependencia | Propósito |
|-------------------------|-----------|
| `scikit-learn` / `joblib` | Modelaje clásico (TF-IDF, NearestNeighbors, SGDClassifier) y carga de modelos serializados (MLOps). |
| `pandas` / `numpy` | Procesamiento vectorizado masivo de catálogos y manipulación analítica de datos tabulares. |
| `SQLAlchemy` / `mysql-connector-python` | Motor ORM transaccional y driver nativo para operaciones directas a la base de datos de producción. |
| `lingua-language-detector` | Detección NLP de idiomas de altísima velocidad y fiabilidad offline (fase 1 del pipeline). |
| `requests` | Comunicación REST asíncrona hacia Google Books, Open Library y el motor local LM Studio. |
| `tqdm` / `colorama` | Control de status visual mediante barras de progreso en terminal e impresión a color. |
| `torch` / `sentence-transformers` | Ecosistema PyTorch y BGE-M3 acelerado por GPU (DirectML) utilizado exclusivamente en entorno I+D (`research/`). |

---

## 📂 Estructura del Directorio

```text
IABookRecommender/
├── .env                          # Variables de base de datos y llaves de acceso
├── .env.example                  # Plantilla de muestra segura
├── README.md                     # Documentación principal
├── requirements.txt              # Conjunto exhaustivo de las librerías a instalar
├── config/
│   ├── __init__.py
│   └── settings.py               # Exposición y unificación lógica del .env
├── docs/
│   └── IMPLEMENTACION.md         # Documentación técnica (fases, pesos del score, etc.)
├── models/                       # Modelos MLOps pre-entrenados (e.g., SGDClassifier, TF-IDF .pkl)
├── output/                       # Cachés negativas de APIs, mismatches e insumos dinámicos
├── research/                     # Scripts de I+D (ej: clasificador de libros con PyTorch BGE-M3)
└── src/
    ├── __init__.py
    ├── book_recommender.py       # Algoritmo de recomendaciones NLP
    └── book_data_enrichment.py   # Pipeline de refinamiento de metadata del catálogo
```

---

## ⚙️ Instalación y Configuración Local

1. **Clona el repositorio** en tu entorno:
   ```bash
   git clone <repo-url>
   cd IABookRecommender
   ```

2. **Creación del Entorno Virtual (Aislamiento de dependencias)**:
   Es vital usar un entorno virtual para que las dependencias de *research* (como PyTorch) no contaminen tu entorno global de Python.
   ```bash
   # Crear el entorno virtual llamado 'venv'
   python -m venv venv
   
   # Activar el entorno (Windows)
   venv\Scripts\activate
   ```

3. **Instalación de paquetes requeridos**:
   Instala las librerías exactas que garantizan el funcionamiento de producción y de I+D.
   ```bash
   pip install -r requirements.txt
   ```
   > **Tip de Mantenimiento:** Si durante el desarrollo instalás nuevos paquetes (por ejemplo, `pip install nueva_libreria`), debés registrar ese cambio actualizando la lista oficial del proyecto con el siguiente comando:
   > ```bash
   > pip freeze > requirements.txt
   > ```

4. **Settear Base de Datos vía variables de entorno**:
   - Clona `.env.example` en formato `.env`
   - Configúralo con los datos locales:
   ```env
   DB_HOST=localhost
   DB_USER=root
   DB_PASSWORD=tu_contraseña_secreta
   DB_NAME=sistema_inventario
   DB_PORT=3306
   CANTIDAD_RECOMENDACIONES=20
   ```

*(Nota técnica: Para el uso del pipeline de enriquecimiento apoyado en IA Generativa, asegúrate de tener ejecutando tu instancia en **LM Studio** local expuesto al loopback local `localhost:1234` usando un modelo capaz como **Qwen2.5-14b-instruct**).*

---

## 🚀 Uso y Ejecución

Se sugiere ejecutar los módulos desde un terminal con el **Entorno Virtual habilitado**.

### Modo 1: Motor de Recomendaciones Puras
Genera desde cero hasta el topo paramétrico (20 recoms) los rankings insertando a la BD.
```bash
python -m src.book_recommender
```

### Modo 2: Enriquecimiento Total del Catálogo Inteligente
Se encarga de levantar y reparar la metadata del catálogo existente.
```bash
# Ejecutar todas las fases (Idioma -> Descripciones -> Autores)
python -m src.book_data_enrichment 

# Ejecutar componentes modulares de manera particular
python -m src.book_data_enrichment --phase language
python -m src.book_data_enrichment --phase description
python -m src.book_data_enrichment --phase author

# Testealo primero. Modo Simulación sin afectar la BD (--dry-run)
python -m src.book_data_enrichment --dry-run
python -m src.book_data_enrichment --limit 20 --dry-run
```

### Modo 3: Utilidad de Normalización de Autores (Script Aislado)
Este script `normalize_authors.py` es una utilidad de mantenimiento que limpia la data existente de autores (corrige capitalización "Title Case", elimina apodos entre comillas, remueve credenciales académicas como "PhD" y reordena formatos tipo "Apellido, Nombre" a "Nombre Apellido"). Opera como un flujo aislado que *sólo* muta la base de datos si hay mejoras evidentes, pero no rellena autores nulos (eso lo hace el Modo 2).
```bash
# 1. Analizar cuánta basura hay y exportar un reporte para auditar
python -m src.normalize_authors --dry-run --export-csv

# 2. Aplicar la limpieza estructural a toda la base de datos de producción
python -m src.normalize_authors
```

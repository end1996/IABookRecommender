# 📚 IABookRecommender

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img alt="MySQL" src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img alt="SQLAlchemy" src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white"/>
</div>

<br/>

**IABookRecommender** es el motor de Inteligencia Artificial para el sistema de inventario de libros. Este proyecto en Python genera recomendaciones automatizadas usando Procesamiento de Lenguaje Natural (NLP) y enriquece la metadata del catálogo integrándose con LLMs locales (LM Studio) y APIs externas (Google Books).

Funciona como un **proceso batch independiente** que se comunica exclusivamente a través de la base de datos MySQL compartida, sin acoplamiento directo con el backend (Spring Boot) ni el frontend (React).

---

## ✨ Características Principales

### 🧠 Motor de Recomendaciones (`book_recommender.py`)
Genera automáticamente un ranking de hasta 20 recomendaciones personalizadas por libro basado en afinidad semántica.
- **Modelo NLP Clásico:** Utiliza **TF-IDF** (Term Frequency–Inverse Document Frequency) combinado con **KNN** (NearestNeighbors con distancia coseno) en lugar de un LLM externo para mayor eficiencia y menor costo.
- **Score Compuesto Ponderado:** Considera el contenido de las descripciones (45%), la similitud de categorías normalizadas (45%), y coincidencias de autores (10%).
- **Tolerancia Bilingüe:** Stopwords especializadas tanto en inglés como en español.
- **Filtros Adaptables:** Evita recomendar libros sin stock o de distintas versiones del mismo título para WooCommerce.

### 🌐 Pipeline de Enriquecimiento (`book_data_enrichment.py`)
Mejora y normaliza la metadata del catálogo en la base de datos dividiéndose en 3 fases:
- **Detección de Idioma:** Offline ultrarrápido usando la librería `lingua` (identifica ~89% de libros instantáneamente), completando con Qwen MLLM (vía LM Studio localmente) para títulos ambiguos o bilingües.
- **Enriquecimiento de Descripciones:** Consulta API de Google Books (vía ISBN) con un *fallback* a LM Studio que genera descripciones orgánicas de 2-3 frases comerciales de alto impacto.
- **Normalización de Autores:** Limpia atributos redundantes (como "PhD", "Dr."), estandariza a *Title Case* y resuelve nombres invertidos (ej: `"Collins, Clifton"` -> `"Clifton Collins"`), rellenando vacíos adicionales vía API/LLM.

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
| `scikit-learn` | Modelaje y distanciación del Coseno (TF-IDF, NearestNeighbors) |
| `pandas` / `numpy` | Procesamiento vectorizado de DataFrames masivos |
| `SQLAlchemy` | Motor ORM transaccional usando `mysql-connector-python` |
| `lingua` | Detección NLP de idiomas de altísima velocidad y fiabilidad offline |
| `requests` | Llamado sin bloqueos a endpoints REST HTTP (LM Studio / Google) |
| `tqdm` | Control local del status de procesos batch mediante barras TUI |

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
└── src/
    ├── __init__.py
    ├── book_recommender.py       # Algoritmo de recomendaciones NLP
    └── book_data_enrichment.py   # Pipeland de refinamiento de metadata del catálogo
```

---

## ⚙️ Instalación y Configuración Local

1. **Clona el repositorio** en tu entorno:
   ```bash
   git clone <repo-url>
   cd IABookRecommender
   ```

2. **Crea e inicializa el entorno virtual de Python** (Instrucciones para Windows):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Carga los paquetes dependientes**:
   ```bash
   pip install -r requirements.txt
   ```

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
# Ejecutar todas las fases
python -m src.book_data_enrichment 

# Ejecutar componentes modulares de manera particular
python -m src.book_data_enrichment --phase language
python -m src.book_data_enrichment --phase description
python -m src.book_data_enrichment --phase author

# Testealo primero. Modo Simulación sin afectar la BD (--dry-run)
python -m src.book_data_enrichment --dry-run
python -m src.book_data_enrichment --limit 20 --dry-run
```

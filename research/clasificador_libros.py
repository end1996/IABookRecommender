import pandas as pd
import torch

# --- PARCHE MÁGICO PARA AMD DIRECTML ---
# Engañamos a PyTorch para evitar el bug del 'version_counter' en la GPU
torch.inference_mode = torch.no_grad 
# ---------------------------------------
import pandas as pd
import torch_directml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def limpiar_html(texto):
    """Elimina etiquetas HTML de un texto."""
    if pd.isna(texto):
        return ""
    # Reemplaza los tags por un espacio para evitar que palabras se peguen (ej: </p><p>Hola ->  Hola)
    return re.sub(r'<[^>]+>', ' ', str(texto)).strip()

# --- 1. CONFIGURACIÓN ---
RUTA_EXCEL_ENTRADA = "best_stock_books_export_2026-04-12T22_05_40.887Z.xlsx" # Cambia esto por tu archivo real
RUTA_EXCEL_SALIDA = "catalogo_clasificado.xlsx"

CATEGORIAS_PERMITIDAS = [
    "Ficción General", "Ciencia Ficción y Fantasía", "Romance", 
    "Misterio y Thriller", "Terror y Suspenso",
    "Literatura Juvenil", "Literatura Infantil", 
    "Cómics, Manga y Novela Gráfica", 
    "Biografías y Memorias", "Historia", "Ciencia y Tecnología", 
    "Salud y Bienestar", "Negocios y Economía", "Desarrollo Personal", 
    "Educación y Referencia", "Religión y Espiritualidad", 
    "Cocina y Gastronomía", "Viajes y Cultura", "Arte y Diseño", 
    "Hogar y Jardinería", "Otros / No clasificable"
]

# --- 2. CARGAR EL MODELO BGE-M3 ---
print("Conectando con la GPU AMD...")
device = torch_directml.device()
model = SentenceTransformer(
    'BAAI/bge-m3', 
    device=device, 
    model_kwargs={"attn_implementation": "eager"} # <--- El parche antimemoria
)
model.max_seq_length = 512

print(f"Modelo cargado en: {device} con límite de {model.max_seq_length} tokens")

# --- 3. VECTORIZAR LAS CATEGORÍAS LIMPIAS ---
print("Calculando embeddings de las categorías...")
# Añadimos un pequeño prefijo para darle contexto al modelo
categorias_texto = [f"Este libro trata sobre: {cat}" for cat in CATEGORIAS_PERMITIDAS]
vectores_categorias = model.encode(categorias_texto, normalize_embeddings=True)

# --- 4. PREPARAR LOS DATOS DEL EXCEL ---
print(f"Leyendo archivo {RUTA_EXCEL_ENTRADA}...")
df = pd.read_excel(RUTA_EXCEL_ENTRADA)

# Limpiar valores nulos para evitar errores
df['Title'] = df['Title'].fillna('')
df['Description'] = df['Description'].fillna('')
df['Category'] = df['Category'].fillna('') # Usaremos la categoría sucia como pista

print("Limpiando etiquetas HTML de las descripciones con regex...")
df['Description'] = df['Description'].apply(limpiar_html)

# Limpiar espacios dobles que deja el regex y saltos de línea
df['Description'] = df['Description'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Textos separados
# A las categorías vacías les ponemos una palabra neutra para que no rompa el cálculo
textos_categoria = df['Category'].apply(lambda x: f"Categoría: {x}" if str(x).strip() else "Categoría: Desconocida").tolist()
textos_contenido = ("Título: " + df['Title'] + ". Descripción: " + df['Description']).tolist()

# --- 5. VECTORIZACIÓN PESADA (WEIGHTED EMBEDDINGS) ---
print(f"Vectorizando {len(df)} categorías originales...")
vectores_cat = model.encode(textos_categoria, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

print(f"Vectorizando {len(df)} títulos y descripciones...")
vectores_cont = model.encode(textos_contenido, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

print("Realizando fusión vectorial (65% Categoría, 35% Contenido)...")
# Aquí está la magia matemática. Si un libro dice "Self-Help" (65%), 
# no importa si el título dice "Muerte, Sangre y Secretos" (35%), el vector final apuntará a Autoayuda.
PESO_CATEGORIA = 0.65
PESO_CONTENIDO = 0.35

vectores_fusionados = (vectores_cat * PESO_CATEGORIA) + (vectores_cont * PESO_CONTENIDO)

# Es importante volver a normalizar el vector resultante después de sumarlo
vectores_fusionados = vectores_fusionados / np.linalg.norm(vectores_fusionados, axis=1, keepdims=True)

print("Calculando similitud y asignando la categoría ganadora...")
# Ahora comparamos nuestro vector super-preciso contra tus 21 categorías limpias
similitudes = cosine_similarity(vectores_fusionados, vectores_categorias)

# Obtener el índice de la categoría con el puntaje más alto para cada libro
indices_ganadores = np.argmax(similitudes, axis=1)

# Obtener también el puntaje de confianza (del 0 al 1)
puntajes_confianza = np.max(similitudes, axis=1)

# Asignar los resultados al DataFrame original
df['Categoria_Limpia'] = [CATEGORIAS_PERMITIDAS[i] for i in indices_ganadores]
df['Nivel_de_Confianza'] = puntajes_confianza

# Lógica condicional: Si el modelo está muy inseguro (ej. < 0.35 de similitud), lo mandamos a "Otros"
UMBRAL_CONFIANZA = 0.35
df.loc[df['Nivel_de_Confianza'] < UMBRAL_CONFIANZA, 'Categoria_Limpia'] = "Otros / No clasificable"

# --- 6. GUARDAR RESULTADOS ---
print("Guardando resultados...")
# Reordenar columnas para ver el resultado fácilmente al principio
columnas_finales = ['Title', 'Categoria_Limpia', 'Nivel_de_Confianza', 'Category', 'Description', 'Language'] 
# Agregar cualquier otra columna original que tuvieras
columnas_finales += [col for col in df.columns if col not in columnas_finales and col != 'texto_a_analizar']

df[columnas_finales].to_excel(RUTA_EXCEL_SALIDA, index=False)
print(f"¡Proceso terminado! Archivo guardado como {RUTA_EXCEL_SALIDA}")
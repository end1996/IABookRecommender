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

# LM Studio (modelo local para generación de descripciones)
# LM Studio expone una API compatible con OpenAI en el puerto 1234
# Recomendado: Qwen 2.5 14B (sin thinking mode, mejor que 3.5 para tareas simples)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
# LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen-2.5-14b")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

# Google Books API (opcional, sin key funciona con cuota reducida)
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "AIzaSyCzJjU-pkb6t56T7vSXyiZ0wAFSQu4iRQw")
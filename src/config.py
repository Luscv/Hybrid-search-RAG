import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")  # troque no .env para comparar modelos
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL) # juiz do RAGAS
#Facilitar as configurações para aplicar em todo o modelo
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DIST = float(os.getenv("MAX_DISTANCE", "0.34")) # ponto de partida: recalibre para a sua base com src/calibrate_threshold.py

#configuração extra de latencia como segurança pro modelo gratuito
# Retry limitado: falha clara em ~2 min no pior caso, em vez de travar por vários minutos
# quando o modelo está sobrecarregado (503/504 observados no plano gratuito)
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=30_000,  # ms: tempo máximo de cada tentativa
        retry_options=types.HttpRetryOptions(
            attempts=3,       # 1 chamada + 2 novas tentativas
            initial_delay=5,  # configurando um delay maior (padrao=1) para evitar de consumir todas as tentativas antes de aceitar requisições
            max_delay=20,     # teto da espera entre tentativas (s)
        ),
    ),
)
# settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    except Exception:
        pass

_load_env()

def _to_bool(s, default=False):
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "on")

MOCK_MODE      = _to_bool(os.getenv("MOCK_MODE"), False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Default to local GROBID so it's never None
GROBID_URL     = os.getenv("GROBID_URL") or "http://127.0.0.1:8071"
REQUIRE_GROBID = _to_bool(os.getenv("REQUIRE_GROBID"), True)

USE_LLM        = _to_bool(os.getenv("USE_LLM"), True)
PORT           = int(os.getenv("PORT", "8001"))

DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

print(f"[startup] MOCK_MODE={MOCK_MODE}  GROBID_URL={GROBID_URL}  USE_LLM={USE_LLM}")

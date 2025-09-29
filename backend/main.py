# main.py  — full file (forces OPENAI_API_KEY override; adds flags; health)

from __future__ import annotations

import os
import uuid
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from extractors import extract_first_pass


# ---------------- Settings ----------------

class Settings(BaseSettings):
    # existing
    MOCK_MODE: bool = False
    PORT: int = 8001
    GROBID_URL: Optional[str] = None

    # LLM-related (safe even if unused)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    USE_LLM: bool = False

    # don’t crash on unknown keys in .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
    )

settings = Settings()


# ---------------- App ----------------

app = FastAPI(title="Trial Abstraction Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# in-memory store
DB: Dict[str, Dict[str, Any]] = {}


def _paths(doc_id: str) -> Dict[str, str]:
    return {
        "pdf": os.path.join(UPLOAD_DIR, f"{doc_id}.pdf"),
    }


# --------------- Models for reviews -----------------

class ReviewerPayload(BaseModel):
    study: Optional[Dict[str, Any]] = None
    arms: Optional[Dict[str, Any]] = None


# --------------- Health helpers ---------------------

def _check_grobid_alive() -> bool:
    if not settings.GROBID_URL:
        return False
    try:
        url = settings.GROBID_URL.rstrip("/") + "/api/isalive"
        r = requests.get(url, timeout=2)
        return r.ok and (r.text.strip().lower() == "true")
    except Exception:
        return False


# --------------- Startup log ------------------------

print(f"[startup] MOCK_MODE={settings.MOCK_MODE}  GROBID_URL={settings.GROBID_URL}  USE_LLM={settings.USE_LLM}")


# --------------- Routes -----------------------------

@app.get("/api/health")
def health():
    """Frontend polls this to show a clear status banner."""
    grobid_alive = _check_grobid_alive()
    llm_configured = bool(settings.OPENAI_API_KEY)
    return {
        "mock_mode": settings.MOCK_MODE,
        "grobid_url": settings.GROBID_URL,
        "grobid_alive": grobid_alive,
        "use_llm": settings.USE_LLM,
        "llm_configured": llm_configured,
        "openai_model": settings.OPENAI_MODEL if llm_configured else None,
        "api_port": settings.PORT,
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    p = _paths(doc_id)
    with open(p["pdf"], "wb") as out:
        out.write(await file.read())

    DB[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "draft": None,
        "reviewA": None,
        "reviewB": None,
        "final": None,
    }
    return DB[doc_id]


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: str):
    return DB.get(doc_id, {"error": "not found", "doc_id": doc_id})


@app.post("/api/extract/{doc_id}")
def extract(doc_id: str):
    doc = DB.get(doc_id)
    if not doc:
        return {"error": "not found", "doc_id": doc_id}

    # --- FORCE the runtime env to use .env values (overrides any stale machine/user var) ---
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY  # hard override
    os.environ["USE_LLM"] = "true" if settings.USE_LLM else "false"
    if settings.OPENAI_MODEL:
        os.environ["OPENAI_MODEL"] = settings.OPENAI_MODEL
    # tiny debug so you can see which one is in play, safely masked:
    masked = (os.environ.get("OPENAI_API_KEY") or "")[:8]
    print(f"[llm] env OPENAI_API_KEY now starts with: {masked}...")

    p = _paths(doc_id)
    # Run the extractor (which may use GROBID and/or LLM depending on config)
    draft = extract_first_pass(p["pdf"], grobid_url=settings.GROBID_URL)

    # Prefer flags from extractor; fallback to notes only if missing
    flags_in = (draft or {}).get("_flags") or {}
    notes_lower = ((draft or {}).get("study", {}).get("notes", "") or "").lower()

    grobid_on = flags_in.get("grobid")
    llm_used = flags_in.get("llm")

    if grobid_on is None:
        grobid_on = ("grobid=on" in notes_lower) or ("grobid tei" in notes_lower)
    if llm_used is None:
        llm_used = ("grobid + llm" in notes_lower) or ("llm=true" in notes_lower) or ("llm" in notes_lower)

    draft["_flags"] = {
        "grobid": bool(grobid_on),
        "llm": bool(llm_used),
    }

    doc["draft"] = draft
    return doc


@app.post("/api/review/{doc_id}/{which}")
def save_review(doc_id: str, which: str, payload: ReviewerPayload):
    doc = DB.get(doc_id)
    if not doc:
        return {"error": "not found", "doc_id": doc_id}

    which = which.upper()
    if which == "A":
        doc["reviewA"] = payload.model_dump()
    elif which == "B":
        doc["reviewB"] = payload.model_dump()
    else:
        return {"error": "which must be A or B"}

    return {"ok": True, "doc_id": doc_id, "which": which}

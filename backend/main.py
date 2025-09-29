# main.py  (full file)

from __future__ import annotations

import os
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from extractors import extract_first_pass

# ---------------- Settings ----------------

class Settings(BaseSettings):
    MOCK_MODE: bool = False
    OPENAI_API_KEY: Optional[str] = None
    PORT: int = 8001
    GROBID_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "forbid"

settings = Settings()

# ---------------- App ----------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    study: Optional[Dict[str, Dict[str, Any]]] = None
    arms: Optional[Dict[str, Dict[str, Any]]] = None

# --------------- Startup log ------------------------

print(f"[startup] MOCK_MODE={settings.MOCK_MODE}  GROBID_URL={settings.GROBID_URL}")

# --------------- Routes -----------------------------

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

    p = _paths(doc_id)
    draft = extract_first_pass(p["pdf"], grobid_url=settings.GROBID_URL)
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

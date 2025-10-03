# main.py
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import settings  # <-- uses backend/settings.py you already have
from extractors import extract_first_pass

# --- DB
from db import engine, SessionLocal, Base
from models import Document, Draft, Review, FinalExtract
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

import requests

# ---------------- App ----------------
app = FastAPI(title="Trial Abstraction Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders & tables exist
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "cache"), exist_ok=True)
Base.metadata.create_all(bind=engine)

# In-memory map for quick UI state (DB is source of truth)
MEM: Dict[str, Dict[str, Any]] = {}

def _paths(doc_id: str) -> Dict[str, str]:
    return {"pdf": os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")}

def _db() -> Session:
    return SessionLocal()

def _check_grobid_alive() -> bool:
    url = settings.GROBID_URL
    if not url:
        return False
    try:
        r = requests.get(url.rstrip("/") + "/api/isalive", timeout=3)
        return r.ok and (r.text.strip().lower() == "true")
    except Exception:
        return False

# ---------------- Schemas ----------------
class ReviewerPayload(BaseModel):
    study: Optional[Dict[str, Any]] = None
    arms: Optional[Dict[str, Any]] = None

class UpsertDoc(BaseModel):
    doc_id: str
    filename: Optional[str] = None

class SaveJSON(BaseModel):
    doc_id: str
    payload: Dict[str, Any]
    meta: Optional[str] = None

class SaveReview(BaseModel):
    doc_id: str
    reviewer: str
    payload: Dict[str, Any]

class ExtractBody(BaseModel):
    doc_id: str

# ---------------- Startup log ----------------
print(f"[startup] MOCK_MODE={settings.MOCK_MODE}  GROBID_URL={settings.GROBID_URL}  USE_LLM={settings.USE_LLM}")

# ---------------- Health ----------------
@app.get("/api/health")
def health():
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

# ---------------- Core: Upload ----------------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    p = _paths(doc_id)
    with open(p["pdf"], "wb") as out:
        out.write(await file.read())

    MEM[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "draft": None,
        "reviewA": None,
        "reviewB": None,
        "final": None,
    }

    # Ensure a Document row exists
    try:
        with _db() as db:
            if not db.get(Document, doc_id):
                db.add(Document(id=doc_id, filename=file.filename))
                db.commit()
    except SQLAlchemyError as e:
        print(f"[db] upload save skipped: {e}")

    print(f"[upload] doc_id={doc_id} filename={file.filename}")
    return MEM[doc_id]

@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: str):
    return MEM.get(doc_id, {"error": "not found", "doc_id": doc_id})

# ---------------- Core: Extract (path param) ----------------
@app.post("/api/extract/{doc_id}")
def extract_path(doc_id: str, force_ocr: bool = False):
    return _do_extract(doc_id, force_ocr=force_ocr)

# ---------------- Core: Extract (JSON body) ----------------
@app.post("/api/extract")
def extract_body(body: ExtractBody, force_ocr: bool = False):
    return _do_extract(body.doc_id, force_ocr=force_ocr)

def _do_extract(doc_id: str, force_ocr: bool = False):
    doc = MEM.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"doc_id not found: {doc_id}")

    # LLM env (force)
    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["USE_LLM"] = "true" if settings.USE_LLM else "false"
    if settings.OPENAI_MODEL:
        os.environ["OPENAI_MODEL"] = settings.OPENAI_MODEL
    masked = (os.environ.get("OPENAI_API_KEY") or "")[:8]
    print(f"[extract] start doc_id={doc_id}  model={settings.OPENAI_MODEL}  key={masked}...")

    # Hard-require GROBID
    if not settings.GROBID_URL:
        raise HTTPException(status_code=503, detail="GROBID_URL not configured; extractor will not run without it.")
    if not _check_grobid_alive():
        raise HTTPException(status_code=503, detail=f"GROBID not reachable at {settings.GROBID_URL}; start it and retry.")

    p = _paths(doc_id)
    draft = extract_first_pass(p["pdf"], grobid_url=settings.GROBID_URL, force_ocr=force_ocr)

    # Normalize flags if missing
    flags_in = (draft or {}).get("_flags") or {}
    grobid_on = bool(flags_in.get("grobid"))
    llm_used = bool(flags_in.get("llm"))

    draft["_flags"] = {"grobid": grobid_on, "llm": llm_used}
    doc["draft"] = draft

    # Persist the draft row
    try:
        with _db() as db:
            drow = Draft(document_id=doc_id, payload=draft, source="grobid+llm" if (grobid_on and llm_used) else ("grobid" if grobid_on else "local"))
            db.add(drow)
            db.commit()
    except SQLAlchemyError as e:
        print(f"[db] draft save skipped: {e}")

    print(f"[extract] done doc_id={doc_id}  flags={draft.get('_flags')}")
    return doc

# ---------------- Reviews (in-memory quick save for UI) ----------------
@app.post("/api/review/{doc_id}/{which}")
def save_review_in_memory(doc_id: str, which: str, payload: ReviewerPayload):
    doc = MEM.get(doc_id)
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

# ---------------- Persistence API (DB) ----------------
@app.post("/api/doc/upsert")
def upsert_document(body: UpsertDoc):
    try:
        with _db() as db:
            doc = db.get(Document, body.doc_id)
            if not doc:
                db.add(Document(id=body.doc_id, filename=body.filename or None))
            else:
                if body.filename and body.filename != doc.filename:
                    doc.filename = body.filename
            db.commit()
        return {"ok": True}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doc/save-draft")
def save_draft(body: SaveJSON):
    try:
        with _db() as db:
            doc = db.get(Document, body.doc_id)
            if not doc:
                doc = Document(id=body.doc_id)
                db.add(doc)
                db.flush()
            d = Draft(document_id=doc.id, payload=body.payload, source=body.meta or None)
            db.add(d)
            db.commit()
            return {"ok": True, "draft_id": d.id}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doc/save-review")
def save_review_db(body: SaveReview):
    try:
        with _db() as db:
            doc = db.get(Document, body.doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            r = Review(document_id=doc.id, reviewer=body.reviewer, payload=body.payload)
            db.add(r)
            db.commit()
            return {"ok": True, "review_id": r.id}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doc/save-final")
def save_final(body: SaveJSON):
    try:
        with _db() as db:
            doc = db.get(Document, body.doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            f = FinalExtract(document_id=doc.id, payload=body.payload)
            db.add(f)
            db.commit()
            return {"ok": True, "final_id": f.id}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/doc/{doc_id}/latest")
def get_latest(doc_id: str):
    try:
        with _db() as db:
            doc = db.get(Document, doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            latest_draft = db.execute(
                select(Draft).where(Draft.document_id == doc_id).order_by(Draft.id.desc())
            ).scalars().first()
            latest_final = db.execute(
                select(FinalExtract).where(FinalExtract.document_id == doc_id).order_by(FinalExtract.id.desc())
            ).scalars().first()
            return {
                "doc_id": doc.id,
                "filename": doc.filename,
                "latest_draft": latest_draft.payload if latest_draft else None,
                "latest_final": latest_final.payload if latest_final else None,
            }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

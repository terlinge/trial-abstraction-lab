# routes_persist.py
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from db import SessionLocal
from models import Document, Draft, Review, FinalExtract

router = APIRouter(prefix="/api/db", tags=["db"])

# ---------------- deps ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- helpers -------------
def _shape_doc(db: Session, doc_id: str) -> Dict[str, Any]:
    doc = db.get(Document, doc_id)
    if not doc:
        return {"error": "not found", "doc_id": doc_id}

    latest_draft = db.execute(
        select(Draft).where(Draft.document_id == doc_id).order_by(desc(Draft.id))
    ).scalars().first()
    reviewA = db.execute(
        select(Review).where(Review.document_id == doc_id, Review.reviewer == "A").order_by(desc(Review.id))
    ).scalars().first()
    reviewB = db.execute(
        select(Review).where(Review.document_id == doc_id, Review.reviewer == "B").order_by(desc(Review.id))
    ).scalars().first()
    final = db.execute(
        select(FinalExtract).where(FinalExtract.document_id == doc_id).order_by(desc(FinalExtract.id))
    ).scalars().first()

    return {
        "doc_id": doc_id,
        "filename": doc.filename,
        "draft": latest_draft.payload if latest_draft else None,
        "reviewA": reviewA.payload if reviewA else None,
        "reviewB": reviewB.payload if reviewB else None,
        "final": final.payload if final else None,
    }

# ---------------- schemas -------------
class SaveDraftIn(BaseModel):
    payload: Dict[str, Any]
    source: Optional[str] = None
    filename: Optional[str] = None

class SaveReviewIn(BaseModel):
    reviewer: str   # "A" or "B"
    payload: Dict[str, Any]

class SaveFinalIn(BaseModel):
    payload: Dict[str, Any]

# ---------------- routes --------------
@router.get("/list")
def list_docs(db: Session = Depends(get_db)):
    rows = db.execute(select(Document).order_by(desc(Document.created_at))).scalars().all()
    return [{"doc_id": d.id, "filename": d.filename, "created_at": str(d.created_at)} for d in rows]

@router.get("/doc/{doc_id}")
def fetch_doc(doc_id: str, db: Session = Depends(get_db)):
    return _shape_doc(db, doc_id)

@router.post("/save_draft/{doc_id}")
def save_draft(doc_id: str, body: SaveDraftIn, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        # create doc if missing (helpful if you come here directly)
        doc = Document(id=doc_id, filename=body.filename or f"{doc_id}.pdf")
        db.add(doc)
    d = Draft(document_id=doc_id, payload=body.payload, source=body.source or "api")
    db.add(d)
    db.commit()
    return {"ok": True, "doc_id": doc_id, "draft_id": d.id}

@router.post("/save_review/{doc_id}")
def save_review(doc_id: str, body: SaveReviewIn, db: Session = Depends(get_db)):
    if body.reviewer not in ("A", "B"):
        return {"error": "reviewer must be 'A' or 'B'"}
    if not db.get(Document, doc_id):
        return {"error": "document not found", "doc_id": doc_id}
    r = Review(document_id=doc_id, reviewer=body.reviewer, payload=body.payload)
    db.add(r)
    db.commit()
    return {"ok": True, "doc_id": doc_id, "review_id": r.id}

@router.post("/save_final/{doc_id}")
def save_final(doc_id: str, body: SaveFinalIn, db: Session = Depends(get_db)):
    if not db.get(Document, doc_id):
        return {"error": "document not found", "doc_id": doc_id}
    f = FinalExtract(document_id=doc_id, payload=body.payload)
    db.add(f)
    db.commit()
    return {"ok": True, "doc_id": doc_id, "final_id": f.id}


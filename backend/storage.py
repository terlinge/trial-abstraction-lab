# backend/storage.py
import os
import json
from typing import Any, Dict

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

def ensure_dirs(paths: Dict[str, str]) -> None:
    for p in paths.values():
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

def get_doc_paths(doc_id: str) -> Dict[str, str]:
    doc_folder = os.path.join(DOCS_DIR, doc_id)
    return {
        "folder": doc_folder,
        "doc": os.path.join(doc_folder, "doc.json"),
        "review_A": os.path.join(doc_folder, "review_A.json"),
        "review_B": os.path.join(doc_folder, "review_B.json"),
        "final": os.path.join(doc_folder, "final.json"),
        "pdf": os.path.join(UPLOADS_DIR, f"{doc_id}.pdf"),
    }

def save_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

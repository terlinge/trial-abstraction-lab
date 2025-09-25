# main.py
import os, uuid, shutil, json, math, re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic_settings import BaseSettings
from typing import Dict, Any
from models import ExtractionDraft, ReviewSubmission, AdjudicationSubmission  # (kept for type hints)
from storage import save_json, load_json, DATA_DIR, append_provenance, read_provenance
from extractors import extract_first_pass


# ---------------------- Settings ----------------------
class Settings(BaseSettings):
    MOCK_MODE: bool = True
    OPENAI_API_KEY: str | None = None
    PORT: int = 8000
    GROBID_URL: str | None = None   # <-- add this line

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")

settings = Settings()

# ---------------------- App / CORS ----------------------
app = FastAPI(title="Trial Abstraction API", version="0.2.0")

origins = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- FS Layout ----------------------
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DRAFTS_DIR  = os.path.join(DATA_DIR, "drafts")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
FINAL_DIR   = os.path.join(DATA_DIR, "final")
PROV_DIR    = os.path.join(DATA_DIR, "provenance")

for d in [UPLOADS_DIR, DRAFTS_DIR, REVIEWS_DIR, FINAL_DIR, PROV_DIR]:
    os.makedirs(d, exist_ok=True)

def doc_paths(doc_id: str) -> Dict[str, str]:
    return {
        "pdf":     os.path.join(UPLOADS_DIR, f"{doc_id}.pdf"),
        "draft":   os.path.join(DRAFTS_DIR,  f"{doc_id}.json"),
        "reviewA": os.path.join(REVIEWS_DIR, f"{doc_id}_A.json"),
        "reviewB": os.path.join(REVIEWS_DIR, f"{doc_id}_B.json"),
        "final":   os.path.join(FINAL_DIR,   f"{doc_id}.json"),
        "prov":    os.path.join(PROV_DIR,    f"{doc_id}.jsonl"),
    }

# ---------------------- Routes ----------------------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    paths = doc_paths(doc_id)
    # ensure uploads dir exists
    os.makedirs(os.path.dirname(paths["pdf"]), exist_ok=True)
    with open(paths["pdf"], "wb") as f:
        shutil.copyfileobj(file.file, f)
    # provenance (4-arg form used earlier in your app)
    append_provenance(doc_id, "system", "upload", {"filename": file.filename})
    return {"doc_id": doc_id, "filename": file.filename}

@app.post("/api/extract/{doc_id}")
async def extract(doc_id: str):
    paths = doc_paths(doc_id)
    if not os.path.exists(paths["pdf"]):
        return PlainTextResponse("PDF not found; upload first.", status_code=404)

    if settings.MOCK_MODE:
        # simple mock draft
        draft = {
            "study": {
                "title": "Example Trial of Drug A vs Placebo in Condition Y",
                "nct_id": "NCT01234567",
                "pmid": "12345678",
                "doi": "10.1000/example.doi",
                "year": 2024,
                "design": "Randomized, double-blind, placebo-controlled",
                "country": "USA",
                "condition": "Condition Y",
                "notes": "Draft extracted by mock mode."
            },
            "arms": [
                {"arm_id": "drug_a", "label": "Drug A", "n_randomized": 100},
                {"arm_id": "placebo", "label": "Placebo", "n_randomized": 98},
            ],
            "outcomes": [
                {
                    "name": "Primary outcome (stub)",
                    "type": "continuous",
                    "timepoints": [{"label": "end of treatment", "measures": []}],
                    "subgroups": []
                }
            ]
        }
        save_json(paths["draft"], draft)
        append_provenance(doc_id, "system", "extract", {"mode": "mock"})
    else:
        # local PDF parsing path
        pdf_path = paths["pdf"]
        try:
            draft = extract_first_pass(pdf_path)
        except Exception as e:
            draft = {
                "study": {"title": f"Extraction error: {e}"},
                "arms": [],
                "outcomes": []
            }
        save_json(paths["draft"], draft)
        append_provenance(doc_id, "system", "extract", {"mode": "local", "pdf": os.path.basename(pdf_path)})

    return {"doc_id": doc_id}

@app.get("/api/doc/{doc_id}")
async def get_doc(doc_id: str):
    paths = doc_paths(doc_id)
    out = {
        "doc_id": doc_id,
        "has_pdf": os.path.exists(paths["pdf"]),
        "draft":   load_json(paths["draft"],   None),
        "reviewA": load_json(paths["reviewA"], None),
        "reviewB": load_json(paths["reviewB"], None),
        "final":   load_json(paths["final"],   None),
        "provenance_count": len(read_provenance(doc_id)),
    }
    return out

@app.post("/api/review/{doc_id}")
async def submit_review(
    doc_id: str,
    reviewer: str = Form(...),
    data: str = Form(...),
    verified: str = Form("{}"),
    evidence: str = Form("{}"),
):
    reviewer = reviewer.upper()
    assert reviewer in ("A", "B"), "reviewer must be A or B"
    payload      = json.loads(data)
    verified_map = json.loads(verified) if verified else {}
    evidence_map = json.loads(evidence) if evidence else {}
    paths = doc_paths(doc_id)
    path  = paths["reviewA"] if reviewer == "A" else paths["reviewB"]
    save_json(path, {"reviewer": reviewer, "data": payload, "verified": verified_map, "evidence": evidence_map})
    append_provenance(doc_id, f"reviewer{reviewer}", {"event": "review_saved", "keys": list(payload.keys())[:5], "verified_ct": sum(1 for v in verified_map.values() if v)})
    return {"ok": True, "doc_id": doc_id, "reviewer": reviewer}

def _flatten(d: dict, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, val in enumerate(v):
                key_i = f"{new_key}[{i}]"
                if isinstance(val, (dict, list)):
                    items.extend(_flatten(val, key_i, sep=sep).items())
                else:
                    items.append((key_i, val))
        else:
            items.append((new_key, v))
    return dict(items)

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

@app.get("/api/conflicts/{doc_id}")
async def get_conflicts(doc_id: str):
    paths = doc_paths(doc_id)
    a = load_json(paths["reviewA"], {"data": {}}) or {"data": {}}
    b = load_json(paths["reviewB"], {"data": {}}) or {"data": {}}
    fa = _flatten(a.get("data", {}))
    fb = _flatten(b.get("data", {}))
    keys = sorted(set(fa.keys()) | set(fb.keys()))
    conflicts = []
    for k in keys:
        va = fa.get(k, None)
        vb = fb.get(k, None)
        if va != vb:
            conflicts.append({
                "key": k, "A": va, "B": vb,
                "type": "numeric" if _num(va) is not None and _num(vb) is not None else "text"
            })
    return {"doc_id": doc_id, "conflicts": conflicts}

def _apply_resolution(obj: Any, res: Dict[str, Any]):
    def set_path(root, path, value):
        tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
        cur = root
        for i, t in enumerate(tokens):
            last = (i == len(tokens)-1)
            if t.startswith('[') and t.endswith(']'):
                idx = int(t[1:-1])
                if not isinstance(cur, list):
                    return
                while len(cur) <= idx:
                    cur.append({})
                if last:
                    cur[idx] = value
                else:
                    if not isinstance(cur[idx], (dict, list)):
                        cur[idx] = {}
                    cur = cur[idx]
            else:
                key = t
                if last:
                    if isinstance(cur, dict):
                        cur[key] = value
                else:
                    if isinstance(cur, dict):
                        if key not in cur or not isinstance(cur[key], (dict, list)):
                            cur[key] = {}
                        cur = cur[key]
                    else:
                        return
    import copy
    root = copy.deepcopy(obj)
    for k, v in res.items():
        set_path(root, k, v)
    return root

@app.post("/api/adjudicate/{doc_id}")
async def adjudicate(doc_id: str, adjudicator: str = Form(...), resolution: str = Form(...), notes: str = Form("")):
    paths = doc_paths(doc_id)
    draft = load_json(paths["draft"], {}) or {}
    res_map = json.loads(resolution) if resolution else {}
    final = _apply_resolution(draft, res_map)
    save_json(paths["final"], {"adjudicator": adjudicator, "final": final, "notes": notes})
    # provenance diff
    f0 = _flatten(draft)
    f1 = _flatten(final)
    diffs = [{"key": k, "old": f0.get(k), "new": f1.get(k)} for k in sorted(set(f0)|set(f1)) if f0.get(k) != f1.get(k)]
    append_provenance(doc_id, adjudicator, {"event": "adjudicated", "diff_count": len(diffs), "changed_keys": [d["key"] for d in diffs[:10]]})
    return {"ok": True, "doc_id": doc_id}

@app.get("/api/provenance/{doc_id}")
async def get_provenance(doc_id: str):
    rows = read_provenance(doc_id)
    return {"doc_id": doc_id, "events": rows}

# --- Export for binary outcomes (netmeta-like contrasts CSV) ---
@app.get("/api/export/contrasts/{doc_id}", response_class=PlainTextResponse)
async def export_contrasts(doc_id: str):
    paths = doc_paths(doc_id)
    final = load_json(paths["final"], None)
    if not final:
        return "error: finalize first"
    data = final.get("final", {})
    outcomes = data.get("outcomes", [])
    arms = data.get("arms", [])
    arm_names = {a.get("arm_id"): (a.get("name") or a.get("treatment") or a.get("arm_id")) for a in arms}
    rows = [["study","treat1","treat2","TE","seTE","outcome_id","effect_type","timepoint_days"]]
    study = data.get("study", {}).get("title", f"doc_{doc_id}")
    for idx, out in enumerate(outcomes):
        if out.get("type") != "dichot":
            continue
        ad = out.get("arm_data", [])
        if len(ad) < 2:
            continue
        a, b = ad[0], ad[1]
        e1, n1 = a.get("events"), a.get("total")
        e0, n0 = b.get("events"), b.get("total")
        if any(v in (None, "") for v in [e1, n1, e0, n0]):
            continue
        rr = (e1 / max(n1, 1)) / (e0 / max(n0, 1) if e0 else 1e-9)
        rr = max(rr, 1e-9)
        TE = math.log(rr)
        def inv(x): return 1.0 / max(x, 1e-9)
        seTE = math.sqrt(inv(e1) - inv(n1) + inv(e0) - inv(n0))
        rows.append([study, arm_names.get(a.get("arm_id"), "A"), arm_names.get(b.get("arm_id"), "B"), TE, seTE, f"OUT{idx+1}", "logRR", out.get("timepoint_days")])
    return "\n".join(",".join([str(c) for c in r]) for r in rows)

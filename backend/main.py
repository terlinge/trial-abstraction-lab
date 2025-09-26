import os, uuid, shutil, json, math, re
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic_settings import BaseSettings

from models import ExtractionDraft, ReviewSubmission, AdjudicationSubmission  # (kept for future use)
from storage import save_json, load_json, DATA_DIR, append_provenance, read_provenance
from extractors import extract_first_pass


# ----------------------------
# Settings (.env backed)
# ----------------------------
class Settings(BaseSettings):
    MOCK_MODE: bool = True
    OPENAI_API_KEY: str | None = None
    PORT: int = 8000
    # add this so .env can contain GROBID_URL without pydantic errors
    GROBID_URL: str | None = None

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")


settings = Settings()


# ----------------------------
# FastAPI app & CORS
# ----------------------------
app = FastAPI(title="Trial Abstraction API", version="0.3.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"  # local development convenience
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Paths helper
# ----------------------------
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DRAFTS_DIR = os.path.join(DATA_DIR, "drafts")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
FINAL_DIR = os.path.join(DATA_DIR, "final")
PROV_DIR = os.path.join(DATA_DIR, "provenance")


def doc_paths(doc_id: str) -> Dict[str, str]:
    return {
        "pdf": os.path.join(UPLOADS_DIR, f"{doc_id}.pdf"),
        "draft": os.path.join(DRAFTS_DIR, f"{doc_id}.json"),
        "reviewA": os.path.join(REVIEWS_DIR, f"{doc_id}_A.json"),
        "reviewB": os.path.join(REVIEWS_DIR, f"{doc_id}_B.json"),
        "final": os.path.join(FINAL_DIR, f"{doc_id}.json"),
        "prov": os.path.join(PROV_DIR, f"{doc_id}.jsonl"),
    }


# ----------------------------
# API: Upload
# ----------------------------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Save the PDF and create an initial JSON shell *at the draft path*.
    (This fixes the earlier bug where we mistakenly saved using the doc_id.)
    """
    doc_id = str(uuid.uuid4())
    paths = doc_paths(doc_id)

    # ensure directories exist
    os.makedirs(os.path.dirname(paths["pdf"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["draft"]), exist_ok=True)

    # save the PDF
    with open(paths["pdf"], "wb") as f:
        shutil.copyfileobj(file.file, f)

    # create initial doc shell and save at the *draft path*
    doc = {"doc_id": doc_id, "filename": file.filename}
    save_json(paths["draft"], doc)

    append_provenance(doc_id, "system", "upload", {"filename": file.filename})
    return {"doc_id": doc_id, "filename": file.filename}


# ----------------------------
# API: Extract (mock or real)
# ----------------------------
@app.post("/api/extract/{doc_id}")
async def extract(doc_id: str):
    """
    Load the draft JSON from its file, add a draft extraction,
    and save back to the same draft path.
    """
    paths = doc_paths(doc_id)
    pdf_path = paths["pdf"]

    # Load existing shell (created during /api/upload). If missing, make one.
    doc = load_json(paths["draft"], {"doc_id": doc_id, "filename": None})

    if settings.MOCK_MODE:
        # Mock data path
        draft = {
            "study": {
                "title": "Example Trial of Drug A vs Placebo in Condition Y",
                "nct_id": None,
                "pmid": None,
                "doi": None,
                "year": None,
                "design": "Randomized, double-blind, placebo-controlled",
                "country": None,
                "condition": "Condition Y",
                "notes": "MOCK_MODE=true"
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
        append_provenance(doc_id, "extract", "mock", {"note": "MOCK_MODE"})
    else:
        # Real local/GROBID parse (extractors.py handles both using grobid_url parameter)
        grobid_url = settings.GROBID_URL  # e.g. http://localhost:8071
        try:
            draft = extract_first_pass(pdf_path, grobid_url=grobid_url)
            append_provenance(
                doc_id,
                "extract",
                "local",
                {"grobid": bool(grobid_url), "pdf": os.path.basename(pdf_path)},
            )
        except Exception as e:
            # Keep UI responsive even if parsing fails
            draft = {
                "study": {"title": f"Extraction error: {e}"},
                "arms": [],
                "outcomes": []
            }
            append_provenance(doc_id, "extract", "error", {"message": str(e)})

    # update & save
    doc["draft"] = draft
    save_json(paths["draft"], doc)

    return {"doc_id": doc_id}


# ----------------------------
# API: Get doc
# ----------------------------
@app.get("/api/doc/{doc_id}")
async def get_doc(doc_id: str):
    paths = doc_paths(doc_id)
    out = {
        "doc_id": doc_id,
        "has_pdf": os.path.exists(paths["pdf"]),
        "draft": load_json(paths["draft"], None),
        "reviewA": load_json(paths["reviewA"], None),
        "reviewB": load_json(paths["reviewB"], None),
        "final": load_json(paths["final"], None),
        "provenance_count": len(read_provenance(doc_id)),
    }
    return out


# ----------------------------
# API: Review save
# ----------------------------
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
    payload = json.loads(data)
    verified_map = json.loads(verified) if verified else {}
    evidence_map = json.loads(evidence) if evidence else {}
    paths = doc_paths(doc_id)
    path = paths["reviewA"] if reviewer == "A" else paths["reviewB"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(
        path,
        {
            "reviewer": reviewer,
            "data": payload,
            "verified": verified_map,
            "evidence": evidence_map,
        },
    )
    append_provenance(
        doc_id,
        f"reviewer{reviewer}",
        "review_saved",
        {"keys": list(payload.keys())[:5], "verified_ct": sum(1 for v in verified_map.values() if v)},
    )
    return {"ok": True, "doc_id": doc_id, "reviewer": reviewer}


# ----------------------------
# Conflicts helper
# ----------------------------
def _flatten(d: dict, parent_key: str = "", sep: str = "."):
    items: List[Tuple[str, Any]] = []
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


# ----------------------------
# API: Conflicts
# ----------------------------
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
            conflicts.append(
                {
                    "key": k,
                    "A": va,
                    "B": vb,
                    "type": "numeric" if _num(va) is not None and _num(vb) is not None else "text",
                }
            )
    return {"doc_id": doc_id, "conflicts": conflicts}


# ----------------------------
# API: Adjudicate
# ----------------------------
def _apply_resolution(obj: Any, res: Dict[str, Any]):
    def set_path(root, path, value):
        tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
        cur = root
        for i, t in enumerate(tokens):
            last = i == len(tokens) - 1
            if t.startswith("[") and t.endswith("]"):
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
async def adjudicate(
    doc_id: str,
    adjudicator: str = Form(...),
    resolution: str = Form(...),
    notes: str = Form(""),
):
    paths = doc_paths(doc_id)
    draft = load_json(paths["draft"], {})
    res_map = json.loads(resolution) if resolution else {}
    final = _apply_resolution(draft, res_map)
    os.makedirs(os.path.dirname(paths["final"]), exist_ok=True)
    save_json(paths["final"], {"adjudicator": adjudicator, "final": final, "notes": notes})

    # provenance diff
    f0 = _flatten(draft)
    f1 = _flatten(final)
    diffs = [
        {"key": k, "old": f0.get(k), "new": f1.get(k)}
        for k in sorted(set(f0) | set(f1))
        if f0.get(k) != f1.get(k)
    ]
    append_provenance(
        doc_id,
        adjudicator,
        "adjudicated",
        {"diff_count": len(diffs), "changed_keys": [d["key"] for d in diffs[:10]]},
    )
    return {"ok": True, "doc_id": doc_id}


# ----------------------------
# API: Provenance
# ----------------------------
@app.get("/api/provenance/{doc_id}")
async def get_provenance(doc_id: str):
    rows = read_provenance(doc_id)
    return {"doc_id": doc_id, "events": rows}


# ----------------------------
# API: Export contrasts (binary outcomes)
# ----------------------------
@app.get("/api/export/contrasts/{doc_id}", response_class=PlainTextResponse)
async def export_contrasts(doc_id: str):
    paths = doc_paths(doc_id)
    final = load_json(paths["final"], None)
    if not final:
        return "error: finalize first"
    data = final.get("final", {})
    outcomes = data.get("outcomes", [])
    arms = data.get("arms", [])
    arm_names = {
        a.get("arm_id"): (a.get("name") or a.get("treatment") or a.get("arm_id"))
        for a in arms
    }
    rows = [
        ["study", "treat1", "treat2", "TE", "seTE", "outcome_id", "effect_type", "timepoint_days"]
    ]
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
        rows.append(
            [
                study,
                arm_names.get(a.get("arm_id"), "A"),
                arm_names.get(b.get("arm_id"), "B"),
                TE,
                seTE,
                f"OUT{idx+1}",
                "logRR",
                out.get("timepoint_days"),
            ]
        )
    out_lines = []
    for r in rows:
        out_lines.append(",".join([str(c) for c in r]))
    return "\n".join(out_lines)

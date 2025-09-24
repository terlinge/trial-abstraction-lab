import os, json, datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROV_DIR = os.path.join(DATA_DIR, "provenance")

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_json(path: str, obj):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def append_provenance(doc_id: str, actor: str, action: str, details: dict):
    os.makedirs(PROV_DIR, exist_ok=True)
    path = os.path.join(PROV_DIR, f"{doc_id}.jsonl")
    rec = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "doc_id": doc_id,
        "actor": actor,
        "action": action,
        "details": details,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def read_provenance(doc_id: str):
    path = os.path.join(PROV_DIR, f"{doc_id}.jsonl")
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows
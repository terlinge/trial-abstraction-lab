
# Trial Abstraction Web App (Local Prototype, extended)

Generated: 2025-09-23T13:06:35.982281Z

This prototype now includes:
- Richer schema (methods, eligibility, centers, funding, outcome definitions)
- **Per-field verification checkboxes** and **evidence pointers**
- **Provenance log** (upload, draft creation, review saves, adjudication diffs)
- Minimal **export to netmeta contrasts** for binary outcomes

## Prereqs
- Python 3.10+
- Node 18+

## Windows (recommended) — Conda quickstart and smoke test

If you're on Windows, using Miniforge/conda with Python 3.11 avoids many compiled build issues (spaCy, thinc, numpy). Example steps:

1. Install Miniforge (https://github.com/conda-forge/miniforge/releases) and open a new PowerShell.
2. Create and activate the env:

```powershell
Push-Location $env:USERPROFILE\Miniforge3\Scripts; .\conda.exe create -n trial-abst python=3.11 -c conda-forge -y; .\conda.exe activate trial-abst; Pop-Location
```

3. Install system binaries (poppler, tesseract, java) and key Python packages:

```powershell
Push-Location $env:USERPROFILE\Miniforge3\Scripts; .\conda.exe run -n trial-abst conda install -c conda-forge poppler tesseract openjdk -y; Pop-Location
Push-Location $env:USERPROFILE\Miniforge3\Scripts; .\conda.exe run -n trial-abst python -m pip install -r backend/requirements.txt; Pop-Location
# If you need spaCy model:
Push-Location $env:USERPROFILE\Miniforge3\Scripts; .\conda.exe run -n trial-abst python -m spacy download en_core_web_sm; Pop-Location
```

4. Run the extraction smoke test (writes `backend/out_draft.json`):

```powershell
Push-Location $env:USERPROFILE\Miniforge3\Scripts; .\conda.exe run -n trial-abst python backend/run_extract.py; Pop-Location
```

Notes:
- The runner will print a short summary (title/authors/_flags). If OCR is required, pass `force_ocr=True` (the runner already sets it).
- Keep `backend/data/` directory in `.gitignore` to avoid committing large files.


## 1) Start the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # keep MOCK_MODE=true for now
uvicorn main:app --reload --port 8000
```

API: http://localhost:8000

## 2) Start the frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

App: http://localhost:5173

## 3) Use the app (step-by-step)
1. **Upload** any PDF (mock mode ignores contents, just stores it).
2. Click **Generate Draft (Mock)** → populates a realistic draft JSON.
3. Reviewer **A**: Edit fields, check **Verified**, add **Evidence pointer**, **Save Review**.
4. Reviewer **B**: Do the same independently.
5. Click **Load Conflicts** → choose A/B per field.
6. **Finalize** → writes `backend/data/final/<doc_id>.json`, logs an adjudication event.
7. **View Provenance** → small modal (alert) with the provenance entries.
8. **Export Contrasts (CSV)** → after finalization, downloads binary outcome contrasts.

## Where things are stored
- Uploads: `backend/data/uploads/<doc_id>.pdf`
- Draft: `backend/data/drafts/<doc_id>.json`
- Reviews: `backend/data/reviews/<doc_id>_A.json`, `<doc_id>_B.json`
- Final: `backend/data/final/<doc_id>.json`
- Provenance: `backend/data/provenance/<doc_id>.jsonl`

## Next steps
- Plug in real PDF parsing + LLM extraction (replace /api/extract with OCR + model call).
- Add auth/roles (Extractor A/B, Adjudicator).
- Add continuous/time-to-event effect exports.
- Add RoB UI and summary exports.

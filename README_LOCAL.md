
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

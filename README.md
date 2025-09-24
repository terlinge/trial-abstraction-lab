\# Trial Abstraction Lab



\## Dev setup

\*\*Backend\*\*

1\. `cd backend`

2\. `.\\.venv\\Scripts\\Activate.ps1`

3\. `python -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1`



\*\*Frontend\*\*

1\. `cd frontend`

2\. `npm install`

3\. `npm run dev` (opens on http://localhost:5173 or 5175)



Set `MOCK\_MODE=false` in `backend/.env` to enable local PDF parsing.




import re
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber

def read_pdf_text(pdf_path, max_pages=5):
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)

def try_simple_tables(pdf_path, max_pages=5):
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            tables = page.extract_tables(
                table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            if not tables:
                continue
            for t in tables:
                rows = [r for r in t if any((c or "").strip() for c in r)]
                if len(rows) >= 2:
                    out.append({"page": i + 1, "rows": rows})
    return out

def extract_first_pass(pdf_path):
    text = read_pdf_text(pdf_path)
    tables = try_simple_tables(pdf_path)

    # title guess = first non-empty line
    first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title_guess = first_lines[0] if first_lines else "Untitled trial"

    # crude arm detection
    arm_names = []
    n_by_arm = {}
    for tbl in tables:
        header = [(c or "").strip() for c in tbl["rows"][0]]
        if any(re.search(r"(arm|group|treat|placebo|dose)", h, re.I) for h in header):
            for cell in header:
                if re.search(r"(arm|group|treat|placebo|dose)", cell, re.I):
                    arm_names.append(cell)
        for row in tbl["rows"][1:]:
            for j, cell in enumerate(row):
                s = (cell or "")
                if re.search(r"\b(N|n)\s*=?\s*\d+", s):
                    m = re.search(r"(\d{2,5})", s)
                    if m:
                        key = header[j] if j < len(header) and header[j] else f"arm{j+1}"
                        n_by_arm[key] = int(m.group(1))

    # dedupe/clean
    cleaned = []
    for a in arm_names:
        a = re.sub(r"\s+", " ", a).strip()
        if a and a not in cleaned:
            cleaned.append(a)
    arm_names = cleaned or ["Arm A", "Arm B"]

    arms = []
    for a in arm_names:
        arm_id = re.sub(r"[^A-Za-z0-9]+", "_", a).strip("_").lower() or "arm"
        arms.append({"arm_id": arm_id, "label": a, "n_randomized": n_by_arm.get(a)})

    outcome = {
        "name": "Primary outcome (stub)",
        "type": "continuous",
        "timepoints": [
            {"label": "end of treatment", "measures": []}
        ],
        "subgroups": []
    }

    return {
        "study": {
            "title": title_guess,
            "design": None,
            "registration": None,
            "population": None
        },
        "arms": arms,
        "outcomes": [outcome],
        "notes": "Local PDF heuristic draft. Please review."
    }

# backend/extractors.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional libs (installed earlier). We'll fail gracefully if missing.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore
try:
    import pdfplumber  # table extraction
except Exception:
    pdfplumber = None  # type: ignore


# ---------------------------
# PDF text helpers
# ---------------------------

def read_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    """
    Grab plain text from the first few pages using PyMuPDF.
    Falls back to empty string if anything goes wrong.
    """
    p = Path(pdf_path)
    if not p.exists() or fitz is None:
        return ""

    chunks: List[str] = []
    try:
        with fitz.open(str(p)) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                chunks.append(page.get_text("text") or "")
    except Exception:
        return ""
    return "\n".join(chunks)


def get_pdf_meta_title(pdf_path: str) -> Optional[str]:
    """Read the PDF metadata title if available."""
    p = Path(pdf_path)
    if not p.exists() or fitz is None:
        return None
    try:
        with fitz.open(str(p)) as doc:
            meta = doc.metadata or {}
            t = meta.get("title")
            if t and isinstance(t, str) and t.strip():
                return t.strip()
    except Exception:
        pass
    return None


def try_simple_tables(pdf_path: str, max_pages: int = 4) -> List[Dict[str, Any]]:
    """
    Very simple table pass using pdfplumber.
    Returns: [{"page": <int>, "rows": [["cell","cell",...], ...]}, ...]
    """
    out: List[Dict[str, Any]] = []
    p = Path(pdf_path)
    if not p.exists() or pdfplumber is None:
        return out

    try:
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                try:
                    tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                        }
                    )
                except Exception:
                    tables = None
                for t in (tables or []):
                    rows = [r for r in t if any(c and str(c).strip() for c in r)]
                    if len(rows) >= 2:
                        out.append({"page": i + 1, "rows": rows})
    except Exception:
        pass
    return out


# ---------------------------
# Smarter title detection
# ---------------------------

_JOURNAL_OR_HEADER = re.compile(
    r"""(?ix)
    \b(
      new\ england\ journal|nejm|jama|lancet|bmj|nature|science|cell|
      annals\ of\ internal\ medicine|plos|springer|wiley|elsevier|
      oxford\ academic|sage|taylor\ &\ francis|wolters\ kluwer|
      clinicaltrials\.?gov|medrxiv|biorxiv|open\ access|
      original\ article|research\ article|review\ article|editorial|
      volume|vol\.|issue|no\.|pp\b|copyright|©|
      american\ college\ of\ rheumatology|
      arthritis\s*&\s*rheumatism|arthritis\sand\srheumatism
    )\b
    |doi\s*10\.
    |https?://
    """,
)

_SECTION_WORDS = re.compile(r"(?i)\b(abstract|introduction|background|methods|results|discussion|conclusions?)\b")

def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # reasonable size & multi-word
    if len(s) < 15 or len(s) > 220 or "  " in s or "...." in s:
        return False
    if _JOURNAL_OR_HEADER.search(s):
        return False
    if _SECTION_WORDS.search(s):
        return False
    # avoid author-ish lines (many commas + initials)
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.){1,2}", s) and s.count(",") >= 1:
        return False
    # need lower-case letters and at least one space
    return any(c.islower() for c in s) and (" " in s)


def _continuation_line(line: str) -> bool:
    """Should this line be joined to the previous title line?"""
    s = line.strip()
    if not _looks_like_title(s):
        return False
    # Disallow if clearly a new section/author/copyright line
    if _SECTION_WORDS.search(s) or _JOURNAL_OR_HEADER.search(s):
        return False
    # Avoid author-like patterns
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.){1,2}", s) and s.count(",") >= 1:
        return False
    return True


def guess_title(lines: List[str], meta_title: Optional[str] = None) -> str:
    # normalize and keep first ~120 non-empty lines
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    head = lines[:120]

    # find the first candidate
    for i, ln in enumerate(head[:40]):  # look near top first
        if _looks_like_title(ln):
            # join subsequent wrapped lines (e.g., 2–3 lines)
            pieces = [ln]
            j = i + 1
            while j < len(head) and len(" ".join(pieces)) < 200 and _continuation_line(head[j]):
                pieces.append(head[j])
                j += 1
            title = " ".join(pieces).strip()
            if 15 <= len(title) <= 220:
                return title

    # fallback to any later candidate
    for i, ln in enumerate(head[40:]):
        if _looks_like_title(ln):
            return ln

    # finally use PDF metadata if it doesn't look like header
    if meta_title and _looks_like_title(meta_title):
        return meta_title.strip()

    return "Untitled trial"


# ---------------------------
# Main extractor
# ---------------------------

def extract_first_pass(pdf_path: str) -> Dict[str, Any]:
    """
    Heuristic, conservative first-pass extractor:
    - Smarter title guess that skips journal headers and joins wrapped lines
    - Detect NCT id / PMID / DOI / year in text
    - Infer arm labels & n from simple tables
    - Create a single stub outcome for reviewers
    """
    text = read_pdf_text(pdf_path)
    tables = try_simple_tables(pdf_path)

    # ---- Title guess (smarter)
    first_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    meta_title = get_pdf_meta_title(pdf_path)
    title_guess = guess_title(first_lines, meta_title)

    # ---- Simple IDs
    nct = None
    m = re.search(r"\bNCT\d{8}\b", text or "", re.I)
    if m:
        nct = m.group(0)

    pmid = None
    m = re.search(r"\bPMID[:\s]*([0-9]{4,})\b", text or "", re.I)
    if m:
        pmid = m.group(1)

    doi = None
    m = re.search(r"\b10\.\d{4,9}/\S+\b", text or "")
    if m:
        doi = m.group(0).rstrip(").,;")

    # crude year near the top
    year = None
    m = re.search(r"(19|20)\d{2}", (text or "")[:3000])
    if m:
        try:
            y = int(m.group(0))
            if 1900 <= y <= 2035:
                year = y
        except Exception:
            pass

    # ---- Arms & N detection from the first suitable table
    arm_names: List[str] = []
    n_by_arm: Dict[str, int] = {}

    for tbl in tables:
        rows = tbl["rows"]
        if not rows or len(rows[0]) < 2:
            continue
        header = [(c or "").strip() for c in rows[0]]
        header_join = " ".join(header)

        if re.search(r"\b(arm|group|treatment|placebo|dose)\b", header_join, re.I):
            # Use header columns as arm hints
            for cell in header:
                if re.search(r"(arm|group|treatment|placebo|dose)", cell, re.I):
                    arm_names.append(cell.strip())

            # Look for numbers that could be N per arm
            for row in rows[1:]:
                for j, cell in enumerate(row):
                    s = str(cell or "")
                    if re.search(r"\bN\s*=?\s*\d+\b|\bn\s*=?\s*\d+\b", s, re.I) or re.search(r"\b\d+\b", s):
                        m = re.search(r"(\d{2,5})", s)
                        if m:
                            key = header[j] if j < len(header) else f"arm{j+1}"
                            val = int(m.group(1))
                            if 5 <= val <= 100000:
                                n_by_arm[key] = val
            break  # use the first suitable table only

    # Clean arm names
    arm_names = [a for a in dict.fromkeys([re.sub(r"\s+", " ", a) for a in arm_names]) if a]
    if not arm_names:
        arm_names = ["Arm A", "Arm B"]

    arms = []
    for a in arm_names:
        arm_id = re.sub(r"[^A-Za-z0-9]+", "_", a).strip("_").lower() or "arm"
        arms.append({
            "arm_id": arm_id,
            "label": a,
            "n_randomized": n_by_arm.get(a)  # may be None; reviewers can fill
        })

    # Minimal outcome stub; reviewers will populate measures
    outcome_stub = {
        "name": "Primary outcome (stub)",
        "type": "continuous",       # reviewers can change to 'dichot' later
        "timepoints": [{"label": "end of treatment", "measures": []}],
        "subgroups": []
    }

    return {
        "study": {
            "title": title_guess,
            "nct_id": nct,
            "pmid": pmid,
            "doi": doi,
            "year": year,
            "design": None,
            "country": None,
            "condition": None,
            "notes": "Draft via local PDF parsing (heuristic). Review carefully."
        },
        "arms": arms,
        "outcomes": [outcome_stub]
    }

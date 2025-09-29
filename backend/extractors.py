# extractors.py  (FULL FILE)
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from grobid_client import GrobidClient


def _read_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    """Read plain text from the first few pages via PyMuPDF."""
    out: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                out.append(page.get_text("text") or "")
    except Exception:
        return ""
    return "\n".join(out)


def _clean_doi(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return re.sub(r"[\s\)\]\}\.,;:]+$", "", s.strip())


def _heuristic_arms(*texts: str) -> List[Dict[str, Any]]:
    # Merge all text we have (TEI abstract + extracted PDF text)
    t = " ".join(texts).replace("\n", " ")
    arms: List[Dict[str, Any]] = []

    def add(arm_id: str, label: str):
        if arm_id not in [a["arm_id"] for a in arms]:
            arms.append({"arm_id": arm_id, "label": label, "n_randomized": None})

    # budesonide 3 mg/day or 3 mg
    if re.search(r"budesonide\s*3\W*mg(?:\s*/\s*day)?", t, flags=re.I):
        add("budesonide_3_mg", "budesonide 3 mg")

    # budesonide 9 mg/day or 9 mg
    if re.search(r"budesonide\s*9\W*mg(?:\s*/\s*day)?", t, flags=re.I):
        add("budesonide_9_mg", "budesonide 9 mg")

    # prednisolone 7.5 mg (/day optional; dot/space tolerated)
    if re.search(r"prednisolone\s*(?:7\.?5|7\W*5)\W*mg(?:\s*/\s*day)?", t, flags=re.I):
        add("prednisolone_7_5_mg", "prednisolone 7.5 mg")

    # placebo anywhere
    if re.search(r"\bplacebo\b", t, flags=re.I):
        add("placebo", "placebo")

    return arms


def _year_from_pdf(pdf_text: str) -> Optional[int]:
    """
    Prefer a 'masthead-like' year in the first ~1500 chars (Vol/No/pp/month names),
    otherwise pick the most common 4-digit year seen in the first 2 pages.
    """
    if not pdf_text:
        return None

    head = pdf_text[:1500]
    # Prefer lines with volume/issue/month patterns
    masthead_years = re.findall(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(?:18|19|20)\d{2}\b"
        r"|(?:Vol\.?|Volume|No\.?|Issue|pp\.?)\s*.*?\b(?:18|19|20)\d{2}\b",
        head,
        flags=re.I | re.S,
    )
    if masthead_years:
        # Extract the final 4-digit year from the match
        m = re.search(r"(?:18|19|20)\d{2}", masthead_years[0])
        if m:
            y = int(m.group(0))
            if 1900 <= y <= 2035:
                return y

    # Otherwise, grab any 4-digit years in first 2 pages
    first_pages = pdf_text[:6000]
    years = re.findall(r"\b(18|19|20)\d{2}\b", first_pages)
    if not years:
        return None

    # Count frequency; pick the most common
    counts: Dict[int, int] = {}
    for m in re.finditer(r"\b((?:18|19|20)\d{2})\b", first_pages):
        y = int(m.group(1))
        if 1800 <= y <= 2035:
            counts[y] = counts.get(y, 0) + 1

    if not counts:
        return None

    # Return the most frequent; tie-breaker = larger count then earlier year
    best = sorted(counts.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[0][0]
    if 1900 <= best <= 2035:
        return best
    return None


def _blank_draft() -> Dict[str, Any]:
    return {
        "study": {
            "title": None,
            "authors": [],
            "nct_id": None,
            "pmid": None,
            "doi": None,
            "year": None,
            "design": None,
            "country": None,
            "condition": None,
            "notes": None,
        },
        "arms": [],
        "outcomes": [
            {
                "name": "Primary outcome (stub)",
                "type": "continuous",
                "timepoints": [{"label": "end of treatment", "measures": []}],
                "subgroups": [],
            }
        ],
    }


def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None) -> Dict[str, Any]:
    draft = _blank_draft()

    # Extract readable text from PDF for dose/arm patterns and fallbacks
    pdf_text = _read_pdf_text(pdf_path, max_pages=6)

    used_grobid = False
    abstract_text = ""

    # ---------------- GROBID ----------------
    if grobid_url:
        client = GrobidClient(grobid_url)
        md = client.header_metadata(pdf_path)
        if md:
            used_grobid = True
            abstract_text = md.get("abstract", "") or ""

            if md.get("title"):
                draft["study"]["title"] = md["title"]
            if md.get("authors"):
                draft["study"]["authors"] = md["authors"]
            if md.get("doi"):
                draft["study"]["doi"] = _clean_doi(md["doi"])
            if md.get("year"):
                draft["study"]["year"] = md["year"]

    # ---------------- Arms ----------------
    draft["arms"] = _heuristic_arms(abstract_text, pdf_text)

    # ---------------- DOI fallback ----------------
    if not draft["study"]["doi"]:
        m_doi = re.search(r"\b10\.\d{4,9}/[^\s<>\)]+", pdf_text, flags=re.I)
        if m_doi:
            draft["study"]["doi"] = _clean_doi(m_doi.group(0))

    # ---------------- Title fallback ----------------
    if not draft["study"]["title"]:
        # fallback: pick a reasonably long line as a title candidate
        m_title = re.search(r"(?m)^[^\r\n]{20,160}$", pdf_text)
        draft["study"]["title"] = m_title.group(0).strip() if m_title else None

    # ---------------- Year fallback / correction ----------------
    y_pdf = _year_from_pdf(pdf_text)
    if draft["study"]["year"] is None and y_pdf:
        draft["study"]["year"] = y_pdf
    else:
        # If GROBID year looks suspicious (e.g., doesn't appear anywhere in PDF text),
        # replace it with the PDF-derived year.
        y_now = draft["study"]["year"]
        if y_now and (str(y_now) not in pdf_text) and y_pdf:
            draft["study"]["year"] = y_pdf

    draft["study"]["notes"] = (
        "Draft via GROBID TEI; GROBID=on" if used_grobid else "Draft via local parsing. GROBID=off"
    )
    return draft

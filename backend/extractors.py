# backend/extractors.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz            # PyMuPDF
import pdfplumber      # tables
import requests
import xml.etree.ElementTree as ET


# -----------------------------
# Utilities (local heuristics)
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def read_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    """Plain text from the first few pages with PyMuPDF."""
    chunks: List[str] = []
    p = Path(pdf_path)
    if not p.exists():
        return ""
    try:
        with fitz.open(str(p)) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                chunks.append(page.get_text("text") or "")
    except Exception:
        return ""
    return "\n".join(chunks)


def try_simple_tables(pdf_path: str, max_pages: int = 4) -> List[Dict[str, Any]]:
    """
    Very simple table pass with pdfplumber.
    Returns: [{"page": int, "rows": [[...], ...]}, ...]
    """
    out: List[Dict[str, Any]] = []
    p = Path(pdf_path)
    if not p.exists():
        return out
    try:
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                for settings in (
                    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                    {"vertical_strategy": "text", "horizontal_strategy": "text"},
                    {"vertical_strategy": "explicit", "horizontal_strategy": "explicit"},
                ):
                    try:
                        tables = page.extract_tables(table_settings=settings) or []
                    except Exception:
                        tables = []
                    for t in tables:
                        if not t or len(t) < 2:
                            continue
                        rows = [r for r in t if any(c and str(c).strip() for c in r)]
                        if len(rows) >= 2:
                            out.append({"page": i + 1, "rows": rows})
    except Exception:
        pass
    return out


# -----------------------------
# GROBID helpers
# -----------------------------
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _parse_tei(xml_text: str) -> Dict[str, Any]:
    """Extract title/authors/doi/year from a TEI XML string."""
    root = ET.fromstring(xml_text)

    # Title
    title = None
    tnode = root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", TEI_NS)
    if tnode is not None and (tnode.text or "").strip():
        title = _norm(tnode.text)

    # Authors – try titleStmt first, then analytic
    authors: List[str] = []
    for a in root.findall(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:author", TEI_NS):
        forename = a.find(".//tei:forename", TEI_NS)
        surname = a.find(".//tei:surname", TEI_NS)
        parts = [p for p in [
            (forename.text if forename is not None else None),
            (surname.text if surname is not None else None)
        ] if p]
        name = _norm(" ".join(parts))
        if name:
            authors.append(name)

    if not authors:
        for a in root.findall(".//tei:analytic/tei:author", TEI_NS):
            forename = a.find(".//tei:forename", TEI_NS)
            surname = a.find(".//tei:surname", TEI_NS)
            parts = [p for p in [
                (forename.text if forename is not None else None),
                (surname.text if surname is not None else None)
            ] if p]
            name = _norm(" ".join(parts))
            if name:
                authors.append(name)

    # DOI
    doi = None
    doi_node = root.find(".//tei:idno[@type='DOI']", TEI_NS)
    if doi_node is not None and (doi_node.text or "").strip():
        doi = _norm(doi_node.text)

    # Year
    year = None
    for dn in root.findall(".//tei:date", TEI_NS):
        when = dn.attrib.get("when", "")
        txt = (dn.text or "").strip()
        cand = when or txt
        m = re.search(r"\b(19|20)\d{2}\b", cand or "")
        if m:
            year = int(m.group(0))
            break

    return {"title": title, "authors": authors, "doi": doi, "year": year}


def _parse_bibtex(bib: str) -> Dict[str, Any]:
    """Minimal BibTeX extractor for title/author/doi/year."""
    out = {"title": None, "authors": [], "doi": None, "year": None}

    def get(field: str) -> Optional[str]:
        m = re.search(rf"{field}\s*=\s*{{(.*?)}}", bib, re.I | re.S)
        if m:
            return _norm(m.group(1))
        return None

    out["title"] = get("title")
    authors_raw = get("author")
    if authors_raw:
        for a in re.split(r"\s+and\s+", authors_raw):
            a = _norm(a)
            if a:
                out["authors"].append(a)
    out["doi"] = get("doi")

    y = get("year")
    if y and re.fullmatch(r"\d{4}", y):
        out["year"] = int(y)

    if out["year"] is None:
        date = get("date")
        if date:
            m = re.search(r"\b(19|20)\d{2}\b", date)
            if m:
                out["year"] = int(m.group(0))

    return out


def grobid_header(pdf_path: str, grobid_url: Optional[str]) -> Dict[str, Any]:
    """
    Call GROBID /api/processHeaderDocument.
    IMPORTANT FIX: reopen/rewind the file for each POST.
    """
    result = {"title": None, "authors": [], "doi": None, "year": None}
    if not grobid_url:
        return result
    p = Path(pdf_path)
    if not p.exists():
        return result

    url = grobid_url.rstrip("/") + "/api/processHeaderDocument"
    params = {"consolidateHeader": "1"}

    # 1) Ask for TEI XML
    try:
        with open(p, "rb") as fh:
            files = {"input": (p.name, fh, "application/pdf")}
            r = requests.post(url, files=files, params=params,
                              headers={"Accept": "application/xml"}, timeout=60)
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.ok and ("xml" in ct or r.text.strip().startswith("<")):
            return _parse_tei(r.text)
    except Exception:
        pass

    # 2) Ask for BibTeX (new file handle!)
    try:
        with open(p, "rb") as fh:
            files = {"input": (p.name, fh, "application/pdf")}
            r2 = requests.post(url, files=files, params=params,
                               headers={"Accept": "application/x-bibtex"}, timeout=60)
        ct2 = (r2.headers.get("Content-Type") or "").lower()
        if r2.ok and ("bibtex" in ct2 or r2.text.strip().startswith("@")):
            return _parse_bibtex(r2.text)
    except Exception:
        pass

    return result


# -----------------------------
# Main extractor
# -----------------------------
def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-source extractor:
      - GROBID for title/authors/DOI/year (preferred)
      - Local heuristics as fallback
    """
    # Read a bit of text for heuristics
    text = read_pdf_text(pdf_path)
    _ = try_simple_tables(pdf_path)  # reserved for future use

    # Local fallbacks
    first_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    local_title = first_lines[0] if first_lines else None

    local_doi = None
    m = re.search(r"\b10\.\d{4,9}/\S+\b", text or "")
    if m:
        local_doi = m.group(0).rstrip(").,;")

    local_year = None
    m = re.search(r"\b(19|20)\d{2}\b", (text or "")[:2000])
    if m:
        try:
            y = int(m.group(0))
            if 1900 <= y <= 2035:
                local_year = y
        except Exception:
            pass

    # GROBID_URL from env unless explicitly provided
    if grobid_url is None:
        grobid_url = os.getenv("GROBID_URL")

    g = grobid_header(pdf_path, grobid_url)

    # Prefer GROBID title; avoid obvious mastheads
    title = g.get("title") or local_title or "Untitled trial"
    if re.search(r"^(ar(?:thritis)?|rheumatism|journal|vol\.?|no\.?)", title, re.I):
        # looks like masthead – if we have a second line, try that
        if len(first_lines) >= 2:
            title = first_lines[1]

    # Tag incomplete endings
    if re.search(r"\b(in|of|with|for|to)$", title, re.I):
        title = title + " [incomplete]"

    authors = g.get("authors") or []
    doi = g.get("doi") or local_doi
    year = g.get("year") or local_year

    # Very light arm detection (placeholder)
    arms: List[Dict[str, Any]] = []
    for pat in (r"budesonide\s*3\s*mg(?:/day)?",
                r"budesonide\s*9\s*mg(?:/day)?",
                r"prednisolone\s*7\.5\s*mg(?:/day)?",
                r"\bplacebo\b"):
        m = re.search(pat, text or "", re.I)
        if m:
            label = _norm(m.group(0))
            arm_id = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            arms.append({"arm_id": arm_id, "label": label, "n_randomized": None})

    # Deduplicate arms
    seen = set()
    dedup_arms = []
    for a in arms:
        k = a["arm_id"]
        if k not in seen:
            seen.add(k)
            dedup_arms.append(a)
    arms = dedup_arms or [
        {"arm_id": "treatment", "label": "Treatment", "n_randomized": None},
        {"arm_id": "control", "label": "Control", "n_randomized": None},
    ]

    # Confidence
    title_conf = 0.9 if g.get("title") else (0.7 if local_title else 0.4)
    authors_conf = 0.9 if authors else 0.0
    arms_conf = 1.0 if arms else 0.3

    notes_bits = [f"Draft via local parsing. Confidence — Title:{title_conf:.2f}, Authors:{authors_conf:.2f}, Arms:{arms_conf:.2f}"]
    notes_bits.append("GROBID=on" if grobid_url else "GROBID=off")

    study = {
        "title": title,
        "authors": authors,
        "nct_id": None,
        "pmid": None,
        "doi": doi,
        "year": year,
        "design": None,
        "country": None,
        "condition": None,
        "notes": " ".join(notes_bits),
    }

    outcome = {
        "name": "Primary outcome (stub)",
        "type": "continuous",
        "timepoints": [{"label": "end of treatment", "measures": []}],
        "subgroups": []
    }

    return {
        "study": study,
        "arms": arms,
        "outcomes": [outcome],
    }

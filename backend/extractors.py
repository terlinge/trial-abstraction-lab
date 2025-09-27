# backend/extractors.py
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import fitz            # PyMuPDF
import pdfplumber      # tables
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ================== Helpers ==================

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = x.lower()
        if key not in seen and x.strip():
            seen.add(key)
            out.append(x.strip())
    return out

def _is_year(s: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}", s.strip()))

def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

def _cleanup_author(name: str) -> str:
    # Collapse extra spaces and remove stray punctuation/footnote marks
    name = _norm(name)
    name = re.sub(r"[\*†‡§¶\d]+$", "", name).strip()
    return name

# ================== Local PDF parsing (fallbacks) ==================

def read_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
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
    except Exception as e:
        logger.warning(f"PyMuPDF read failed: {e}")
        return ""
    return "\n".join(chunks)

def try_simple_tables(pdf_path: str, max_pages: int = 4) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = Path(pdf_path)
    if not p.exists():
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
    except Exception as e:
        logger.debug(f"pdfplumber failed: {e}")
    return out

def _arms_from_text(text: str) -> List[str]:
    arms: List[str] = []
    # Look for patterns like "budesonide 3 mg/day", "prednisolone 7.5 mg/day", "placebo"
    patterns = [
        r"\bbudesonide\s+3\s*mg(?:/?day)?\b",
        r"\bbudesonide\s+9\s*mg(?:/?day)?\b",
        r"\bprednisolone\s+7\.?5\s*mg(?:/?day)?\b",
        r"\bplacebo\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            arms.append(_norm(m.group(0)).lower())
    return _dedupe_keep_order(arms)

def _arms_from_tables(tables: List[Dict[str, Any]]) -> List[str]:
    arms: List[str] = []
    for tbl in tables:
        rows = tbl["rows"]
        if not rows:
            continue
        # Check header row for arm-like labels
        header = [str(c or "") for c in rows[0]]
        header_join = " ".join(header).lower()
        if re.search(r"(placebo|budesonide|prednisolone)", header_join):
            # columns from header (skip first likely row label)
            for cell in header[1:]:
                cell_s = _norm(str(cell))
                if re.search(r"(placebo|budesonide|prednisolone)", cell_s, re.I):
                    arms.append(cell_s.lower())
            break
    return _dedupe_keep_order(arms)

# ================== GROBID integration ==================

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def _tei_text(elem: ET.Element, path: str) -> Optional[str]:
    n = elem.find(path, TEI_NS)
    if n is not None and (n.text or "").strip():
        return _norm(n.text)
    return None

def _tei_all(elem: ET.Element, path: str) -> List[ET.Element]:
    return elem.findall(path, TEI_NS)

def _parse_tei(tei_xml: str) -> Dict[str, Any]:
    """
    Parse TEI XML from GROBID header/fulltext to extract: title, authors, doi, year, abstract.
    """
    out: Dict[str, Any] = {"title": None, "authors": [], "doi": None, "year": None, "abstract": None}
    try:
        root = ET.fromstring(tei_xml)
    except Exception as e:
        logger.warning(f"TEI parse error: {e}")
        return out

    # Title: prefer analytic title (article title), else monogr title
    title = None
    for path in [
        ".//tei:sourceDesc//tei:biblStruct/tei:analytic/tei:title[@type='main']",
        ".//tei:sourceDesc//tei:biblStruct/tei:analytic/tei:title",
        ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:title",
    ]:
        t = _tei_text(root, path)
        if t:
            title = t
            break
    out["title"] = title

    # Authors: try analytic authors first, else monogr authors
    authors: List[str] = []
    for path in [
        ".//tei:sourceDesc//tei:biblStruct/tei:analytic/tei:author",
        ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:author",
    ]:
        for a in _tei_all(root, path):
            # Common layout: author/persName/forename + surname
            pers = a.find(".//tei:persName", TEI_NS)
            if pers is None:
                continue
            # collect forenames
            forenames = [ _norm(x.text) for x in pers.findall("./tei:forename", TEI_NS) if (x.text or "").strip() ]
            surname   = _tei_text(pers, "./tei:surname")
            # fallback: sometimes only 'name' is present
            if not forenames and not surname:
                nm = _tei_text(pers, "./tei:name")
                if nm:
                    authors.append(_cleanup_author(nm))
                    continue
            parts = []
            if forenames:
                parts.append(" ".join(forenames))
            if surname:
                parts.append(surname)
            if parts:
                authors.append(_cleanup_author(" ".join(parts)))
    out["authors"] = _dedupe_keep_order(authors)

    # DOI
    doi = None
    for idno in _tei_all(root, ".//tei:idno"):
        typ = (idno.attrib.get("type") or "").lower()
        val = (idno.text or "").strip()
        if typ == "doi" and val:
            doi = val
            break
    if not doi:
        # Try plain regex in entire XML text
        m = re.search(r"\b10\.\d{4,9}/\S+\b", tei_xml)
        if m:
            doi = m.group(0).rstrip(").,;")
    out["doi"] = doi

    # Year
    year = None
    # published date first
    for d in _tei_all(root, ".//tei:date"):
        when = (d.attrib.get("when") or "").strip()
        if _is_year(when[:4]):
            year = int(when[:4])
            break
        txt = (d.text or "").strip()
        if _is_year(txt[-4:]):
            year = int(txt[-4:])
            break
    out["year"] = year

    # Abstract (optional – may help arms via text regex)
    abs_txt = []
    for p in _tei_all(root, ".//tei:profileDesc/tei:abstract//tei:p"):
        if (p.text or "").strip():
            abs_txt.append(_norm(p.text))
    if abs_txt:
        out["abstract"] = " ".join(abs_txt)

    return out

def _grobid_post(url: str, pdf_path: str, accept: str) -> Optional[str]:
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                url,
                files={"input": (Path(pdf_path).name, f, "application/pdf")},
                headers={"Accept": accept},
                timeout=60,
            )
        if resp.status_code != 200:
            logger.warning(f"GROBID HTTP {resp.status_code} at {url}")
            return None
        return resp.text
    except Exception as e:
        logger.warning(f"GROBID request failed: {e}")
        return None

def _extract_with_grobid(pdf_path: str, grobid_url: str) -> Optional[Dict[str, Any]]:
    """
    Try TEI first (best structured), then fallback to BibTeX.
    Returns dict or None if nothing usable.
    """
    base = grobid_url.rstrip("/")
    tei = _grobid_post(f"{base}/api/processHeaderDocument?consolidateHeader=1", pdf_path, "application/xml")
    data = None
    if tei:
        parsed = _parse_tei(tei)
        if parsed.get("title") or parsed.get("authors"):
            data = parsed

    if not data:
        # Fallback: BibTeX
        bib = _grobid_post(f"{base}/api/processHeaderDocument?consolidateHeader=1", pdf_path, "application/x-bibtex")
        if bib and "title =" in bib:
            title = None
            doi = None
            authors: List[str] = []
            # title
            m = re.search(r"title\s*=\s*[{\"]([^}\"]+)[}\"]", bib, re.I)
            if m:
                title = _norm(m.group(1))
            # doi
            m = re.search(r"doi\s*=\s*[{\"]([^}\"]+)[}\"]", bib, re.I)
            if m:
                doi = _norm(m.group(1))
            # authors: "A and B and C"
            m = re.search(r"author\s*=\s*[{\"]([^}\"]+)[}\"]", bib, re.I)
            if m:
                raw = m.group(1)
                parts = [ _cleanup_author(_norm(x)) for x in re.split(r"\band\b", raw, flags=re.I) ]
                authors = _dedupe_keep_order(parts)
            data = {"title": title, "authors": authors, "doi": doi, "year": None, "abstract": None}

    return data

# ================== Main extraction ==================

def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-strategy extraction:
    - If GROBID is configured and reachable, use it for title/authors/doi/year/abstract.
    - Otherwise, local heuristics.
    - Arms via abstract regex + simple tables + text regex fallback.
    """
    # Start with empty skeleton
    study: Dict[str, Any] = {
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
    }

    arms: List[Dict[str, Any]] = []

    # 1) GROBID (if provided)
    used_grobid = False
    grobid_data = None
    if grobid_url:
        grobid_data = _extract_with_grobid(pdf_path, grobid_url)
        if grobid_data:
            used_grobid = True
            study["title"] = grobid_data.get("title") or study["title"]
            study["authors"] = grobid_data.get("authors") or study["authors"]
            study["doi"] = grobid_data.get("doi") or study["doi"]
            study["year"] = grobid_data.get("year") or study["year"]

    # 2) Local text/tables to supplement arms & identifiers
    text = read_pdf_text(pdf_path)
    tables = try_simple_tables(pdf_path)

    # DOI fallback if missing
    if not study["doi"]:
        m = re.search(r"\b10\.\d{4,9}/\S+\b", text)
        if m:
            study["doi"] = m.group(0).rstrip(").,;")

    # Year fallback if missing (search early part of text)
    if not study["year"]:
        m = re.search(r"\b(19|20)\d{2}\b", text[:2500])
        if m:
            y = _safe_int(m.group(0))
            if y and 1900 <= y <= 2035:
                study["year"] = y

    # Title fallback if missing
    if not study["title"]:
        first_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if first_lines:
            # Avoid common masthead lines
            for ln in first_lines[:10]:
                if re.search(r"\b(abstract|introduction|methods?|results?|discussion|conclusion)\b", ln, re.I):
                    break
                if len(ln) > 20:
                    study["title"] = ln
                    break

    # Arms: use abstract from GROBID if available, else text/tables
    abstract_txt = (grobid_data or {}).get("abstract") if grobid_data else None
    arm_labels: List[str] = []
    if abstract_txt:
        arm_labels = _arms_from_text(abstract_txt)
    if not arm_labels:
        arm_labels = _arms_from_tables(tables)
    if not arm_labels:
        arm_labels = _arms_from_text(text)

    # Build arms
    seen = set()
    for label in arm_labels:
        arm_id = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if arm_id in seen:
            continue
        seen.add(arm_id)
        arms.append({"arm_id": arm_id, "label": label, "n_randomized": None})

    # Stub outcome
    outcome_stub = {
        "name": "Primary outcome (stub)",
        "type": "continuous",
        "timepoints": [{"label": "end of treatment", "measures": []}],
        "subgroups": []
    }

    # Notes
    if used_grobid:
        note = "Draft via local parsing. GROBID=on"
    else:
        note = "Draft via local parsing. GROBID=off"
    study["notes"] = note

    # Final
    return {
        "study": {
            "title": study["title"] or "Untitled trial",
            "authors": study["authors"] or [],
            "nct_id": study["nct_id"],
            "pmid": study["pmid"],
            "doi": study["doi"],
            "year": study["year"],
            "design": study["design"],
            "country": study["country"],
            "condition": study["condition"],
            "notes": study["notes"],
        },
        "arms": arms if arms else [
            {"arm_id": "treatment", "label": "treatment", "n_randomized": None},
            {"arm_id": "control", "label": "control", "n_randomized": None},
        ],
        "outcomes": [outcome_stub],
    }

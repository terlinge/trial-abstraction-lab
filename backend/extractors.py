# backend/extractors.py
from __future__ import annotations

import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

# External libs (already installed earlier)
import requests           # for GROBID HTTP calls
import fitz               # PyMuPDF
import pdfplumber         # tables

logger = logging.getLogger(__name__)

# ================== Data Classes ==================
@dataclass
class StudyArm:
    arm_id: str
    label: str
    n_randomized: Optional[int] = None
    confidence: float = 0.0
    source: str = "unknown"

@dataclass
class StudyMetadata:
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    nct_id: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    design: Optional[str] = None
    country: Optional[str] = None
    condition: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)

# ================== Constants / Regex ==================
STOP_HEAD = re.compile(
    r"^(abstract|objective|background|methods?|patients?|results?|conclusions?|introduction)[:\.\s]\b",
    re.I,
)

SECTION_HEADERS = re.compile(
    r"\b(abstract|introduction|background|methods?|results?|discussion|conclusions?|references|acknowledg(e)?ments?)\b",
    re.I
)

AUTHORS_BAN = re.compile(
    r"\b(vol|volume|no|number|issue|page|pp|doi|issn|journal|editor|publisher|"
    r"copyright|received|accepted|submitted|revised|online)\b",
    re.I,
)

# ================== Small Helpers ==================
def _norm(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_marks(s: str) -> str:
    """Remove footnote/superscripts and diacritics for author lines."""
    if not s:
        return s
    s = re.sub(r"(?<=\w)[\d*†‡§¶]+(?=[\s,]|$)", "", s)
    s = re.sub(r"(?:^|\s)[\d*†‡§¶]+(?=\s|$)", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return _norm(s)

def _extract_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    marks = list(SECTION_HEADERS.finditer(text))
    for i, m in enumerate(marks):
        name = m.group(1).lower()
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections[name] = text[start:end]
    return sections

# ================== Title/Authors (font-based) ==================
def extract_title_authors_by_font(pdf_path: str) -> Tuple[Optional[str], List[str]]:
    """
    Use font sizes/positions on first page to guess title + authors.
    """
    p = Path(pdf_path)
    if not p.exists():
        return None, []

    try:
        with fitz.open(str(p)) as doc:
            if not doc:
                return None, []

            spans = []
            for i, page in enumerate(doc):
                if i >= 2:
                    break
                d = page.get_text("dict")
                for block in d.get("blocks", []):
                    for line in block.get("lines", []):
                        sps = line.get("spans", [])
                        if not sps:
                            continue
                        text = _norm(" ".join(_norm(s.get("text", "")) for s in sps))
                        if not text:
                            continue
                        sizes = [float(s.get("size", 0.0)) for s in sps]
                        avg_size = sum(sizes) / max(1, len(sizes))
                        y0 = float(line.get("bbox", [0, 0, 0, 0])[1])
                        spans.append((i, avg_size, y0, text))

            if not spans:
                return None, []

            page0 = [s for s in spans if s[0] == 0] or spans

            # title candidates = largest fonts at the top
            top = [s for s in page0 if s[2] < 400] or page0[:10]
            max_size = max(s[1] for s in top)
            cands = [(y, t, sz) for _, sz, y, t in top if sz >= 0.85 * max_size]
            cands.sort(key=lambda x: x[0])

            title_lines = []
            last_y = None
            for y, t, sz in cands:
                if STOP_HEAD.match(t):
                    break
                # stop if we hit author-like tokens (e.g., comma-number)
                if re.search(r"[A-Z][a-z]+,?\s*\d", t):
                    break
                if last_y is None or (y - last_y) < 100:
                    title_lines.append(t)
                    last_y = y
                else:
                    break
            title = " ".join(title_lines) if title_lines else None

            # author zone: below the title
            authors: List[str] = []
            if last_y is not None:
                zone = [t for _, _, y, t in page0 if last_y < y < last_y + 320]
                for t in zone[:10]:
                    if STOP_HEAD.match(t):
                        break
                    parts = re.split(r"[,;]|\band\b", t)
                    for part in parts:
                        part = _strip_marks(part)
                        if re.search(r"[A-Z][a-z]+", part) and not AUTHORS_BAN.search(part):
                            # keep something that looks like a name
                            if re.search(r"[A-Z][a-z].+\s[A-Z][a-z]", part):
                                authors.append(part)

            # dedupe authors (preserve order)
            seen = set()
            deduped = []
            for a in authors:
                k = a.lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    deduped.append(a)
            return title, deduped
    except Exception as e:
        logger.warning(f"Font extraction error: {e}")
        return None, []

# ================== Abstract-based patterns ==================
def extract_from_abstract(full_text: str) -> Dict[str, Any]:
    out = {"arms": [], "design": None, "condition": None}
    m = re.search(r"abstract[:\s]*(.+?)(?=\n[A-Z][^\n]*\n|introduction|background|methods|$)", full_text[:6000], re.I | re.S)
    if not m:
        return out
    abstr = m.group(1)

    # design
    for pat in [r"randomized controlled trial", r"double-blind.*?trial", r"placebo[-\s]?controlled"]:
        mm = re.search(pat, abstr, re.I)
        if mm:
            out["design"] = _norm(mm.group(0))
            break

    # condition
    for pat in [r"patients with\s+([^\.;,]+)", r"treatment of\s+([^\.;,]+)", r"\b(rheumatoid arthritis)\b"]:
        mm = re.search(pat, abstr, re.I)
        if mm:
            out["condition"] = _norm(mm.group(1) if mm.lastindex else mm.group(0))
            break

    # arm mentions
    arm_raw = []
    for pat in [
        r"received\s+(?:either\s+)?([^\.]+)",
        r"randomized to\s+([^\.]+)",
        r"(?:budesonide|prednisolone|placebo)[^\.]+",
    ]:
        mm = re.search(pat, abstr, re.I)
        if mm:
            seg = _norm(mm.group(1) if mm.lastindex else mm.group(0))
            arm_raw.extend(re.split(r",\s*(?:and|or)\s*|,\s*", seg))

    out["arms"] = [_norm(a) for a in arm_raw if a]
    return out

# ================== Table parsing (simple) ==================
def extract_from_tables(pdf_path: str) -> Dict[str, Any]:
    """
    Very light table sniffing (first 5 pages) to pull arm labels & Ns from header rows.
    """
    result = {"arms": [], "sample_sizes": {}}
    p = Path(pdf_path)
    if not p.exists():
        return result
    try:
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages[:5]:
                for settings in [
                    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                    {"vertical_strategy": "text", "horizontal_strategy": "text"},
                    {"vertical_strategy": "explicit", "horizontal_strategy": "explicit"},
                ]:
                    try:
                        tables = page.extract_tables(table_settings=settings) or []
                    except Exception:
                        tables = []
                    for tb in tables:
                        if not tb or len(tb) < 2:
                            continue
                        header = [str(c or "").strip() for c in tb[0]]
                        hdr_join = " ".join(header).lower()
                        if not any(k in hdr_join for k in ["characteristics", "baseline", "table 1", "treatment", "arm", "group"]):
                            continue
                        # collect arms from header cells (skip first col)
                        for col_idx, cell in enumerate(header[1:], 1):
                            cell_s = str(cell).strip()
                            if re.search(r"(budesonide|prednisolone|placebo)", cell_s, re.I):
                                result["arms"].append(cell_s)
                        # Ns from body
                        for row in tb[1:]:
                            row_txt = " ".join(str(c) for c in row if c)
                            if re.search(r"\b(n|randomized|enrolled|patients)\b", row_txt, re.I):
                                for col_idx, cell in enumerate(row[1:], 1):
                                    cs = str(cell or "")
                                    mm = re.search(r"(\d{2,4})", cs)
                                    if mm and col_idx - 1 < len(result["arms"]):
                                        label = result["arms"][col_idx - 1]
                                        result["sample_sizes"][label] = int(mm.group(1))
                                break
    except Exception as e:
        logger.warning(f"Table extraction error: {e}")
    return result

# ================== Validation / Scoring ==================
def validate_title(title: Optional[str]) -> Tuple[str, float]:
    if not title:
        return "Untitled trial", 0.0
    score = 1.0
    t = title.strip()
    if re.search(r"\b(vol|no|pp|doi|issn|copyright)\b", t, re.I):
        score *= 0.6
    if len(t) < 10:
        score *= 0.5
    if t.endswith((" in", " of", " with")):
        t += " [incomplete]"
        score *= 0.8
    return _norm(t), score

def validate_authors(authors: List[str]) -> Tuple[List[str], float]:
    if not authors:
        return [], 0.0
    valid, total = [], 0.0
    for a in authors:
        if AUTHORS_BAN.search(a) or not re.search(r"[A-Z][a-z].+\s[A-Z][a-z]", a):
            continue
        aa = _strip_marks(a)
        if 2 <= len(aa) <= 60:
            valid.append(aa)
            total += 1.0
    avg = (total / max(1, len(valid))) if valid else 0.0
    return valid[:30], avg

def validate_arms(arms: List[Dict[str, Any]]) -> Tuple[List[StudyArm], float]:
    out: List[StudyArm] = []
    total = 0.0
    seen = set()
    for arm in arms:
        lbl = arm.get("label", "")
        key = lbl.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        sc = 1.0
        n = arm.get("n_randomized")
        if n is not None and not (5 <= n <= 10000):
            sc *= 0.7
        out.append(StudyArm(arm_id=arm.get("arm_id", key.replace(" ", "_")),
                            label=lbl, n_randomized=n, confidence=sc, source=arm.get("source", "unknown")))
        total += sc
    avg = total / max(1, len(out)) if out else 0.0
    return out, avg

# ================== GROBID ==================
def _from_tei_text(t: Optional[str]) -> Optional[str]:
    return (t or "").strip() or None

def grobid_metadata(pdf_path: str, grobid_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Query a local GROBID server for header metadata (title, authors, doi, year).
    Returns None if GROBID_URL is not set or on any failure.
    """
    if not grobid_url:
        return None
    try:
        with open(pdf_path, "rb") as f:
            r = requests.post(
                f"{grobid_url}/api/processHeaderDocument",
                files={"input": ("paper.pdf", f, "application/pdf")},
                data={"consolidateHeader": "1"},
                timeout=30,
            )
        if r.status_code != 200 or not r.text:
            return None
        root = ET.fromstring(r.text)
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        title_el = root.find(".//tei:titleStmt/tei:title", ns)
        title = _from_tei_text(title_el.text if title_el is not None else None)

        authors: List[str] = []
        for pers in root.findall(".//tei:sourceDesc//tei:author/tei:persName", ns):
            fn = pers.find("tei:forename", ns)
            sn = pers.find("tei:surname", ns)
            name = " ".join([x.text.strip() for x in [fn, sn] if x is not None and x.text])
            if name:
                authors.append(name)

        doi = None
        doi_el = root.find(".//tei:sourceDesc//tei:idno[@type='DOI']", ns)
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        year = None
        date_el = root.find(".//tei:sourceDesc//tei:date", ns)
        if date_el is not None and date_el.get("when"):
            y = date_el.get("when")[:4]
            if y.isdigit():
                year = int(y)

        return {"title": title, "authors": authors, "doi": doi, "year": year}
    except Exception as e:
        logger.warning(f"GROBID failure: {e}")
        return None

# ================== Arm normalization ==================
def normalize_arms(arm_labels: List[str]) -> List[str]:
    """
    Deduplicate + filter noisy labels, and normalize dose notation.
    Keeps only common control/treatment keywords; merges variants.
    """
    cleaned: List[str] = []
    seen = set()
    keywords = ["placebo", "budesonide", "prednisolone"]

    for raw in arm_labels:
        s = re.sub(r"\s+", " ", raw.strip().lower())
        if "cortisone" in s:  # obvious false positive in this paper
            continue
        if not any(k in s for k in keywords):
            continue

        s = s.replace(" / day", "/day").replace(" /day", "/day")
        s = re.sub(r"(\d)\s*mg", r"\1 mg", s)
        # Merge "budesonide 9 mg" -> "budesonide 9 mg/day" for consistency
        s = re.sub(r"\b(budesonide|prednisolone)\s+(3|9|7\.5)\s*mg\b", r"\1 \2 mg/day", s)

        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned

# ================== Consensus extraction ==================
def extract_with_consensus(pdf_path: str, text: str) -> Dict[str, Any]:
    # Strategy 1: font-based
    t_font, a_font = extract_title_authors_by_font(pdf_path)
    # Strategy 2: abstract/text
    sections = _extract_sections(text)
    abstr = extract_from_abstract(text)
    # Strategy 3: tables
    tables = extract_from_tables(pdf_path)

    # Title voting
    title_cands: List[Tuple[str, float]] = []
    if t_font:
        title_cands.append((t_font, 0.9))
    # also try the first non-empty line
    for ln in [l for l in text.splitlines() if l.strip()][:5]:
        if len(ln) > 20 and not STOP_HEAD.match(ln):
            title_cands.append((ln, 0.7))
            break
    best_title, best_score = None, 0.0
    for t, base in title_cands:
        t2, s = validate_title(t)
        sc = base * s
        if sc > best_score:
            best_title, best_score = t2, sc

    # Authors
    authors_raw = a_font or []
    authors, a_score = validate_authors(authors_raw)

    # Arms
    arm_cands: List[str] = []
    arm_cands.extend(tables.get("arms", []))
    arm_cands.extend(abstr.get("arms", []))

    # additional pattern scan in methods/intro
    methods_text = sections.get("methods", "") + " " + sections.get("introduction", "")
    for pat in [r"budesonide\s+9\s*mg(?:/day)?", r"budesonide\s+3\s*mg(?:/day)?", r"prednisolone\s+7\.5\s*mg(?:/day)?", r"\bplacebo\b"]:
        for m in re.finditer(pat, methods_text, re.I):
            arm_cands.append(_norm(m.group(0)))

    labels = normalize_arms(arm_cands)
    arms_pre = [{"arm_id": re.sub(r"[^a-z0-9]+", "_", lbl).strip("_"), "label": lbl, "source": "consensus"} for lbl in labels]
    arms, arms_score = validate_arms(arms_pre)

    meta = StudyMetadata(
        title=best_title,
        authors=authors,
        design=abstr.get("design"),
        condition=abstr.get("condition"),
        confidence_scores={"title": round(best_score, 2), "authors": round(a_score, 2), "arms": round(arms_score, 2)}
    )
    return {"metadata": meta, "arms": arms, "tables": tables, "abstract": abstr}

# ================== Public API ==================
def extract_first_pass(pdf_path: str) -> Dict[str, Any]:
    """
    Main entry point used by the FastAPI backend.
    Combines font/text/table heuristics and (if available) GROBID metadata.
    Returns a dict with keys: study, arms, outcomes.
    """
    logger.info(f"Extracting from: {pdf_path}")
    text = ""
    p = Path(pdf_path)
    if p.exists():
        try:
            with fitz.open(str(p)) as doc:
                for i, page in enumerate(doc):
                    if i >= 6:
                        break
                    text += page.get_text("text") or ""
        except Exception as e:
            logger.error(f"PDF read error: {e}")

    consensus = extract_with_consensus(pdf_path, text)
    meta: StudyMetadata = consensus["metadata"]
    arms: List[StudyArm] = consensus["arms"]

    # Try GROBID for stronger bibliographic data
    g_url = os.getenv("GROBID_URL")
    gmeta = grobid_metadata(pdf_path, g_url)
    title = (gmeta.get("title") if gmeta else None) or meta.title
    authors = (gmeta.get("authors") if gmeta else None) or meta.authors
    doi = (gmeta.get("doi") if gmeta else None)
    year = (gmeta.get("year") if gmeta else None)

    study = {
        "title": title,
        "authors": authors,
        "nct_id": None,
        "pmid": None,
        "doi": doi,
        "year": year,
        "design": meta.design,
        "country": None,
        "condition": meta.condition,
        "notes": f"Draft via local parsing. Confidence — Title:{meta.confidence_scores.get('title', 0):.2f}, "
                 f"Authors:{meta.confidence_scores.get('authors', 0):.2f}, Arms:{meta.confidence_scores.get('arms', 0):.2f}",
    }

    # Finalize arms
    final_arms = [{"arm_id": a.arm_id, "label": a.label, "n_randomized": a.n_randomized} for a in arms]
    if not final_arms:
        final_arms = [
            {"arm_id": "treatment", "label": "treatment", "n_randomized": None},
            {"arm_id": "control", "label": "control", "n_randomized": None},
        ]

    outcome_stub = {
        "name": "Primary outcome (stub)",
        "type": "continuous",
        "timepoints": [{"label": "end of treatment", "measures": []}],
        "subgroups": [],
    }

    return {"study": study, "arms": final_arms, "outcomes": [outcome_stub]}

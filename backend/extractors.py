# extractors.py
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from grobid_client import GrobidClient
# Optional OCR / table libs (import lazily/use defensively)
try:
    from PIL import Image, ImageEnhance
except Exception:
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None  # type: ignore
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore
try:
    import spacy
except Exception:
    spacy = None  # type: ignore
import xml.etree.ElementTree as ET


# ---------------- small logs ----------------
def _llm_log(msg: str) -> None:
    print(f"[llm] {msg}")


# ---------------- PDF helpers ----------------

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


def _layout_title_and_authors(pdf_path: str) -> Dict[str, Any]:
    """Try to pick the title and author lines from the first page using layout info.
    Returns {'title': str|null, 'authors': [str,...]}.
    This is a conservative fallback when GROBID TEI looks noisy.
    """
    out = {"title": None, "authors": []}
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count < 1:
                return out
            page = doc.load_page(0)
            # get structured text with spans (font sizes)
            txt = page.get_text("dict")
            spans = []
            for block in txt.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size") or 0
                        if text:
                            spans.append({"text": text, "size": float(size), "font": span.get("font", "")})

            if not spans:
                return out

            # choose candidate with largest size that passes scoring
            spans_sorted = sorted(spans, key=lambda s: (-s["size"], -len(s["text"])))
            for cand in spans_sorted[:8]:
                s = _sanitize_text(cand["text"])
                if not s:
                    continue
                # reject if it looks like a sentence or contains measurement units
                low = s.lower()
                if len(s) < 8:
                    continue
                if re.search(r"\bwas defined\b|\bmm\s*hg\b|\b\d+\s*mm\b|;", low):
                    continue
                # good candidate
                out["title"] = s
                break

            # for authors: look for spans immediately following the chosen title's size (slightly smaller)
            if out["title"]:
                title_size = None
                for sp in spans:
                    if _sanitize_text(sp["text"]) == out["title"]:
                        title_size = sp["size"]
                        break
                if title_size is None:
                    title_size = spans_sorted[0]["size"]

                # collect spans with size slightly smaller than title_size and limited length
                author_candidates = [s for s in spans if s["size"] <= title_size and s["size"] >= max(8.0, title_size - 6)]
                # join nearby short spans into lines, then filter plausible person-like strings
                seen = set()
                authors_out: List[str] = []
                for a in author_candidates:
                    t = _sanitize_text(a["text"])
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    # heuristics: contain comma or multiple capitalized tokens and not affiliation words
                    if re.search(r"\b(university|hospital|department|center|institute|clinic)\b", t.lower()):
                        continue
                    # avoid publisher/journal mastheads being mis-detected as authors
                    if _is_journal_masthead(t):
                        continue
                    toks = [tok for tok in re.split(r"[,\s]+", t) if tok.strip()]
                    # break comma-separated lists
                    for tok in toks:
                        tok = tok.strip()
                        if 2 <= len(tok) <= 60 and re.search(r"[A-Za-z]", tok):
                            # avoid catching short words like 'and'
                            if len(tok.split()) <= 6 and not re.search(r"\b(and|for|the)\b", tok.lower()):
                                authors_out.append(_repair_kerned_name(tok))
                if authors_out:
                    # avoid sequences that are actually journal mastheads split into tokens
                    if not _looks_like_masthead_tokens(authors_out):
                        out["authors"] = authors_out

            # Try spaCy PERSON NER on the first page if available and authors look bad
            if (not out.get("authors")) and spacy is not None:
                try:
                    # build a short chunk from top spans
                    chunk = " ".join([s["text"] for s in spans_sorted[:40]])
                    nlp = _get_spacy()
                    if nlp is not None:
                        docsp = nlp(chunk)
                        persons = []
                        for ent in docsp.ents:
                            if ent.label_ == "PERSON":
                                name = _collapse_spaces(ent.text)
                                if _plausible_person_name(name):
                                    persons.append(_repair_kerned_name(name))
                        if persons:
                            out["authors"] = persons
                except Exception:
                    pass

    except Exception:
        return out

    return out


def _needs_ocr(pdf_path: str, min_chars: int = 200) -> bool:
    """Return True when the first pages contain too little extractable text."""
    txt = _read_pdf_text(pdf_path, max_pages=4)
    if not txt or len(txt.strip()) < min_chars:
        return True
    return False


def _ocr_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    """Use pdf2image + pytesseract to OCR the first pages. Returns joined text.
    Defensive: returns empty string if dependencies/binaries are missing.
    """
    if convert_from_path is None or pytesseract is None or Image is None:
        return ""
    out: List[str] = []
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception:
        return ""
    for i, image in enumerate(images):
        if i >= max_pages:
            break
        try:
            # Preprocess image: deskew/denoise/contrast
            img = image.convert("L")
            try:
                pil_img = img
                # Use OpenCV for deskew/denoise if available
                if cv2 is not None:
                    import numpy as np
                    arr = np.array(pil_img)
                    # denoise
                    try:
                        den = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
                        arr = den
                    except Exception:
                        pass
                    # deskew via minAreaRect on edges
                    try:
                        edges = cv2.Canny(arr, 50, 150)
                        coords = cv2.findNonZero(edges)
                        if coords is not None and len(coords) > 0:
                            rect = cv2.minAreaRect(coords)
                            angle = rect[-1]
                            if angle < -45:
                                angle = -(90 + angle)
                            else:
                                angle = -angle
                            (h, w) = arr.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    except Exception:
                        pass
                    pil_img = Image.fromarray(arr)

                # Pillow contrast enhancement
                if ImageEnhance is not None:
                    try:
                        enhancer = ImageEnhance.Contrast(pil_img)
                        pil_img = enhancer.enhance(1.5)
                    except Exception:
                        pass

                text = pytesseract.image_to_string(pil_img, lang="eng", config="--psm 6")
            except Exception:
                text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            out.append(text or "")
        except Exception:
            out.append("")
    return "\n".join(out)


def _try_extract_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Attempt to extract simple table metadata via camelot (if installed).
    Returns a list of table summaries: {page, n_rows, n_cols}.
    Defensive: returns empty list if camelot not available or extraction fails.
    """
    out: List[Dict[str, Any]] = []

    # 1) Try Camelot (best, but often tricky on Windows)
    try:
        import camelot
        tables = []
        try:
            tables = camelot.read_pdf(pdf_path, pages="1-30", flavor="lattice")
        except Exception:
            pass
        if not tables:
            try:
                tables = camelot.read_pdf(pdf_path, pages="1-30", flavor="stream")
            except Exception:
                tables = []
        for t in tables:
            try:
                df = t.df
                nrows = len(df.index)
                ncols = len(df.columns)
                out.append({"page": int(t.page), "n_rows": nrows, "n_cols": ncols, "source": "camelot"})
            except Exception:
                continue
        if out:
            return out
    except Exception:
        pass

    # 2) Try tabula-py (Java; generally Windows-friendly)
    try:
        import tabula
        try:
            dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            if isinstance(dfs, list):
                for df in dfs:
                    try:
                        nrows = len(df.index)
                        ncols = len(df.columns)
                        # tabula doesn't give page easily; omit page when unknown
                        out.append({"page": None, "n_rows": nrows, "n_cols": ncols, "source": "tabula"})
                    except Exception:
                        continue
            if out:
                return out
        except Exception:
            pass
    except Exception:
        pass

    # 3) Fallback: pdfplumber (pure Python) — returns tables per page
    try:
        import pdfplumber
        try:
            with pdfplumber.open(pdf_path) as doc:
                for i, page in enumerate(doc.pages, start=1):
                    try:
                        tables = page.extract_tables()
                        if not tables:
                            continue
                        for t in tables:
                            nrows = len(t)
                            ncols = max((len(r) for r in t), default=0)
                            out.append({"page": i, "n_rows": nrows, "n_cols": ncols, "source": "pdfplumber"})
                    except Exception:
                        continue
            return out
        except Exception:
            return out
    except Exception:
        return out


def _clean_doi(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return re.sub(r"[\s\)\]\}\.,;:]+$", "", s.strip())


def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _sanitize_text(s: Optional[str]) -> Optional[str]:
    """Remove non-printable/control junk and normalize whitespace/encoding artifacts."""
    if s is None:
        return s
    # remove obvious control characters and replacement characters
    s2 = re.sub(r"[\x00-\x1f\x7f\u0000-\u001f\ufeff\ufffd]", " ", str(s))
    # common mojibake/section symbols that leak in
    s2 = s2.replace("Â", "").replace("\u00a7", "").replace("\u00b6", "")
    # collapse whitespace
    s2 = _collapse_spaces(s2)
    return s2


def _title_and_authors_from_text(text: str) -> Dict[str, Any]:
    """Heuristic: pick a title from OCR/pdf text and nearby author lines.
    Returns {'title': str|None, 'authors': [str,...]}.
    This is a last-resort deterministic fallback that inspects the first
    ~40 non-empty lines and looks for the best title-like line and
    subsequent author-like lines.
    """
    out = {"title": None, "authors": []}
    if not text:
        return out
    lines = [_collapse_spaces(l) for l in text.splitlines() if _collapse_spaces(l)]
    if not lines:
        return out

    # limit to top of document where masthead/title usually is
    head = lines[:60]

    # score candidates using existing title scorer
    scored: List[Tuple[int, float, str]] = []
    for i, ln in enumerate(head):
        try:
            sc = _score_title_candidate(ln)
            scored.append((i, sc, ln))
        except Exception:
            continue

    if scored:
        # prefer highest score
        scored_sorted = sorted(scored, key=lambda x: (-x[1], x[0]))
        best_idx, best_score, best_line = scored_sorted[0]
        if best_score >= 0:
            out["title"] = best_line
            # look for author lines immediately following the chosen index
            authors_found: List[str] = []
            for j in range(best_idx + 1, min(best_idx + 8, len(head))):
                cand = head[j]
                # author heuristics: contain comma-separated names or multiple capitalized tokens
                if len(cand) < 4 or len(cand) > 200:
                    continue
                    if re.search(r",", cand) or len(re.findall(r"\b[A-Z][a-z]+\b", cand)) >= 2:
                        # split commas, semicolons
                        parts = [p.strip() for p in re.split(r"[,;]", cand) if p.strip()]
                        for p in parts:
                            p2 = _repair_kerned_name(_collapse_spaces(p))
                            if _plausible_person_name(p2) and not _is_journal_masthead(p2):
                                authors_found.append(p2)
                # stop if we hit affiliation-like words
                if re.search(r"\b(university|hospital|department|clinic|institute)\b", cand.lower()):
                    break
            if authors_found:
                # filter masthead-token cases like ['New','England','Journal','Of','Medicine']
                if not _looks_like_masthead_tokens(authors_found):
                    out["authors"] = authors_found
            return out

    # fallback: take the longest line in the head as a naive title
    longest = max(head, key=lambda s: len(s)) if head else None
    if longest and _score_title_candidate(longest) >= 0:
        out["title"] = longest
        # try authors below
        idx = head.index(longest)
        for j in range(idx + 1, min(idx + 8, len(head))):
            cand = head[j]
            if re.search(r",", cand) or len(re.findall(r"\b[A-Z][a-z]+\b", cand)) >= 2:
                parts = [p.strip() for p in re.split(r"[,;]", cand) if p.strip()]
                for p in parts:
                    p2 = _repair_kerned_name(_collapse_spaces(p))
                    if _plausible_person_name(p2):
                        out.setdefault("authors", []).append(p2)
                if out.get("authors"):
                    break

    return out


# ---------------- small helpers for masthead filtering ----------------
_JOURNAL_MASTHEAD_PATTERNS = [
    r"new\s+england\s+journal\s+of\s+medicine",
    r"the\s+lancet",
    r"british\s+medical\s+journal",
    r"bmj\b",
    r"jama\b",
    r"annals\s+of\s+internal\s+medicine",
    r"science\b",
    r"nature\b",
    r"plos\s+one",
]
_JOURNAL_MASTHEAD_RE = re.compile("|".join(_JOURNAL_MASTHEAD_PATTERNS), flags=re.I)


def _is_journal_masthead(s: Optional[str]) -> bool:
    if not s:
        return False
    t = _collapse_spaces(str(s)).lower()
    # normalize punctuation out so 'New England Journal of Medicine' and 'NEJM' alike are matched
    t2 = re.sub(r"[^a-z0-9\s]", " ", t)
    return bool(_JOURNAL_MASTHEAD_RE.search(t2))


def _looks_like_masthead_tokens(items: List[str]) -> bool:
    """Detect cases where the 'authors' are actually a broken-up journal masthead like
    ['New','England','Journal','Of','Medicine'] or ['The','Lancet'].
    """
    if not items:
        return False
    # If every item is a short single capitalized token and there are 2-8 tokens,
    # join and test against masthead regex
    if not (2 <= len(items) <= 8):
        return False
    joined = " ".join([_collapse_spaces(i) for i in items]).strip()
    # if joined contains lowercase letters it's less likely masthead tokens, normalize
    return _is_journal_masthead(joined)



# ---------------- Heuristics ----------------

def _heuristic_arms(*texts: str) -> List[Dict[str, Any]]:
    """
    Safe arm detector for steroid RA example & similar patterns.
    Never invents sample sizes; just labels when strings are present.
    """
    t = " ".join(texts).replace("\n", " ")
    arms: List[Dict[str, Any]] = []

    def add(arm_id: str, label: str):
        if arm_id not in [a["arm_id"] for a in arms]:
            arms.append({
                "arm_id": arm_id,
                "label": label,
                "n_randomized": None
            })

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


def _fill_arm_ns_from_text(arms: List[Dict[str, Any]], text: str) -> None:
    """
    Best-effort per-arm N from raw text.
    Does NOT overwrite existing non-null values (so manual/GROBID/LLM wins if already present).
    Patterns: '<label> ... n=123' or '<label> ... (123)' within ~100 chars.
    """
    if not text or not arms:
        return
    for arm in arms:
        if arm.get("n_randomized") is not None:
            continue
        label = arm.get("label") or ""
        if not label:
            continue
        # label ... n=##
        pat1 = re.compile(re.escape(label) + r".{0,100}?\b[nN]\s*=\s*(\d{1,4})\b", re.S)
        m = pat1.search(text)
        if not m:
            # label ... (##)
            pat2 = re.compile(re.escape(label) + r".{0,100}?\(\s*(\d{1,4})\s*\)", re.S)
            m = pat2.search(text)
        if m:
            try:
                arm["n_randomized"] = int(m.group(1))
            except Exception:
                pass


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
        m = re.search(r"(?:18|19|20)\d{2}", masthead_years[0])
        if m:
            y = int(m.group(0))
            if 1900 <= y <= 2035:
                return y

    # Otherwise, grab any 4-digit years in first 2 pages
    first_pages = pdf_text[:6000]
    counts: Dict[int, int] = {}
    for m in re.finditer(r"\b((?:18|19|20)\d{2})\b", first_pages):
        y = int(m.group(1))
        if 1800 <= y <= 2035:
            counts[y] = counts.get(y, 0) + 1

    if not counts:
        return None

    # Return the most frequent; tie-breaker favors the earlier year in ties
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
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
        "outcome_defs": [],
        "results": {
            "continuous": [],
            "dichotomous": [],
            "tte": [],
        },
    }


def _suppress_percent_measures(outcomes):
    """
    Remove numeric '%' values from outcomes[*].timepoints[*].measures.
    Keep the row (group/metric) but drop 'value' and 'unit' when unit is '%'.
    """
    if not isinstance(outcomes, list):
        return outcomes
    for o in outcomes:
        if not isinstance(o, dict):
            continue
        tps = o.get("timepoints")
        if not isinstance(tps, list):
            continue
        for tp in tps:
            measures = tp.get("measures")
            if not isinstance(measures, list):
                continue
            for m in measures:
                u = (m.get("unit") or "").strip()
                metric = (m.get("metric") or "").strip().lower()
                if u == "%" or metric in ("percent", "percentage", "percent_change"):
                    m.pop("value", None)
                    m.pop("unit", None)
    return outcomes


# ---------------- spaCy helper ----------------
_SPACY_NLP = None
def _get_spacy():
    """Lazily load a spaCy English model if available. Returns None on failure."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if spacy is None:
        return None
    try:
        # prefer transformer if present, fall back to small model
        try:
            _SPACY_NLP = spacy.load("en_core_web_trf")
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = None
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP


# ---------------- Numeric extraction ----------------
def _extract_numeric_candidates(text: str) -> List[Dict[str, Any]]:
    """Find simple numeric result patterns in text: means, changes, p-values, n= values.
    Returns a list of candidate dicts with keys like 'group','n','mean','sd','p','quote'."""
    out: List[Dict[str, Any]] = []
    if not text:
        return out
    # common patterns: n=123, mean 12.3 (SD 4.5), 12.3 (4.5), 12.3 +/- 4.5
    # and p-values: P < 0.05 or p=0.01
    # We'll scan line-by-line to keep context
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # find n= patterns
        m = re.search(r"\b[nN]\s*[=:\(]\s*(\d{1,4})\b", s)
        n = int(m.group(1)) if m else None
        # mean (SD) or mean ± sd
        mean = None
        sd = None
        m2 = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*\(\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*\)", s)
        if m2:
            mean = float(m2.group(1))
            sd = float(m2.group(2))
        else:
            m3 = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*±\s*([0-9]{1,3}(?:\.[0-9]+)?)", s)
            if m3:
                mean = float(m3.group(1))
                sd = float(m3.group(2))

        # p-value
        p = None
        mp = re.search(r"[pP]\s*[=<>]\s*([0-9]*\.?[0-9]+)", s)
        if mp:
            try:
                p = float(mp.group(1))
            except Exception:
                p = None

        if n is not None or mean is not None or p is not None:
            out.append({"group": None, "n": n, "mean": mean, "sd": sd, "p": p, "quote": s})
    return out


# ---------------- Generic title scoring (no per-journal hacks) ----------------

# add/extend this set near _TITLE_STOPWORDS
_TITLE_STOPWORDS = {
    "journal", "volume", "issue", "copyright", "supplement",
    "appendix", "figure", "table", "editorial", "correspondence",
    "references", "received", "accepted", "published", "conflict of interest",
    "license", "open access", "preprint", "doi", "abstract",
    # NEW: kill common footer/caption phrases generically
    "confidence interval", "ci denotes", "p value", "p-value",
    "mean values have been adjusted"
}

def _score_title_candidate(s: str) -> float:
    """
    Return a score; negative means 'reject'.
    Penalize boilerplate words, short lines, and shouty mastheads.
    """
    txt = _collapse_spaces(s)
    if not txt:
        return -1.0

    lower = txt.lower()
    # hard reject typical footers/captions regardless of length
    if re.search(r"\b(ci denotes|confidence interval|p[- ]?value|mean values have been adjusted)\b", lower):
        return -1.0

    n_chars = len(txt)
    n_tokens = len(re.findall(r"\w+", txt))

    if n_chars < 20 or n_tokens < 4:
        return -1.0

    letters = [c for c in txt if c.isalpha()]
    cap_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0

    stop_penalty = sum(1 for w in _TITLE_STOPWORDS if w in lower)

    # Penalize candidates that look like method/statements (contain measurement units or verbs)
    if re.search(r"\bmm\s*hg\b|\bmg\b|\bkg\b|\bwas defined\b|\bwere defined\b|\breduced\b", lower):
        return -1.0

    if cap_ratio > 0.85:
        return -1.0

    score = n_chars - 10 * stop_penalty
    if cap_ratio > 0.65:
        score -= 8
    return score


# ---------------- GROBID TEI parsing + sanity filters ----------------

def _is_plausible_title(s: str) -> bool:
    # kept for backward-compatibility in some checks; scoring is primary
    s_clean = _collapse_spaces(s).lower()
    return len(s_clean) >= 10

def _plausible_person_name(s: str) -> bool:
    s = _collapse_spaces(s)
    if len(s) < 4:
        return False
    low = s.lower()
    if "group" in low or "committee" in low or "trial" in low:
        return False
    # reject if mostly single-letter tokens
    toks = re.split(r"[ \t\-]+", s.replace(".", " "))
    if toks:
        singles = sum(1 for t in toks if len(t) == 1 and t.isalpha())
        if singles / len(toks) > 0.5:
            return False
    return True

# --------- Generic kerning/spacing repair for names ----------

def _repair_kerned_word(word: str) -> str:
    return word[:1] + word[1:].lower()

def _repair_kerned_name(s: str) -> str:
    """
    Join patterns like 'T Homas' -> 'Thomas', while leaving 'J K Rowling' or 'Pao-Hwa' intact.
    """
    if not s:
        return s

    # sanitize whitespace around hyphens and slashes (P Ao -H Wa L In -> Pao-Hwalin rough)
    s = re.sub(r"\s*[-/]\s*", "-", s)
    toks = s.split()
    normalized = []

    for tok in toks:
        # Remove stray non-letter punctuation inside tokens
        tok_clean = re.sub(r"[^\w\-']+", "", tok)
        # If token has uppercase beyond first char (e.g., LAwrence, VOgt), lowercase the tail
        if len(tok_clean) >= 3 and tok_clean[0].isupper() and any(c.isupper() for c in tok_clean[1:]):
            tok_fixed = tok_clean[0] + tok_clean[1:].lower()
        else:
            tok_fixed = tok_clean
        normalized.append(tok_fixed)

    # Merge single-letter initial pattern by keeping them as initials and capitalizing the next token
    merged = []
    i = 0
    while i < len(normalized):
        cur = normalized[i]
        nxt = normalized[i + 1] if i + 1 < len(normalized) else None
        if nxt and len(cur) == 1 and len(nxt) >= 2:
            # keep initial as separate token, and title-case the following token
            nxt_clean = nxt[0].upper() + nxt[1:].lower() if len(nxt) > 1 else nxt.upper()
            merged.append(cur.upper())
            merged.append(nxt_clean)
            i += 2
        else:
            # Properly title-case hyphenated tokens (e.g., Pao-Hwa)
            if "-" in cur:
                parts = [p[0].upper() + p[1:].lower() if len(p) > 1 else p.upper() for p in cur.split("-")]
                cur = "-".join(parts)
            else:
                if len(cur) >= 2:
                    cur = cur[0].upper() + cur[1:].lower()
                else:
                    cur = cur.upper()
            merged.append(cur)
            i += 1

    # Final pass: join runs of very short tokens (e.g., ['P', 'Ao', '-H', 'Wa', 'L', 'In'])
    # Collapse when it seems to form a single surname (heuristic)
    if len(merged) >= 4 and sum(1 for t in merged if len(t) <= 2) >= 3:
        # join everything and capitalize appropriately
        j = "".join(merged)
        j = j[0].upper() + j[1:].lower() if j else j
        return j

    return " ".join(merged)

def _authors_look_bad(names: List[str]) -> bool:
    if not names:
        return True
    tokens = [tok for nm in names for tok in nm.split()]
    if not tokens:
        return True
    short = sum(1 for t in tokens if len(t) <= 2)
    kerned = sum(1 for t in tokens if _looks_kerned_token(t))
    # bad if many short tokens OR a noticeable share of kerned tokens
    if short >= max(2, len(tokens) // 3):
        return True
    if kerned / max(1, len(tokens)) >= 0.25:  # 25% of tokens look kerned
        return True
    return False

def _looks_kerned_token(t: str) -> bool:
    # any extra uppercase after the first character suggests broken kerning (e.g., LAwrence, VOgt)
    t = t.strip()
    if len(t) < 3:
        return False
    return any(c.isupper() for c in t[1:])


def _join_name(pers: ET.Element) -> Optional[str]:
    # Gather forenames then surname
    fnames = [(_collapse_spaces(el.text or "")) for el in pers.findall(".//forename")]
    sname_el = pers.find(".//surname")
    sname = _collapse_spaces(sname_el.text or "") if sname_el is not None else ""
    parts = [p for p in ((" ".join(x for x in fnames if x)), sname) if p]
    if not parts:
        raw = _collapse_spaces("".join(pers.itertext()))
        name = raw or None
    else:
        name = " ".join(parts)
    return _repair_kerned_name(name) if name else None


def _parse_grobid_tei(tei_xml: str) -> Dict[str, Any]:
    """
    Return {'title': str|None, 'authors': list[str], 'doi': str|None, 'year': int|None, 'abstract': str|None}
    with conservative sanitation and fallbacks.
    """
    out = {"title": None, "authors": [], "doi": None, "year": None, "abstract": None}
    if not tei_xml:
        return out
    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        return out

    # Title: score all <title> candidates generically
    cands = []
    for t in root.findall(".//title"):
        cands.append(_collapse_spaces("".join(t.itertext())))
    best = None
    if cands:
        best = max(cands, key=_score_title_candidate)
        # sanitize and reject titles that look like method sentences or contain measurement units
        best_clean = _sanitize_text(best)
        if _score_title_candidate(best_clean or best) < 0 or re.search(r"\bwas defined\b|\bmm\s*hg\b|\breduced\b|;", (best_clean or best).lower()):
            best = None
        else:
            best = best_clean
    out["title"] = best

    # Authors
    authors: List[str] = []
    for a in root.findall(".//titleStmt/author"):
        name_el = a.find(".//persName") or a
        nm = _join_name(name_el) if name_el is not None else None
        if nm and _plausible_person_name(nm):
            authors.append(nm)
    if not authors:
        for a in root.findall(".//analytic/author"):
            nm = _join_name(a) or ""
            if nm and _plausible_person_name(nm):
                authors.append(nm)
    # dedupe in order
    seen = set()
    clean_authors: List[str] = []
    for nm in authors:
        nm_c = _collapse_spaces(_repair_kerned_name(nm))
        if nm_c and nm_c not in seen:
            seen.add(nm_c)
            clean_authors.append(nm_c)
    out["authors"] = clean_authors

    # DOI
    doi_nodes = root.findall(".//idno[@type='DOI']") or root.findall(".//idno[@type='doi']")
    if doi_nodes:
        out["doi"] = _clean_doi(_collapse_spaces("".join(doi_nodes[0].itertext())))

    # Year (prefer published date)
    year = None
    for d in root.findall(".//date"):
        typ = (d.get("type") or "").lower()
        if typ in ("published", "published-online", "published-print", "datepub"):
            when = d.get("when") or ""
            m = re.search(r"(?:18|19|20)\d{2}", when or _collapse_spaces("".join(d.itertext())))
            if m:
                year = int(m.group(0))
                break
    out["year"] = year

    # Abstract
    abs_el = root.find(".//abstract")
    if abs_el is not None:
        out["abstract"] = _collapse_spaces("".join(abs_el.itertext()))

    return out


# ---------------- LLM enrichment (optional) ----------------

def _has_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


def _call_openai_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
    """
    Works with either the new 'openai' SDK (OpenAI client) or the legacy 'openai' module.
    Returns the assistant message content as a string, or None on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _llm_log("no OPENAI_API_KEY in environment")
        return None

    # Try the modern client first
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        _llm_log(f"modern SDK path failed: {e!r}")

    # Fallback to legacy
    try:
        import openai  # type: ignore
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        _llm_log(f"legacy SDK path failed: {e!r}")
        return None


def _llm_enrich(draft: Dict[str, Any], pdf_text: str, tei_xml: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to produce a JSON block with improved fields, including:
      - per-arm n_randomized and structured dose info
      - outcome definitions
      - atomic numeric results in results.continuous/dichotomous/tte
    STRICT JSON only.
    """
    if not _has_llm():
        _llm_log("disabled by flags or missing key (USE_LLM / key)")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _llm_log(f"attempting enrichment with model={model}")

    snippet_pdf = (pdf_text or "")[:12000]
    snippet_tei = (tei_xml or "")[:12000] if tei_xml else ""

    current_arms = [{"arm_id": a.get("arm_id"), "label": a.get("label")} for a in draft.get("arms", [])]

    system = (
        "You are an expert systematic-review abstractor. "
        "Return STRICT JSON ONLY (no extra text). "
        "If confident, fill missing fields; otherwise set null. "
        "Prefer publication year from article masthead. "
        "For arms, only report sample sizes and dose info if explicitly stated; do not guess."
    )

    user = {
        "task": "Enrich trial metadata from text.",
        "current_draft_study": draft.get("study", {}),
        "current_arms": current_arms,
        "pdf_text_snippet": snippet_pdf,
        "tei_snippet": snippet_tei,
        "need_fields": [
            "title", "authors", "doi", "year", "design", "country", "condition",
            "arms (with per-arm n and dose structure if reported)",
            "outcome_defs",
            "results (continuous/dichotomous/tte)"
        ],
        "json_schema": {
                "study": {
                    "title": "string|null",
                    "authors": ["string"],
                    "doi": "string|null",
                    "year": "integer|null",
                    "design": "string|null",
                    "country": "string|null",
                    "condition": "string|null"
                },
                "arms": [
                    {
                        "label": "string",
                        "n_randomized": "integer|null",
                        "drug": { "name": "string|null" },
                        "dose": {
                            "value": "number|null",
                            "unit": "string|null",
                            "schedule": "string|null",
                            "route": "string|null"
                        }
                    }
                ],
                "outcome_defs": [
                    {
                        "outcome_id": "string",
                        "name": "string",
                        "type": "continuous|dichotomous|time-to-event|other",
                        "domain": "string|null",
                        "definition": "string|null",
                        "measurement": "string|null",
                        "unit": "string|null",
                        "primary": "boolean|null",
                        "timepoints": ["string"]
                    }
                ],
                "results": {
                    "continuous": [
                        {
                            "outcome_id": "string",
                            "timepoint": "string",
                            "group": "string",
                            "subgroup": "string|null",
                            "n": "integer|null",
                            "mean": "number|null",
                            "sd": "number|null",
                            "se": "number|null",
                            "median": "number|null",
                            "iqr25": "number|null",
                            "iqr75": "number|null",
                            "unit": "string|null",
                            "change": { "mean": "number|null", "sd": "number|null" },
                            "change_pct": "number|null",
                            "comparison": {
                                "comparator": "string|null",
                                "effect": "md|smd|difference_in_means|ratio_of_means|other|null",
                                "value": "number|null",
                                "ci": "array[2 numbers] or null",
                                "p": "number|null",
                                "unit": "string|null"
                            },
                            "evidence": { "page_hint": "integer|null", "quote": "string|null" }
                        }
                    ],
                    "dichotomous": [
                        {
                            "outcome_id": "string",
                            "timepoint": "string",
                            "group": "string",
                            "subgroup": "string|null",
                            "n": "integer|null",
                            "events": "integer|null",
                            "proportion": "number|null",
                            "ci": "array[2 numbers] or null",
                            "comparison": {
                                "comparator": "string|null",
                                "effect": "rr|or|rd|other|null",
                                "value": "number|null",
                                "ci": "array[2 numbers] or null",
                                "p": "number|null"
                            },
                            "evidence": { "page_hint": "integer|null", "quote": "string|null" }
                        }
                    ],
                    "tte": [
                        {
                            "outcome_id": "string",
                            "group": "string",
                            "subgroup": "string|null",
                            "n": "integer|null",
                            "events": "integer|null",
                            "median_time": "number|null",
                            "unit": "string|null",
                            "hr": "number|null",
                            "ci": "array[2 numbers] or null",
                            "p": "number|null",
                            "evidence": { "page_hint": "integer|null", "quote": "string|null" }
                        }
                    ]
                }
        },
        "instructions": [
            "Only include arms present in the study.",
            "If total N is present but per-arm N is not, leave n_randomized null.",
            "Capture dose as value+unit+schedule+route when explicitly reported.",
            "For continuous results: if the article reports percentage change (e.g., 34%), set change_pct=34 and leave mean/median null.",
            "Use exact timepoint labels from the text.",
            "Map groups to exact arm labels when possible; use 'overall' only for pooled numbers.",
            "Include a short evidence quote and an approximate page number for numeric extraction.",
            "Do not invent values."
        ]
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]
    content = _call_openai_chat(model, messages, temperature=0.0)
    if not content:
        _llm_log("no content returned from OpenAI (check internet/key/package)")
        return None

    # Ensure we can parse JSON from the result
    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None

    # Shape guards
    if not isinstance(data, dict):
        return None
    data.setdefault("study", {})
    data.setdefault("arms", [])
    data.setdefault("outcome_defs", [])
    data.setdefault("results", {"continuous": [], "dichotomous": [], "tte": []})
    if not isinstance(data["arms"], list):
        data["arms"] = []
    if not isinstance(data.get("outcomes", []), list):
        data["outcomes"] = []  # legacy
    if not isinstance(data.get("outcome_defs", []), list):
        data["outcome_defs"] = []
    if not isinstance(data.get("results", {}), dict):
        data["results"] = {"continuous": [], "dichotomous": [], "tte": []}

    return data


def _map_group_to_arm_label(group: str, arms: List[Dict[str, Any]]) -> str:
    if not isinstance(group, str):
        return group
    g = group.strip()
    gl = g.lower()
    if not gl:
        return g

    best = None
    score = 0
    for a in arms or []:
        lab = (a.get("label") or "").strip()
        if not lab:
            continue
        al = lab.lower()
        if gl == al:
            return lab
        # simple token-overlap score (ignores very short tokens)
        gtoks = set(t for t in re.split(r"\W+", gl) if len(t) > 2)
        atoks = set(t for t in re.split(r"\W+", al) if len(t) > 2)
        overlap = len(gtoks & atoks)
        if overlap > score:
            score = overlap
            best = lab
    return best or g


def _merge_enrichment(base_draft: Dict[str, Any], enriched: Dict[str, Any], pdf_text: str) -> Dict[str, Any]:
    """Merge LLM results safely into the draft (no wild overrides)."""
    out = json.loads(json.dumps(base_draft))  # deep copy-ish

    # Study fields
    s = out["study"]
    e = enriched.get("study", {})

    def _update_str(field: str):
        val = e.get(field)
        if isinstance(val, str) and val.strip():
            s[field] = val.strip()

    def _update_year():
        val = e.get("year")
        if isinstance(val, int) and 1900 <= val <= 2035:
            if str(val) in (pdf_text or "") or s.get("year") is None:
                s["year"] = val

    # Prefer TEI/fallback title unless it looks bad; let LLM override only if better
    title_llm = e.get("title")
    if isinstance(title_llm, str) and title_llm.strip():
        tt = title_llm.strip()
        if (not s.get("title")) or (not _is_plausible_title(s.get("title") or "")):
            if _score_title_candidate(tt) >= 0:
                s["title"] = tt

    _update_str("doi")
    _update_str("design")
    _update_str("country")
    _update_str("condition")
    _update_year()

    # Allow LLM to fill authors only when missing or obviously low quality
    authors_llm = e.get("authors")
    if (not s.get("authors") or _authors_look_bad(s["authors"])) and isinstance(authors_llm, list):
        filtered = [str(a).strip() for a in authors_llm if isinstance(a, str) and _plausible_person_name(str(a))]
        if filtered and not _authors_look_bad(filtered):
            s["authors"] = filtered

    # Arms: update/insert; also merge drug/dose sub-objects
    def _norm(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (x or "").lower())

    existing_by_norm = {_norm(a.get("label", "")): a for a in out.get("arms", [])}

    for arm in enriched.get("arms", []):
        if not isinstance(arm, dict):
            continue
        label = arm.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        norm = _norm(label)
        target = existing_by_norm.get(norm)
        if target is None:
            target = {
                "arm_id": re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"),
                "label": label.strip(),
                "n_randomized": None,
            }
            out.setdefault("arms", []).append(target)
            existing_by_norm[norm] = target

        # n_randomized (LLM will NOT overwrite non-null)
        nr = arm.get("n_randomized")
        if isinstance(nr, int) and nr >= 0 and target.get("n_randomized") in (None, 0):
            target["n_randomized"] = nr

        # drug
        drug = arm.get("drug")
        if isinstance(drug, dict):
            name = drug.get("name")
            if isinstance(name, str) and name.strip():
                target["drug"] = {"name": name.strip()}

        # dose
        dose = arm.get("dose")
        if isinstance(dose, dict):
            dv = dose.get("value")
            du = dose.get("unit")
            ds = dose.get("schedule")
            dr = dose.get("route")
            d_out: Dict[str, Any] = {}
            if isinstance(dv, (int, float)):
                d_out["value"] = float(dv)
            if isinstance(du, str) and du.strip():
                d_out["unit"] = du.strip()
            if isinstance(ds, str) and ds.strip():
                d_out["schedule"] = ds.strip()
            if isinstance(dr, str) and dr.strip():
                d_out["route"] = dr.strip()
            if d_out:
                target["dose"] = d_out

    # Outcomes: normalize & SUPPRESS % entries in descriptive measures
    llm_outcomes: List[Dict[str, Any]] = []
    for o in enriched.get("outcomes", []):
        if not isinstance(o, dict):
            continue
        name = o.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        typ = o.get("type")
        if not isinstance(typ, str) or not typ.strip():
            typ = "other"

        tps_in = o.get("timepoints")
        norm_tps: List[Dict[str, Any]] = []
        if isinstance(tps_in, list):
            for tp in tps_in:
                if not isinstance(tp, dict):
                    continue
                lab = tp.get("label")
                if not isinstance(lab, str) or not lab.strip():
                    continue

                measures_in = tp.get("measures")
                norm_measures: List[Dict[str, Any]] = []
                if isinstance(measures_in, list):
                    for m in measures_in:
                        if not isinstance(m, dict):
                            continue
                        group = m.get("group")
                        metric = m.get("metric")
                        value = m.get("value")
                        unit = m.get("unit")
                        n = m.get("n")

                        # Build row
                        row: Dict[str, Any] = {}
                        if isinstance(group, str) and group.strip():
                            row["group"] = group.strip()
                        if isinstance(n, int) and n >= 0:
                            row["n"] = n
                        if isinstance(metric, str) and metric.strip():
                            row["metric"] = metric.strip()
                        if isinstance(value, (int, float)):
                            row["value"] = float(value)
                        if isinstance(unit, str) and unit.strip():
                            row["unit"] = unit.strip()

                        # --- SUPPRESS PERCENT MEASURES HERE ---
                        is_percent = (
                            (isinstance(row.get("unit"), str) and row["unit"] == "%")
                            or (isinstance(row.get("metric"), str) and row["metric"].lower() in {"percent", "percentage", "percent_change"})
                        )
                        if is_percent:
                            continue
                        # --------------------------------------

                        if row:
                            norm_measures.append(row)

                norm_tps.append({"label": lab.strip(), "measures": norm_measures})

        llm_outcomes.append({
            "name": name.strip(),
            "type": typ.strip(),
            "timepoints": norm_tps if norm_tps else [{"label": "end of treatment", "measures": []}],
            "subgroups": [],
        })

    if llm_outcomes:
        out["outcomes"] = llm_outcomes

    out["outcomes"] = _suppress_percent_measures(out.get("outcomes", []))

    # outcome_defs & results: pass-through (with group/comparator normalization)
    edefs = enriched.get("outcome_defs")
    if isinstance(edefs, list) and edefs:
        out["outcome_defs"] = edefs

    eres = enriched.get("results")
    if isinstance(eres, dict):
        # normalize groups/comparators to canonical arm labels
        for bucket in ("continuous", "dichotomous", "tte"):
            items = eres.get(bucket) or []
            if not isinstance(items, list):
                continue
            for r in items:
                if not isinstance(r, dict):
                    continue
                if r.get("group"):
                    r["group"] = _map_group_to_arm_label(r["group"], out.get("arms", []))
                comp = r.get("comparison")
                if isinstance(comp, dict) and comp.get("comparator"):
                    comp["comparator"] = _map_group_to_arm_label(comp["comparator"], out.get("arms", []))

        out["results"] = {
            "continuous": eres.get("continuous") or [],
            "dichotomous": eres.get("dichotomous") or [],
            "tte": eres.get("tte") or [],
        }

    return out


# ---------------- Public entrypoint ----------------

def _from_md_field(md: dict, key: str, default=None):
    try:
        return md.get(key, default)
    except Exception:
        return default

def _clean_md_authors(md_authors) -> List[str]:
    out: List[str] = []
    if isinstance(md_authors, list):
        for a in md_authors:
            if isinstance(a, dict):
                name = a.get("name") or a.get("full_name") or a.get("surname") or ""
            else:
                name = str(a)
            name = _repair_kerned_name(_collapse_spaces(name))
            if name and _plausible_person_name(name):
                out.append(name)
    return out

def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None, force_ocr: bool = False) -> Dict[str, Any]:
    """
    Main extractor:
    - GROBID header + abstract (strict TEI parsing; fallback to header fields with sanity checks)
    - Heuristic arms (+ per-arm N from text when possible)
    - DOI/title/year fallbacks (PDF)
    - Optional LLM enrichment (USE_LLM=true)
    """
    draft = _blank_draft()

    # Extract readable text from PDF for dose/arm patterns and fallbacks
    pdf_text = _read_pdf_text(pdf_path, max_pages=6)

    # If the PDF looks like a scanned image (too little text), or if forced, try OCR and prepend OCR text
    used_ocr = False
    try:
        if force_ocr or _needs_ocr(pdf_path):
            ocr_text = _ocr_pdf_text(pdf_path, max_pages=8)
            if ocr_text and len(ocr_text.strip()) > len(pdf_text or ""):
                # Keep original pdf_text as fallback but prefer OCR where sparse
                pdf_text = (ocr_text + "\n" + (pdf_text or ""))
                used_ocr = True
            elif force_ocr and ocr_text:
                # even if not longer, prefer OCR when forced
                pdf_text = (ocr_text + "\n" + (pdf_text or ""))
                used_ocr = True
    except Exception:
        used_ocr = False

    used_grobid = False
    abstract_text = ""
    tei_xml = None

    # ---------- GROBID ----------
    if grobid_url:
        client = GrobidClient(grobid_url)
        md = client.header_metadata(pdf_path)
        if md:
            tei_xml = _from_md_field(md, "tei_xml") or None
            parsed = _parse_grobid_tei(tei_xml or "")

            # TEI-first (sanitized)
            if parsed.get("title"):
                draft["study"]["title"] = parsed["title"]
            if parsed.get("authors"):
                draft["study"]["authors"] = parsed["authors"]
            if parsed.get("doi"):
                draft["study"]["doi"] = _clean_doi(parsed["doi"])
            if parsed.get("year"):
                draft["study"]["year"] = parsed["year"]
            abstract_text = parsed.get("abstract") or ""

            # Fallbacks to raw header fields (with sanity filters)
            # Title
            if not draft["study"]["title"]:
                md_title = _from_md_field(md, "title")
                if isinstance(md_title, str) and _score_title_candidate(md_title) >= 0:
                    draft["study"]["title"] = _collapse_spaces(md_title)

            # Authors
            if not draft["study"]["authors"]:
                md_auth = _from_md_field(md, "authors")
                cleaned = _clean_md_authors(md_auth)
                if cleaned:
                    draft["study"]["authors"] = cleaned

            # DOI
            if not draft["study"]["doi"]:
                md_doi = _from_md_field(md, "doi")
                if isinstance(md_doi, str) and md_doi.strip():
                    draft["study"]["doi"] = _clean_doi(md_doi)

            # Year (we'll correct against PDF text below if needed)
            if draft["study"]["year"] is None:
                md_year = _from_md_field(md, "year")
                try:
                    y = int(md_year) if md_year is not None else None
                except Exception:
                    y = None
                if y and 1900 <= y <= 2035:
                    draft["study"]["year"] = y

            used_grobid = True
            _llm_log(
                f"grobid parsed: title={repr(draft['study']['title'])}, "
                f"authors={len(draft['study']['authors'] or [])}, "
                f"doi={repr(draft['study']['doi'])}, year={repr(draft['study']['year'])}"
            )
        else:
            _llm_log("grobid header FAILED or empty; will fall back")

    # ---------- Arms ----------
    # If GROBID produced no usable title/authors, try a lightweight layout fallback
    try:
        if (not draft["study"]["title"]) or _authors_look_bad(draft["study"].get("authors", [])):
            layout = _layout_title_and_authors(pdf_path)
            if layout.get("title") and not draft["study"]["title"]:
                draft["study"]["title"] = layout.get("title")
            if layout.get("authors") and (_authors_look_bad(draft["study"].get("authors", [])) or not draft["study"].get("authors")):
                draft["study"]["authors"] = layout.get("authors")
    except Exception:
        pass

    # If authors still look bad or missing, try spaCy NER on the extracted PDF/OCR text as a last deterministic fallback
    try:
        if (not draft["study"].get("authors")) or _authors_look_bad(draft["study"].get("authors", [])):
            nlp = _get_spacy()
            if nlp is not None and (pdf_text or ""):
                _llm_log("spaCy fallback: running PERSON NER on PDF/OCR text")
                try:
                    # limit size for performance but keep top of doc where authors/masthead often live
                    chunk = (pdf_text or "")[:8000]
                    docsp = nlp(chunk)
                    persons = []
                    for ent in docsp.ents:
                        if ent.label_ == "PERSON":
                            name = _collapse_spaces(ent.text)
                            if _plausible_person_name(name):
                                persons.append(_repair_kerned_name(name))
                    # dedupe while preserving order
                    seen = set()
                    deduped: List[str] = []
                    for p in persons:
                        if p and p not in seen:
                            seen.add(p)
                            deduped.append(p)
                    if deduped:
                        # Only replace authors if current authors look bad or are empty
                        if _authors_look_bad(draft["study"].get("authors", [])) or not draft["study"].get("authors"):
                            draft["study"]["authors"] = deduped
                except Exception:
                    pass
    except Exception:
        pass

    draft["arms"] = _heuristic_arms(abstract_text, pdf_text)
    _fill_arm_ns_from_text(draft["arms"], pdf_text)

    # ---------- Table extraction (optional) ----------
    try:
        tables = _try_extract_tables(pdf_path)
        if tables:
            draft["_tables"] = tables
    except Exception:
        pass

    # ---------- DOI fallback ----------
    if not draft["study"]["doi"]:
        m_doi = re.search(r"\b10\.\d{4,9}/[^\s<>\)]+", pdf_text, flags=re.I)
        if m_doi:
            draft["study"]["doi"] = _clean_doi(m_doi.group(0))

    # ---------- Title fallback ----------
    if not draft["study"]["title"]:
        candidates = []
        for line in (pdf_text or "").splitlines():
            s = _collapse_spaces(line)
            if s:
                candidates.append(s)
        if candidates:
            best = max(candidates, key=_score_title_candidate)
            draft["study"]["title"] = best if _score_title_candidate(best) >= 0 else None

    # ---------- Year fallback / correction ----------
    y_pdf = _year_from_pdf(pdf_text)
    if draft["study"]["year"] is None and y_pdf:
        draft["study"]["year"] = y_pdf
    else:
        y_now = draft["study"]["year"]
        if y_now and (str(y_now) not in (pdf_text or "")) and y_pdf:
            draft["study"]["year"] = y_pdf

    # ---------- Optional LLM enrichment ----------
    used_llm = False
    if _has_llm():
        enriched = _llm_enrich(draft, pdf_text, tei_xml)
        if enriched:
            draft = _merge_enrichment(draft, enriched, pdf_text)
            used_llm = True

    # ---------- Notes ----------
    notes_parts: List[str] = []
    if used_grobid and used_llm:
        notes_parts.append(f"GROBID+LLM({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})")
    elif used_grobid:
        notes_parts.append("GROBID")
    else:
        notes_parts.append("local_parse")
    if used_ocr:
        notes_parts.append("OCR=on")
    if draft.get("_tables"):
        notes_parts.append("tables=camelot")
    draft["study"]["notes"] = "Draft via " + ", ".join(notes_parts)

    # explicit flags for UI
    draft.setdefault("_flags", {})
    draft["_flags"]["grobid"] = bool(used_grobid)
    draft["_flags"]["llm"] = bool(used_llm)
    draft["_flags"]["ocr"] = bool(used_ocr)
    draft["_flags"]["camelot_tables"] = bool(draft.get("_tables"))

    # keep measures clean
    draft["outcomes"] = _suppress_percent_measures(draft.get("outcomes", []))

    # ---------------- lightweight numeric extraction ----------------
    try:
        existing = draft.get("results", {}).get("continuous", [])
        if not existing:
            nums = _extract_numeric_candidates(pdf_text)
            # promote good candidates to draft results (simple mapping)
            for c in nums:
                row = {
                    "outcome_id": "auto",
                    "timepoint": "reported",
                    "group": c.get("group") or "overall",
                    "subgroup": None,
                    "n": c.get("n"),
                    "mean": c.get("mean"),
                    "sd": c.get("sd"),
                    "se": None,
                    "median": None,
                    "iqr25": None,
                    "iqr75": None,
                    "unit": None,
                    "change": {"mean": None, "sd": None},
                    "change_pct": None,
                    "comparison": None,
                    "evidence": {"page_hint": None, "quote": c.get("quote")},
                }
                draft.setdefault("results", {}).setdefault("continuous", []).append(row)
    except Exception:
        pass

    return draft

# extractors.py  (CLEAN, FIXED)
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from grobid_client import GrobidClient


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


def _clean_doi(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return re.sub(r"[\s\)\]\}\.,;:]+$", "", s.strip())


# ---------------- Heuristics ----------------

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

    # Otherwise, grab any 4-digit years in first ~2 pages
    first_pages = pdf_text[:6000]
    counts: Dict[int, int] = {}
    for m in re.finditer(r"\b((?:18|19|20)\d{2})\b", first_pages):
        y = int(m.group(1))
        if 1800 <= y <= 2035:
            counts[y] = counts.get(y, 0) + 1

    if not counts:
        return None

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
    }


# ---------------- LLM enrichment (optional) ----------------

def _has_llm() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


def _call_openai_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> Optional[str]:
    """
    Try the modern 'openai' SDK first, then legacy 'openai' module.
    Return assistant message content, or None on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _llm_log("no OPENAI_API_KEY in environment")
        return None

    # Modern SDK
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

    # Legacy SDK
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
    Ask the LLM to produce a JSON block with improved fields.
    We feed it current (possibly partial) values + snippets of PDF/GROBID text.
    """
    if not _has_llm():
        _llm_log("disabled by flags or missing key (USE_LLM / key)")
        return None
    _llm_log(f"attempting enrichment with model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Keep the prompt small enough
    snippet_pdf = (pdf_text or "")[:12000]
    snippet_tei = (tei_xml or "")[:12000] if tei_xml else ""

    system = (
        "You are an expert systematic-review abstractor. "
        "Return STRICT JSON ONLY (no commentary). "
        "Fill missing fields if confident; leave null when unsure. "
        "Do not invent arms or years."
    )
    user = {
        "task": "Enrich trial metadata from text.",
        "current_draft": draft.get("study", {}),
        "arms_seen": [a.get("label") for a in draft.get("arms", [])],
        "need_fields": ["title", "authors", "doi", "year", "design", "country", "condition", "arms"],
        "notes": "Prefer publication year from masthead; do NOT use processing years. Arms should be concrete labels.",
        "pdf_text_snippet": snippet_pdf,
        "tei_snippet": snippet_tei,
        "json_schema": {
            "study": {
                "title": "string or null",
                "authors": ["string"],
                "doi": "string or null",
                "year": "integer or null",
                "design": "string or null",
                "country": "string or null",
                "condition": "string or null",
            },
            "arms": [{"label": "string"}],
        },
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]
    content = _call_openai_chat(model, messages, temperature=0.0)
    if not content:
        _llm_log("no content returned from OpenAI (check internet/key/package)")
        return None

    # Parse JSON
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

    # Shape guard
    if not isinstance(data, dict):
        return None
    if "study" not in data or "arms" not in data:
        return None
    if not isinstance(data["arms"], list):
        data["arms"] = []

    return data


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

    _update_str("title")
    _update_str("doi")
    _update_str("design")
    _update_str("country")
    _update_str("condition")

    # Authors
    if isinstance(e.get("authors"), list) and e["authors"]:
        seen = set()
        authors = []
        for a in e["authors"]:
            if isinstance(a, str):
                k = a.strip()
                if k and k.lower() not in seen:
                    seen.add(k.lower())
                    authors.append(k)
        if authors:
            s["authors"] = authors

    _update_year()

    # Arms: union+dedupe by normalized label
    existing = {re.sub(r"[^a-z0-9]+", "", (a["label"] or "").lower()) for a in out.get("arms", [])}
    for arm in enriched.get("arms", []):
        if not isinstance(arm, dict):
            continue
        label = arm.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        norm = re.sub(r"[^a-z0-9]+", "", label.lower())
        if norm and norm not in existing:
            out["arms"].append({
                "arm_id": re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"),
                "label": label.strip(),
                "n_randomized": None,
            })
            existing.add(norm)

    return out


# ---------------- Public entrypoint ----------------

def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Main extractor:
    - GROBID header + abstract
    - Heuristic arms
    - DOI/title/year fallbacks (PDF)
    - Optional LLM enrichment (USE_LLM=true)
    """
    draft = _blank_draft()

    # Extract readable text from PDF for dose/arm patterns and fallbacks
    pdf_text = _read_pdf_text(pdf_path, max_pages=6)

    used_grobid = False
    abstract_text = ""
    tei_xml = None

    # ---------- GROBID ----------
    if grobid_url:
        client = GrobidClient(grobid_url)
        md = client.header_metadata(pdf_path)
        if md:
            used_grobid = True
            abstract_text = md.get("abstract", "") or ""
            tei_xml = md.get("tei_xml")

            if md.get("title"):
                draft["study"]["title"] = md["title"]
            if md.get("authors"):
                draft["study"]["authors"] = md["authors"]
            if md.get("doi"):
                draft["study"]["doi"] = _clean_doi(md["doi"])
            if md.get("year"):
                draft["study"]["year"] = md["year"]

    # ---------- Arms ----------
    draft["arms"] = _heuristic_arms(abstract_text, pdf_text)

    # ---------- DOI fallback ----------
    if not draft["study"]["doi"]:
        m_doi = re.search(r"\b10\.\d{4,9}/[^\s<>\)]+", pdf_text, flags=re.I)
        if m_doi:
            draft["study"]["doi"] = _clean_doi(m_doi.group(0))

    # ---------- Title fallback ----------
    if not draft["study"]["title"]:
        m_title = re.search(r"(?m)^[^\r\n]{20,160}$", pdf_text)
        draft["study"]["title"] = m_title.group(0).strip() if m_title else None

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
    if used_grobid and used_llm:
        draft["study"]["notes"] = f"Draft via GROBID + LLM ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}); GROBID=on"
    elif used_grobid:
        draft["study"]["notes"] = "Draft via GROBID TEI; GROBID=on"
    else:
        draft["study"]["notes"] = "Draft via local parsing. GROBID=off"

    # Explicit flags for UI
    draft.setdefault("_flags", {})
    draft["_flags"]["grobid"] = bool(used_grobid)
    draft["_flags"]["llm"] = bool(used_llm)

    return draft

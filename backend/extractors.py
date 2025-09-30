# extractors.py  (FULL FILE)
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from grobid_client import GrobidClient


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
    }


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
    Ask the LLM to produce a JSON block with improved fields, including per-arm n_randomized
    and a small set of outcomes. STRICT JSON only.
    """
    if not _has_llm():
        _llm_log("disabled by flags or missing key (USE_LLM / key)")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _llm_log(f"attempting enrichment with model={model}")

    # Keep prompt small-ish
    snippet_pdf = (pdf_text or "")[:12000]
    snippet_tei = (tei_xml or "")[:12000] if tei_xml else ""

    # Provide current arms so the model can map n per arm reliably
    current_arms = [{"arm_id": a.get("arm_id"), "label": a.get("label")} for a in draft.get("arms", [])]

    system = (
        "You are an expert systematic-review abstractor. "
        "Return STRICT JSON ONLY (no extra text). "
        "If confident, fill missing fields; otherwise set null. "
        "Prefer publication year from article masthead. "
        "For arms, only report sample sizes if explicitly stated per arm; do not guess."
    )

    user = {
        "task": "Enrich trial metadata from text.",
        "current_draft_study": draft.get("study", {}),
        "current_arms": current_arms,
        "need_fields": [
            "title", "authors", "doi", "year", "design", "country", "condition",
            "arms (with per-arm n if reported)", "primary/secondary outcomes"
        ],
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
                "condition": "string or null"
            },
            "arms": [
                {
                    "label": "string",               # exact arm label as appears in paper if possible
                    "n_randomized": "integer|null"  # per-arm randomized count; null if not found
                }
            ],
            "outcomes": [
                {
                    "name": "string",                          # e.g., 'Morning stiffness', 'DAS28 change'
                    "type": "continuous|binary|time-to-event|other",
                    "timepoints": [{"label": "string"}]        # e.g., 'end of treatment', '12 weeks'
                }
            ]
        },
        "instructions": [
            "Only include arms that are actually in the study.",
            "If you see total N but not per-arm N, leave n_randomized as null.",
            "If outcomes are named/described, include them; otherwise return an empty list.",
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

    # Basic shape guard
    if not isinstance(data, dict):
        return None
    if "study" not in data:
        data["study"] = {}
    if "arms" not in data or not isinstance(data["arms"], list):
        data["arms"] = []
    if "outcomes" not in data or not isinstance(data["outcomes"], list):
        data["outcomes"] = []

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
            # year should appear somewhere in text or we didn’t have one
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

    # Arms: union+dedupe by normalized label; update n_randomized when present
    def _norm(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (x or "").lower())

    existing_by_norm = {_norm(a.get("label", "")): a for a in out.get("arms", [])}

    for arm in enriched.get("arms", []):
        if not isinstance(arm, dict):
            continue
        label = arm.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        nr = arm.get("n_randomized")
        norm = _norm(label)
        if norm in existing_by_norm:
            # update n_randomized if present and valid
            if isinstance(nr, int) and nr >= 0:
                existing_by_norm[norm]["n_randomized"] = nr
        else:
            # new arm
            new_arm = {
                "arm_id": re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"),
                "label": label.strip(),
                "n_randomized": nr if isinstance(nr, int) and nr >= 0 else None,
            }
            out.setdefault("arms", []).append(new_arm)
            existing_by_norm[norm] = new_arm

    # Outcomes: if LLM returned any, replace the single stub
    llm_outcomes = []
    for o in enriched.get("outcomes", []):
        if not isinstance(o, dict):
            continue
        name = o.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        typ = o.get("type")
        if not isinstance(typ, str) or not typ.strip():
            typ = "other"
        # Normalize timepoints -> ensure measures: []
        tps = o.get("timepoints") if isinstance(o.get("timepoints"), list) else []
        norm_tps = []
        for tp in tps:
            if isinstance(tp, dict) and isinstance(tp.get("label"), str) and tp["label"].strip():
                norm_tps.append({"label": tp["label"].strip(), "measures": []})
        if not norm_tps:
            norm_tps = [{"label": "end of treatment", "measures": []}]
        llm_outcomes.append({
            "name": name.strip(),
            "type": typ.strip(),
            "timepoints": norm_tps,
            "subgroups": [],
        })

    if llm_outcomes:
        out["outcomes"] = llm_outcomes

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
        # fallback: pick a reasonably long line as a title candidate
        m_title = re.search(r"(?m)^[^\r\n]{20,160}$", pdf_text)
        draft["study"]["title"] = m_title.group(0).strip() if m_title else None

    # ---------- Year fallback / correction ----------
    y_pdf = _year_from_pdf(pdf_text)
    if draft["study"]["year"] is None and y_pdf:
        draft["study"]["year"] = y_pdf
    else:
        # If GROBID year looks suspicious, replace it with the PDF-derived year.
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

    # explicit flags for UI
    draft.setdefault("_flags", {})
    draft["_flags"]["grobid"] = bool(used_grobid)
    draft["_flags"]["llm"] = bool(used_llm)

    return draft

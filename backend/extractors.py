# extractors.py  (DROP-IN REPLACEMENT)
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
        # keep old 'outcomes' for UI compatibility
        "outcomes": [
            {
                "name": "Primary outcome (stub)",
                "type": "continuous",
                "timepoints": [{"label": "end of treatment", "measures": []}],
                "subgroups": [],
            }
        ],
        # NEW: normalized layer for DB-ready capture
        "outcome_defs": [],  # [{outcome_id,name,type,domain,definition,measurement,unit,primary,timepoints:[]}]
        "results": {
            "continuous": [],   # [{outcome_id,timepoint,group,n,mean,sd,se,median,iqr25,iqr75,unit,change:{mean,sd},comparison:{comparator,effect,value,ci:[l,u],p,unit}, evidence:{page_hint,quote}}]
            "dichotomous": [],  # [{outcome_id,timepoint,group,n,events,proportion,ci:[l,u],comparison:{...}, evidence:{...}}]
            "tte": []           # [{outcome_id,group,n,events,median_time,unit,hr,ci:[l,u],p,evidence:{...}}]
        },
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
    Ask the LLM to produce a JSON block with improved fields, including:
      - per-arm n_randomized and structured dose info
      - normalized outcome_defs
      - structured results in continuous/dichotomous/tte buckets
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
        "For arms, only report sample sizes and dose info if explicitly stated; do not guess. "
        "Whenever you report a numeric result, include an evidence object with a brief quote and page hint if possible."
    )

    # JSON schema the model should follow
    user = {
        "task": "Enrich trial metadata and structure numeric results.",
        "current_draft_study": draft.get("study", {}),
        "current_arms": current_arms,
        "pdf_text_snippet": snippet_pdf,
        "tei_snippet": snippet_tei,
        "need_fields": [
            "study core fields",
            "per-arm n_randomized and dose structure",
            "outcome definitions",
            "results (continuous, dichotomous, time-to-event)"
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
                    "drug": {"name": "string|null"},
                    "dose": {
                        "value": "number|null",
                        "unit": "string|null",       # mg, mg/day, etc.
                        "schedule": "string|null",   # qd, bid, daily, etc.
                        "route": "string|null"       # oral, IV, etc.
                    }
                }
            ],
            "outcome_defs": [
                {
                    "outcome_id": "string",      # slug or short id, e.g., 'cortisol'
                    "name": "string",            # short descriptor
                    "type": "continuous|dichotomous|time-to-event|other",
                    "domain": "string|null",     # efficacy|safety|QoL|... (optional)
                    "definition": "string|null", # how measured
                    "measurement": "string|null",# mean/SD|median/IQR|N/%|HR|...
                    "unit": "string|null",       # measurement unit if relevant
                    "primary": "boolean|null",   # primary outcome?
                    "timepoints": ["string"]     # e.g., ['baseline','3 months']
                }
            ],
            "results": {
                "continuous": [
                    {
                        "outcome_id": "string",
                        "timepoint": "string",
                        "group": "string",            # arm label or 'overall'
                        "n": "integer|null",
                        "mean": "number|null",
                        "sd": "number|null",
                        "se": "number|null",
                        "median": "number|null",
                        "iqr25": "number|null",
                        "iqr75": "number|null",
                        "unit": "string|null",
                        "change": {
                            "mean": "number|null",
                            "sd": "number|null"
                        },
                        "comparison": {
                            "comparator": "string|null",    # other group label
                            "effect": "md|smd|difference_in_means|ratio_of_means|other|null",
                            "value": "number|null",
                            "ci": ["number","number"] if True else None,
                            "p": "number|null",
                            "unit": "string|null"
                        },
                        "evidence": {
                            "page_hint": "integer|null",
                            "quote": "string|null"
                        }
                    }
                ],
                "dichotomous": [
                    {
                        "outcome_id": "string",
                        "timepoint": "string",
                        "group": "string",
                        "n": "integer|null",
                        "events": "integer|null",
                        "proportion": "number|null",
                        "ci": ["number","number"] if True else None,
                        "comparison": {
                            "comparator": "string|null",
                            "effect": "rr|or|rd|other|null",
                            "value": "number|null",
                            "ci": ["number","number"] if True else None,
                            "p": "number|null"
                        },
                        "evidence": {
                            "page_hint": "integer|null",
                            "quote": "string|null"
                        }
                    }
                ],
                "tte": [
                    {
                        "outcome_id": "string",
                        "group": "string",
                        "n": "integer|null",
                        "events": "integer|null",
                        "median_time": "number|null",
                        "unit": "string|null",
                        "hr": "number|null",
                        "ci": ["number","number"] if True else None,
                        "p": "number|null",
                        "evidence": {
                            "page_hint": "integer|null",
                            "quote": "string|null"
                        }
                    }
                ]
            }
        },
        "instructions": [
            "Only include arms present in the study.",
            "If total N is present but per-arm N is not, leave n_randomized null.",
            "Capture dose as value+unit+schedule+route when explicitly reported.",
            "For results, prefer numeric measures bound to a group and timepoint.",
            "If no numeric data are present, you may still define outcomes but leave numeric fields null.",
            "Always include evidence.page_hint and evidence.quote for numeric rows when feasible.",
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
    if not isinstance(data["outcome_defs"], list):
        data["outcome_defs"] = []
    if not isinstance(data["results"], dict):
        data["results"] = {"continuous": [], "dichotomous": [], "tte": []}
    for k in ("continuous", "dichotomous", "tte"):
        if not isinstance(data["results"].get(k), list):
            data["results"][k] = []

    return data


def _merge_enrichment(base_draft: Dict[str, Any], enriched: Dict[str, Any], pdf_text: str) -> Dict[str, Any]:
    """Merge LLM results safely into the draft (no wild overrides)."""
    out = json.loads(json.dumps(base_draft))  # deep copy-ish

    # --- Study fields ---
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
    _update_year()

    # --- Arms: update or insert; merge drug/dose ---
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

        # n_randomized
        nr = arm.get("n_randomized")
        if isinstance(nr, int) and nr >= 0:
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

    # --- Outcomes (legacy UI compatibility) ---
    # If LLM provided better structured 'outcomes' we could merge, but we leave current behavior as-is.

    # --- NEW: outcome_defs (normalized) ---
    if isinstance(enriched.get("outcome_defs"), list):
        out["outcome_defs"] = []
        for od in enriched["outcome_defs"]:
            if not isinstance(od, dict):
                continue
            oid = od.get("outcome_id")
            name = od.get("name")
            typ = od.get("type")
            if not (isinstance(oid, str) and oid.strip() and isinstance(name, str) and name.strip()):
                continue
            row: Dict[str, Any] = {
                "outcome_id": oid.strip(),
                "name": name.strip(),
                "type": typ.strip() if isinstance(typ, str) and typ.strip() else "other",
                "domain": od.get("domain") if isinstance(od.get("domain"), str) else None,
                "definition": od.get("definition") if isinstance(od.get("definition"), str) else None,
                "measurement": od.get("measurement") if isinstance(od.get("measurement"), str) else None,
                "unit": od.get("unit") if isinstance(od.get("unit"), str) else None,
                "primary": bool(od.get("primary")) if od.get("primary") is not None else None,
                "timepoints": [t for t in od.get("timepoints", []) if isinstance(t, str) and t.strip()],
            }
            out["outcome_defs"].append(row)

    # --- NEW: results buckets (continuous / dichotomous / tte) ---
    def _norm_evidence(ev: Any) -> Dict[str, Any]:
        page = ev.get("page_hint") if isinstance(ev, dict) else None
        quote = ev.get("quote") if isinstance(ev, dict) else None
        ev_out: Dict[str, Any] = {}
        if isinstance(page, int):
            ev_out["page_hint"] = page
        if isinstance(quote, str) and quote.strip():
            ev_out["quote"] = quote.strip()
        return ev_out

    out.setdefault("results", {"continuous": [], "dichotomous": [], "tte": []})

    # Continuous
    cont_in = enriched.get("results", {}).get("continuous", [])
    if isinstance(cont_in, list):
        out["results"]["continuous"] = []
        for r in cont_in:
            if not isinstance(r, dict):
                continue
            row: Dict[str, Any] = {}
            for k in ("outcome_id", "timepoint", "group", "unit"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    row[k] = v.strip()
            for k in ("n",):
                v = r.get(k)
                if isinstance(v, int) and v >= 0:
                    row[k] = v
            for k in ("mean", "sd", "se", "median", "iqr25", "iqr75"):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    row[k] = float(v)
            ch = r.get("change")
            if isinstance(ch, dict):
                ch_out: Dict[str, Any] = {}
                if isinstance(ch.get("mean"), (int, float)):
                    ch_out["mean"] = float(ch["mean"])
                if isinstance(ch.get("sd"), (int, float)):
                    ch_out["sd"] = float(ch["sd"])
                if ch_out:
                    row["change"] = ch_out
            cmp_in = r.get("comparison")
            if isinstance(cmp_in, dict):
                cmp_out: Dict[str, Any] = {}
                if isinstance(cmp_in.get("comparator"), str) and cmp_in["comparator"].strip():
                    cmp_out["comparator"] = cmp_in["comparator"].strip()
                if isinstance(cmp_in.get("effect"), str) and cmp_in["effect"].strip():
                    cmp_out["effect"] = cmp_in["effect"].strip()
                if isinstance(cmp_in.get("value"), (int, float)):
                    cmp_out["value"] = float(cmp_in["value"])
                ci = cmp_in.get("ci")
                if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                    cmp_out["ci"] = [float(ci[0]), float(ci[1])]
                if isinstance(cmp_in.get("p"), (int, float)):
                    cmp_out["p"] = float(cmp_in["p"])
                if isinstance(cmp_in.get("unit"), str) and cmp_in["unit"].strip():
                    cmp_out["unit"] = cmp_in["unit"].strip()
                if cmp_out:
                    row["comparison"] = cmp_out
            ev = _norm_evidence(r.get("evidence"))
            if ev:
                row["evidence"] = ev
            if row:
                out["results"]["continuous"].append(row)

    # Dichotomous
    dich_in = enriched.get("results", {}).get("dichotomous", [])
    if isinstance(dich_in, list):
        out["results"]["dichotomous"] = []
        for r in dich_in:
            if not isinstance(r, dict):
                continue
            row: Dict[str, Any] = {}
            for k in ("outcome_id", "timepoint", "group"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    row[k] = v.strip()
            v = r.get("n")
            if isinstance(v, int) and v >= 0:
                row["n"] = v
            v = r.get("events")
            if isinstance(v, int) and v >= 0:
                row["events"] = v
            v = r.get("proportion")
            if isinstance(v, (int, float)):
                row["proportion"] = float(v)
            ci = r.get("ci")
            if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                row["ci"] = [float(ci[0]), float(ci[1])]
            cmp_in = r.get("comparison")
            if isinstance(cmp_in, dict):
                cmp_out: Dict[str, Any] = {}
                if isinstance(cmp_in.get("comparator"), str) and cmp_in["comparator"].strip():
                    cmp_out["comparator"] = cmp_in["comparator"].strip()
                if isinstance(cmp_in.get("effect"), str) and cmp_in["effect"].strip():
                    cmp_out["effect"] = cmp_in["effect"].strip()
                if isinstance(cmp_in.get("value"), (int, float)):
                    cmp_out["value"] = float(cmp_in["value"])
                ci2 = cmp_in.get("ci")
                if isinstance(ci2, list) and len(ci2) == 2 and all(isinstance(x, (int, float)) for x in ci2):
                    cmp_out["ci"] = [float(ci2[0]), float(ci2[1])]
                if isinstance(cmp_in.get("p"), (int, float)):
                    cmp_out["p"] = float(cmp_in["p"])
                if cmp_out:
                    row["comparison"] = cmp_out
            ev = _norm_evidence(r.get("evidence"))
            if ev:
                row["evidence"] = ev
            if row:
                out["results"]["dichotomous"].append(row)

    # Time-to-event
    tte_in = enriched.get("results", {}).get("tte", [])
    if isinstance(tte_in, list):
        out["results"]["tte"] = []
        for r in tte_in:
            if not isinstance(r, dict):
                continue
            row: Dict[str, Any] = {}
            for k in ("outcome_id", "group", "unit"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    row[k] = v.strip()
            for k in ("n", "events"):
                v = r.get(k)
                if isinstance(v, int) and v >= 0:
                    row[k] = v
            v = r.get("median_time")
            if isinstance(v, (int, float)):
                row["median_time"] = float(v)
            v = r.get("hr")
            if isinstance(v, (int, float)):
                row["hr"] = float(v)
            ci = r.get("ci")
            if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                row["ci"] = [float(ci[0]), float(ci[1])]
            v = r.get("p")
            if isinstance(v, (int, float)):
                row["p"] = float(v)
            ev = _norm_evidence(r.get("evidence"))
            if ev:
                row["evidence"] = ev
            if row:
                out["results"]["tte"].append(row)

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

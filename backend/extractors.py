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
        "outcomes": [
            {
                "name": "Primary outcome (stub)",
                "type": "continuous",
                "timepoints": [{"label": "end of treatment", "measures": []}],
                "subgroups": [],
            }
        ],
        # Slots for defs + numeric results
        "outcome_defs": [],
        "results": {
            "continuous": [],
            "dichotomous": [],
            "tte": [],
        },
    }


# ---------------- Heuristic: per-arm n extraction ----------------
def _tolerant_label_regex(label: str) -> str:
    """Turn an arm label into a whitespace/hyphen tolerant regex."""
    pat = re.escape(label)
    pat = pat.replace(r"\ ", r"\s+")
    pat = pat.replace(r"\-", r"[\-\u2011\u2013\u2014]?")
    return pat


def _dose_number_regex(val: Any) -> Optional[str]:
    """Regex for a dose number allowing 7.5 -> 7[.,·]?5; 9 -> 9(?:\.\d+)?"""
    if not isinstance(val, (int, float)):
        return None
    f = float(val)
    if f.is_integer():
        return rf"{int(f)}(?:\.\d+)?"
    s = f"{f}"
    if "." in s:
        a, b = s.split(".", 1)
        return rf"{re.escape(a)}[.,·]?{re.escape(b)}"
    return re.escape(s)


def _fill_arm_ns_from_text(arms: List[Dict[str, Any]], *texts: str) -> None:
    """
    For each arm with n_randomized=None, try to find a nearby '(n=35)' or similar
    around the arm label or a drug+dose synonym in the provided texts.
    Does NOT overwrite non-null values.
    """
    t = " ".join(texts).replace("\n", " ")
    if not t or not arms:
        return

    for arm in arms:
        if not isinstance(arm, dict):
            continue
        if arm.get("n_randomized") is not None:
            continue

        label = arm.get("label") or ""
        label_pat = _tolerant_label_regex(label)
        pats = [label_pat]

        # Add a drug+dose synonym if available
        drug = (arm.get("drug") or {}).get("name")
        dose = arm.get("dose") or {}
        dv, du = dose.get("value"), dose.get("unit")
        if isinstance(drug, str) and drug.strip() and dv is not None and isinstance(du, str) and du.strip():
            num_pat = _dose_number_regex(dv)
            if num_pat:
                dd_pat = rf"{re.escape(drug)}\s+{num_pat}\s*{re.escape(du)}(?:\s*/\s*day)?"
                pats.append(dd_pat)

        found = None
        for base in pats:
            # label ... (n=35)
            rx1 = rf"{base}[^\.]{{0,80}}\(\s*[nN]\s*=\s*(\d{{1,4}})\s*\)"
            m = re.search(rx1, t, flags=re.I)
            if m:
                found = int(m.group(1)); break

            # label ... n=35
            rx2 = rf"{base}[^\.]{{0,80}}[nN]\s*=\s*(\d{{1,4}})"
            m = re.search(rx2, t, flags=re.I)
            if m:
                found = int(m.group(1)); break

            # (n=35) ... label
            rx3 = rf"[nN]\s*=\s*(\d{{1,4}})[^\.]{{0,80}}{base}"
            m = re.search(rx3, t, flags=re.I)
            if m:
                found = int(m.group(1)); break

            # label ... (35 patients)
            rx4 = rf"{base}[^\.]{{0,80}}\(\s*(\d{{1,4}})\s*(?:patients?|subjects?)\s*\)"
            m = re.search(rx4, t, flags=re.I)
            if m:
                found = int(m.group(1)); break

            # label ... (35)  — sometimes trial descriptions do this
            rx5 = rf"{base}\s*\(\s*(\d{{1,4}})\s*\)"
            m = re.search(rx5, t, flags=re.I)
            if m:
                found = int(m.group(1)); break

        if isinstance(found, int) and found >= 0:
            arm["n_randomized"] = found


# ---------------- LLM helpers ----------------
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


def _normalize_llm_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common LLM quirks: percent changes placed as mean + unit, add 'post-ACTH' hint if needed."""
    try:
        results = data.get("results", {})
        cont = results.get("continuous", []) if isinstance(results, dict) else []
        if isinstance(cont, list):
            for row in cont:
                if not isinstance(row, dict):
                    continue
                ev = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
                quote = (ev.get("quote") or "").lower()

                # If quote contains a '%' and row has change.mean in [-100, 100], treat it as percent change.
                ch = row.get("change") if isinstance(row.get("change"), dict) else None
                if ch and isinstance(ch.get("mean"), (int, float)) and -100 <= ch["mean"] <= 100:
                    if "%" in quote or "percent" in quote:
                        row["change_pct"] = abs(float(ch["mean"]))
                        row["change"] = {"mean": None, "sd": None}
                        if "unit" in row:
                            row["unit"] = None

                # Normalize timepoint strings if ACTH context is present in the quote
                tp = row.get("timepoint")
                if isinstance(tp, str) and ("acth" in quote) and ("acth" not in tp.lower()):
                    row["timepoint"] = f"{tp} post-ACTH"

            data.setdefault("results", {})["continuous"] = cont
    except Exception:
        # Don't let normalization kill the pipeline
        pass
    return data


def _llm_enrich(draft: Dict[str, Any], pdf_text: str, tei_xml: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to produce a JSON block with improved fields, including:
      - per-arm n_randomized and structured dose info
      - outcomes with atomic measures and optional comparisons
      - outcome_defs + results (continuous/dichotomous/tte)
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
            "outcome_defs and results (continuous/dichotomous/tte)"
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
                        "unit": "string|null",
                        "schedule": "string|null",
                        "route": "string|null"
                    }
                }
            ],
            "outcomes": [
                {
                    "name": "string",
                    "type": "continuous|binary|time-to-event|other",
                    "timepoints": [
                        {
                            "label": "string",
                            "measures": [
                                {
                                    "group": "string",
                                    "n": "integer|null",
                                    "metric": "mean|sd|se|median|iqr|proportion|event_count|total|change_mean|change_sd",
                                    "value": "number|null",
                                    "unit": "string|null"
                                }
                            ]
                        }
                    ],
                    "comparisons": [
                        {
                            "groups": ["string", "string"],
                            "effect": "difference_in_means|ratio_of_means|risk_ratio|odds_ratio|hazard_ratio|md|smd|rd|other",
                            "value": "number|null",
                            "ci": "array[2 numbers]|null",
                            "p": "number|null",
                            "unit": "string|null"
                        }
                    ]
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
                        "change": {"mean": "number|null", "sd": "number|null"},
                        "change_pct": "number|null",
                        "comparison": {
                            "comparator": "string|null",
                            "effect": "md|smd|difference_in_means|ratio_of_means|other|null",
                            "value": "number|null",
                            "ci": "array[2 numbers]|null",
                            "p": "number|null",
                            "unit": "string|null"
                        },
                        "evidence": {"page_hint": "integer|null", "quote": "string|null"}
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
                        "ci": "array[2 numbers]|null",
                        "comparison": {
                            "comparator": "string|null",
                            "effect": "rr|or|rd|other|null",
                            "value": "number|null",
                            "ci": "array[2 numbers]|null",
                            "p": "number|null"
                        },
                        "evidence": {"page_hint": "integer|null", "quote": "string|null"}
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
                        "ci": "array[2 numbers]|null",
                        "p": "number|null",
                        "evidence": {"page_hint": "integer|null", "quote": "string|null"}
                    }
                ]
            }
        },
        "instructions": [
            "Only include arms present in the study.",
            "If total N is present but per-arm N is not, leave n_randomized null.",
            "Capture dose as value+unit+schedule+route when explicitly reported.",
            "For continuous results: if the article reports percentage change (e.g., 34%), set change_pct to the numeric percent (0–100) and leave mean/median null; do NOT set unit for percentage change rows.",
            "For continuous results with absolute values, set mean/sd/etc. and set unit accordingly (e.g., µg/dL). Do not place percent changes into mean.",
            "Use timepoints that exactly match the evidence (e.g., '30 minutes', '60 minutes', 'baseline', '3 months'). If the text is clearly 'after ACTH', append 'post-ACTH'.",
            "Map groups to exact arm labels when possible (‘budesonide 9 mg’, ‘placebo’). Use ‘overall’ only if the article gives a pooled number.",
            "Include a short evidence quote and an approximate page number for any numeric extraction.",
            "If data are not explicitly present, leave fields null; do not invent values."
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

    # Normalize common issues
    data = _normalize_llm_results(data)

    # Shape guards
    if not isinstance(data, dict):
        return None
    data.setdefault("study", {})
    data.setdefault("arms", [])
    data.setdefault("outcomes", [])
    if not isinstance(data["arms"], list):
        data["arms"] = []
    if not isinstance(data["outcomes"], list):
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
            if str(val) in (pdf_text or "") or s.get("year") is None:
                s["year"] = val

    _update_str("title")
    _update_str("doi")
    _update_str("design")
    _update_str("country")
    _update_str("condition")
    _update_year()

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

        # n_randomized — DO NOT overwrite a non-null value discovered by heuristics/manual
        nr = arm.get("n_randomized")
        if isinstance(nr, int) and nr >= 0:
            current = target.get("n_randomized")
            if current is None or current == nr:
                target["n_randomized"] = nr
            else:
                _llm_log(f"keeping existing n_randomized={current} for '{target.get('label')}', LLM suggested {nr}")

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

    # Outcomes: merge with structured measures and optional comparisons
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

        # normalize timepoints
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

                        if row:
                            norm_measures.append(row)

                norm_tps.append({"label": lab.strip(), "measures": norm_measures})

        # comparisons (optional)
        comps_out: List[Dict[str, Any]] = []
        comps_in = o.get("comparisons")
        if isinstance(comps_in, list):
            for c in comps_in:
                if not isinstance(c, dict):
                    continue
                groups = c.get("groups")
                effect = c.get("effect")
                value = c.get("value")
                ci = c.get("ci")
                p = c.get("p")
                unit = c.get("unit")
                row: Dict[str, Any] = {}
                if isinstance(groups, list) and len(groups) == 2 and all(isinstance(g, str) and g.strip() for g in groups):
                    row["groups"] = [groups[0].strip(), groups[1].strip()]
                if isinstance(effect, str) and effect.strip():
                    row["effect"] = effect.strip()
                if isinstance(value, (int, float)):
                    row["value"] = float(value)
                if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
                    row["ci"] = [float(ci[0]), float(ci[1])]
                if isinstance(p, (int, float)):
                    row["p"] = float(p)
                if isinstance(unit, str) and unit.strip():
                    row["unit"] = unit.strip()
                if row:
                    comps_out.append(row)

        llm_outcomes.append({
            "name": name.strip(),
            "type": typ.strip(),
            "timepoints": norm_tps if norm_tps else [{"label": "end of treatment", "measures": []}],
            "subgroups": [],
            **({"comparisons": comps_out} if comps_out else {}),
        })

    if llm_outcomes:
        out["outcomes"] = llm_outcomes

    # Pass through outcome_defs & results if provided by the LLM
    edefs = enriched.get("outcome_defs")
    if isinstance(edefs, list) and edefs:
        out["outcome_defs"] = edefs

    eres = enriched.get("results")
    if isinstance(eres, dict):
        out["results"] = {
            "continuous": eres.get("continuous") or [],
            "dichotomous": eres.get("dichotomous") or [],
            "tte": eres.get("tte") or [],
        }

    return out


# ---------------- Public entrypoint ----------------
def extract_first_pass(pdf_path: str, grobid_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Main extractor:
    - GROBID header + abstract
    - Heuristic arms
    - Per-arm n heuristic from text
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

    # ---------- Arms (heuristic) ----------
    draft["arms"] = _heuristic_arms(abstract_text, pdf_text)

    # ---------- Per-arm n (heuristic) ----------
    _fill_arm_ns_from_text(draft["arms"], abstract_text, pdf_text)

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

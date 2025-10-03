# patch_suppress_percent.py
import io, os, re, sys

HERE = os.path.dirname(__file__)
TARGET = os.path.join(HERE, "extractors.py")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

if "_suppress_percent_measures(" in src:
    print("Already patched; nothing to do.")
    sys.exit(0)

helper = r'''
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
                if u == "%" or u.lower() in ("percent", "percentage"):
                    m.pop("value", None)
                    m.pop("unit", None)
    return outcomes
'''

# inject helper just after '_blank_draft' definition
src = re.sub(
    r"(\ndef _blank_draft\(\).*?\n    }\n\n)",
    r"\1" + helper + "\n",
    src,
    flags=re.S
)

# call helper right after we assign llm_outcomes to out["outcomes"]
src = re.sub(
    r'(\n\s*if\s+llm_outcomes:\s*\n\s*out\["outcomes"\]\s*=\s*llm_outcomes\s*\n)',
    r'\1    out["outcomes"] = _suppress_percent_measures(out.get("outcomes", []))\n',
    src,
    flags=re.S
)

with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
    f.write(src)

print("Patched extractors.py to suppress percent measures.")

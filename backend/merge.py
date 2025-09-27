# backend/merge.py
"""
Merge helpers for Trial Abstraction.

What this does
--------------
- Never overwrite extracted draft values with empty/null reviewer values.
- Prefer Reviewer A/B non-empty values; if both provided and differ, mark a conflict.
- Optionally respect "verified" maps: only use a reviewer's value if they marked it verified.
- Works on nested dict/list structures via dot-and-index paths (e.g. "arms[0].label").

Primary entry points
--------------------
merge_reviews(draft, reviewA=None, reviewB=None, verifiedA=None, verifiedB=None)
    -> (merged: dict, conflicts: list[Conflict])

list_conflicts(reviewA=None, reviewB=None)  # independent of draft
    -> list[Conflict]

apply_resolution(draft, resolution_map)
    -> dict  # draft updated with adjudicator's path->value overrides

You can import in main.py, for example:
    from merge import merge_reviews, list_conflicts, apply_resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import re
import copy


# ---------------------------- utilities ----------------------------

def _is_empty(v: Any) -> bool:
    """True if v is an 'empty' value (None, '', only-whitespace string, empty list/dict)."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _canon(v: Any) -> Any:
    """
    Canonicalize values for comparison:
    - Trim strings; map common NA tokens to None
    - Parse numeric strings to numbers
    - Leave dicts/lists as-is
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"na", "n/a", "none", "null", "(not reported)"}:
            return None
        # numeric?
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
            try:
                # return int when possible, else float
                f = float(s)
                return int(f) if f.is_integer() else f
            except Exception:
                pass
        return s
    return v


def _equal(a: Any, b: Any) -> bool:
    """
    Robust equality: canonicalize then compare.
    Numeric values compared with small tolerance.
    """
    ca, cb = _canon(a), _canon(b)
    if isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
        return math.isclose(float(ca), float(cb), rel_tol=1e-9, abs_tol=1e-9)
    return ca == cb


# -------------------------- path handling --------------------------

_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")

def _flatten(obj: Any, parent_key: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Flatten nested dict/list into path->value pairs.
    Lists use [i] indices. Example: arms[0].label
    """
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else k
            _flatten(v, key, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{parent_key}[{i}]"
            _flatten(v, key, out)
    else:
        out[parent_key] = obj
    return out


def _unflatten(flat: Dict[str, Any]) -> Any:
    """Build a nested object back from _flatten output."""
    root: Any = {}
    for path, value in flat.items():
        if not path:
            continue
        tokens = _PATH_TOKEN.findall(path)
        cur = root
        for i, t in enumerate(tokens):
            last = (i == len(tokens) - 1)
            if t.startswith('['):
                idx = int(t[1:-1])
                if not isinstance(cur, list):
                    # convert empty spot to list
                    cur_parent = cur
                    # Determine the parent key to place a list under
                    # If parent was dict, we must have set it already in prior step.
                    # We can't infer the key here, so assume structure already exists.
                    # As a fallback, skip malformed paths gracefully.
                    return root
                while len(cur) <= idx:
                    cur.append({})
                if last:
                    cur[idx] = value
                else:
                    if not isinstance(cur[idx], (dict, list)):
                        cur[idx] = {}
                    cur = cur[idx]
            else:
                key = t
                if last:
                    if isinstance(cur, dict):
                        cur[key] = value
                else:
                    if isinstance(cur, dict):
                        if key not in cur or not isinstance(cur[key], (dict, list)):
                            # decide next container type by peeking next token
                            nxt = tokens[i + 1]
                            cur[key] = [] if nxt.startswith('[') else {}
                        cur = cur[key]
                    else:
                        # malformed path for current structure; ignore
                        return root
    return root


def _set_path(root: Any, path: str, value: Any) -> None:
    """Set a single path in a nested object."""
    tokens = _PATH_TOKEN.findall(path)
    cur = root
    for i, t in enumerate(tokens):
        last = (i == len(tokens) - 1)
        if t.startswith('['):
            idx = int(t[1:-1])
            if not isinstance(cur, list):
                return
            while len(cur) <= idx:
                cur.append({})
            if last:
                cur[idx] = value
            else:
                if not isinstance(cur[idx], (dict, list)):
                    cur[idx] = {}
                cur = cur[idx]
        else:
            key = t
            if last:
                if isinstance(cur, dict):
                    cur[key] = value
            else:
                if isinstance(cur, dict):
                    if key not in cur or not isinstance(cur[key], (dict, list)):
                        nxt = tokens[i + 1]
                        cur[key] = [] if nxt.startswith('[') else {}
                    cur = cur[key]
                else:
                    return


# ------------------------- conflict object -------------------------

@dataclass
class Conflict:
    key: str
    A: Any
    B: Any
    draft: Any

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "A": self.A, "B": self.B, "draft": self.draft}


# ---------------------------- merging -----------------------------

def _consider(v: Any, verified: Optional[bool]) -> Optional[Any]:
    """
    Decide if a review value should be considered:
    - If verified is provided, only consider when verified is True.
    - Ignore empty/null values (treat as no-op).
    """
    if verified is False:
        return None
    if _is_empty(v):
        return None
    return v


def merge_reviews(
    draft: Dict[str, Any],
    reviewA: Optional[Dict[str, Any]] = None,
    reviewB: Optional[Dict[str, Any]] = None,
    verifiedA: Optional[Dict[str, bool]] = None,
    verifiedB: Optional[Dict[str, bool]] = None,
) -> Tuple[Dict[str, Any], List[Conflict]]:
    """
    Merge draft with reviewers' data (A then B), without letting empty/null override.
    If both A and B provide non-empty, non-equal values -> conflict.

    Parameters
    ----------
    draft : dict
        Extracted draft object.
    reviewA, reviewB : dict or None
        Expected shape: {"data": {...}} as saved by your /api/review endpoint.
        If you already pass just the inner dict, that's fine too.
    verifiedA, verifiedB : dict or None
        Maps of path -> bool, to only accept reviewer values marked verified.

    Returns
    -------
    merged : dict
        The merged nested object.
    conflicts : list[Conflict]
        List of conflicts discovered (non-empty, non-equal A vs B values).
    """
    # Normalize incoming structures
    ra_data = (reviewA or {}).get("data") if isinstance(reviewA, dict) and "data" in reviewA else (reviewA or {}) or {}
    rb_data = (reviewB or {}).get("data") if isinstance(reviewB, dict) and "data" in reviewB else (reviewB or {}) or {}
    va = (reviewA or {}).get("verified") if isinstance(reviewA, dict) else (verifiedA or {})
    vb = (reviewB or {}).get("verified") if isinstance(reviewB, dict) else (verifiedB or {})

    f_d = _flatten(draft or {})
    f_a = _flatten(ra_data)
    f_b = _flatten(rb_data)

    keys = sorted(set(f_d.keys()) | set(f_a.keys()) | set(f_b.keys()))
    out: Dict[str, Any] = {}
    conflicts: List[Conflict] = []

    for k in keys:
        v_d = f_d.get(k)
        v_a_raw = f_a.get(k, None)
        v_b_raw = f_b.get(k, None)

        # Respect verified flags if they exist; otherwise treat as not provided when unverified
        v_a = _consider(v_a_raw, (va or {}).get(k) if isinstance(va, dict) else None)
        v_b = _consider(v_b_raw, (vb or {}).get(k) if isinstance(vb, dict) else None)

        if v_a is not None and v_b is not None:
            if _equal(v_a, v_b):
                out[k] = v_a  # consistent override
            else:
                # conflict: keep draft (or A by policy if you prefer),
                # but record conflict so adjudication can resolve.
                out[k] = v_d
                conflicts.append(Conflict(key=k, A=v_a_raw, B=v_b_raw, draft=v_d))
        elif v_a is not None:
            out[k] = v_a
        elif v_b is not None:
            out[k] = v_b
        else:
            # Neither reviewer gave a usable override -> keep draft
            out[k] = v_d

    merged = _unflatten(out)
    return merged, conflicts


def list_conflicts(
    reviewA: Optional[Dict[str, Any]] = None,
    reviewB: Optional[Dict[str, Any]] = None,
) -> List[Conflict]:
    """
    Compare reviewer A vs B directly (ignores draft), reporting only true disagreements
    where both provided non-empty values and they differ.
    """
    ra_data = (reviewA or {}).get("data") if isinstance(reviewA, dict) and "data" in reviewA else (reviewA or {}) or {}
    rb_data = (reviewB or {}).get("data") if isinstance(reviewB, dict) and "data" in reviewB else (reviewB or {}) or {}

    f_a = _flatten(ra_data)
    f_b = _flatten(rb_data)
    keys = sorted(set(f_a.keys()) | set(f_b.keys()))

    out: List[Conflict] = []
    for k in keys:
        va, vb = f_a.get(k), f_b.get(k)
        if _is_empty(va) or _is_empty(vb):
            continue
        if not _equal(va, vb):
            out.append(Conflict(key=k, A=va, B=vb, draft=None))
    return out


# ---------------------- adjudication helpers ----------------------

def apply_resolution(draft: Dict[str, Any], resolution_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply adjudicator's path->value overrides to a copy of the draft.
    Uses same path grammar as _flatten/_unflatten (dots and [i] indices).
    """
    root = copy.deepcopy(draft or {})
    for k, v in (resolution_map or {}).items():
        _safe_set(root, k, v)
    return root


def _safe_set(root: Any, path: str, value: Any) -> None:
    """
    A safer set that will create intermediate containers when possible
    (dict vs list decided by next token).
    """
    tokens = _PATH_TOKEN.findall(path)
    cur = root
    for i, t in enumerate(tokens):
        last = (i == len(tokens) - 1)
        if t.startswith('['):
            idx = int(t[1:-1])
            if not isinstance(cur, list):
                # Try to convert dict slot to list if possible (when path allows)
                return
            while len(cur) <= idx:
                cur.append({})
            if last:
                cur[idx] = value
            else:
                if not isinstance(cur[idx], (dict, list)):
                    # decide next container type
                    nxt = tokens[i + 1]
                    cur[idx] = [] if nxt.startswith('[') else {}
                cur = cur[idx]
        else:
            key = t
            if last:
                if isinstance(cur, dict):
                    cur[key] = value
            else:
                if isinstance(cur, dict):
                    if key not in cur or not isinstance(cur[key], (dict, list)):
                        nxt = tokens[i + 1]
                        cur[key] = [] if nxt.startswith('[') else {}
                    cur = cur[key]
                else:
                    return

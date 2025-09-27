# backend/grobid_client.py
from __future__ import annotations

import re
from typing import Dict, List, Optional

import requests
import xml.etree.ElementTree as ET

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def grobid_is_alive(grobid_url: str, timeout: int = 5) -> bool:
    """
    Quick health check for GROBID.
    """
    try:
        url = grobid_url.rstrip("/") + "/api/isalive"
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200 and (r.text or "").strip().lower().startswith("true")
    except Exception:
        return False


def fetch_tei(pdf_path: str, grobid_url: str, timeout: int = 60) -> Optional[str]:
    """
    Call GROBID /api/processHeaderDocument and return TEI XML as a string.
    We force Accept: application/xml so we don't get BibTeX back.
    Returns None on any error.
    """
    url = grobid_url.rstrip("/") + "/api/processHeaderDocument?consolidateHeader=1"
    headers = {"Accept": "application/xml"}  # important: force XML
    files = {"input": (pdf_path, open(pdf_path, "rb"), "application/pdf")}
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=timeout)
    finally:
        # always close the file handle
        try:
            files["input"][1].close()
        except Exception:
            pass

    if resp.status_code != 200:
        # 406 happens when server can't satisfy Accept; forcing XML should prevent it
        return None

    txt = (resp.text or "").lstrip()
    # Ensure it's actually XML (sometimes GROBID returns BibTeX if Accept wasn't set)
    if not txt.startswith("<"):
        return None
    return txt


# ----------------- TEI parsing helpers -----------------

def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"\s+", " ", s.strip())


def _text(el: Optional[ET.Element]) -> Optional[str]:
    return _norm(el.text) if el is not None else None


def _valid_year(y: str | int) -> Optional[int]:
    try:
        i = int(str(y)[:4])
        if 1900 <= i <= 2035:
            return i
    except Exception:
        pass
    return None


def _pick_title(root: ET.Element) -> Optional[str]:
    # Prefer explicit main title if present
    for xp in [
        ".//tei:fileDesc/tei:titleStmt/tei:title[@type='main']",
        ".//tei:fileDesc/tei:titleStmt/tei:title[1]",
    ]:
        el = root.find(xp, NS)
        if el is not None:
            t = _text(el)
            if t:
                return t
    return None


def _pick_authors(root: ET.Element) -> List[str]:
    out: List[str] = []

    # Prefer authors in analytic/monogr
    candidates = root.findall(".//tei:biblStruct//tei:author", NS)
    if not candidates:
        candidates = root.findall(".//tei:author", NS)

    def dedup_lower(name: str) -> bool:
        lower = name.lower()
        for existing in out:
            if existing.lower() == lower:
                return True
        return False

    for au in candidates:
        surname = _text(au.find(".//tei:surname", NS)) or ""
        forename = (
            _text(au.find(".//tei:forename", NS))
            or _text(au.find(".//tei:persName/tei:forename", NS))
            or ""
        )
        name = (forename + " " + surname).strip()
        if not name:
            # Fallback to any persName text
            name = _text(au.find(".//tei:persName", NS)) or ""
            name = name.strip()

        # Filter out obvious junk
        if not name or len(name) < 3 or len(name) > 80:
            continue
        if dedup_lower(name):
            continue

        out.append(name)

    return out[:30]


def _pick_doi(root: ET.Element) -> Optional[str]:
    for xp in [
        ".//tei:idno[@type='DOI']",
        ".//tei:biblStruct//tei:idno[@type='DOI']",
    ]:
        el = root.find(xp, NS)
        if el is not None:
            doi = _text(el)
            if doi:
                return doi.rstrip(").,;")
    return None


def _pick_year(root: ET.Element) -> Optional[int]:
    """
    Prefer published date in monogr/imprint. Otherwise, scan imprint text and
    pick the most plausible year.
    """
    # 1) explicit published date with @when
    for xp in [
        ".//tei:biblStruct/tei:monogr/tei:imprint/tei:date[@type='published']",
        ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:imprint/tei:date[@type='published']",
    ]:
        el = root.find(xp, NS)
        if el is not None:
            when = el.get("when")
            if when:
                y = _valid_year(when)
                if y:
                    return y
            txt = _text(el)
            if txt:
                m = re.search(r"(19|20)\d{2}", txt)
                if m:
                    y = _valid_year(m.group(0))
                    if y:
                        return y

    # 2) any @when inside imprint
    for xp in [
        ".//tei:biblStruct/tei:monogr/tei:imprint/tei:date[@when]",
        ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:imprint/tei:date[@when]",
    ]:
        for el in root.findall(xp, NS):
            y = _valid_year(el.get("when", ""))
            if y:
                return y

    # 3) scan imprint text and choose max year (commonly the volume year)
    imprint = root.find(".//tei:biblStruct/tei:monogr/tei:imprint", NS)
    if imprint is None:
        imprint = root.find(".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:imprint", NS)

    if imprint is not None:
        txt = " ".join(imprint.itertext())
        years = [int(m.group(0)) for m in re.finditer(r"(19|20)\d{2}", txt)]
        years = [y for y in years if 1900 <= y <= 2035]
        if years:
            return max(years)

    return None


def tei_to_metadata(tei_xml: str) -> Dict[str, object]:
    """
    Convert TEI XML into a light metadata dict:
    { title, authors[], doi, year }
    """
    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        return {}

    return {
        "title": _pick_title(root),
        "authors": _pick_authors(root),
        "doi": _pick_doi(root),
        "year": _pick_year(root),
    }

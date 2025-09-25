# backend/grobid_client.py
from __future__ import annotations
import re
from xml.etree import ElementTree as ET

try:
    import requests  # pip install requests
except Exception:
    requests = None  # graceful fallback

def fetch_grobid_metadata(pdf_path: str, base_url: str = "http://localhost:8070") -> dict:
    """
    Calls a local GROBID server and returns {title, authors, doi, year} when available.
    Falls back to {} if requests is missing or the server is not reachable.
    """
    if requests is None:
        return {}

    url = base_url.rstrip("/") + "/api/processHeaderDocument"
    try:
        with open(pdf_path, "rb") as f:
            r = requests.post(url, files={"input": f}, data={"consolidateHeader": "1"}, timeout=25)
        if r.status_code != 200 or not r.text:
            return {}
        tei = r.text
        # Parse TEI XML
        root = ET.fromstring(tei)
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        out = {}

        title = root.findtext(".//tei:fileDesc/tei:titleStmt/tei:title", namespaces=ns)
        if title:
            out["title"] = title.strip()

        authors = []
        for pers in root.findall(".//tei:fileDesc/tei:titleStmt/tei:author/tei:persName", ns):
            forenames = [el.text for el in pers.findall("tei:forename", ns) if el is not None and el.text]
            surname = pers.findtext("tei:surname", default="", namespaces=ns)
            name = " ".join([*(forenames or []), surname or ""]).strip()
            if name:
                authors.append(name)
        if authors:
            out["authors"] = authors

        doi = None
        for node in root.findall(".//tei:fileDesc/tei:publicationStmt/tei:idno", ns):
            typ = (node.get("type") or "").lower()
            txt = (node.text or "").strip()
            if "doi" in typ or txt.lower().startswith("10."):
                doi = txt
                break
        if doi:
            out["doi"] = doi

        year = None
        date_node = root.find(".//tei:fileDesc/tei:publicationStmt/tei:date", ns)
        if date_node is not None:
            when = date_node.get("when") or ""
            if when[:4].isdigit():
                year = int(when[:4])
            else:
                txt = (date_node.text or "")
                m = re.search(r"(19|20)\d{2}", txt)
                if m:
                    year = int(m.group(0))
        if year:
            out["year"] = year

        return out
    except Exception:
        return {}

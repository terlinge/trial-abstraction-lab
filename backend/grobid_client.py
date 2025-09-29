# grobid_client.py  (FULL FILE)
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional

import requests
import xml.etree.ElementTree as ET

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _txt(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _pick_year(s: str) -> Optional[int]:
    m = re.search(r"\b(?:18|19|20)\d{2}\b", s or "")
    if m:
        y = int(m.group(0))
        if 1800 <= y <= 2100:
            return y
    return None


def _clean_doi(s: str) -> str:
    # remove trailing punctuation/spaces like ")" "," "." ";" etc.
    return re.sub(r"[\s\)\]\}\.,;:]+$", "", (s or "").strip())


class GrobidClient:
    """
    Minimal GROBID client: calls /api/processHeaderDocument and parses TEI.
    Returns dict with: title, authors[], doi, year (from imprint/publication only),
    abstract, tei_xml (string)
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # ---------------- HTTP ----------------

    def _post_header(self, pdf_path: str) -> Optional[str]:
        url = f"{self.base_url}/api/processHeaderDocument?consolidateHeader=1"
        try:
            with open(pdf_path, "rb") as f:
                files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
                r = requests.post(url, files=files, headers={"Accept": "application/xml"}, timeout=60)
            if r.status_code != 200:
                return None
            text = r.text.strip()
            if not text.startswith("<"):
                return None
            return text
        except Exception:
            return None

    # ---------------- Parsing helpers ----------------

    def _authors(self, root: ET.Element) -> List[str]:
        # Prefer analytic authors first, then titleStmt
        sets = [
            root.findall(".//tei:analytic/tei:author//tei:persName", TEI_NS),
            root.findall(".//tei:titleStmt/tei:author//tei:persName", TEI_NS),
            root.findall(".//tei:author//tei:persName", TEI_NS),
        ]
        out: List[str] = []
        for nodes in sets:
            for p in nodes:
                forenames = [_txt(f) for f in p.findall("./tei:forename", TEI_NS) if _txt(f)]
                surname = _txt(p.find("./tei:surname", TEI_NS))
                name = " ".join([*forenames, surname]).strip()
                if name and name not in out:
                    out.append(name)
            if out:
                break
        return out

    def _title(self, root: ET.Element) -> Optional[str]:
        nodes = (
            root.findall(".//tei:analytic/tei:title", TEI_NS)
            or root.findall(".//tei:titleStmt/tei:title", TEI_NS)
            or root.findall(".//tei:monogr/tei:title", TEI_NS)
        )
        if not nodes:
            return None
        titles = sorted((_txt(n) for n in nodes), key=len, reverse=True)
        return titles[0] if titles and titles[0] else None

    def _doi(self, root: ET.Element, tei_xml: str) -> Optional[str]:
        node = root.find(".//tei:idno[@type='DOI']", TEI_NS)
        if node is not None:
            return _clean_doi(_txt(node))
        m = re.search(r"\b10\.\d{4,9}/[^\s<>\)]+", tei_xml)
        return _clean_doi(m.group(0)) if m else None

    def _year_from_nodes(self, root: ET.Element) -> Optional[int]:
        # Only trust publication/imprint-ish nodes. If not found, return None
        xps = [
            ".//tei:monogr/tei:imprint/tei:date[@type='published']",
            ".//tei:monogr/tei:imprint/tei:date[@type='print']",
            ".//tei:monogr/tei:imprint/tei:date[@type='ePublished']",
            ".//tei:monogr/tei:imprint/tei:date",
            ".//tei:imprint/tei:date[@type='published']",
            ".//tei:imprint/tei:date[@type='print']",
            ".//tei:imprint/tei:date[@type='ePublished']",
            ".//tei:imprint/tei:date",
            ".//tei:publicationStmt/tei:date",
        ]
        for xp in xps:
            for dn in root.findall(xp, TEI_NS):
                y = _pick_year(dn.attrib.get("when", "")) or _pick_year(_txt(dn))
                if y:
                    return y
        return None

    def _abstract(self, root: ET.Element) -> str:
        parts = root.findall(".//tei:abstract//tei:p", TEI_NS)
        if not parts:
            parts = root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS)
        return " ".join(_txt(p) for p in parts if _txt(p))

    # ---------------- Public ----------------

    def header_metadata(self, pdf_path: str) -> Dict[str, Any]:
        tei_xml = self._post_header(pdf_path)
        if not tei_xml:
            return {}
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError:
            return {}

        md: Dict[str, Any] = {
            "title": self._title(root),
            "authors": self._authors(root),
            "doi": None,
            "year": None,  # will be set only from imprint/publication nodes
            "abstract": self._abstract(root),
            "tei_xml": tei_xml,
        }
        md["doi"] = self._doi(root, tei_xml)
        md["year"] = self._year_from_nodes(root)  # no fallback to "any year in the TEI"
        return md

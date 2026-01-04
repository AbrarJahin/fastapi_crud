from __future__ import annotations

import re
from .text import normalize_whitespace


def strip_html_tags_fallback(html: str) -> str:
    txt = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", html, flags=re.I)
    txt = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return normalize_whitespace(txt)


def extract_readable_text(html: str) -> str:
    """
    Uses BeautifulSoup if available (no lxml required).
    Falls back to a simple regex stripper if bs4 isn't installed.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return strip_html_tags_fallback(html)

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "canvas", "header", "footer", "nav", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body
    if main is None:
        return ""

    return normalize_whitespace(main.get_text(separator="\n"))

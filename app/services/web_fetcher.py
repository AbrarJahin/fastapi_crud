from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import httpx

from app.utils.html_extract import extract_readable_text


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    user_agent: str,
    max_bytes: int,
) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        r = await client.get(url, headers=headers)
    except Exception:
        return ""

    if r.status_code >= 400:
        return ""

    ctype = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype and "application/xml" not in ctype:
        return ""

    html = r.content[:max_bytes].decode(errors="ignore")
    return extract_readable_text(html)


async def fetch_many(
    client: httpx.AsyncClient,
    urls: List[str],
    user_agent: str,
    max_bytes: int,
    timeout_s: float,
    concurrency: int,
) -> Dict[str, str]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(u: str) -> Tuple[str, str]:
        async with sem:
            try:
                # Per-request read timeout; connect should be fast
                client.timeout = httpx.Timeout(connect=5.0, read=timeout_s, write=30.0, pool=30.0)
                return u, await _fetch_one(client, u, user_agent, max_bytes)
            except Exception:
                return u, ""

    results = await asyncio.gather(*[_one(u) for u in urls])
    return {u: txt for u, txt in results}

from __future__ import annotations

import asyncio
import random
from typing import List

from fastapi import HTTPException

from app.schemas.agent import WebSource


def _ddg_search_sync(query: str, max_results: int) -> List[WebSource]:
    try:
        from duckduckgo_search import DDGS
    except Exception as e:
        raise RuntimeError("duckduckgo_search is not installed. Install: pip install duckduckgo-search") from e

    out: List[WebSource] = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=max_results), start=1):
            url = (r.get("href") or "").strip()
            if not url:
                continue
            out.append(
                WebSource(
                    id=i,
                    title=((r.get("title") or "").strip() or "Untitled"),
                    url=url,
                    snippet=((r.get("body") or "").strip() or None),
                )
            )
    return out


async def ddg_search(query: str, max_results: int) -> List[WebSource]:
    delays = [1.0, 2.0, 4.0]
    last_err: Exception | None = None

    for d in delays:
        try:
            return await asyncio.to_thread(_ddg_search_sync, query, max_results)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "ratelimit" in msg or "rate limit" in msg or "202" in msg:
                await asyncio.sleep(d + random.uniform(0.0, 0.5))
                continue
            raise HTTPException(status_code=502, detail=f"DuckDuckGo search failed: {e}") from e

    raise HTTPException(
        status_code=429,
        detail="DuckDuckGo rate-limited this server. Wait a bit, or add caching / switch provider.",
    ) from last_err

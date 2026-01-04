from __future__ import annotations

from typing import List, Optional, Tuple

import httpx
from fastapi import HTTPException
from duckduckgo_async_search import top_n_result

from app.config import settings
from app.schemas.agent import AskWebResponse, WebSource
from app.services.ollama_client import ollama_post
from app.services.prompt_builder import build_system_prompt, context_block
from app.services.web_fetcher import fetch_many


# -----------------------------
# Public service API
# -----------------------------
async def ask_web_with_ollama(
    query: str,
    max_search_results: int,
    *,
    max_fetch: int = 3,
    include_snippets: bool = True,
    max_chars_per_page: int = 4000,
    system_prompt: Optional[str] = None,
) -> AskWebResponse:
    """
    Search the web (DDG), fetch top pages, and ask Ollama with citations.

    Public signature is intentionally simple:
      ask_web_with_ollama(query, max_search_results)

    Optional kwargs keep route flexibility without bloating the main signature.
    """
    sources = await _search_sources(query, max_search_results)
    if not sources:
        raise HTTPException(status_code=502, detail="No search results found.")

    fetch_sources = sources[: min(max_fetch, len(sources))]

    async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
        context_blocks, used_sources = await _build_context(
            client=client,
            sources=fetch_sources,
            include_snippets=include_snippets,
            max_chars_per_page=max_chars_per_page,
        )

    if not context_blocks:
        raise HTTPException(status_code=502, detail="Unable to fetch readable web content or snippets.")

    answer = await _ollama_answer(
        question=query,
        context_blocks=context_blocks,
        system_prompt=system_prompt,
    )

    return AskWebResponse(
        model=settings.ollama_chat_model,
        answer=answer,
        sources=used_sources,
    )


# -----------------------------
# Small, focused helpers
# -----------------------------
async def _search_sources(query: str, n: int) -> List[WebSource]:
    try:
        items = await top_n_result(query, n=n)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DuckDuckGo search failed: {e}") from e

    sources: List[WebSource] = []
    for i, item in enumerate(items or [], start=1):
        url = (getattr(item, "url", "") or "").strip()
        if not url:
            continue

        title = (getattr(item, "title", "") or "").strip() or "Untitled"
        snippet = getattr(item, "snippet", None)
        if isinstance(snippet, str):
            snippet = snippet.strip() or None

        sources.append(WebSource(id=i, title=title, url=url, snippet=snippet))

    return sources


async def _build_context(
    *,
    client: httpx.AsyncClient,
    sources: List[WebSource],
    include_snippets: bool,
    max_chars_per_page: int,
) -> Tuple[List[str], List[WebSource]]:
    fetched = await fetch_many(
        client=client,
        urls=[s.url for s in sources],
        user_agent=settings.ask_web_user_agent,
        max_bytes=settings.ask_web_max_page_bytes,
        timeout_s=settings.ask_web_fetch_timeout_s,
        concurrency=settings.ask_web_fetch_concurrency,
    )

    context_blocks: List[str] = []
    used_sources: List[WebSource] = []

    for s in sources:
        txt = (fetched.get(s.url) or "").strip()
        if txt:
            context_blocks.append(context_block(s, txt[:max_chars_per_page], "CONTENT"))
            used_sources.append(s)
        elif include_snippets and s.snippet:
            snip = s.snippet.strip()
            context_blocks.append(context_block(s, snip[:max_chars_per_page], "SNIPPET"))
            used_sources.append(s)

    return context_blocks, used_sources


async def _ollama_answer(
    *,
    question: str,
    context_blocks: List[str],
    system_prompt: Optional[str],
) -> str:
    system = build_system_prompt(system_prompt)
    user = (
        f"User question:\n{question}\n\n"
        "WEB CONTEXT:\n"
        + "\n\n---\n\n".join(context_blocks)
        + "\n\nAnswer with citations like [1], [2]."
    )

    payload = {
        "model": settings.ollama_chat_model,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }

    resp = await ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected chat response: {resp}")

    return msg.strip()

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.config import settings


async def ollama_get(path: str, timeout: float = 10.0) -> Dict[str, Any]:
    url = settings.ollama_url(path)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.get(url)
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Timeout contacting Ollama at {url}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
    return r.json()


async def ollama_post(path: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    url = settings.ollama_url(path)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.post(url, json=payload)
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Timeout contacting Ollama at {url}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")
    return r.json()


async def chat(question: str, system_prompt: str) -> str:
    payload = {
        "model": settings.ollama_chat_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }
    resp = await ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {resp}")
    return msg.strip()

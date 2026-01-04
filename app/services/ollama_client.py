from __future__ import annotations

from typing import Any, Dict, Optional, Union

import httpx
from fastapi import HTTPException

from app.config import settings

TimeoutArg = Union[float, httpx.Timeout]  # float = override READ timeout seconds


def _build_timeout(read_s: float) -> httpx.Timeout:
    """
    Build an httpx.Timeout using split values from .env.
    'read_s' is endpoint-specific (chat can be long).
    """
    return httpx.Timeout(
        connect=float(settings.ollama_timeout_connect_s),
        read=float(read_s),
        write=float(settings.ollama_timeout_write_s),
        pool=float(settings.ollama_timeout_pool_s),
    )


def _default_read_timeout_for_get(path: str) -> float:
    """
    Choose sensible read timeout defaults for GET endpoints, from .env.
    """
    # Health/tags should be quick
    if path.startswith("/api/tags"):
        return float(settings.ollama_tags_read_timeout_s)

    # Fallback for other GET endpoints
    return float(settings.ollama_timeout_get_read_s)


def _default_read_timeout_for_post(path: str) -> float:
    """
    Choose sensible read timeout defaults for POST endpoints, from .env.
    """
    if path.startswith("/api/chat"):
        return float(settings.ollama_chat_read_timeout_s)

    if path.startswith("/api/embeddings"):
        return float(settings.ollama_embeddings_read_timeout_s)

    # Fallback for other POST endpoints
    return float(settings.ollama_timeout_post_read_s)


def _resolve_timeout(path: str, method: str, timeout: Optional[TimeoutArg]) -> httpx.Timeout:
    """
    Resolve httpx timeout:
      - None -> use endpoint-specific defaults from .env
      - float -> override READ timeout only (connect/write/pool still from .env)
      - httpx.Timeout -> use as-is
    """
    if isinstance(timeout, httpx.Timeout):
        return timeout

    if method.upper() == "GET":
        read_s = _default_read_timeout_for_get(path)
    else:
        read_s = _default_read_timeout_for_post(path)

    if isinstance(timeout, (int, float)):
        read_s = float(timeout)

    return _build_timeout(read_s)


async def ollama_get(path: str, timeout: Optional[TimeoutArg] = None) -> Dict[str, Any]:
    """
    GET to Ollama using:
      - default timeouts from .env (endpoint-specific)
      - optional per-call override via `timeout=5.0` (READ timeout seconds)
    """
    url = settings.ollama_url(path)
    t = _resolve_timeout(path, "GET", timeout)

    try:
        async with httpx.AsyncClient(timeout=t, trust_env=False) as client:
            r = await client.get(url)
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Timeout contacting Ollama at {url}: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"HTTP error contacting Ollama at {url}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")

    return r.json()


async def ollama_post(
    path: str,
    payload: Dict[str, Any],
    timeout: Optional[TimeoutArg] = None,
) -> Dict[str, Any]:
    """
    POST to Ollama using:
      - default timeouts from .env (endpoint-specific)
      - optional per-call override via `timeout=600.0` (READ timeout seconds)
    """
    url = settings.ollama_url(path)
    t = _resolve_timeout(path, "POST", timeout)

    try:
        async with httpx.AsyncClient(timeout=t, trust_env=False) as client:
            r = await client.post(url, json=payload)
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Timeout contacting Ollama at {url}: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"HTTP error contacting Ollama at {url}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text}")

    return r.json()


# Optional helper: keep your existing service calls simpler
async def chat(question: str, system_prompt: str) -> str:
    """
    Convenience wrapper using /api/chat defaults from .env.
    Override per-call by calling ollama_post(..., timeout=...) directly if needed.
    """
    payload = {
        "model": settings.ollama_chat_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "options": {"num_predict": settings.ollama_num_predict},
        "keep_alive": settings.ollama_keep_alive,
    }

    resp = await ollama_post("/api/chat", payload)
    msg = (resp.get("message") or {}).get("content")

    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {resp}")

    return msg.strip()

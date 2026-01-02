from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(prefix="/agent", tags=["Agent"])


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    embedding: list[float]


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None


class QAResponse(BaseModel):
    model: str
    answer: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=2, description="Language name or ISO code, e.g. 'en', 'bn', 'ar'")
    source_language: Optional[str] = Field(None, description="Optional source language hint")


class TranslateResponse(BaseModel):
    model: str
    translated_text: str


def _ollama_base() -> str:
    """
    Normalize base URL.
    - Removes trailing slashes
    - If user mistakenly sets .../api, remove it (because we add /api/...)
    """
    base = (settings.ollama_base_url or "").strip().rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base


def _build_url(path: str) -> str:
    base = _ollama_base()
    if not base:
        raise HTTPException(status_code=500, detail="OLLAMA_BASE_URL is not configured.")
    # Ensure path starts with "/"
    if not path.startswith("/"):
        path = "/" + path
    # urljoin needs a trailing slash on base to work correctly
    return urljoin(base + "/", path.lstrip("/"))


async def _ollama_get(path: str, timeout: float = 10.0) -> Dict[str, Any]:
    url = _build_url(path)
    try:
        # trust_env=False is CRITICAL on many Windows setups to ignore proxy env vars
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.get(url)
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to Ollama at {url}. "
                   f"Check OLLAMA_BASE_URL in .env and that the host/port are reachable. Error: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=504,
            detail=f"Timeout contacting Ollama at {url}. Error: {e}",
        ) from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code} from {url}: {r.text}")
    return r.json()


async def _ollama_post(path: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    url = _build_url(path)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.post(url, json=payload)
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to Ollama at {url}. "
                   f"Check OLLAMA_BASE_URL in .env and that the host/port are reachable. Error: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=504,
            detail=f"Timeout contacting Ollama at {url}. Error: {e}",
        ) from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code} from {url}: {r.text}")
    return r.json()


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Check that Ollama is reachable."""
    data = await _ollama_get("/api/tags", timeout=5.0)
    return {
        "ok": True,
        "ollama_base_url": _ollama_base(),
        "models_count": len(data.get("models", [])),
        "models": [m.get("name") for m in data.get("models", []) if isinstance(m, dict)],
    }


@router.get("/config")
async def config() -> Dict[str, Any]:
    """Return the configured model names (useful for debugging)."""
    return {
        "ollama_base_url": settings.ollama_base_url,
        "chat_model": settings.ollama_chat_model,
        "embed_model": settings.ollama_embed_model,
        "translate_model": settings.ollama_translate_model,
    }


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Generate embeddings via Ollama."""
    resp = await _ollama_post(
        "/api/embeddings",
        {"model": settings.ollama_embed_model, "prompt": req.text},
        timeout=60.0,
    )
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama embeddings response: {resp}")
    return EmbedResponse(model=settings.ollama_embed_model, embedding=emb)


@router.post("/qa", response_model=QAResponse)
async def qa(req: QARequest) -> QAResponse:
    """Basic QA chat completion using the configured chat model."""
    system = req.system_prompt or (
        "You are a helpful assistant. Answer in the same language as the user question. "
        "Be concise and accurate."
    )

    payload = {
        "model": settings.ollama_chat_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": req.question},
        ],
    }
    resp = await _ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama chat response: {resp}")
    return QAResponse(model=settings.ollama_chat_model, answer=msg.strip())


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """Translate text using the configured translation model."""
    src = f" from {req.source_language}" if req.source_language else ""
    system = (
        "You are a translation engine. Translate the user's text" + src +
        f" into {req.target_language}. Output ONLY the translated text, no explanations."
    )
    payload = {
        "model": settings.ollama_translate_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": req.text},
        ],
    }
    resp = await _ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama chat response: {resp}")
    return TranslateResponse(model=settings.ollama_translate_model, translated_text=msg.strip())

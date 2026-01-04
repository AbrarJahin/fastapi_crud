from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from duckduckgo_async_search import top_n_result
import httpx
from app.services.ask_web_service import ask_web_with_ollama
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.schemas.agent import (
    AskWebRequest, AskWebResponse, EmbedRequest, EmbedResponse,
    QARequest, QAResponse, TranslateRequest, TranslateResponse, WebSource,
)
from app.services.ollama_client import ollama_get, ollama_post
from app.services.prompt_builder import build_system_prompt, context_block
from app.services.web_fetcher import fetch_many
from app.services.web_search_ddg import ddg_search

router = APIRouter(prefix="/agent", tags=["Agent"])

@router.get("/health")
async def health() -> Dict[str, Any]:
    # health should always be fast: use tags endpoint + short read timeout from env
    data = await ollama_get("/api/tags", timeout=5.0)
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "ollama_base_url": settings.ollama_base_url_norm,
        "models_count": len(data.get("models", [])),
        "models": [m.get("name") for m in data.get("models", []) if isinstance(m, dict)],
    }


@router.get("/config")
async def config() -> Dict[str, Any]:
    return {
        "ollama_base_url": settings.ollama_base_url_norm,
        "chat_model": settings.ollama_chat_model,
        "embed_model": settings.ollama_embed_model,
        "translate_model": settings.ollama_translate_model,
        "ask_web_fetch_timeout_s": settings.ask_web_fetch_timeout_s,
        "ask_web_user_agent": settings.ask_web_user_agent,
        "ask_web_max_page_bytes": settings.ask_web_max_page_bytes,
        "ask_web_fetch_concurrency": settings.ask_web_fetch_concurrency,
        # expose the effective timeouts (helps debugging)
        "ollama_timeout_connect_s": settings.ollama_timeout_connect_s,
        "ollama_timeout_get_read_s": settings.ollama_timeout_get_read_s,
        "ollama_timeout_post_read_s": settings.ollama_timeout_post_read_s,
        "ollama_timeout_write_s": settings.ollama_timeout_write_s,
        "ollama_timeout_pool_s": settings.ollama_timeout_pool_s,
        "ollama_chat_read_timeout_s": settings.ollama_chat_read_timeout_s,
        "ollama_embeddings_read_timeout_s": settings.ollama_embeddings_read_timeout_s,
        "ollama_tags_read_timeout_s": settings.ollama_tags_read_timeout_s,
        "ollama_keep_alive": settings.ollama_keep_alive,
        "ollama_num_predict": settings.ollama_num_predict,
    }


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    resp = await ollama_post("/api/embeddings", {"model": settings.ollama_embed_model, "prompt": req.text}, timeout=60.0)
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise HTTPException(status_code=502, detail=f"Unexpected embeddings response: {resp}")
    return EmbedResponse(model=settings.ollama_embed_model, embedding=emb)


@router.post("/qa", response_model=QAResponse)
async def qa(req: QARequest) -> QAResponse:
    system = req.system_prompt or "You are helpful. Answer in the user's language. Be concise."
    payload = {
        "model": settings.ollama_chat_model,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": req.question}],
    }
    resp = await ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected chat response: {resp}")
    return QAResponse(model=settings.ollama_chat_model, answer=msg.strip())


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    system = f"Translate into {req.target_language}. Output only translation."
    payload = {
        "model": settings.ollama_translate_model,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": req.text}],
    }
    resp = await ollama_post("/api/chat", payload, timeout=180.0)
    msg = (resp.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise HTTPException(status_code=502, detail=f"Unexpected translation response: {resp}")
    return TranslateResponse(model=settings.ollama_translate_model, translated_text=msg.strip())


@router.post("/ask-web", response_model=AskWebResponse)
async def ask_web(req: AskWebRequest) -> AskWebResponse:
    return await ask_web_with_ollama(
        req.question,
        req.max_results,
        max_fetch=req.max_fetch,
        include_snippets=req.include_snippets,
        max_chars_per_page=req.max_chars_per_page,
        system_prompt=req.system_prompt,
    )

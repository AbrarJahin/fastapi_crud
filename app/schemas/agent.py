from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


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
    target_language: str = Field(..., min_length=2)
    source_language: Optional[str] = None


class TranslateResponse(BaseModel):
    model: str
    translated_text: str


class AskWebRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_results: int = Field(5, ge=1, le=10)
    max_fetch: int = Field(3, ge=1, le=6)
    max_chars_per_page: int = Field(2500, ge=500, le=8000)
    include_snippets: bool = True
    system_prompt: Optional[str] = None


class WebSource(BaseModel):
    id: int
    title: str
    url: str
    snippet: Optional[str] = None


class AskWebResponse(BaseModel):
    model: str
    answer: str
    sources: List[WebSource]

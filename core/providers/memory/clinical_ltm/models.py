from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryLayer(str, Enum):
    WORKING = "working"
    FACTUAL = "factual"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class WorkingTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    session_id: str
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredMemory(BaseModel):
    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    layer: MemoryLayer
    entity: str
    attribute: str
    value: str
    content: str
    source: str
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    importance: float = 0.5
    weight: float = 0.5
    locked: bool = False
    dedupe_key: str = ""
    evidence: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class FactualMemory(StructuredMemory):
    layer: MemoryLayer = MemoryLayer.FACTUAL
    importance: float = 1.0
    weight: float = 1.0
    locked: bool = True


class EpisodicMemory(StructuredMemory):
    layer: MemoryLayer = MemoryLayer.EPISODIC
    importance: float = 0.6
    weight: float = 0.6


class SemanticMemory(StructuredMemory):
    layer: MemoryLayer = MemoryLayer.SEMANTIC
    importance: float = 0.85
    weight: float = 0.85


class ExtractionResult(BaseModel):
    is_noise: bool = False
    noise_reason: str = ""
    factual_memories: list[FactualMemory] = Field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
    semantic_memories: list[SemanticMemory] = Field(default_factory=list)
    raw_response: str = ""


class RetrievedMemory(BaseModel):
    memory_id: str
    user_id: str
    layer: MemoryLayer
    content: str
    source: str
    observed_at: datetime
    weight: float
    importance: float
    locked: bool
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    working_memory: list[WorkingTurn] = Field(default_factory=list)
    factual_memories: list[RetrievedMemory] = Field(default_factory=list)
    episodic_memories: list[RetrievedMemory] = Field(default_factory=list)
    semantic_memories: list[RetrievedMemory] = Field(default_factory=list)


class ExtractionEnvelope(BaseModel):
    user_id: str
    session_id: str
    messages: list[dict[str, str]]
    query_hint: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)

"""
Pydantic models shared across the graph and the API layer.

EmailPayload mirrors the format Person A's classifier will eventually
receive too, so the contract stays identical between the mocked
score_email() used now and the real one swapped in on integration day.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmailPayload(BaseModel):
    """A single email to run through the pipeline."""

    sender: str = Field(..., description="Sender's email address")
    display_name: str = Field(..., description="Sender's display name")
    subject: str
    body: str
    urls: list[str] = Field(default_factory=list)


class Layer1Score(BaseModel):
    """Output contract for Person A's score_email(). Mocked for now."""

    score: float = Field(..., ge=0.0, le=1.0, description="Phishing probability")
    features: dict = Field(default_factory=dict, description="Feature breakdown, model-internal")


class Verdict(str, Enum):
    ALLOW = "allow"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


class ContextBundle(BaseModel):
    """Gathered by Layer 2's context tools before reasoning."""

    sender_history: dict = Field(default_factory=dict)
    domain_age: dict = Field(default_factory=dict)
    threat_intel: dict = Field(default_factory=dict)


class ReasoningResult(BaseModel):
    """Structured output of the reasoning node — doubles as the SOC ticket note."""

    decision: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str
    evidence_used: list[str] = Field(default_factory=list)
    mitre_technique_ids: list[str] = Field(default_factory=list)
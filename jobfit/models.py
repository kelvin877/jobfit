"""Data models shared across the pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """One piece of evidence from the candidate profile (usually one bullet)."""

    id: str
    section: str
    text: str


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The requirement, rewritten as one short sentence")
    importance: Literal["must", "nice"] = Field(
        description="'must' for required qualifications, 'nice' for preferred/bonus ones"
    )


class RequirementList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[Requirement]


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_index: int = Field(description="Index of the requirement being judged")
    verdict: Literal["met", "partial", "missing"]
    evidence_ids: list[str] = Field(
        description="IDs of the evidence chunks that support the verdict; empty if missing"
    )
    reason: str = Field(description="One sentence explaining the verdict")


class JudgementList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[Judgement]


class RequirementResult(BaseModel):
    requirement: Requirement
    verdict: Literal["met", "partial", "missing"]
    evidence: list[Chunk]
    reason: str
    downgraded: bool = False  # True when the grounding check rejected the model's claim


class FitReport(BaseModel):
    score: int  # 0-100
    band: Literal["strong", "moderate", "weak"]
    results: list[RequirementResult]
    ungrounded_claims: int  # claims rejected by the grounding check

    @property
    def gaps(self) -> list[RequirementResult]:
        return [r for r in self.results if r.verdict == "missing"]

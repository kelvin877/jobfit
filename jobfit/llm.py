"""LLM backends.

`ClaudeLLM` calls the Claude API with structured outputs (JSON schema), so every
response is validated against a Pydantic model. `OfflineLLM` is a deterministic
keyword heuristic used for tests, CI and running the eval without an API key.
"""
from __future__ import annotations

import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .models import Chunk, Judgement, JudgementList, Requirement, RequirementList
from .retrieval import tokenize

T = TypeVar("T", bound=BaseModel)

MODEL = "claude-opus-5-5"

EXTRACT_SYSTEM = """You read job postings and list the qualifications a candidate is \
judged on. Return each distinct technical or experience requirement as one short \
sentence. Mark it "must" if the posting presents it as required, "nice" if it is \
preferred, a bonus or an asset. Skip benefits, company description and generic \
traits such as "team player"."""

JUDGE_SYSTEM = """You assess whether a candidate meets job requirements, using ONLY \
the evidence snippets provided for each requirement. Rules:
- "met": the evidence directly shows the requirement.
- "partial": the evidence shows related or adjacent experience, not the exact thing.
- "missing": the evidence does not support it. Do not infer skills that are not written.
- Cite the evidence IDs you relied on. Never cite an ID that was not provided for that \
requirement. A "met" or "partial" verdict must cite at least one ID."""


class LLM(Protocol):
    def extract_requirements(self, posting: str) -> RequirementList: ...

    def judge(
        self, requirements: list[Requirement], evidence: list[list[Chunk]]
    ) -> JudgementList: ...


class RefusalError(RuntimeError):
    pass


class ClaudeLLM:
    def __init__(self, client=None, model: str = MODEL, effort: str = "medium"):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def _structured(self, system: str, user: str, schema: type[T]) -> T:
        # Server-side fallbacks: if the model declines, the API retries the
        # request on a fallback model inside the same call.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema.model_json_schema()},
            },
        )
        if response.stop_reason == "refusal":
            raise RefusalError(f"Model declined the request (request id {response._request_id})")
        text = next(b.text for b in response.content if b.type == "text")
        return schema.model_validate_json(text)

    def extract_requirements(self, posting: str) -> RequirementList:
        return self._structured(EXTRACT_SYSTEM, f"<posting>\n{posting}\n</posting>", RequirementList)

    def judge(self, requirements: list[Requirement], evidence: list[list[Chunk]]) -> JudgementList:
        parts = []
        for i, (req, chunks) in enumerate(zip(requirements, evidence)):
            ev = "\n".join(f"  [{c.id}] ({c.section}) {c.text}" for c in chunks) or "  (no evidence found)"
            parts.append(f"Requirement {i} ({req.importance}): {req.text}\nEvidence:\n{ev}")
        return self._structured(JUDGE_SYSTEM, "\n\n".join(parts), JudgementList)


class OfflineLLM:
    """Keyword heuristic stand-in for the model. Deterministic; no network."""

    NICE = re.compile(r"\b(nice to have|bonus|asset|preferred|plus)\b", re.I)
    HEADER_NICE = re.compile(r"(nice to have|preferred|bonus|assets?)\s*:?$", re.I)
    HEADER_ANY = re.compile(r":\s*$")

    def extract_requirements(self, posting: str) -> RequirementList:
        reqs: list[Requirement] = []
        in_nice = False
        for raw in posting.splitlines():
            line = raw.strip()
            if not line:
                continue
            if self.HEADER_ANY.search(line) and not line.startswith(("-", "*", "•")):
                in_nice = bool(self.HEADER_NICE.search(line))
                continue
            if not line.startswith(("-", "*", "•")):
                continue
            text = line.lstrip("-*• ").strip()
            nice = in_nice or bool(self.NICE.search(text))
            reqs.append(Requirement(text=text, importance="nice" if nice else "must"))
        return RequirementList(requirements=reqs)

    def judge(self, requirements: list[Requirement], evidence: list[list[Chunk]]) -> JudgementList:
        out = []
        for i, (req, chunks) in enumerate(zip(requirements, evidence)):
            want = set(tokenize(req.text))
            best, best_overlap = [], 0.0
            for c in chunks:
                overlap = len(want & set(tokenize(c.text))) / max(len(want), 1)
                if overlap > best_overlap:
                    best, best_overlap = [c.id], overlap
            if best_overlap >= 0.5:
                verdict = "met"
            elif best_overlap >= 0.2:
                verdict = "partial"
            else:
                verdict, best = "missing", []
            out.append(
                Judgement(
                    requirement_index=i,
                    verdict=verdict,
                    evidence_ids=best,
                    reason=f"keyword overlap {best_overlap:.0%}",
                )
            )
        return JudgementList(judgements=out)

"""The job-fit pipeline: extract -> retrieve -> judge -> ground -> score."""
from __future__ import annotations

from .llm import LLM
from .models import Chunk, FitReport, RequirementResult
from .retrieval import BM25Index

WEIGHTS = {"must": 2.0, "nice": 1.0}
CREDIT = {"met": 1.0, "partial": 0.5, "missing": 0.0}


def band_for(score: int) -> str:
    if score >= 70:
        return "strong"
    if score >= 45:
        return "moderate"
    return "weak"


def analyze(posting: str, chunks: list[Chunk], llm: LLM, k: int = 3) -> FitReport:
    requirements = llm.extract_requirements(posting).requirements
    index = BM25Index(chunks)
    evidence = [index.search(r.text, k=k) for r in requirements]
    judgements = {j.requirement_index: j for j in llm.judge(requirements, evidence).judgements}

    results: list[RequirementResult] = []
    ungrounded = 0
    for i, (req, retrieved) in enumerate(zip(requirements, evidence)):
        j = judgements.get(i)
        if j is None:
            results.append(RequirementResult(requirement=req, verdict="missing", evidence=[], reason="not judged"))
            continue
        allowed = {c.id: c for c in retrieved}
        cited = [allowed[e] for e in j.evidence_ids if e in allowed]
        verdict, downgraded = j.verdict, False
        # Grounding check: a positive verdict must rest on evidence that was
        # actually retrieved for this requirement. Otherwise it is a claim the
        # profile cannot back up, so it is downgraded to "missing".
        if verdict != "missing" and (not cited or len(cited) != len(j.evidence_ids)):
            ungrounded += 1
            verdict, downgraded, cited = "missing", True, []
        results.append(
            RequirementResult(requirement=req, verdict=verdict, evidence=cited, reason=j.reason, downgraded=downgraded)
        )

    total = sum(WEIGHTS[r.requirement.importance] for r in results)
    earned = sum(WEIGHTS[r.requirement.importance] * CREDIT[r.verdict] for r in results)
    score = round(100 * earned / total) if total else 0
    return FitReport(score=score, band=band_for(score), results=results, ungrounded_claims=ungrounded)

from pathlib import Path

import pytest

from jobfit.analyze import analyze, band_for
from jobfit.cli import main
from jobfit.llm import OfflineLLM
from jobfit.models import Judgement, JudgementList, Requirement, RequirementList
from jobfit.profile import parse_profile
from jobfit.retrieval import BM25Index, tokenize

ROOT = Path(__file__).resolve().parent.parent
PROFILE = """
## Backend Developer - Acme (2021-2024)
- Built C# .NET REST APIs backed by SQL Server
- Ran services in Docker on AWS
## Skills
- React, TypeScript
"""


def test_parse_profile_sections_and_ids():
    chunks = parse_profile(PROFILE)
    assert [c.id for c in chunks] == ["E1", "E2", "E3"]
    assert chunks[0].section.startswith("Backend Developer")
    assert chunks[2].section == "Skills"


def test_tokenize_aliases():
    assert "csharp" in tokenize("Strong C# skills")
    assert "dotnet" in tokenize(".NET Core")
    assert "nodejs" in tokenize("Node.js services")


def test_bm25_ranks_relevant_chunk_first():
    index = BM25Index(parse_profile(PROFILE))
    assert index.search("Docker and AWS", k=1)[0].id == "E2"
    assert index.search("Haskell compilers") == []


class LyingLLM:
    """Claims every requirement is met and cites evidence that was never retrieved."""

    def extract_requirements(self, posting):
        return RequirementList(requirements=[
            Requirement(text="C# REST APIs", importance="must"),
            Requirement(text="Kubernetes operators in Go", importance="must"),
        ])

    def judge(self, requirements, evidence):
        return JudgementList(judgements=[
            Judgement(requirement_index=0, verdict="met", evidence_ids=["E1"], reason="ok"),
            Judgement(requirement_index=1, verdict="met", evidence_ids=["E99"], reason="made up"),
        ])


def test_grounding_check_rejects_uncited_claims():
    report = analyze("posting", parse_profile(PROFILE), LyingLLM())
    assert report.results[0].verdict == "met"
    assert report.results[1].verdict == "missing"
    assert report.results[1].downgraded
    assert report.ungrounded_claims == 1
    assert report.score == 50


@pytest.mark.parametrize("score,band", [(90, "strong"), (70, "strong"), (69, "moderate"), (45, "moderate"), (44, "weak")])
def test_bands(score, band):
    assert band_for(score) == band


def test_offline_extraction_marks_nice_to_have():
    posting = "Requirements:\n- Python\n- SQL\n\nNice to have:\n- Kafka\n"
    reqs = OfflineLLM().extract_requirements(posting).requirements
    assert [(r.text, r.importance) for r in reqs] == [("Python", "must"), ("SQL", "must"), ("Kafka", "nice")]


def test_cli_offline(tmp_path, capsys):
    posting = tmp_path / "job.txt"
    posting.write_text("Requirements:\n- C# .NET REST APIs\n- Swift iOS apps\n", encoding="utf-8")
    profile = tmp_path / "profile.md"
    profile.write_text(PROFILE, encoding="utf-8")
    assert main(["analyze", str(posting), "--profile", str(profile), "--offline"]) == 0
    out = capsys.readouterr().out
    assert "Fit score:" in out and "Swift iOS apps" in out

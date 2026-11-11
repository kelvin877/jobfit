"""Command line: python -m jobfit analyze POSTING --profile PROFILE [--offline] [--json]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import analyze
from .llm import ClaudeLLM, OfflineLLM
from .profile import load_profile

ICON = {"met": "[x]", "partial": "[~]", "missing": "[ ]"}


def render(report) -> str:
    lines = [f"Fit score: {report.score}/100 ({report.band})", ""]
    for r in report.results:
        tag = " (ungrounded claim rejected)" if r.downgraded else ""
        lines.append(f"{ICON[r.verdict]} [{r.requirement.importance}] {r.requirement.text}{tag}")
        for c in r.evidence:
            lines.append(f"      <- {c.id} {c.section}: {c.text}")
    if report.gaps:
        lines += ["", "Gaps to address:"] + [f"  - {g.requirement.text}" for g in report.gaps]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jobfit", description="Score a job posting against your profile.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("posting", help="Path to a job posting (.txt/.md), or - for stdin")
    a.add_argument("--profile", required=True, help="Path to your Markdown profile")
    a.add_argument("--offline", action="store_true", help="Use the keyword heuristic instead of Claude")
    a.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args(argv)

    posting = sys.stdin.read() if args.posting == "-" else Path(args.posting).read_text(encoding="utf-8")
    llm = OfflineLLM() if args.offline else ClaudeLLM()
    report = analyze(posting, load_profile(args.profile), llm)
    print(report.model_dump_json(indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

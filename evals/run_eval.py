"""Evaluate the pipeline on labelled postings.

Metrics:
  band_accuracy   - share of cases whose fit band (strong/moderate/weak) matches the label
  grounding_rate  - share of positive verdicts that cite evidence actually retrieved for
                    that requirement (after the grounding check this should be 100%; the
                    raw rate before the check is reported as model_grounding_rate)
  ungrounded      - number of model claims the grounding check had to reject

Usage:
  python evals/run_eval.py --offline      # deterministic heuristic, no API key
  python evals/run_eval.py                # Claude (needs ANTHROPIC_API_KEY or `ant auth login`)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobfit.analyze import analyze  # noqa: E402
from jobfit.llm import ClaudeLLM, OfflineLLM  # noqa: E402
from jobfit.profile import load_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile", default=str(ROOT / "examples" / "profile.md"))
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="Exit 1 if band accuracy is below this")
    args = parser.parse_args()

    llm = OfflineLLM() if args.offline else ClaudeLLM()
    chunks = load_profile(args.profile)
    cases = sorted((ROOT / "evals" / "cases").glob("*.json"))

    correct = positives = ungrounded = 0
    print(f"{'case':<18}{'expected':<10}{'got':<10}{'score':>6}  ungrounded")
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        report = analyze(case["posting"], chunks, llm)
        ok = report.band == case["expected_band"]
        correct += ok
        positives += sum(r.verdict != "missing" for r in report.results) + report.ungrounded_claims
        ungrounded += report.ungrounded_claims
        mark = "" if ok else "  <-- mismatch"
        print(f"{case['name']:<18}{case['expected_band']:<10}{report.band:<10}{report.score:>6}  {report.ungrounded_claims}{mark}")

    accuracy = correct / len(cases)
    model_grounding = 1 - ungrounded / positives if positives else 1.0
    print(f"\nband_accuracy={accuracy:.0%}  model_grounding_rate={model_grounding:.0%}  "
          f"grounding_rate_after_check=100%  ungrounded_rejected={ungrounded}")
    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())

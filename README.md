# jobfit — grounded job-fit analysis with Claude

`jobfit` reads a job posting and your profile (a Markdown CV) and tells you, requirement
by requirement, whether your experience covers it — **citing the exact line of your
profile that proves it**. Claims the model can't back with retrieved evidence are
rejected, so the report never invents skills you don't have.

```
$ jobfit analyze posting.txt --profile examples/profile.md
Fit score: 56/100 (moderate)

[x] [must] Built retrieval-augmented generation (RAG) features with an LLM API
      <- E8 Projects: Built a retrieval-augmented chatbot over travel FAQ documents using an LLM API
[~] [must] Python backend development
      <- E1 Senior Software Developer - Northwind Travel: Build and maintain C# .NET backend services ...
[ ] [must] Vector databases such as pgvector or Pinecone
[x] [nice] TypeScript
      <- E10 Skills: C#, .NET, ASP.NET Core, Python, TypeScript, React, ...

Gaps to address:
  - Vector databases such as pgvector or Pinecone
```

## How it works

```
posting ──► 1. extract requirements (Claude, JSON-schema structured output)
                │
profile.md ──► 2. retrieve evidence per requirement (BM25 over CV bullets, tech-term aliases)
                │
            3. judge each requirement using ONLY its retrieved evidence (Claude, structured output)
                │
            4. grounding check: a "met"/"partial" verdict must cite evidence that was actually
               retrieved for that requirement — otherwise it is downgraded to "missing"
                │
            5. weighted score (must = 2, nice = 1; met = 1, partial = 0.5) → strong / moderate / weak
```

Design choices:

- **Structured outputs, not prompt-and-parse.** Both model calls use `output_config.format`
  with a JSON schema generated from Pydantic models, and responses are validated on return.
- **Grounding is enforced in code, not just requested in the prompt.** The prompt asks the
  model to cite evidence; `analyze.py` verifies the citations and rejects unsupported claims.
  The eval reports how often that check had to step in.
- **Retrieval without extra infrastructure.** A CV is small, so an in-memory BM25 index with an
  alias table (`C#`/`csharp`, `.NET`/`dotnet`, `k8s`/`kubernetes`, ...) is enough — no vector DB.
- **Refusal handling.** Requests opt into server-side fallbacks, and a `refusal` stop reason
  raises a typed error instead of failing on an empty response.
- **Offline mode.** A deterministic keyword heuristic implements the same interface, so tests,
  CI and the eval run without an API key — and it doubles as a baseline to beat.

## Quick start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...            # or `ant auth login`
jobfit analyze my_posting.txt --profile my_profile.md
jobfit analyze my_posting.txt --profile my_profile.md --json   # full report
jobfit analyze my_posting.txt --profile my_profile.md --offline   # no API key
```

Profile format: `## Section` headings (jobs, projects, skills) with `- bullet` lines.
See [`examples/profile.md`](examples/profile.md). Keep your real profile in `private/`
(git-ignored).

## Evaluation

`evals/cases/` holds labelled postings (expected band: strong / moderate / weak) scored
against the sample profile.

```bash
python evals/run_eval.py --offline   # keyword baseline
python evals/run_eval.py             # Claude
```

| Backend | Band accuracy | Notes |
|---|---|---|
| Keyword baseline (offline) | 5/6 (83%) | misses `python_data` (scores 44, one point under "moderate") |
| Claude (`claude-opus-5-5`) | run `python evals/run_eval.py` | also reports `model_grounding_rate` |

The eval also tracks **grounding**: the share of positive verdicts whose citations survive
the grounding check, before the check is applied.

## Tests

```bash
pytest -q
```

Covers profile parsing, tokenizer aliases, BM25 ranking, the grounding check (with a stub
model that cites evidence it was never given), scoring bands, the CLI, and the Claude request
shape (with a stubbed client — no network).

## Project layout

```
jobfit/
  analyze.py    pipeline + grounding check + scoring
  llm.py        ClaudeLLM (Claude API) and OfflineLLM (heuristic baseline)
  retrieval.py  BM25 index and tokenizer
  profile.py    Markdown profile -> evidence chunks
  models.py     Pydantic models (also the JSON schemas sent to Claude)
  cli.py        command line
evals/          labelled cases + eval runner
tests/          pytest suite
```

## Roadmap

- Add more eval cases, including hand-labelled per-requirement verdicts
- Suggest truthful CV rewording for "partial" matches
- Optional embedding retrieval for larger profiles

Built by Kelvin Ling with AI-assisted development (Claude Code). MIT licensed.

"""BM25 retrieval over profile chunks, with a small alias table for tech terms.

No external services: the index is built in memory, which is plenty for a
CV-sized corpus (tens to hundreds of chunks).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .models import Chunk

# Map spelling variants onto one canonical token so "C#", "csharp" and
# ".NET Core" match each other.
ALIASES = {
    "c#": "csharp", "csharp": "csharp", ".net": "dotnet", "dotnet": "dotnet",
    "asp.net": "dotnet", "node.js": "nodejs", "node": "nodejs", "nodejs": "nodejs",
    "react.js": "react", "reactjs": "react", "ts": "typescript", "js": "javascript",
    "postgres": "postgresql", "postgresql": "postgresql", "mssql": "sqlserver",
    "k8s": "kubernetes", "ci/cd": "cicd", "cicd": "cicd", "llm": "llm", "llms": "llm",
    "gen-ai": "genai", "genai": "genai", "aws": "aws", "gcp": "gcp",
}

STOPWORDS = set(
    "a an and are as at be by for from has have in into is it of on or our the to "
    "with you your we will this that experience years year strong ability work "
    "working using use knowledge skills plus".split()
)

TOKEN_RE = re.compile(r"[a-z0-9#+./-]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        tok = raw.strip(".,/-")
        tok = ALIASES.get(raw, ALIASES.get(tok, tok))
        if tok and tok not in STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, i: int) -> float:
        doc, length = self.docs[i], self.lengths[i]
        total = 0.0
        for tok in set(tokenize(query)):
            tf = doc.get(tok, 0)
            if not tf:
                continue
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            total += self.idf.get(tok, 0.0) * norm
        return total

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        scored = [(self.score(query, i), i) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [self.chunks[i] for _, i in scored[:k]]

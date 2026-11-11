"""Parse a Markdown profile into evidence chunks.

Format: each `## Heading` starts a section (a job, a project, "Skills", ...);
each `- bullet` under it becomes one chunk. Plain paragraphs are also chunked.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk


def parse_profile(markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "General"
    counter = 0
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        text = re.sub(r"^[-*]\s+", "", line)
        if len(text) < 3:
            continue
        counter += 1
        chunks.append(Chunk(id=f"E{counter}", section=section, text=text))
    return chunks


def load_profile(path: str | Path) -> list[Chunk]:
    return parse_profile(Path(path).read_text(encoding="utf-8"))

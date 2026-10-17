"""The requirements gate, read by rules: is there a problem statement, and is there a data source with a way to reach it?

`requirement_agent` asks a model the same two questions (see `prompts/requirements.md`). A large
model answers them in the JSON the prompt asks for. The 0.5B model this project is benchmarked with
(`benchmarks/RESULTS.md`) does not: it wrote `requirements: true, problem: ...` as prose in the
first run, said `false` to a message that carried both halves in the second, and once named a
different dataset than the one in the message. Nothing downstream can start on a verdict like that,
and a verdict that cannot be parsed is read as "not yet" — so a clear request never got past the
gate.

The two questions have answers that can be read straight off the message, so they are, here:

    data source   a path that exists (relative to the working directory or the project root), else a file- or URL-looking token,
                  else a `scheme://` connection string
    problem       an intent verb (predict, forecast, classify...) and at least a few words that are not the path

`read_request` returns the same dict the model is asked for, or None when either half is missing.
`console.run.chat` takes it over the model's verdict only when the two disagree about the data (a
missing, invented or unreachable source) or the model's reply did not parse: a model that answers
correctly keeps the last word.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# A file-looking token (`customers.csv`, `data/titanic/train.csv`) or a `db://...`-style connection
# string.
PATH_PATTERN = re.compile(
    r"(?:[\w./~-]+/)?[\w.-]+\.(?:csv|tsv|json|jsonl|parquet|pq|xlsx?|db|sqlite3?)\b|\w+://[\w./:@?=&%-]+",
    re.IGNORECASE,
)

INTENT = re.compile(
    r"\b(predict|forecast|classif\w*|estimate|detect|score|rank|recommend|identify|determine|model|segment|cluster|churn|fraud|default|"
    r"build|prepare|engineer|features?|train|decide|find out|understand)\b",
    re.IGNORECASE,
)

ROOT = Path(__file__).resolve().parent.parent


def reachable(token: str) -> bool:
    """Whether `token` is a file that exists, looked up from the working directory and from the project root."""
    if "://" in token:
        return True
    path = Path(token).expanduser()
    return path.is_file() or (ROOT / path).is_file()


def find_source(text: str) -> str | None:
    """The data source a message names: a file that exists first, else the first file- or URL-looking token."""
    tokens = [m.group(0).rstrip(".,;:)") for m in PATH_PATTERN.finditer(text)]
    for token in tokens:
        if reachable(token):
            return token
    return tokens[0] if tokens else None


def find_problem(text: str, source: str | None) -> str | None:
    """The problem a message states: its first sentences with the source taken out, when they carry an intent and more than a topic's worth of words."""
    cleaned = text.replace(source, "") if source else text
    cleaned = re.sub(r"\bUser:\s*", "", cleaned)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    stated = [s for s in sentences if INTENT.search(s) and len(re.findall(r"[A-Za-z]{2,}", s)) >= 4]
    if not stated:
        return None
    return " ".join(stated[:2]).strip()


def read_request(text: str) -> dict[str, Any] | None:
    """`{"requirements": True, "problem": ..., "data_details": ...}` when `text` carries both halves, else None."""
    source = find_source(text)
    if source is None:
        return None
    problem = find_problem(text, source)
    if problem is None:
        return None
    return {"requirements": True, "problem": problem, "data_details": source}


def agrees(verdict: dict[str, Any] | None, ruled: dict[str, Any]) -> bool:
    """Whether the model's verdict is usable next to what the rules read: met, and naming the same source."""
    if not verdict or str(verdict.get("requirements")).strip().lower() != "true":
        return False
    return ruled["data_details"] in str(verdict.get("data_details", ""))

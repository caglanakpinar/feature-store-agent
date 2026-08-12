"""Async DuckDuckGo web search, backed by a per-agent ChromaDB cache.

Neither agent's knowledge base can answer a question about something that happened after
it was built, and re-searching the web for a prompt seen recently wastes a round trip on
an answer already sitting in storage. This module is the shared piece both agents bind to
their own `dbs/chroma_db`: it runs the search itself, and it decides whether the search
needs to run at all.

    from console.web_search import WebSearchCache, handle_prompt

    cache = WebSearchCache.open(chroma_dir, embedder=my_embedder)   # per agent, once
    results, from_cache = await handle_prompt("current price of ...", cache)

A cached entry is keyed by a normalised hash of the query text, not by similarity search
— the point of the cache is "did we already ask this," not "did we already ask something
like this." Every entry carries `searched_at_epoch` plus two derived booleans,
`is_web_searched_within_7d` and `is_web_searched_within_30d`, so a lookup filters on
recency the way it would filter on any other Chroma metadata field. Those booleans go
stale as time passes — `WebSearchCache.refresh_recency_flags` recomputes them against the
current time, and every read path calls it first.

`data_engineer/dbs/web_search_cache.py` and `feature_engineering/dbs/web_search_cache.py`
are the thin per-agent bindings: each opens this cache against its own `chroma_db`
directory and its own embedder, so the two agents' cached searches never mix.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from uilts.logger import logger

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DEFAULT_MAX_RESULTS = 8
DEFAULT_TIMEOUT = 10.0

SEVEN_DAYS_SECONDS = 7 * 24 * 60 * 60
THIRTY_DAYS_SECONDS = 30 * 24 * 60 * 60

# The HTML endpoint's result markup, not a documented API: DuckDuckGo has no key-free JSON
# search API, and this is the surface every key-free client (browser included) actually
# uses. Two targeted regexes rather than a full HTML parser, since the markup is simple and
# stable enough not to justify a new dependency.
_RESULT_LINK_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


async def duckduckgo_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[WebSearchResult]:
    """Query DuckDuckGo's HTML endpoint and parse the organic results out of the response."""
    if not query.strip():
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; feature-store-agent/1.0; "
            "+https://github.com/your-org/feature-store-agent)"
        ),
    }
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        response = await client.post(DUCKDUCKGO_HTML_URL, data={"q": query})
        response.raise_for_status()
        body = response.text

    links = list(_RESULT_LINK_RE.finditer(body))
    snippets = list(_RESULT_SNIPPET_RE.finditer(body))

    results: list[WebSearchResult] = []
    for position, link in enumerate(links[:max_results]):
        href = html.unescape(link.group("href"))
        title = _strip_tags(link.group("title"))
        snippet = _strip_tags(snippets[position].group("snippet")) if position < len(snippets) else ""
        if title and href:
            results.append(WebSearchResult(title=title, url=href, snippet=snippet))

    logger.info(f"duckduckgo_search: {len(results)} result(s) for {query!r}.")
    return results


# ---------------------------------------------------------------------------
# ChromaDB cache
# ---------------------------------------------------------------------------

def _patch_chromadb_otel() -> None:
    """Let `chromadb` import even when its optional OTLP exporter is missing.

    Chroma only instantiates the exporter when telemetry is enabled, which nothing here
    does, so a harmless stub is enough. Same workaround as `*/dbs/dataset.py` — repeated
    here rather than shared, since this module must not require importing either agent's
    package to load.
    """
    target = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    if target in sys.modules:
        return
    mod = types.ModuleType(target)

    class OTLPSpanExporter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    mod.OTLPSpanExporter = OTLPSpanExporter  # type: ignore[attr-defined]
    sys.modules[target] = mod


class _Embedder(Protocol):
    """What `WebSearchCache` needs from an embedder: batch text -> vectors."""

    def encode(self, texts: list[str]) -> Any: ...


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def cache_key(query: str) -> str:
    """Deterministic cache id for a query, stable across whitespace and case variation."""
    return hashlib.sha256(_normalize_query(query).encode()).hexdigest()


@dataclass
class CachedSearch:
    query: str
    results: list[WebSearchResult]
    searched_at: datetime
    within_7d: bool
    within_30d: bool


class WebSearchCache:
    """A ChromaDB collection of past web searches, filterable by recency.

    Extra params:
        collection: an already-open `chromadb.Collection` named `web_search_cache`.
        embedder: optional; when given, cached entries also carry a vector so they can be
            found by similarity later, not only by exact-query cache key. The cache works
            without one — recency filtering and exact-key lookup do not need vectors.
    """

    COLLECTION_NAME = "web_search_cache"

    def __init__(self, collection: Any, embedder: _Embedder | None = None) -> None:
        self._collection = collection
        self._embedder = embedder

    @classmethod
    def open(cls, chroma_dir: Path, embedder: _Embedder | None = None) -> "WebSearchCache":
        """Open (creating if needed) the `web_search_cache` collection under `chroma_dir`."""
        _patch_chromadb_otel()
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(chroma_dir), settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(name=cls.COLLECTION_NAME)
        return cls(collection, embedder=embedder)

    def _vector(self, text: str) -> list[float] | None:
        if self._embedder is None:
            return None
        return self._embedder.encode([text])[0].tolist()

    def refresh_recency_flags(self, *, now: float | None = None) -> int:
        """Recompute `is_web_searched_within_7d/30d` for every entry. Returns count changed."""
        now = now if now is not None else time.time()
        existing = self._collection.get(include=["metadatas"])
        ids = existing.get("ids", []) or []
        metadatas = existing.get("metadatas", []) or []
        if not ids:
            return 0

        changed_ids: list[str] = []
        changed_metadatas: list[dict[str, Any]] = []
        for doc_id, metadata in zip(ids, metadatas):
            metadata = dict(metadata or {})
            searched_at_epoch = metadata.get("searched_at_epoch")
            if searched_at_epoch is None:
                continue
            age_seconds = now - float(searched_at_epoch)
            within_7d = age_seconds <= SEVEN_DAYS_SECONDS
            within_30d = age_seconds <= THIRTY_DAYS_SECONDS
            if (
                metadata.get("is_web_searched_within_7d") == within_7d
                and metadata.get("is_web_searched_within_30d") == within_30d
            ):
                continue
            metadata["is_web_searched_within_7d"] = within_7d
            metadata["is_web_searched_within_30d"] = within_30d
            changed_ids.append(doc_id)
            changed_metadatas.append(metadata)

        if changed_ids:
            self._collection.update(ids=changed_ids, metadatas=changed_metadatas)
            logger.info(f"WebSearchCache: refreshed recency flags on {len(changed_ids)} entr(y/ies).")
        return len(changed_ids)

    def get(self, query: str, *, max_age_days: int = 7) -> CachedSearch | None:
        """Return the cached search for `query` if one exists within `max_age_days`, else None."""
        if max_age_days not in (7, 30):
            raise ValueError("max_age_days must be 7 or 30 — the two recency flags that are indexed.")

        key = cache_key(query)
        found = self._collection.get(ids=[key], include=["metadatas", "documents"])
        ids = found.get("ids", []) or []
        if not ids:
            return None

        metadata = (found.get("metadatas") or [{}])[0] or {}
        flag = "is_web_searched_within_7d" if max_age_days == 7 else "is_web_searched_within_30d"
        if not metadata.get(flag, False):
            return None

        document = (found.get("documents") or [""])[0] or "[]"
        results = [WebSearchResult(**item) for item in json.loads(document)]
        searched_at = datetime.fromtimestamp(float(metadata["searched_at_epoch"]), tz=timezone.utc)
        return CachedSearch(
            query=metadata.get("query", query),
            results=results,
            searched_at=searched_at,
            within_7d=bool(metadata.get("is_web_searched_within_7d", False)),
            within_30d=bool(metadata.get("is_web_searched_within_30d", False)),
        )

    def put(self, query: str, results: list[WebSearchResult]) -> None:
        """Cache `results` for `query`, timestamped now — freshly within both recency windows."""
        key = cache_key(query)
        now = time.time()
        document = json.dumps([r.to_dict() for r in results])
        metadata: dict[str, Any] = {
            "query": query,
            "searched_at_epoch": now,
            "searched_at_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "source": "duckduckgo",
            "result_count": len(results),
            "is_web_searched_within_7d": True,
            "is_web_searched_within_30d": True,
        }
        vector = self._vector(f"{query}\n\n{document}")

        self._collection.upsert(
            ids=[key],
            documents=[document],
            metadatas=[metadata],
            embeddings=[vector] if vector is not None else None,
        )
        logger.info(f"WebSearchCache: cached {len(results)} result(s) for {query!r}.")

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Prompt-facing orchestration
# ---------------------------------------------------------------------------

async def search_with_cache(
    query: str,
    cache: WebSearchCache,
    *,
    max_age_days: int = 7,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> tuple[list[WebSearchResult], bool]:
    """Serve `query` from the cache if a search within `max_age_days` exists, else search live.

    Returns `(results, from_cache)`. The cache read and write are synchronous ChromaDB
    calls run in a thread via `asyncio.to_thread`, so this coroutine never blocks the event
    loop on either the cache path or the live-search path — the reason to make this async
    at all is so a prompt handler can fire it off without stalling either way.
    """
    await asyncio.to_thread(cache.refresh_recency_flags)
    cached = await asyncio.to_thread(cache.get, query, max_age_days=max_age_days)
    if cached is not None:
        logger.info(f"search_with_cache: cache hit for {query!r} (searched {cached.searched_at.isoformat()}).")
        return cached.results, True

    results = await duckduckgo_search(query, max_results=max_results)
    if results:
        await asyncio.to_thread(cache.put, query, results)
    return results, False


async def handle_prompt(
    prompt: str,
    cache: WebSearchCache,
    *,
    max_age_days: int = 7,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> tuple[list[WebSearchResult], bool]:
    """Entry point for an incoming prompt that needs web search: cache-first, DuckDuckGo on a miss."""
    return await search_with_cache(prompt, cache, max_age_days=max_age_days, max_results=max_results)

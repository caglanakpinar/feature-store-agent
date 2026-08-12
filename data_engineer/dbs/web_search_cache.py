"""The data-engineering agent's web-search cache: a `web_search_cache` collection living
inside this package's `chroma_db`, alongside the knowledge-base collection `dataset.py`
builds.

The search and caching logic itself is `console.web_search` — shared with
`feature_engineering.dbs.web_search_cache` — this module only binds it to this agent's own
storage and embedder, so the two agents' cached searches never mix.

    from data_engineer.dbs.web_search_cache import handle_prompt

    results, from_cache = await handle_prompt("current interest rate")
"""

from __future__ import annotations

from console.web_search import (  # noqa: F401
    WebSearchCache,
    WebSearchResult,
    cache_key,
    duckduckgo_search,
)
from console.web_search import handle_prompt as _handle_prompt
from console.web_search import search_with_cache as _search_with_cache
from data_engineer.dbs.dataset import CHROMA_DIR, FAISS_DIR, SyntheticEmbedder

__all__ = [
    "open_cache",
    "handle_prompt",
    "search_with_cache",
    "WebSearchCache",
    "WebSearchResult",
    "cache_key",
    "duckduckgo_search",
]

_cache: WebSearchCache | None = None


def open_cache() -> WebSearchCache:
    """Open (creating if needed) this agent's `web_search_cache` collection.

    Reuses the embedder already fitted and saved by `dataset.py`'s knowledge-base build,
    if one has been built yet, so cached search results land in the same vector space as
    the rest of this agent's `chroma_db` — and so this module works with no embedder at
    all (exact-key lookup only) before that build has run.
    """
    embedder = None
    embedder_dir = FAISS_DIR / "embedder"
    if embedder_dir.exists():
        embedder = SyntheticEmbedder.load(embedder_dir)
    return WebSearchCache.open(CHROMA_DIR, embedder=embedder)


def _get_cache() -> WebSearchCache:
    global _cache
    if _cache is None:
        _cache = open_cache()
    return _cache


async def handle_prompt(
    prompt: str,
    *,
    max_age_days: int = 7,
    max_results: int = 8,
) -> tuple[list[WebSearchResult], bool]:
    """Cache-first web search for an incoming prompt, scoped to this agent's cache."""
    return await _handle_prompt(
        prompt, _get_cache(), max_age_days=max_age_days, max_results=max_results
    )


async def search_with_cache(
    query: str,
    *,
    max_age_days: int = 7,
    max_results: int = 8,
) -> tuple[list[WebSearchResult], bool]:
    return await _search_with_cache(
        query, _get_cache(), max_age_days=max_age_days, max_results=max_results
    )

"""Thin, named handles onto the agents `agentic_configurations.yaml` declares under `agents:`.

Building one is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    agent = build_agent("chat_agent", configs)

— wrapped the same way `data_engineer.agents.AgentHandle` and `feature_engineering.agents.AgentHandle`
wrap their own packages' agents, so a caller writes `ChatAgent.run(question=...)` rather than resolving
the name and the config directory itself. Not imported from either of those packages: `console` is what
assembles their steps into a pipeline, so the dependency runs the other way — copying the same dozen
lines here keeps that direction intact instead of reaching back into a leaf package for a base class.

    from console.agentic import ChatAgent, RequirementAgent

    reply = ChatAgent.run(question="what can you help me with?")
    gate = RequirementAgent.run(question=reply)   # a JSON string — see requirements.md's own contract

`save_question` is the other thing this module owns: writing an incoming chat question into a
package's own knowledge base (`ds_knowledge_db`, `ds_knowledge_text_db`) so a future retrieval can find
it, regardless of whether the agent that would eventually read that store is in `"rag"` mode yet.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
import types
from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_agent, build_vector_db, load_configs
from uilts.logger import logger

# `agentic_configurations.yaml` lives at the root of this package.
CONFIG_DIR = Path(__file__).resolve().parent


class AgentHandle:
    """Base for a named handle onto one `agents:` entry. A subclass sets `NAME` and nothing else.

    Args (class attributes a subclass sets):
        NAME: The key this agent is registered under in `agentic_configurations.yaml`'s `agents:`
            block — exactly as written there, since that is what `build_agent` looks it up by.
    """

    NAME: ClassVar[str]
    _agent: ClassVar[Any] = None

    @classmethod
    def build(cls, configs: Any = None, **overrides: Any) -> Any:
        """Build (or return the cached) `BaseAgent` this handle names, via `agent_builder.build_agent`."""
        if cls._agent is None or configs is not None or overrides:
            cls._agent = build_agent(cls.NAME, configs or CONFIG_DIR, **overrides)
        return cls._agent

    @classmethod
    def run(
        cls,
        question: str,
        context: str = "",
        agent_outputs: dict[str, str] | None = None,
        configs: Any = None,
        **kwargs: Any,
    ) -> str:
        """Build this agent if it is not already built, and run it — the one-call form most callers want."""
        agent = cls.build(configs)
        return agent.run(question, context, agent_outputs or {}, **kwargs)


class ChatAgent(AgentHandle):
    """`chat_agent` — the front door: the agent `cli.py`'s `generate` command talks to."""

    NAME = "chat_agent"


class RequirementAgent(AgentHandle):
    """`requirement_agent` — the gate before any pipeline starts.

    Checks that a problem statement and a data location/connection are both on the record, and reports
    the verdict as one JSON object — see `prompts/requirements.md` for the exact contract:

        {"requirements": true, "problem": "...", "data_details": "..."}

    `.run()` still returns the raw model text (a `BaseAgent` always does); parse it as JSON to get the
    dict this docstring describes, e.g. `json.loads(RequirementAgent.run(question=...))`.
    """

    NAME = "requirement_agent"


class CodeGeneratorAgent(AgentHandle):
    """`code_generator_agent` — runs once, after every pipeline stage is approved.

    Its `dependency_agent` in `agentic_configurations.yaml` lists every worker agent the pipeline ran,
    in order, so `{agent_output}` in `prompts/code_generator.md` renders each stage's own text, labelled
    by who produced it. What it is actually meant to reproduce faithfully — the real function each stage
    called and the exact arguments it ran with — does not come from that prose at all: `run_pipeline`
    passes it separately as `context`, rendered by `CodeTracker.render_calls()`. See that module's own
    docstring for why: an agent's text is a description of what it did, not a record of it.
    """

    NAME = "code_generator_agent"


def _patch_chromadb_otel() -> None:
    """Let `chromadb` import even when its optional OTLP exporter is missing.

    The same workaround `*/dbs/dataset.py` and `console/web_search.py` already apply before importing
    `chromadb` — repeated here rather than shared, so this module still loads with neither of those
    packages installed. `db_connector.vector.ChromaDB` imports `chromadb` directly with no patch of its
    own, so this has to run before the first `build_vector_db(..., db="chroma")` call.
    """
    target = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    if target in sys.modules:
        return
    module = types.ModuleType(target)

    class OTLPSpanExporter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    module.OTLPSpanExporter = OTLPSpanExporter  # type: ignore[attr-defined]
    sys.modules[target] = module


def save_question(package: str, question: str, answer: str = "") -> None:
    """Embed `question` and upsert it into `package`'s knowledge base.

    Writes to every vector db that package's `dbs:` block declares under the names
    `ds_knowledge_db` (FAISS) and `ds_knowledge_text_db` (Chroma) — whichever of the two actually
    exist there — keyed by a hash of the question text, so asking the same thing twice replaces the
    entry rather than duplicating it.

    Embeds with `<package>.dbs.dataset.SyntheticEmbedder` — the same offline, deterministic embedder
    `dataset.py`'s own knowledge-base build fitted and `dbs/web_search_cache.py`'s `open_cache()`
    already reuses — not the `rag_embeddings` entry in `agentic_configurations.yaml` (the real Gemini
    embeddings API). The two are not interchangeable here: the existing `ds_knowledge_db` index was
    built at `SyntheticEmbedder`'s width (256 dimensions), and `rag_embeddings` answers in a different
    one (3072) that FAISS would refuse to add beside it. Once retrieval actually moves to real
    embeddings this write path needs to move with it, or the two will drift.

    Best effort throughout: no fitted embedder yet, an unbuildable db, or a write failure is logged and
    skipped rather than raised. This runs alongside the real pipeline work on every chat turn, and a
    knowledge-base write is never worth stalling a run over.
    """
    _patch_chromadb_otel()

    try:
        dataset = importlib.import_module(f"{package}.dbs.dataset")
        embedder_dir = dataset.FAISS_DIR / "embedder"
        if not embedder_dir.exists():
            logger.warning(f"save_question: no fitted embedder yet for {package!r} at {embedder_dir}.")
            return
        vector = dataset.SyntheticEmbedder.load(embedder_dir).encode([question])[0].tolist()
    except Exception as error:
        logger.warning(f"save_question: could not embed a question for {package!r}: {error}")
        return

    configs = load_configs(package)
    document_id = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    metadata = {"question": question, "answer": answer, "saved_at": time.time()}

    for db_name in ("ds_knowledge_db", "ds_knowledge_text_db"):
        if db_name not in configs.db_confgs:
            continue

        try:
            db = build_vector_db(db_name, configs)
            _ensure_id_mapped(db)
            db.upsert(ids=[document_id], vectors=[vector], metadatas=[metadata], documents=[question])
            if hasattr(db, "save"):  # FAISS is in-memory until this is called; Chroma writes through
                db.save()
        except Exception as error:
            logger.warning(f"save_question: could not save to {package}.{db_name}: {error}")


def _ensure_id_mapped(db: Any) -> None:
    """Rebuild a loaded `FAISSDB`'s index as `faiss.IndexIDMap` when it isn't already one.

    `FAISSDB.upsert` assumes `self.client.index` exists, which is only true of an index it built and
    saved itself (`_initialize_connection` wraps a fresh index in `IndexIDMap` before assigning it). An
    index built directly with `faiss` — as this project's own `dbs/dataset.py` knowledge-base fixtures
    are — loads back as a plain `IndexFlatIP`/`IndexFlatL2`, with no `.index` attribute, and `upsert`
    raises `AttributeError` on it.

    `faiss.IndexIDMap` refuses to wrap an index that already holds vectors (`index must be empty on
    input`), so an in-place wrap is not an option here — this reconstructs every existing vector out of
    the loaded flat index and re-adds it, in the same order, to a fresh empty index of the same type
    and dimension, wrapped in `IndexIDMap` from the start. Their positional index (`0..ntotal-1`)
    becomes their id; `FAISSDB`'s own `_id_map`/`_metadata`/`_documents` still won't carry a string id
    or metadata for them, since those never existed for vectors this connector did not itself insert —
    only a lookup that goes through this connector rather than the array position they were originally
    built at is affected.
    """
    if type(db).__name__ != "FAISSDB":
        return

    import faiss
    import numpy as np

    client = getattr(db, "client", None)
    if client is None or isinstance(client, faiss.IndexIDMap):
        return

    wrapped = faiss.IndexIDMap(type(client)(client.d))
    if client.ntotal:
        vectors = client.reconstruct_n(0, client.ntotal)
        wrapped.add_with_ids(vectors, np.arange(client.ntotal, dtype="int64"))

    db.client = wrapped

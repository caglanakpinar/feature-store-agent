"""Thin, named handle onto the embeddings caller `agentic_configurations.yaml` declares under `embeddings:`.

Building it is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    embeddings = build_embeddings("rag_embeddings", configs)

— which is what a `"rag"`/`"rag_builder"`/`"retriever"` agent needs to turn a question into the vector
`ds_knowledge_db` and `ds_knowledge_text_db` are searched with, once `rag_data_engineer_decider_agent`'s
`type` switches from `"thinker"` to one of those. `RAGBuilderAgent` already builds and wires its own
embeddings connector from the config (see `agent_builder.build_agent`'s `embeddings=` argument); the
handle here is for everything else that needs the same vector without going through an agent — indexing
a document into `ds_knowledge_db`/`ds_knowledge_text_db`, or a script that queries one directly.

    from data_engineer.embeders import RAGEmbeddings

    vector = RAGEmbeddings.embed("what does a duplicate id column mean for training data?")

The underlying `BaseEmbeddings` is built once per process and cached on the class, for the same reason
`AgentHandle` and `LLMHandle` cache theirs. Pass `configs=` or any override to `.build()`/`.embed()` to
force a rebuild with that change applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_embeddings

# `agentic_configurations.yaml` lives at the root of this package — the same directory
# `data_engineer/agents/__init__.py`'s own `CONFIG_DIR` resolves against.
CONFIG_DIR = Path(__file__).resolve().parent


class EmbeddingsHandle:
    """Base for a named handle onto one `embeddings:` entry. A subclass sets `NAME` and nothing else.

    Args (class attributes a subclass sets):
        NAME: The key this caller is registered under in `agentic_configurations.yaml`'s `embeddings:`
            block — exactly as written there, since that is what `build_embeddings` looks it up by.
    """

    NAME: ClassVar[str]
    _embeddings: ClassVar[Any] = None

    @classmethod
    def build(cls, configs: Any = None, **overrides: Any) -> Any:
        """Build (or return the cached) `BaseEmbeddings` this handle names, via `agent_builder.build_embeddings`.

        Args:
            configs: An already-read `Configs` to build against, when a caller is assembling several
                handles and wants the YAML read once rather than once per handle. Defaults to reading
                `agentic_configurations.yaml` from `CONFIG_DIR`.
            **overrides: `build_embeddings`'s own — `provider`, `model_name`, `api_key`, `settings` —
                each taking precedence over what the YAML says. Passing any override forces a rebuild
                rather than returning the cached caller.

        Returns:
            The `BaseEmbeddings` subclass instance this entry's model resolves to.
        """
        if cls._embeddings is None or configs is not None or overrides:
            cls._embeddings = build_embeddings(cls.NAME, configs or CONFIG_DIR, **overrides)
        return cls._embeddings

    @classmethod
    def embed(cls, text: str, configs: Any = None, **kwargs: Any) -> list[float]:
        """Build this caller if it is not already built, and embed one text into its vector."""
        return cls.build(configs)._call(text, **kwargs)


class RAGEmbeddings(EmbeddingsHandle):
    """`rag_embeddings` — the embedder `ds_knowledge_db`/`ds_knowledge_text_db` are searched with."""

    NAME = "rag_embeddings"

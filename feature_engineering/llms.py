"""Thin, named handles onto the LLM callers `agentic_configurations.yaml` declares under `llms:`.

Building one is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    llm = build_llm("generator_llm_1", configs)

— except every agent this config's `agents:` block declares already resolves its own `llm:`/
`substitute_llm:` entry through `build_agent`, so a class here is for what that doesn't cover: a tool
or script that needs a raw prompt-in/text-out call without going through an agent, or several agents
that should share one already-built caller rather than each construct their own — pass the built
instance in as `llm=` to `build_agent(..., llm=GeneratorLLM.build())` to wire that up.

    from feature_engineering.llms import GeneratorLLM

    answer = GeneratorLLM.call("summarize this feature candidate in two sentences: ...")

The underlying `BaseLLM` is built once per process and cached on the class: constructing the provider
SDK client is not free, and `agentic_configurations.yaml` does not change while a pipeline runs. Pass
`configs=` or any override to `.build()`/`.call()` to force a rebuild with that change applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_llm

# `agentic_configurations.yaml` lives at the root of this package, and every path a builder resolves
# for an entry in it — prompts, tools — is resolved relative to this same directory.
CONFIG_DIR = Path(__file__).resolve().parent


class LLMHandle:
    """Base for a named handle onto one `llms:` entry. A subclass sets `NAME` and nothing else.

    Args (class attributes a subclass sets):
        NAME: The key this caller is registered under in `agentic_configurations.yaml`'s `llms:`
            block — exactly as written there, since that is what `build_llm` looks it up by.
    """

    NAME: ClassVar[str]
    _llm: ClassVar[Any] = None

    @classmethod
    def build(cls, configs: Any = None, **overrides: Any) -> Any:
        """Build (or return the cached) `BaseLLM` this handle names, via `agent_builder.build_llm`.

        Args:
            configs: An already-read `Configs` to build against, when a caller is assembling several
                of these handles and wants the YAML read once rather than once per handle. Defaults to
                reading `agentic_configurations.yaml` from `CONFIG_DIR`.
            **overrides: `build_llm`'s own — `provider`, `model_name`, `api_key`, `temperature`,
                `max_tokens`, `settings`, ... — each taking precedence over what the YAML says, exactly
                as `build_llm` documents. Passing any override forces a rebuild rather than returning
                the cached caller.

        Returns:
            The `BaseLLM` subclass instance this entry's model resolves to.
        """
        if cls._llm is None or configs is not None or overrides:
            cls._llm = build_llm(cls.NAME, configs or CONFIG_DIR, **overrides)
        return cls._llm

    @classmethod
    def call(cls, prompt: str, configs: Any = None, **kwargs: Any) -> str:
        """Build this caller if it is not already built, and run one prompt-in/text-out generation."""
        return cls.build(configs)._call(prompt, **kwargs)


class RAGLLM(LLMHandle):
    """`rag_llm` — answers the RAG-backed agents once their `type` is switched to retrieve first."""

    NAME = "rag_llm"


class GeneratorLLM(LLMHandle):
    """`generator_llm_1` — the workhorse behind the specialists; most agents' primary caller."""

    NAME = "generator_llm_1"


class StructuredLLM(LLMHandle):
    """`generator_llm_2` — for short structured answers, sized to still leave room to think first."""

    NAME = "generator_llm_2"


class ThinkerLLM(LLMHandle):
    """`generator_llm_thinker` — every agent's fallback (`substitute_llm`) when its primary call fails."""

    NAME = "generator_llm_thinker"


class JudgerLLM(LLMHandle):
    """`judger_agent_1` — reviews with a stronger model than the work it is reviewing."""

    NAME = "judger_agent_1"

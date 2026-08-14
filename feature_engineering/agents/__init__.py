"""Thin, named handles onto the agents `agentic_configurations.yaml` declares under `agents:`.

Building one is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    agent = build_agent("feature_prep_agent", configs)

— except every caller of the classes in `decider.py` and `workers.py` writes
`FeaturePrep.run(question=...)` rather than resolving the name and the config directory itself and
risking getting either wrong. `build_agent` decides which `BaseAgent` subclass actually runs a name
(`PlannerAgent`, `WorkerAgent`, ...) from that entry's configured `type`; nothing here second-guesses
that — a class in this package is a name and a config directory, not a type.

    from feature_engineering.agents.workers import FeatureSelector

    result = FeatureSelector.run(question="select the features worth keeping", context=features_summary)

The underlying `BaseAgent` is built once per process and cached on the class: `agentic_configurations.yaml`
does not change while a pipeline runs, and everything `build_agent` resolves for it — the LLM caller,
the toolbox, the prompt file — is expensive enough that rebuilding it on every `.run()` call would cost
real time for no benefit. Pass `configs=` or any override to `.build()`/`.run()` to force a rebuild with
that change applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_agent

# `agentic_configurations.yaml` lives at the root of this package — the same directory
# `feature_engineering/llms.py` and `feature_engineering/embeders.py` resolve their own entries against.
CONFIG_DIR = Path(__file__).resolve().parent.parent


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
        """Build (or return the cached) `BaseAgent` this handle names, via `agent_builder.build_agent`.

        Args:
            configs: An already-read `Configs` to build against, when a caller is assembling several
                of these handles and wants the YAML read once rather than once per handle. Defaults to
                reading `agentic_configurations.yaml` from `CONFIG_DIR`.
            **overrides: `build_agent`'s own — `llm`, `substitute_llm`, or any `AgentConfigs` field —
                each taking precedence over what the YAML says, exactly as `build_agent` documents.
                Passing any override forces a rebuild rather than returning the cached agent.

        Returns:
            The `BaseAgent` subclass instance this entry's `type` resolves to.
        """
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

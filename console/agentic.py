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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_agent

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

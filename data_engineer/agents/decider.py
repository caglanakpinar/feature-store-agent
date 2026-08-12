"""The agent that starts the run: `rag_data_engineer_decider_agent` in `agentic_configurations.yaml`.

It has no `dependency_agent` — every other agent in this config either depends on it directly or on
something that traces back to it — and its configured `type` is `"thinker"`, which `agent_builder`
runs as a `PlannerAgent`. Switching that entry's `type` to `"rag"` (or `"rag_builder"`/`"retriever"`)
turns it into a `RAGBuilderAgent` instead, retrieving from `db_vector: "ds_knowledge_db"` before it
answers — the class here does not change either way, since `build_agent` is what reads `type` and
picks the class; `Decider` is only ever the name.

    from data_engineer.agents.decider import Decider

    plan = Decider.run(question="what does this request need before anything else runs?")
"""

from __future__ import annotations

from data_engineer.agents import AgentHandle


class Decider(AgentHandle):
    """`rag_data_engineer_decider_agent` — decides what the run needs before any other agent starts."""

    NAME = "rag_data_engineer_decider_agent"

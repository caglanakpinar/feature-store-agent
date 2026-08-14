"""The agent that starts the run: `problem_analyzer_agent` in `agentic_configurations.yaml`.

It has no `dependency_agent` — every other agent in this config either depends on it directly or on
something that traces back to it — and its configured `type` is `"thinker"`, which `agent_builder` runs
as a `PlannerAgent`. Switching that entry's `type` to `"rag"` (or `"rag_builder"`/`"retriever"`) turns it
into a `RAGBuilderAgent` instead, retrieving from `db_vector: "ds_knowledge_db"` before it answers — the
class here does not change either way, since `build_agent` is what reads `type` and picks the class;
`ProblemAnalyzer` is only ever the name.

    from feature_engineering.agents.decider import ProblemAnalyzer

    statement = ProblemAnalyzer.run(question="predict churn next quarter", context=dataset_profile)
"""

from __future__ import annotations

from feature_engineering.agents import AgentHandle


class ProblemAnalyzer(AgentHandle):
    """`problem_analyzer_agent` — starts the run: frames the decision, target, evaluation and hand-off.

    Escalates rather than guessing where the request cannot carry a target or a decision — see
    `Escalation:` in its output.
    """

    NAME = "problem_analyzer_agent"

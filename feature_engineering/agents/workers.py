"""The three `"generator"` agents in `agentic_configurations.yaml` — `agent_builder` runs a
`"generator"` as a `WorkerAgent`, and these are the ones that actually touch the dataset, one stage
each, chained by `dependency_agent`:

    ProblemAnalyzer -> MissingValueImputer -> FeaturePrep -> FeatureSelector

`MissingValueImputer` and `FeaturePrep` both read `problem_analyzer_agent`'s output; `FeatureSelector`
is deliberately split from `FeaturePrep`, which builds features without ever reading the target, so
nothing that can leak the label lives in the same agent as the tools that build from it — see
`feature_selection.md`'s own header for why the split matters.

    from feature_engineering.agents.workers import MissingValueImputer, FeaturePrep

    imputed = MissingValueImputer.run(question="fill the holes", context=decider_output)
    features = FeaturePrep.run(question="build features", agent_outputs={"missing_value_agent": imputed})

`FeatureSelector`'s prompt also declares a `{target}` placeholder (as does `ProblemAnalyzer`'s, along
with a `{human_feedback}` one) that `agent_builder.BasePrompt` has no wiring for yet: `_substitute`
fills a `{name}` from another configured agent's output, or from a matching attribute the prompt
instance sets — `question`, `context`, `agent_output`, a threshold name — and nothing sets `target` or
`human_feedback`. Passing `target=`/`human_feedback=` to `.run()` does not reach the prompt either —
`**kwargs` there flows to the LLM call, not the template. Until `agent_builder` grows a way to set
them, fold the target column and any human answer into `context` when calling these two, or that
section renders unfilled.
"""

from __future__ import annotations

from feature_engineering.agents import AgentHandle


class MissingValueImputer(AgentHandle):
    """`missing_value_agent` — decides what a hole in the data means before feature prep sees it."""

    NAME = "missing_value_agent"


class FeaturePrep(AgentHandle):
    """`feature_prep_agent` — turns cleaned, imputed columns into the features a model can learn from."""

    NAME = "feature_prep_agent"


class FeatureSelector(AgentHandle):
    """`feature_selection_agent` — ranks built features against the target and prunes what doesn't earn its place."""

    NAME = "feature_selection_agent"

"""Thin, named handles onto the agents `agentic_configurations.yaml` declares under `agents:`.

Building one is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    agent = build_agent("feature_prep_agent", configs)

— except every caller of the classes below writes `FeaturePrep.run(question=...)` rather than resolving
the name and the config directory itself and risking getting either wrong. `build_agent` decides which
`BaseAgent` subclass actually runs a name (`PlannerAgent`, `WorkerAgent`, ...) from that entry's
configured `type`; nothing here second-guesses that — a class in this module is a name and a config
directory, not a type.

Every agent here is backed by a real prompt in `prompts/` — this config went through the same cleanup
`data_engineer`'s own config already had: the "benchmarks" example's agents (`data_engineer`,
`feature_preprocessing`, `model_developer`, `evaluator`, `mlops`, `benchmark`, a classifier, and five
judgers) were removed rather than left pointing at a `benchmarks/prompts/*.md` file this repo does not
have. What is left is the chain a real prompt actually documents:

    ProblemAnalyzer -> MissingValueImputer -> FeaturePrep -> FeatureSelector

`ProblemAnalyzer` has no dependency — every other agent here either depends on it directly or on
something that traces back to it. `MissingValueImputer` and `FeaturePrep` both read `problem_analyzer`'s
output; `FeatureSelector` is deliberately split from `FeaturePrep`, which builds features without ever
reading the target, so nothing that can leak the label lives in the same agent as the tools that build
from it — see `feature_selection.md`'s own header for why the split matters.

    from feature_engineering.agents import ProblemAnalyzer, FeatureSelector

    statement = ProblemAnalyzer.run(question="predict churn next quarter", context=dataset_profile)
    selected = FeatureSelector.run(
        question="select the features worth keeping",
        agent_outputs={"problem_analyzer_agent": statement, "feature_prep_agent": features},
    )

`ProblemAnalyzer` and `FeatureSelector`'s prompts also declare a `{target}` placeholder (and
`ProblemAnalyzer`'s a `{human_feedback}` one) that `agent_builder.BasePrompt` has no wiring for yet:
`_substitute` fills a `{name}` from another configured agent's output, or from a matching attribute the
prompt instance sets — `question`, `context`, `agent_output`, a threshold name — and nothing sets
`target` or `human_feedback`. Passing `target=`/`human_feedback=` to `.run()` does not reach the prompt
either — `**kwargs` there flows to the LLM call, not the template. Until `agent_builder` grows a way to
set them, fold the target column and any human answer into `context` when calling these two, or that
section renders unfilled.

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


class ProblemAnalyzer(AgentHandle):
    """`problem_analyzer_agent` — starts the run: frames the decision, target, evaluation and hand-off.

    Type `"thinker"` (`PlannerAgent`), switchable to `"rag"` to retrieve from `ds_knowledge_db` before
    answering. Escalates rather than guessing where the request cannot carry a target or a decision —
    see `Escalation:` in its output.
    """

    NAME = "problem_analyzer_agent"


class MissingValueImputer(AgentHandle):
    """`missing_value_agent` — decides what a hole in the data means before feature prep sees it."""

    NAME = "missing_value_agent"


class FeaturePrep(AgentHandle):
    """`feature_prep_agent` — turns cleaned, imputed columns into the features a model can learn from."""

    NAME = "feature_prep_agent"


class FeatureSelector(AgentHandle):
    """`feature_selection_agent` — ranks built features against the target and prunes what doesn't earn its place."""

    NAME = "feature_selection_agent"

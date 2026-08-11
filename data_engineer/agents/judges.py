"""The five `"judger"` agents in `agentic_configurations.yaml` — `agent_builder` runs a `"judger"` as
a `JudgerAgent`, and each one holds a stage to the numeric bars in its own `thresholds:` block rather
than to a free-form opinion: a `min_`/`max_` prefix carries the direction, the stem names a measurement
the pipeline's tools compute, and `judger.run()` argues about whether that measurement clears the bar.

    Decider, DataReader/DataAnalyzer/... -> ProblemJudger / DataJudger / ModelJudger / DeliveryJudger
                                                                                            |
                                                                                      FinalJudger

`FinalJudger` (`judger_agent_1`) is the arbiter: its `dependency_agent` is the whole run, work and
every other judge's verdict together, and its one threshold — `max_failed_gates: 0` — passes only if
every gate before it already did.

Two of these — `ModelJudger` and `DeliveryJudger` — currently name a `dependency_agent`
(`model_developer`, `evaluator`, `mlops`, `benchmark`, `feature_preprocessing`) that is not itself
defined anywhere in this config yet; that is a gap in the YAML, not in these classes, and building
either handle will only surface it once `.run()` actually tries to resolve those names.

    from data_engineer.agents.judges import DataJudger

    verdict = DataJudger.run(question="does the prepared data clear its gates?", agent_outputs=stage_outputs)
"""

from __future__ import annotations

from data_engineer.agents import AgentHandle


class ProblemJudger(AgentHandle):
    """`problem_judger` — gates the problem statement: row count, target balance, missingness ceiling."""

    NAME = "problem_judger"


class DataJudger(AgentHandle):
    """`data_judger` — gates data prep: no duplicates left, rows survived cleaning, features carry signal."""

    NAME = "data_judger"


class ModelJudger(AgentHandle):
    """`model_judger` — gates the trained model: cross-validated AUC, fold spread, train/holdout gap."""

    NAME = "model_judger"


class DeliveryJudger(AgentHandle):
    """`delivery_judger` — gates delivery: lift and precision at depth, gain over the existing baseline."""

    NAME = "delivery_judger"


class FinalJudger(AgentHandle):
    """`judger_agent_1` — the final arbiter: passes only if every gate above it already passed."""

    NAME = "judger_agent_1"

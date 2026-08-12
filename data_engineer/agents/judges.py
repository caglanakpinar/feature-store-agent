"""The two `"judger"` agents in `agentic_configurations.yaml` — `agent_builder` runs a `"judger"` as a
`JudgerAgent`, and each one holds a stage to the numeric bars in its own `thresholds:` block rather
than to a free-form opinion: a `min_`/`max_` prefix carries the direction, the stem names a measurement
the pipeline's tools compute, and `judger.run()` argues about whether that measurement clears the bar.

    Decider -> ProblemJudger -> DataReader/DataAnalyzer/DataPreprocessor/DataQuality/... -> DataJudger

`model_judger`, `delivery_judger` and a final arbiter used to live in this file too, gating a
modelling/evaluation/delivery pipeline this config never actually defined — leftover from the
"benchmarks" example this yaml was cloned from. They were removed from the YAML rather than left
dangling; add their classes back here once an agent that trains and evaluates a model exists for them
to gate.

    from data_engineer.agents.judges import DataJudger

    verdict = DataJudger.run(question="does the prepared data clear its gates?", agent_outputs=stage_outputs)
"""

from __future__ import annotations

from data_engineer.agents import AgentHandle


class ProblemJudger(AgentHandle):
    """`problem_judger` — gates the decision step: row count, target balance, missingness ceiling."""

    NAME = "problem_judger"


class DataJudger(AgentHandle):
    """`data_judger` — gates data prep: no duplicates left, rows survived cleaning, features carry signal."""

    NAME = "data_judger"

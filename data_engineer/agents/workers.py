"""The five `"generator"` agents in `agentic_configurations.yaml` — `agent_builder` runs a
`"generator"` as a `WorkerAgent`, and these are the ones that actually touch a dataset, one stage
each, chained by `dependency_agent`:

    Decider -> DataReader -> DataAnalyzer -> DataPreprocessor -> DataQuality

`PipelineCreator` sits beside that chain rather than in it — its `dependency_agent` is the first three
stages together (`data_preprocessor_agent`, `data_analyzer_agent`, `data_reader_agent`), and its job is
to assemble what they produced into a pipeline, not to add a further transformation of its own.

    from data_engineer.agents.workers import DataReader, DataAnalyzer

    reader_report = DataReader.run(question="read customers.csv", context=decider_output)
    analysis = DataAnalyzer.run(question="analyse what was read", agent_outputs={"data_reader_agent": reader_report})
"""

from __future__ import annotations

from data_engineer.agents import AgentHandle


class DataReader(AgentHandle):
    """`data_reader_agent` — reads a source into the dataset every later stage works on."""

    NAME = "data_reader_agent"


class DataAnalyzer(AgentHandle):
    """`data_analyzer_agent` — profiles what `DataReader` read, before anything changes it."""

    NAME = "data_analyzer_agent"


class DataPreprocessor(AgentHandle):
    """`data_preprocessor_agent` — cleans and transforms the dataset `DataAnalyzer` profiled."""

    NAME = "data_preprocessor_agent"


class DataQuality(AgentHandle):
    """`data_quality_agent` — validates what `DataPreprocessor` left, against fitness-to-train checks."""

    NAME = "data_quality_agent"


class PipelineCreator(AgentHandle):
    """`data_engineer_pipeline_creator` — assembles the reading, analysis and prep stages into a pipeline."""

    NAME = "data_engineer_pipeline_creator"

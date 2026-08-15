# feature-store-agent

A conversational agent pipeline for data science work: talk to it about a problem and a dataset, and
it drives that dataset through data engineering and feature engineering, one agent at a time, stopping
for your approval after each one.

```
chat  ->  requirement_agent  ->  data engineering  ->  feature engineering
                |                   (4 agents)            (4 agents)
                └── requirements: false ── back to chat for what is missing
```

Nothing runs until `requirement_agent` confirms two things are on the record: a problem statement, and
a data source with a way to connect to it. Until then the console just keeps talking, carrying the
whole conversation forward as context — the two halves usually arrive in different turns ("predict
churn" now, "it's in `data/train.csv`" next).

Once the gate passes, every agent in the two stages runs in dependency order. After each one:

- **you approve it** — its output is recorded and handed to the next agent as `{agent_output}`.
- **you reject it** — the console runs a web search on that agent's own topic, then the same agent
  retries with the search results and its own rejected attempt in front of it, so it can go deeper
  rather than just rephrase.
- **the agent escalates** (`Escalation: true` in its own result) — it's blocked on missing information,
  not quality, so there's no approve/reject: you're asked directly for what it needs, and it retries
  with your answer folded into context.

## Pipeline

| Stage | Agents | What they do |
|---|---|---|
| **data engineering** | `data_reader_agent` → `data_analyzer_agent` → `data_preprocessor_agent` → `data_quality_agent` | Read the source, profile it, clean it, and check it's fit to train on. |
| **feature engineering** | `problem_analyzer_agent` → `missing_value_agent` → `feature_prep_agent` → `feature_selection_agent` | Frame the problem and target, decide what a hole in the data means, build features, and select what earns a place in the model. |

Each agent is a real, tool-calling worker — not a single prompt-and-hope call. Its tools are real,
importable Python functions (`data_engineer/tools/`, `feature_engineering/tools/`), so what an agent
claims can be checked against what its tools actually returned.

## Project layout

```
cli.py                          the `feature-store-agent` console script
console/                        the orchestration layer — chat, the requirements gate, the pipeline
  run.py                        the pipeline itself: STAGES, approval/rejection/escalation loop
  agentic.py                    ChatAgent, RequirementAgent
  web_search.py                 cached DuckDuckGo search, used on a rejection
  prompts/                      chat.md, requirements.md, not_approved.md

data_engineer/                  reads, analyses, cleans and quality-checks a dataset
feature_engineering/            frames the problem, imputes, builds and selects features
  agents/                       named handles onto each package's agents (decider.py / workers.py)
  tools/                        the real functions an agent's tools call
  prompts/                      one .md file per agent, documenting its tools and its output contract
  agentic_configurations.yaml   llms:, agents:, tools:, orchestrators:, pipeline: — the whole config

data/                           wherever you point a run's data at (not version controlled)
```

Every agent is built from its package's `agentic_configurations.yaml` through
[`agent-builder`](https://github.com/caglanakpinar/agentic_ai_builder) — the shared library behind
every LLM caller, tool, prompt and agent type in this repo — via a thin per-package `AgentHandle`, e.g.:

```python
from feature_engineering.agents.workers import FeatureSelector

FeatureSelector.run(question="select the features worth keeping", agent_outputs={...})
```

## Setup

```bash
poetry install
```

Every LLM in this repo currently runs on Gemini (`gemini-3.1-flash-lite`), configured under `llms:` in
each of the three configs below, so out of the box you need:

```bash
export GEMINI=<your Gemini API key>
```

### Using a different model

Each `llms:` entry sets a `model: "<provider>/<model-id>"` and an `api_key: <ENV_VAR_NAME>` — change
either to change what that entry calls. The provider has to be one `agent-builder` actually implements
a caller for:

| Provider | `model:` prefix | Native tool-calling |
|---|---|---|
| Anthropic (Claude) | `claude/` or `anthropic/` | ✅ |
| OpenAI | `openai/` | ✅ |
| xAI (Grok) | `grok/` or `xai/` | ✅ |
| Ollama (local) | `ollama/` | ✅ |
| Mistral | `mistral/` | ✅ |
| Hugging Face Inference | `huggingface/` or `hf/` | ✅ |
| Google (Gemini) | `google/` or `gemini/` | ❌ — see below |

Every `llms:` block lives in one of these three files — update whichever agents you want on a
different model:

- [`console/agentic_configurations.yaml`](console/agentic_configurations.yaml) — `chat_agent`,
  `requirement_agent`
- [`data_engineer/agentic_configurations.yaml`](data_engineer/agentic_configurations.yaml) —
  `data_reader_agent`, `data_analyzer_agent`, `data_preprocessor_agent`, `data_quality_agent`
- [`feature_engineering/agentic_configurations.yaml`](feature_engineering/agentic_configurations.yaml)
  — `problem_analyzer_agent`, `missing_value_agent`, `feature_prep_agent`, `feature_selection_agent`

Then export whatever `api_key:` names for the entries you changed — `CLAUDE`, `OPENAI`, `MISTRAL`,
however you name the variable — alongside or instead of `GEMINI`.

Gemini has no native tool-calling in `agent-builder`, so tool-bearing agents on a `google`/`gemini`
entry run through a manual prompted tool loop instead (`console/run.py`'s `run_with_prompted_tools`) —
real tool execution, just driven by a `TOOL_CALL: {...}` text convention rather than the provider's own
function-calling API. Every other provider in the table above gets the native round trip for free.

Some readers need extra dependencies:

```bash
poetry install -E files       # Parquet, Excel
poetry install -E selection   # the sklearn-backed feature-selection scorers
poetry install -E all
```

## Usage

```bash
poetry run feature-store-agent generate
# or: python cli.py generate
```

```
Hey, I am Feature Engineer Helper feature-store-agent. How can I help you?
You: I want to predict Titanic survival. The data is in data/titanic/train.csv and test.csv.

Requirements met — starting the pipeline.
  problem       predict whether a passenger survived
  data details  data/titanic/train.csv, data/titanic/test.csv

=== data engineering ===

[data_reader_agent] working...
...
Approve data_reader_agent? [y/n]:
```

## Known limitations

- **Gemini can't run tools natively** in `agent-builder` — see the prompted-tool-loop note above. It
  works, but is a text convention rather than the provider's own protocol, and a model can occasionally
  fail to follow it.
- **The two stages aren't wired together via `agent_outputs`.** `feature_engineering`'s first agent has
  no `dependency_agent`, so it only ever sees the raw conversation (`{context}`), not
  `data_engineer`'s actual output — both stages point at the same original source rather than feature
  engineering building on data engineering's cleaned result.
- **Some agents have no real prompt yet** (`rag_data_engineer_decider_agent`,
  `data_engineer_pipeline_creator`, and the judger agents) — each still points at a `benchmarks/`
  prompt directory this repo doesn't have. They're deliberately left out of `console/run.py`'s
  `STAGES` rather than run against an empty prompt.

## Development

```bash
poetry run pytest
poetry run black . && poetry run isort .
poetry run mypy .
```

"""The pipeline `cli.py generate` drives: chat, a requirements gate, then the two stages, agent by agent.

    chat  ->  requirement_agent  ->  data engineering  ->  feature engineering
                    |                  (4 agents)            (4 agents)
                    └── requirements: false ── back to chat for what is missing

Nothing starts until `requirement_agent` says both of its rules are satisfied — a problem statement, and
a data source with a way to connect to it (see `prompts/requirements.md`). Until then the console keeps
talking to the user, carrying the whole conversation forward as context, because the two halves usually
arrive in different turns ("predict churn" now, "it's in customers.csv" next).

Once the gate passes, every agent in the two stages runs in dependency order, and each one stops at the
console for a human verdict before the next begins:

  - **approved** — its output is recorded in `agent_outputs` under its own configured name, which is
    what the next agent's `{agent_output}` renders from, and the run moves on.
  - **not approved** — the console runs a web search on that agent's own topic, then the same agent
    runs again with `prompts/not_approved.md` prepended to its rendered prompt, carrying both its
    rejected attempt (so "go deeper" names what it is going deeper than) and what the search found. It
    keeps its turn until it is approved; nothing downstream sees the rejected text.

The search goes through the rejected agent's own package cache — `data_engineer.dbs.web_search_cache`
or `feature_engineering.dbs.web_search_cache`, whichever stage it belongs to — so a repeated query is
served from that agent's ChromaDB rather than re-fetched, and the two packages' caches stay separate.
Both of those modules import `chromadb` and `faiss` through their `dbs/dataset.py`, so they are
imported inside `web_search` rather than at the top of this file: a missing driver costs the run its
search results, which is a degraded retry, not a failed one.

`rag_data_engineer_decider_agent` and `data_engineer_pipeline_creator` are deliberately not in `STAGES`.
Both are configured with a `prompt_path` under `benchmarks/prompts/`, a directory this repo does not
have, and `BasePrompt` answers a prompt it cannot find with a warning and an empty string — so running
them would send the model an empty prompt and return whatever it said to nothing at all. `run_agent`
refuses an empty prompt outright rather than trusting the list here to stay correct.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import click

# Already an implicit dependency of every agent built here — `build_llm` imports the provider class an
# agent's `llm:` entry names straight out of this module — so importing it again here costs nothing new.
from models.llms import ToolFailure, as_tool_content

from console.agentic import ChatAgent, CodeGeneratorAgent, RequirementAgent, save_question
from console import gate as gate_rules
from console import playbook, tool_turn
from console.code_tracker import CodeTracker
from data_engineer.agents.workers import (
    DataAnalyzer,
    DataPreprocessor,
    DataQuality,
    DataReader,
)
from feature_engineering.agents.decider import ProblemAnalyzer
from feature_engineering.agents.workers import (
    FeaturePrep,
    FeatureSelector,
    MissingValueImputer,
)

# The stages, in the order they run: the label, the package whose web-search cache its agents use, and
# the agents themselves in dependency order.
STAGES: list[tuple[str, str, list[Any]]] = [
    (
        "data engineering",
        "data_engineer",
        [DataReader, DataAnalyzer, DataPreprocessor, DataQuality],
    ),
    (
        "feature engineering",
        "feature_engineering",
        [ProblemAnalyzer, MissingValueImputer, FeaturePrep, FeatureSelector],
    ),
]

# What a rejected agent searches the web for, by agent name. An agent's own name makes a poor query
# ("data reader agent best practices"), so each one names the subject it would actually want to read
# about; the problem statement is appended to it at search time.
SEARCH_TOPICS: dict[str, str] = {
    DataReader.NAME: "reading and profiling a dataset for machine learning",
    DataAnalyzer.NAME: "exploratory data analysis of a training dataset",
    DataPreprocessor.NAME: "data cleaning and preprocessing for machine learning",
    DataQuality.NAME: "data quality checks before training a model",
    ProblemAnalyzer.NAME: "framing a machine learning problem, target definition and evaluation metric",
    MissingValueImputer.NAME: "missing value imputation methods",
    FeaturePrep.NAME: "feature engineering techniques",
    FeatureSelector.NAME: "feature selection methods and leakage detection",
}

# The note prepended to a rejected agent's own prompt, with `{rejected}` and `{web_search}` filled in.
NOT_APPROVED_PROMPT = Path(__file__).resolve().parent / "prompts" / "not_approved.md"

# `data_reader_agent` declares `rag_data_engineer_decider_agent` as its dependency, and that agent is
# not in `STAGES` (no prompt file — see this module's docstring), so its slot would render as "has not
# run yet". The requirements gate produces what that agent was there to decide — the problem, and where
# the data is — so its verdict is seeded under that name instead of leaving the first agent's
# `{agent_output}` reading like something upstream failed.
DECIDER_SLOT = "rag_data_engineer_decider_agent"


# Set by `chat(auto=True)` (`cli.py generate --yes`): every approval is granted and nothing waits on
# the keyboard, so the whole console can be driven unattended — by `benchmarks/run.py chat`, for
# one.
AUTO = False
# Unattended, an agent that keeps asking a question nobody can answer would otherwise stall the run
# (a person would eventually give up): it is answered from what the run already knows this many
# times, and its last reply is taken as it stands after that.
AUTO_MAX_ESCALATIONS = 2
# A turn of a small model that does not stop would otherwise run to the configured `max_tokens`
# (16,000 for these agents: minutes, per turn, per agent). Each model turn is cut at this many
# tokens, and a tool loop at this many rounds; a large model's own, smaller `max_tokens` wins.
MODEL_TURN_TOKENS = 500
MODEL_ROUNDS = 2
# An agent whose model made no tool call that ran is given its playbook (`console/playbook.py`).
# `cli.py generate --no-playbook` turns that off, to see what the model does on its own.
PLAYBOOK = True
# After this many agents in a row whose model made no tool call that ran, the console stops giving
# the model a turn and goes straight to the playbook: a model that has not called a tool in two
# tries is not about to (the 0.5B model made none in any of its turns, minutes each).
# `--no-playbook` never skips, so what the model does on its own is still measured in full.
MODEL_PATIENCE = 2
# How a model gets its turn at a tool-bearing agent: `auto` gives a small model run in this process a short *tool turn* (`console/tool_turn.py`)
# instead of the agent's own multi-thousand-token prompt, `on` and `off` force it either way. `cli.py generate --tool-turns`.
TOOL_TURNS = "auto"
# Where an unattended run leaves `run.json` and `main.py` (`cli.py generate --run-dir`); None keeps
# the interactive behaviour — `scripts/main.py` only.
RUN_DIR: Path | None = None


def ask(prompt: str) -> str | None:
    """Read one line from the user, or None when they end the session with Ctrl-C/Ctrl-D (or when running unattended)."""
    if AUTO:
        return None
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        click.echo()
        return None


def as_json(text: str) -> dict[str, Any] | None:
    """Parse the requirements gate's reply as the JSON object its prompt asks for.

    The prompt is explicit about returning bare JSON, but a model will still wrap it in a markdown fence
    or a sentence often enough to matter, and one stray fence is not a reason to stall the run — so the
    outermost `{...}` is what gets parsed rather than the whole reply.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None

    return parsed if isinstance(parsed, dict) else None


def truthy(value: Any) -> bool:
    """Read a JSON field's boolean intent, whether the model sent an actual bool or the word "true"."""
    return value is True or str(value).strip().lower() == "true"


def is_met(verdict: dict[str, Any]) -> bool:
    """Read the gate's `requirements` field."""
    return truthy(verdict.get("requirements"))


def escalation_envelope(output: str) -> tuple[bool, str] | None:
    """Read an agent's `{"Escalation": ..., "output": ...}` envelope — see `data_reader.md`'s own
    "## Result format" for the contract this reads.

    Returns `(escalation, text)` when `output` is that envelope, `None` when it is not — plain prose
    from an agent whose prompt does not use this convention yet, which is read as-is rather than as a
    (false) escalation.
    """
    parsed = as_json(output)
    if not parsed or "Escalation" not in parsed or "output" not in parsed:
        return None

    return truthy(parsed["Escalation"]), str(parsed["output"])


# A file-looking token (`customers.csv`, `data/titanic/train.csv`) or a `db://...`-style connection
# string, wherever it turns up in an agent's own prose.
PATH_PATTERN = re.compile(
    r"(?:[\w./~-]+/)?[\w.-]+\.(?:csv|tsv|json|jsonl|parquet|pq|xlsx?|db)\b|\w+://[\w./:@-]+",
    re.IGNORECASE,
)


def extract_path(text: str, fallback: str) -> str:
    """The last file- or connection-looking token an agent's output mentions, or `fallback`.

    Every prompt from `data_reader.md` on is told to restate the exact `data_source` it worked from —
    that was supposed to be enough on its own, and in practice a faster, smaller model does not always
    do it where the prompt expects, which is what actually sends `None` into a tool call. This is not
    the fix for that: `run_pipeline` injects an explicit `Current data path:` fact into every agent's
    context regardless, and this is what keeps that fact current — reading whatever new path an
    approved agent's own text seems to report, so a later agent gets the cleaned file a preprocessing
    step wrote rather than the original raw source, without needing to parse that out of prose either.
    Best-effort only: `fallback` (the path already known) wins when nothing matches.
    """
    matches = PATH_PATTERN.findall(text)
    return matches[-1] if matches else fallback


def web_search(package: str, query: str, max_results: int = 5) -> str:
    """Search the web through `package`'s own cache, rendered as the block a prompt can read.

    Best effort by design: the caller is already handling a rejection, and a driver that is not
    installed, a cache that will not open or a search that times out are all reasons to retry with no
    search results rather than to fail the run. Every failure comes back as an empty string.
    """
    try:
        cache = importlib.import_module(f"{package}.dbs.web_search_cache")
        results, from_cache = asyncio.run(cache.handle_prompt(query, max_results=max_results))
    except Exception as error:  # a missing driver, an unopenable cache, a network failure
        click.secho(f"  (web search unavailable: {error})", fg="yellow")
        return ""

    if not results:
        return ""

    click.secho(
        f"  (web search: {len(results)} result(s){' from cache' if from_cache else ''})", fg="yellow"
    )
    return "\n\n".join(
        f"{position}. {result.title}\n   {result.url}\n   {result.snippet}"
        for position, result in enumerate(results, start=1)
    )


def rejection_prefix(handle: Any, package: str, problem: str, rejected: str) -> str:
    """Build the block prepended to a rejected agent's prompt: the note, its attempt, and a search."""
    topic = SEARCH_TOPICS.get(handle.NAME, handle.NAME.replace("_", " "))
    found = web_search(package, f"{topic} {problem}".strip())

    # `.replace` rather than `.format`: the note is a markdown file that may hold literal braces, and
    # only these two placeholders are meant to be substituted.
    return (
        NOT_APPROVED_PROMPT.read_text()
        .replace("{rejected}", rejected)
        .replace("{web_search}", found or "(no results — the search returned nothing or was unavailable)")
    )


# --------------------------------------------------------------------------------------------------
# Prompted tool-calling: the ask -> execute -> answer loop for a provider with no native tool-calling
# support. `BaseAgent.generate` only runs an agent's tools when `llm.runs_tools` is True — Claude and
# the OpenAI-dialect callers, not `GoogleLLM` — so on Gemini it falls back to a single call with the
# tools "offered" and never run (the `... cannot run a tool-use loop ...` warning). This is the same
# loop `BaseAgent.run_with_tools` runs for a provider that speaks native tool-calling, done in plain
# text instead of a provider protocol: the model is told what it can call and asked to say so on one
# `TOOL_CALL:` line, `agent.toolbox` runs whatever it names — the same real, importable tool functions
# a native round trip would call — and the result goes back as the next round's context.
# --------------------------------------------------------------------------------------------------

TOOL_CALL_MARKER = "TOOL_CALL:"
PROMPTED_TOOL_ROUNDS = 8
PARSE_FAILED = object()  # a `TOOL_CALL:` line was there, but its JSON did not parse


def render_tool_catalogue(agent: Any) -> str:
    """Render every tool an agent declares as text a model can read and act on."""
    lines: list[str] = []
    for tool in agent.toolbox.agent_tools.values():
        parameters = tool.declaration()["parameters"]
        properties, required = parameters["properties"], set(parameters["required"])
        args = ", ".join(
            f"{name}{'' if name in required else '?'}: {schema.get('type', 'string')}"
            for name, schema in properties.items()
        )
        lines.append(f"- {tool.name}({args or 'no arguments'}) — {tool.description}")
    return "\n".join(lines)


def tool_instructions(agent: Any) -> str:
    """The block appended to a prompt telling a non-tool-calling provider how to ask for one anyway."""
    return (
        "Your provider has no native tool-calling, so tools work like this instead: to call one, "
        f"respond with a line starting with exactly `{TOOL_CALL_MARKER}` followed by one JSON object — "
        '`{"tool": "<name>", "arguments": {...}}` — and nothing else in that response. You will be '
        "given the tool's real result and asked again. Call as many tools, across as many turns, as "
        "you actually need, and quote what they return rather than guessing. When you are done, answer "
        f"normally with no `{TOOL_CALL_MARKER}` line; that response is taken as final.\n\n"
        f"Tools available to you:\n{render_tool_catalogue(agent)}"
    )


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Read a `TOOL_CALL:` line out of a response.

    Returns `(tool_name, arguments)` when one parses, `PARSE_FAILED` when a `TOOL_CALL:` line is there
    but its JSON is broken, and `None` when there is no such line at all — the model's final answer.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(TOOL_CALL_MARKER):
            continue

        try:
            call = json.loads(line[len(TOOL_CALL_MARKER):].strip())
        except ValueError:
            return PARSE_FAILED

        return (str(call["tool"]), dict(call.get("arguments") or {})) if isinstance(call, dict) and call.get("tool") else PARSE_FAILED

    return None


# The argument names a tool declares its dataset under — every tool in this project uses one or the
# other. Checked in `with_known_data_path` against what a tool actually declares, not assumed present.
DATA_ARGUMENT_NAMES = ("data_source", "data")


def with_known_data_path(tool: Any, arguments: dict[str, Any], data_path: str) -> dict[str, Any]:
    """Fill in a tool call's `data`/`data_source` argument from the pipeline's own known path, when the
    model left it out or passed something falsy.

    Every prompt in this project's `prompts/` is now explicit about where that path lives in
    `{context}`, and it still is not enough on its own — a smaller, faster model omits it, or passes
    `null`, often enough that this stopped being the rare case. This is the deterministic backstop:
    not a request the model can decline, applied only when the tool actually declares one of
    `DATA_ARGUMENT_NAMES` and `data_path` is itself known. It never overrides a value the model did
    supply, even a wrong one — a wrong path is a different failure than no path, and still the model's
    call to make.
    """
    if not data_path:
        return arguments

    properties = tool.declaration()["parameters"]["properties"]
    filled = dict(arguments)
    for name in DATA_ARGUMENT_NAMES:
        if name in properties and not filled.get(name):
            filled[name] = data_path
    return filled


def run_with_prompted_tools(
    agent: Any,
    prompt: str,
    data_path: str = "",
    max_rounds: int = PROMPTED_TOOL_ROUNDS,
    tracker: CodeTracker | None = None,
    stage: str = "",
    agent_name: str = "",
) -> str:
    """Drive the text-only tool loop described above until the model answers instead of asking for a tool.

    Bounded by `max_rounds` so a model that keeps asking for tools forever stops eventually, returning
    whatever it last said. `agent._generate` (not `.generate`) is used for each round: it is the one
    method that still gets the primary/substitute fallback without also trying `agent.llm`'s own (here,
    nonexistent) native tool-use round trip. `data_path`, when given, is applied to every tool call via
    `with_known_data_path` before it runs.

    `tracker`, when given, records every call that actually succeeds — see `console.code_tracker` for
    why a failed call is not recorded: reproducing something that did not work is not the point of the
    script this feeds. `stage` and `agent_name` are only there to label what gets recorded; this function
    does not otherwise need to know its caller's position in the pipeline.
    """
    conversation = f"{prompt}\n\n---\n\n{tool_instructions(agent)}"
    text = ""

    for _ in range(max_rounds):
        text = agent._generate(conversation)
        call = parse_tool_call(text)

        if call is None:
            return text

        if call is PARSE_FAILED:
            conversation += (
                f'\n\n---\nYour `{TOOL_CALL_MARKER}` line did not parse as `{{"tool": "...", '
                f'"arguments": {{...}}}}`. Try again, or answer normally with no `{TOOL_CALL_MARKER}` '
                "line to give your final answer."
            )
            continue

        name, arguments = call
        tool = agent.toolbox.agent_tools.get(name)
        if tool is not None:
            arguments = with_known_data_path(tool, arguments, data_path)

        try:
            result = agent.toolbox.call(name, arguments)
            # The tools catch their own errors and answer `{"status": "error", "message": ...}`
            # rather than raising, so a call that failed that way is not a call worth reproducing
            # either.
            failed = isinstance(result, dict) and result.get("status") == "error"
            if tool is not None and tracker is not None:
                if failed:
                    tracker.record_failure(
                        stage, agent_name, name, arguments, str(result.get("message", ""))
                    )
                else:
                    tracker.record(stage, agent_name, tool, arguments, result)
            rendered = as_tool_content(result)
        except Exception as error:
            if tracker is not None:
                tracker.record_failure(
                    stage, agent_name, name, arguments, f"{type(error).__name__}: {error}"
                )
            rendered = as_tool_content(ToolFailure(f"{type(error).__name__}: {error}"))

        conversation += (
            f"\n\n---\nYou called {name} with {json.dumps(arguments)}. Result:\n\n{rendered}\n\n"
            f"Call another tool the same way, or answer normally with no `{TOOL_CALL_MARKER}` line to "
            "give your final answer."
        )

    return text


def capped(agent: Any, tokens: int = MODEL_TURN_TOKENS) -> dict[str, int]:
    """`max_tokens` for one model call: `tokens`, unless the agent's own configured ceiling is lower."""
    configured = int(getattr(getattr(agent, "llm", None), "max_tokens", 0) or tokens)
    return {"max_tokens": min(configured, tokens)}


@contextmanager
def recorded_calls(
    agent: Any, tracker: CodeTracker | None, stage: str, agent_name: str, data_path: str
) -> Iterator[None]:
    """Every tool the agent runs while this is open — whoever asked for it — is recorded, with the data path filled in when it was left out.

    `run_with_prompted_tools` does this itself for a provider with no tool-calling of its own. A
    provider that has it (this project's local model, Claude, OpenAI...) runs the loop inside
    `agent-builder`, which has no hook for `CodeTracker`, so the toolbox's `call` is wrapped for the
    duration instead: the model's calls, and the playbook's, reach `scripts/main.py` and `run.json`
    the same way.
    """
    if tracker is None or not agent.toolbox:
        yield
        return

    original = agent.toolbox.call
    # a toolbox with its own instance-level `call` gets exactly that back, not nothing
    shadowed = "call" in vars(agent.toolbox)

    def call(name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = agent.toolbox.agent_tools.get(name)
        used = (
            with_known_data_path(tool, dict(arguments or {}), data_path)
            if tool is not None
            else dict(arguments or {})
        )
        try:
            result = original(name, used)
        except Exception as error:
            tracker.record_failure(
                stage, agent_name, name, used, f"{type(error).__name__}: {error}"
            )
            raise
        if isinstance(result, dict) and result.get("status") == "error":
            tracker.record_failure(stage, agent_name, name, used, str(result.get("message", "")))
        elif tool is not None:
            tracker.record(stage, agent_name, tool, used, result)
        return result

    agent.toolbox.call = call
    try:
        yield
    finally:
        if shadowed:
            agent.toolbox.call = original
        else:
            del agent.toolbox.call


def run_agent(
    handle: Any,
    question: str,
    context: str,
    agent_outputs: dict[str, str],
    prefix: str = "",
    data_path: str = "",
    tracker: CodeTracker | None = None,
    stage: str = "",
) -> str:
    """Run one agent — through `.run()`, the same one-call interface `RequirementAgent`/`ChatAgent` use —
    prepending `prefix` ahead of its own prompt on a retry, and routing through `run_with_prompted_tools`
    when this agent has tools its own LLM cannot call natively.

    `.run()` renders and generates in one call, with nowhere to inject `prefix` ahead of the agent's own
    instructions: the only hook it exposes is `context`, which lands wherever `{context}` happens to sit
    in that agent's prompt — not necessarily first. A rejection needs it first, so a retry builds the
    prompt itself instead of going through `.run()`. The empty-prompt guard runs either way, since
    neither `.run()` nor `run_with_prompted_tools` has one of its own.

    `data_path`, when given, only reaches `run_with_prompted_tools` — the deterministic backstop that
    fills a tool's `data`/`data_source` argument when the model leaves it out (see
    `with_known_data_path`). The native tool-calling path (`.run()`/`.generate()`, for a Claude-backed
    agent) has no such hook to apply it through; that path relies on the prompt alone. `tracker` and
    `stage` have the same limitation, for the same reason: `CodeTracker` only sees a call by being handed
    it directly, and only `run_with_prompted_tools` does that.
    """
    agent = handle.build()
    prompt = agent.build_prompt(question, context, agent_outputs)
    if not prompt.strip():
        raise ValueError(
            f"{handle.NAME} rendered an empty prompt — its configured `prompt_path` does not exist. "
            "Fix that entry in `agentic_configurations.yaml` before running it."
        )

    full_prompt = f"{prefix}\n\n{prompt}" if prefix else prompt

    if agent.toolbox and not getattr(agent.llm, "runs_tools", False):
        return run_with_prompted_tools(
            agent, full_prompt, data_path, tracker=tracker, stage=stage, agent_name=handle.NAME
        )

    limits = {**capped(agent), **({"max_rounds": MODEL_ROUNDS} if agent.toolbox else {})}
    with recorded_calls(agent, tracker, stage, handle.NAME, data_path):
        if not prefix:
            return agent.run(question, context, agent_outputs, **limits)

        return agent.generate(full_prompt, **limits)


def where_the_table_is(
    played: bool, model_calls: int, text: str, current_path: str, play_state: "playbook.State"
) -> str:
    """The data path the next agent should work from, once an agent's output is approved.

    Whoever did the work says where it left the table. When the playbook ran, that is its own state
    — never a path-looking word in some model's prose, which can be the original `data.csv` quoted
    back and would send every later stage to the raw file. When the model ran tools, it is the last
    path its text names, as before, if that file exists (and the playbook follows it); one that does
    not exist is ignored. An agent that ran nothing (a model's framing text) leaves the table where
    it was.
    """
    if played:
        return play_state.path
    if model_calls:
        named = extract_path(text, current_path)
        if Path(named).is_file():
            play_state.path = named
            return named
        # a connection string is not a file, and is where a database source stays
        if "://" in named:
            return named
    return current_path


def small_model_turn(
    handle: Any, play_state: "playbook.State", tracker: CodeTracker, stage: str
) -> tuple[str, dict[str, Any]]:
    """A small model's turn at a tool-bearing agent: one sentence, the stage's tools in short form, whatever it calls (`console/tool_turn.py`).

    With the playbook on, the stage runs its own calls afterwards -- their arguments come from what the earlier stages found -- and what the
    model chose is counted: `resolved` proposals named a real tool of the stage, `arguments_ok` of them needed nothing taken back. With the
    playbook off the model's calls are run as its own, after `repair_arguments` has fixed the path and dropped what the tool does not take,
    and what they returned is the agent's output.
    """
    agent = handle.build()
    proposals, text, prompt = tool_turn.propose(
        agent, play_state.path, play_state.target, MODEL_TURN_TOKENS, play_state.columns or None
    )
    named = [p for p in proposals if p.tool]
    record = {
        "prompt": prompt,
        "proposals": tool_turn.record(proposals),
        "resolved": len(named),
        "unresolved": len(proposals) - len(named),
        "arguments_ok": sum(p.usable for p in proposals),
    }
    if PLAYBOOK:
        return "", record

    outputs = []
    with recorded_calls(agent, tracker, stage, handle.NAME, play_state.path):
        for proposal in named:
            tool = agent.toolbox.agent_tools[proposal.tool]
            arguments = tool_turn.repair_arguments(
                proposal.tool, tool, proposal.arguments, play_state.path, play_state.target, play_state.columns or None
            )
            try:
                result = agent.toolbox.call(proposal.tool, arguments)
            except Exception as error:  # a model's own call that raises is its output, not the end of the run
                outputs.append(f"{proposal.tool} raised {type(error).__name__}: {error}")
                continue
            playbook.learn(play_state, proposal.tool, result)
            outputs.append(tool_turn.summarize(proposal.tool, result, play_state.path))
    return "\n\n".join(outputs) or text, record


def backstop(
    handle: Any,
    output: str,
    play_state: "playbook.State",
    tracker: CodeTracker,
    calls_before: int,
    stage: str,
) -> str:
    """Give an agent its playbook when its own turn did not do the work, and return the text that stands as its output.

    A tool-bearing agent has done the work when at least one of its calls ran and answered; a
    thinking agent (no tools) when its text names the target the playbook found. Otherwise the
    model's text is a claim about work that did not happen, and the playbook's own text — built from
    what its calls returned — replaces it.
    """
    if not PLAYBOOK or handle.NAME not in playbook.PLAYBOOKS:
        return output

    agent = handle.build()
    if agent.toolbox:
        if len(tracker.calls) > calls_before:
            return output
    elif play_state.target and play_state.target in output and len(output.strip()) > 40:
        return output

    click.secho(
        f"  (console: {handle.NAME} made no tool call that ran, so its default calls run instead)",
        fg="yellow",
    )
    with recorded_calls(agent, tracker, stage, handle.NAME, play_state.path):
        call = (
            (lambda name, arguments: agent.toolbox.call(name, arguments))
            if agent.toolbox
            else (lambda name, arguments: {})
        )
        try:
            return playbook.PLAYBOOKS[handle.NAME](play_state, call)
        # a playbook that falls over must not end an unattended run: say so, keep the model's text
        except Exception as error:
            click.secho(
                f"  (the default calls for {handle.NAME} failed: {type(error).__name__}: {error})",
                fg="red",
            )
            return output


def approve(name: str) -> bool | None:
    """Ask the human to approve what an agent just produced. None when they end the session."""
    if AUTO:
        click.secho(f"\nApprove {name}? [y/n]: y (--yes)", fg="yellow")
        return True

    while True:

        answer = ask(f"\nApprove {name}? [y/n]: ")
        if answer is None:
            return None

        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        click.echo("Please answer y or n.")


SCRIPTS_DIR = Path("scripts")


def write_script(content: str, path: Path = SCRIPTS_DIR / "main.py") -> Path:
    """Write `content` to `path`, creating its parent directory if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def generate_code(
    question: str, agent_outputs: dict[str, str], tracker: CodeTracker, use_model: bool = True
) -> None:
    """Turn everything `tracker` recorded into `scripts/main.py`, once every stage is approved.

    `code_generator_agent` is asked to write the script itself — see `prompts/code_generator.md` — so
    its narrative comments can come from what each stage actually reported (`agent_outputs`, rendered
    through the agent's own `dependency_agent` list as `{agent_output}`) rather than from nothing. Its
    answer is trusted only as far as `CodeTracker.covers()` can check: every call the tracker recorded
    must still be named in the script it wrote back, or this falls back to `tracker.render_script()`
    instead — the same calls, faithfully reproduced, with no narrative around them. Either way something
    gets written; a plainer script is a smaller loss than one that silently reproduces less than the
    pipeline actually did.

    Best effort, like the rest of a pipeline's tail end: a model failure here does not undo an already
    approved run, so it is logged to the console and swallowed rather than raised.
    """
    if not tracker.calls:
        click.secho("\nNo tool calls were recorded; nothing to generate a script from.", fg="yellow")
        return

    script = tracker.render_script()
    if not use_model:  # a small model's script has never covered the recorded calls; the recorded one is written as it is
        path = write_script(script, RUN_DIR / "main.py" if RUN_DIR else SCRIPTS_DIR / "main.py")
        click.secho(f"\nWrote {path} — {len(tracker.calls)} step(s).", fg="green", bold=True)
        return

    try:
        reply = CodeGeneratorAgent.run(
            question=question,
            context=tracker.render_calls(),
            agent_outputs=agent_outputs,
            **capped(CodeGeneratorAgent.build(), 2500),
        )
        parsed = as_json(reply)
        candidate = str(parsed["script"]) if parsed and parsed.get("script") else ""
        if candidate and tracker.covers(candidate):
            script = candidate
        else:
            click.secho(
                "  (code_generator_agent's script did not cover every call; using the recorded one)",
                fg="yellow",
            )
    except Exception as error:
        click.secho(f"  (code generation unavailable: {error})", fg="yellow")

    path = write_script(script, RUN_DIR / "main.py" if RUN_DIR else SCRIPTS_DIR / "main.py")
    click.secho(f"\nWrote {path} — {len(tracker.calls)} step(s).", fg="green", bold=True)


def auto_answer(agent_name: str, problem: str, data_path: str) -> str:
    """What an unattended run says to an agent that stopped to ask: there is nobody to ask, so it is told
    what the run already knows and to carry on with a stated assumption instead of asking again."""
    return (
        f"No one is available to answer questions. Work from what is already on the record: the problem is "
        f"'{problem}' and the data is at {data_path}. Where something is not stated, pick the most "
        "reasonable default, say in your answer that it is an assumption, and do not ask again."
    )


def save_record(
    agent_outputs: dict[str, str],
    log: list[dict[str, Any]],
    tracker: CodeTracker,
    verdict: dict[str, Any],
    finished: bool,
    started: float,
) -> None:
    """Write `run.json` into `RUN_DIR`: what an unattended run did, for `benchmarks/score.py` to read.

    Only when `RUN_DIR` is set. Every tool call that ran carries its arguments and what it returned,
    so a score can read what the run actually found rather than what an agent said it found.
    """
    if RUN_DIR is None:
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "finished": finished,
        "seconds": round(time.time() - started, 1),
        "problem": verdict.get("problem", ""),
        "data_details": verdict.get("data_details", ""),
        "agents": log,
        "agent_outputs": agent_outputs,
        "calls": [
            {
                "stage": call.stage,
                "agent": call.agent,
                "tool": call.tool,
                "function": f"{call.module}.{call.function}",
                "arguments": call.arguments,
                "result": call.result,
            }
            for call in tracker.calls
        ],
        "failed_calls": tracker.failures,
    }
    (RUN_DIR / "run.json").write_text(json.dumps(record, indent=1, default=str))
    click.echo(f"Saved run record {RUN_DIR / 'run.json'}.")


def run_pipeline(question: str, context: str, verdict: dict[str, Any]) -> bool:
    """Run every stage agent by agent, stopping at the console for approval after each one.

    Returns True when the whole pipeline was approved through to the end, False when the user ended the
    session partway. On a full approval, `generate_code` also writes `scripts/main.py` from everything
    `tracker` recorded — see that function's own docstring.

    Unattended (`AUTO`), every output is approved and an agent that escalates is answered from what
    the run already knows — see `auto_answer`.
    """
    agent_outputs: dict[str, str] = {
        DECIDER_SLOT: (
            f"Problem: {verdict.get('problem', '')}\n"
            f"Data details: {verdict.get('data_details', '')}"
        )
    }

    problem = str(verdict.get("problem", ""))
    # The path every agent is told about explicitly, alongside whatever `{agent_output}` says — see
    # `extract_path`'s own docstring for why relying on prose alone was not holding up in practice.
    current_path = str(verdict.get("data_details", ""))
    tracker = CodeTracker()
    log: list[dict[str, Any]] = []
    started = time.time()
    play_state = playbook.State(problem=problem, source=current_path)
    idle_turns = 0  # consecutive tool-bearing agents whose model made no call that ran
    small_model_run = False

    def finish(finished: bool) -> bool:
        save_record(agent_outputs, log, tracker, verdict, finished, started)
        return finished

    try:
        for stage, package, handles in STAGES:
            click.secho(f"\n=== {stage} ===", fg="cyan", bold=True)

            for handle in handles:
                prefix = ""
                escalations = 0
                while True:
                    click.secho(f"\n[{handle.NAME}] working...", fg="yellow")
                    agent_context = f"{context}\n\nCurrent data path: {current_path}"
                    calls_before, failures_before = len(tracker.calls), len(tracker.failures)
                    began = time.time()
                    agent = handle.build()
                    compact = tool_turn.uses_tool_turn(agent, TOOL_TURNS)
                    # A small model has nothing to add to a framing that reads the problem back (and wrote one that invented its target in
                    # most runs); the playbook's is built from what the tools found.
                    framing = PLAYBOOK and not agent.toolbox and tool_turn.is_local_model(agent, TOOL_TURNS)
                    skipped = framing or (
                        PLAYBOOK
                        and idle_turns >= MODEL_PATIENCE
                        and handle.NAME in playbook.PLAYBOOKS
                        and agent.toolbox
                    )
                    turn: dict[str, Any] = {}
                    try:
                        if skipped:
                            if not framing:
                                click.secho(
                                    f"  (console: the model made no tool call in its last {idle_turns} turns, so {handle.NAME} goes straight to its default calls)",
                                    fg="yellow",
                                )
                            output = ""
                        elif compact:
                            output, turn = small_model_turn(handle, play_state, tracker, stage)
                            click.secho(
                                f"  (tool turn: the model called {[p['tool'] for p in turn['proposals']] or 'nothing'})",
                                fg="yellow",
                            )
                        else:
                            output = run_agent(
                                handle,
                                question,
                                agent_context,
                                agent_outputs,
                                prefix,
                                current_path,
                                tracker=tracker,
                                stage=stage,
                            )
                    except ValueError as error:
                        click.secho(f"{handle.NAME} could not run: {error}", fg="red")
                        return finish(False)

                    model_calls, model_failures = (
                        len(tracker.calls) - calls_before,
                        len(tracker.failures) - failures_before,
                    )
                    if turn and PLAYBOOK:  # the model's choice counts as its call; the stage's own calls come next
                        model_calls, model_failures = turn["resolved"], turn["unresolved"]
                    if agent.toolbox and not skipped:
                        idle_turns = 0 if model_calls else idle_turns + 1
                    small_model_run = small_model_run or bool(turn) or framing
                    model_output = output
                    output = backstop(handle, output, play_state, tracker, calls_before, stage)
                    envelope = escalation_envelope(output)
                    escalated, text = envelope if envelope else (False, output)

                    click.secho(f"\n[{handle.NAME}]", fg="green", bold=True)
                    click.echo(text)
                    click.echo(
                        f"[console] {handle.NAME}: {len(tracker.calls) - calls_before} tool call(s) ran "
                        f"in {time.time() - began:.0f}s"
                    )
                    log.append(
                        {
                            "agent": handle.NAME,
                            "stage": stage,
                            "escalated": escalated,
                            "tool_calls": len(tracker.calls) - calls_before,
                            "model_calls": model_calls,
                            "model_failed_calls": model_failures,
                            "model_skipped": bool(skipped),
                            "tool_turn": turn or None,
                            "playbook": output is not model_output,
                            "model_output": model_output[:1200],
                            "seconds": round(time.time() - began, 1),
                            "output": text,
                        }
                    )

                    if escalated:
                        if AUTO:
                            escalations += 1
                            if escalations > AUTO_MAX_ESCALATIONS:
                                click.secho(
                                    f"  ({handle.NAME} kept asking; taking its last reply as it stands)",
                                    fg="yellow",
                                )
                                agent_outputs[handle.NAME] = text
                                break
                            answer = auto_answer(handle.NAME, problem, current_path)
                            click.secho(
                                f"\n{handle.NAME} is blocked — answered unattended.", fg="yellow"
                            )
                        else:
                            # Blocked on the human, not on quality — go back to the prompt for an
                            # answer instead of asking for y/n, and retry with nothing else changed.
                            answer = ask(f"\n{handle.NAME} is blocked — your answer: ")
                            if answer is None:
                                click.secho("\nEnding the run here.", fg="yellow")
                                return finish(False)

                        context = f"{context}\n\n{handle.NAME} asked:\n{text}\n\nAnswer: {answer}"
                        current_path = extract_path(answer, current_path)
                        continue

                    verdict_given = approve(handle.NAME)
                    if verdict_given is None:
                        click.secho("\nEnding the run here.", fg="yellow")
                        return finish(False)

                    if verdict_given:
                        agent_outputs[handle.NAME] = text
                        current_path = where_the_table_is(
                            output is not model_output, model_calls, text, current_path, play_state
                        )
                        break

                    # Its next turn goes deeper than exactly this attempt, with a search on its own
                    # topic.
                    click.secho(
                        f"Not approved — {handle.NAME} will search and go deeper.", fg="yellow"
                    )
                    prefix = rejection_prefix(handle, package, problem, text)

    # an unexpected error ends the run, but not before what it did so far is on the record
    except Exception:
        finish(False)
        raise

    click.secho("\nEvery stage approved. The run is complete.", fg="green", bold=True)
    generate_code(question, agent_outputs, tracker, use_model=not small_model_run)
    return finish(True)


def chat(
    message: str | None = None,
    auto: bool = False,
    run_dir: Path | None = None,
    use_playbook: bool = True,
    tool_turns: str = "auto",
) -> bool:
    """Talk to the user until the requirements gate passes, then run the pipeline.

    Args:
        message: A first user turn to send instead of reading one — the problem as the user would type it.
        auto: Run unattended: approve every stage and never wait on the keyboard. If the requirements
            gate is not met by `message`, the session ends (there is nobody to answer the follow-up
            question).
        run_dir: Where an unattended run leaves `run.json` and `main.py`; `scripts/main.py` when None.
        use_playbook: Give an agent whose model made no tool call that ran its default calls (`console/playbook.py`); False to see the model alone.
        tool_turns: `auto` gives a small local model a short tool turn (`console/tool_turn.py`) instead of each agent's own prompt; `on`/`off` force it.

    Returns whether the pipeline ran to the end.
    """
    global AUTO, RUN_DIR, PLAYBOOK, TOOL_TURNS
    AUTO = auto
    RUN_DIR = run_dir
    PLAYBOOK = use_playbook
    TOOL_TURNS = tool_turns
    click.echo("Hey, I am Feature Engineer Helper feature-store-agent. How can I help you?")
    transcript: list[str] = []
    pending = message

    while True:
        if pending is not None:
            question, pending = pending, None
            click.echo(f"You: {question}")
        else:
            question = ask("You: ")
        if question is None:
            return False

        if not question.strip():
            continue

        transcript.append(f"User: {question}")
        context = "\n".join(transcript)

        # Every new question is indexed into both packages' knowledge bases as it arrives, regardless
        # of whether `rag_data_engineer_decider_agent`/`problem_analyzer_agent` are in "rag" mode yet —
        # see `save_question`'s own docstring for why this is best-effort and never blocks the turn.
        # An unattended run does not write: benchmark questions are not the user's to remember.
        if not auto:
            save_question("data_engineer", question)
            save_question("feature_engineering", question)

        try:
            verdict = as_json(
                RequirementAgent.run(
                    question=question, context=context, **capped(RequirementAgent.build(), 700)
                )
            )
        except ValueError as error:  # an unresolvable key or model — nothing to retry against
            click.secho(f"Feature-Store Agent: ({error})", fg="red")
            return False

        # A small model often answers this gate in prose, or names a source nobody mentioned. The
        # two things it checks can be read off the conversation, so when the model's verdict is
        # missing, unmet or names a different source than the conversation does, the rules' reading
        # is taken instead (see `console.gate`).
        ruled = gate_rules.read_request(context)
        if ruled and not gate_rules.agrees(verdict, ruled):
            click.secho(
                "  (the rules read the request: a problem and a reachable data source are both there)",
                fg="yellow",
            )
            verdict = ruled

        # A reply that would not parse is treated as "not yet": the chat agent asks for what is
        # missing, which is the same thing that happens on an honest `requirements: false`.
        if verdict and is_met(verdict):
            click.secho("\nRequirements met — starting the pipeline.", fg="green", bold=True)
            click.echo(f"  problem       {verdict.get('problem', '')}")
            click.echo(f"  data details  {verdict.get('data_details', '')}")
            return run_pipeline(question, context, verdict)

        gate = json.dumps(verdict) if verdict else "the requirements check did not return a verdict"
        if auto:
            click.secho(
                f"Requirements not met by the message, and nobody is there to answer: {gate}",
                fg="red",
            )
            return False

        reply = ChatAgent.run(
            **capped(ChatAgent.build(), 700),
            question=question,
            context=f"{context}\n\nRequirements check: {gate}\n\n"
            "Requirements are not met yet. Ask the user for whatever is missing — the problem "
            "statement, or where the data is and how to connect to it — in one short question.",
        )
        transcript.append(f"Assistant: {reply}")
        click.echo(f"Feature-Store Agent: {reply}")

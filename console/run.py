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
from pathlib import Path
from typing import Any

import click

# Already an implicit dependency of every agent built here — `build_llm` imports the provider class an
# agent's `llm:` entry names straight out of this module — so importing it again here costs nothing new.
from models.llms import ToolFailure, as_tool_content

from console.agentic import ChatAgent, RequirementAgent, save_question
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


def ask(prompt: str) -> str | None:
    """Read one line from the user, or None when they end the session with Ctrl-C/Ctrl-D."""
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


def run_with_prompted_tools(agent: Any, prompt: str, max_rounds: int = PROMPTED_TOOL_ROUNDS) -> str:
    """Drive the text-only tool loop described above until the model answers instead of asking for a tool.

    Bounded by `max_rounds` so a model that keeps asking for tools forever stops eventually, returning
    whatever it last said. `agent._generate` (not `.generate`) is used for each round: it is the one
    method that still gets the primary/substitute fallback without also trying `agent.llm`'s own (here,
    nonexistent) native tool-use round trip.
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
        try:
            rendered = as_tool_content(agent.toolbox.call(name, arguments))
        except Exception as error:
            rendered = as_tool_content(ToolFailure(f"{type(error).__name__}: {error}"))

        conversation += (
            f"\n\n---\nYou called {name} with {json.dumps(arguments)}. Result:\n\n{rendered}\n\n"
            f"Call another tool the same way, or answer normally with no `{TOOL_CALL_MARKER}` line to "
            "give your final answer."
        )

    return text


def run_agent(
    handle: Any,
    question: str,
    context: str,
    agent_outputs: dict[str, str],
    prefix: str = "",
) -> str:
    """Run one agent — through `.run()`, the same one-call interface `RequirementAgent`/`ChatAgent` use —
    prepending `prefix` ahead of its own prompt on a retry, and routing through `run_with_prompted_tools`
    when this agent has tools its own LLM cannot call natively.

    `.run()` renders and generates in one call, with nowhere to inject `prefix` ahead of the agent's own
    instructions: the only hook it exposes is `context`, which lands wherever `{context}` happens to sit
    in that agent's prompt — not necessarily first. A rejection needs it first, so a retry builds the
    prompt itself instead of going through `.run()`. The empty-prompt guard runs either way, since
    neither `.run()` nor `run_with_prompted_tools` has one of its own.
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
        return run_with_prompted_tools(agent, full_prompt)

    if not prefix:
        return agent.run(question, context, agent_outputs)

    return agent.generate(full_prompt)


def approve(name: str) -> bool | None:
    """Ask the human to approve what an agent just produced. None when they end the session."""
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


def run_pipeline(question: str, context: str, verdict: dict[str, Any]) -> bool:
    """Run every stage agent by agent, stopping at the console for approval after each one.

    Returns True when the whole pipeline was approved through to the end, False when the user ended the
    session partway.
    """
    agent_outputs: dict[str, str] = {
        DECIDER_SLOT: (
            f"Problem: {verdict.get('problem', '')}\n"
            f"Data details: {verdict.get('data_details', '')}"
        )
    }

    problem = str(verdict.get("problem", ""))

    for stage, package, handles in STAGES:
        click.secho(f"\n=== {stage} ===", fg="cyan", bold=True)

        for handle in handles:
            prefix = ""
            while True:
                click.secho(f"\n[{handle.NAME}] working...", fg="yellow")
                try:
                    output = run_agent(handle, question, context, agent_outputs, prefix)
                except ValueError as error:
                    click.secho(f"{handle.NAME} could not run: {error}", fg="red")
                    return False

                envelope = escalation_envelope(output)
                escalated, text = envelope if envelope else (False, output)

                click.secho(f"\n[{handle.NAME}]", fg="green", bold=True)
                click.echo(text)

                if escalated:
                    # Blocked on the human, not on quality — go back to the prompt for an answer
                    # instead of asking for y/n, and retry with nothing else changed.
                    answer = ask(f"\n{handle.NAME} is blocked — your answer: ")
                    if answer is None:
                        click.secho("\nEnding the run here.", fg="yellow")
                        return False

                    context = f"{context}\n\n{handle.NAME} asked:\n{text}\n\nAnswer: {answer}"
                    continue

                verdict_given = approve(handle.NAME)
                if verdict_given is None:
                    click.secho("\nEnding the run here.", fg="yellow")
                    return False

                if verdict_given:
                    agent_outputs[handle.NAME] = text
                    break

                # Its next turn goes deeper than exactly this attempt, with a search on its own topic.
                click.secho(f"Not approved — {handle.NAME} will search and go deeper.", fg="yellow")
                prefix = rejection_prefix(handle, package, problem, text)

    click.secho("\nEvery stage approved. The run is complete.", fg="green", bold=True)
    return True


def chat() -> None:
    """Talk to the user until the requirements gate passes, then run the pipeline."""
    click.echo("Hey, I am Feature Engineer Helper feature-store-agent. How can I help you?")
    transcript: list[str] = []

    while True:
        question = ask("You: ")
        if question is None:
            return

        if not question.strip():
            continue

        transcript.append(f"User: {question}")
        context = "\n".join(transcript)

        # Every new question is indexed into both packages' knowledge bases as it arrives, regardless
        # of whether `rag_data_engineer_decider_agent`/`problem_analyzer_agent` are in "rag" mode yet —
        # see `save_question`'s own docstring for why this is best-effort and never blocks the turn.
        save_question("data_engineer", question)
        save_question("feature_engineering", question)

        try:
            verdict = as_json(RequirementAgent.run(question=question, context=context))
        except ValueError as error:  # an unresolvable key or model — nothing to retry against
            click.secho(f"Feature-Store Agent: ({error})", fg="red")
            return

        # A reply that would not parse is treated as "not yet": the chat agent asks for what is
        # missing, which is the same thing that happens on an honest `requirements: false`.
        if verdict and is_met(verdict):
            click.secho("\nRequirements met — starting the pipeline.", fg="green", bold=True)
            click.echo(f"  problem       {verdict.get('problem', '')}")
            click.echo(f"  data details  {verdict.get('data_details', '')}")
            run_pipeline(question, context, verdict)
            return

        gate = json.dumps(verdict) if verdict else "the requirements check did not return a verdict"
        reply = ChatAgent.run(
            question=question,
            context=f"{context}\n\nRequirements check: {gate}\n\n"
            "Requirements are not met yet. Ask the user for whatever is missing — the problem "
            "statement, or where the data is and how to connect to it — in one short question.",
        )
        transcript.append(f"Assistant: {reply}")
        click.echo(f"Feature-Store Agent: {reply}")

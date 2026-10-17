"""A short turn in which a small model picks the tool for an agent's stage.

An agent's own prompt is two to three thousand tokens of method, and its tools carry nine or more
arguments each. A 0.5B model given both writes about calling the tool instead of calling it: the
tool-trained build of `Caglana/qwen0.5b-tinylora-ds-assistant` made no call in 7 of 7 such turns, where
the same model asked in one sentence, with each tool cut to the argument or two that matter, called
the right tool in 35 of 35 (`benchmarks/RESULTS.md`). Its arguments are the weak part: it invents
column names, option names and steps (`analyzers: ["train"]`), mangles a path, and once in a while
cuts a JSON call short.

So a small model gets a *tool turn*:

    prompt    one sentence from `prompts/tool_turns.md`, with the table and the target filled in
    tools     only the tools this stage can use, each with a one-line description and the arguments
              named in `VIEWS` -- a path and, where there is one, a target
    proposal  whatever it calls, its name resolved to a real tool (`feature_selection_tools` is
              `feature_selection_tool`) and its arguments checked and recorded

What happens to a proposal is the caller's choice (`console/run.py`): the pipeline runs the stage's
calls with arguments taken from what the earlier stages found, and counts the model's choice; or,
with `--no-playbook`, runs the model's own call after `repair_arguments` has put the path right and
dropped what the tool does not take, to see how far the model gets by itself.

A model that is not a small local one keeps its agents' own prompts and tools: `uses_tool_turn` says
which.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS = Path(__file__).resolve().parent / "prompts" / "tool_turns.md"

# Callers of a model run in this process: the ones a tool turn is for. Matched by class name anywhere in
# the caller's bases, so a subclass (`LayerLoraAdapterHuggingFaceLLM`) counts.
LOCAL_MODELS = ("HuggingFaceLocalLLM",)

NAME_CUTOFF = 0.75  # how alike a tool name the model wrote must be to a real one to be read as it


@dataclass(frozen=True)
class View:
    """What a small model is shown of a tool: one line, and the arguments worth choosing."""

    description: str
    arguments: tuple[str, ...]


# Every tool a tool turn can offer. `arguments` come first-required: the first is the table, and the
# second, where there is one, the target. Everything else a tool takes is left to its defaults or to
# the pipeline, which knows the roles of the columns and a small model does not.
VIEWS: dict[str, View] = {
    "data_reader_tool": View(
        "Read a dataset and report its rows, columns and column types.", ("data_source",)
    ),
    "data_analyzer_tool": View(
        "Analyse a dataset: missing values, duplicates, outliers, correlations, and which columns give the target away.",
        ("data_source", "target"),
    ),
    "data_prep_tool": View(
        "Clean a dataset: fix the types, strip whitespace and drop duplicate rows.",
        ("data_source",),
    ),
    "missing_report_tool": View(
        "Report which columns of a dataset have missing values, and how to fill each.", ("data",)
    ),
    "impute_missing_tool": View("Fill the missing values of a dataset.", ("data",)),
    "feature_prep_tool": View(
        "Build features from a dataset: dates, ratios, encodings and history per entity.", ("data",)
    ),
    "feature_screening_tool": View(
        "Check which columns of a dataset can be features at all.", ("data",)
    ),
    "feature_selection_tool": View(
        "Rank the features against the target and keep the best ones.", ("data", "target")
    ),
}

# The tools each tool-bearing agent is offered in a tool turn, in the order its stage runs them.
OFFERED: dict[str, tuple[str, ...]] = {
    "data_reader_agent": ("data_reader_tool",),
    "data_analyzer_agent": ("data_analyzer_tool",),
    "data_preprocessor_agent": ("data_prep_tool",),
    "data_quality_agent": ("data_analyzer_tool",),
    "missing_value_agent": ("missing_report_tool", "impute_missing_tool"),
    "feature_prep_agent": ("feature_prep_tool",),
    "feature_selection_agent": ("feature_screening_tool", "feature_selection_tool"),
}

PATH_ARGUMENTS = ("data_source", "data")


@dataclass
class Proposal:
    """One call a model wrote: as written, as read, and what is wrong with its arguments."""

    written: str
    tool: str | None
    arguments: dict[str, Any]
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """A real tool, called with nothing the tool turn would have to take back."""
        return self.tool is not None and not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "tool": self.tool,
            "arguments": self.arguments,
            "problems": self.problems,
        }


def is_local_model(agent: Any, mode: str = "auto") -> bool:
    """Whether this agent's model is one a tool turn is for: `on`, `off`, or `auto` for a model run in this process."""
    if mode in ("on", "off"):
        return mode == "on"
    return any(cls.__name__ in LOCAL_MODELS for cls in type(getattr(agent, "llm", None)).__mro__)


def uses_tool_turn(agent: Any, mode: str = "auto") -> bool:
    """Whether this agent's model gets a tool turn instead of the agent's own prompt."""
    if getattr(agent, "name", None) not in OFFERED or not getattr(agent, "toolbox", None):
        return False
    return is_local_model(agent, mode)


@lru_cache(maxsize=1)
def prompts() -> dict[str, str]:
    """`prompts/tool_turns.md`: the text under each `## agent_name` heading."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in PROMPTS.read_text().splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def render_prompt(agent_name: str, path: str, target: str | None) -> str:
    """The sentence an agent's tool turn opens with. A sentence about the target is left out when none is known."""
    text = prompts()[agent_name]
    if target is None:
        text = " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if "{target}" not in s)
    return text.format(path=path, target=target or "")


def compact_schemas(agent: Any, tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
    """The agent's own tool schemas, cut to `VIEWS`: one line each, and only the arguments a small model should choose."""
    schemas = agent.toolbox.schemas_for(agent.llm)
    out = []
    for schema in schemas:
        function = schema.get("function", schema)
        name = function.get("name")
        if name not in tool_names or name not in VIEWS:
            continue
        view = VIEWS[name]
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        kept = {
            a: {**properties[a], "description": str(properties[a].get("description", ""))[:100]}
            for a in view.arguments
            if a in properties
        }
        compact = {
            **function,
            "description": view.description,
            "parameters": {**parameters, "properties": kept, "required": [view.arguments[0]]},
        }
        out.append({**schema, "function": compact} if "function" in schema else compact)
    return out


def resolve_name(written: str, offered: tuple[str, ...]) -> str | None:
    """The real tool a written name means: exact, then ignoring case and `_tool`/`s`, then the closest above `NAME_CUTOFF`."""
    if written in offered:
        return written
    squashed = {re.sub(r"(_tools?|s)$", "", n.lower()): n for n in offered}
    key = re.sub(r"(_tools?|s)$", "", written.strip().lower())
    if key in squashed:
        return squashed[key]
    close = difflib.get_close_matches(
        written.strip().lower(), [n.lower() for n in offered], n=1, cutoff=NAME_CUTOFF
    )
    return next((n for n in offered if n.lower() == close[0]), None) if close else None


def declared(tool: Any) -> dict[str, dict[str, Any]]:
    """The arguments a tool takes, with their declared types."""
    return tool.declaration()["parameters"]["properties"]


def is_path(value: Any) -> bool:
    text = str(value)
    return "://" in text or Path(text).expanduser().is_file()


def check_arguments(
    tool: Any, arguments: dict[str, Any], path: str, columns: list[str] | None = None
) -> list[str]:
    """What is wrong with the arguments a model wrote, as plain sentences: an argument the tool does not take, a path that is not the table's, a
    column that is not in it, a value of the wrong kind."""
    problems = []
    takes = declared(tool)
    kinds = {
        "array": list,
        "object": dict,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "string": str,
    }
    for name, value in arguments.items():
        if name not in takes:
            problems.append(f"{name}: the tool takes no such argument")
            continue
        expected = kinds.get(str(takes[name].get("type", "")).lower())
        if expected and not isinstance(value, expected):
            problems.append(
                f"{name}: expected {takes[name].get('type')}, got {type(value).__name__}"
            )
        elif name in PATH_ARGUMENTS and value and str(value) != str(path) and not is_path(value):
            problems.append(f"{name}: {value} is not a file")
        elif name == "target" and columns and value not in columns:
            problems.append(f"target: no column named {value!r}")
        elif (
            columns
            and isinstance(value, list)
            and name.endswith("columns")
            and any(v not in columns for v in value)
        ):
            problems.append(
                f"{name}: names columns the table does not have: {[v for v in value if v not in columns][:3]}"
            )
    return problems


def repair_arguments(
    tool_name: str,
    tool: Any,
    arguments: dict[str, Any],
    path: str,
    target: str | None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """The call a model's call becomes when it is run as its own: only the arguments the view offers, a path that exists, a target that is a column.

    The target is added for any tool that takes one. Anything else the model added is dropped, so the tool's own defaults decide it; a path or target it got wrong, or left out, is the pipeline's.
    """
    view = VIEWS[tool_name]
    takes = declared(tool)
    kept = {k: v for k, v in arguments.items() if k in view.arguments and k in takes}
    for name in PATH_ARGUMENTS:
        if (
            name in takes
            and name in view.arguments
            and (not kept.get(name) or not is_path(kept[name]))
        ):
            kept[name] = path
    # The target is the pipeline's to supply for every tool that takes one, offered to the model or not: a build tool that is not told
    # which column is the answer folds it into its row totals and group means.
    if "target" in takes:
        if columns and kept.get("target") not in columns:
            kept.pop("target", None)
        if target and not kept.get("target"):
            kept["target"] = target
    return kept


def propose(
    agent: Any, path: str, target: str | None, max_tokens: int, columns: list[str] | None = None
) -> tuple[list[Proposal], str, str]:
    """Run an agent's tool turn. Returns `(proposals, the model's text, the prompt it was given)`."""
    offered = OFFERED[agent.name]
    prompt = render_prompt(agent.name, path, target)
    schemas = compact_schemas(agent, offered)
    response = agent.llm.converse(
        [{"role": "user", "content": prompt}], tools=schemas, max_tokens=max_tokens
    )
    text, calls = agent.llm.read_turn(response)
    proposals = []
    for call in calls:
        tool = resolve_name(call.name, offered)
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        problems = (
            []
            if tool is None
            else check_arguments(agent.toolbox.agent_tools[tool], arguments, path, columns)
        )
        if tool is None:
            problems = [f"{call.name}: not one of {list(offered)}"]
        elif tool != call.name:
            problems.insert(0, f"{call.name}: read as {tool}")
        proposals.append(
            Proposal(written=call.name, tool=tool, arguments=arguments, problems=problems)
        )
    return proposals, text, prompt


def summarize(tool: str, result: dict[str, Any], path: str) -> str:
    """What a tool returned, as a few plain lines for the next agent: counts, where the table went, the tool's own summary."""
    if not isinstance(result, dict) or result.get("status") != "ok":
        message = (
            result.get("message", "unknown error") if isinstance(result, dict) else str(result)
        )
        return f"{tool} failed: {message}"
    lines = [f"{tool} ran."]
    counts = [
        f"{k}={result[k]:,}" if isinstance(result[k], int) else f"{k}={result[k]}"
        for k in (
            "row_count",
            "rows",
            "column_count",
            "filled",
            "remaining",
            "feature_count",
            "selected_count",
        )
        if k in result and not isinstance(result[k], (list, dict))
    ]
    if counts:
        lines.append(", ".join(counts) + ".")
    if isinstance(result.get("selected"), list):
        lines.append("Selected: " + ", ".join(map(str, result["selected"][:20])) + ".")
    if result.get("summary_text"):
        lines.append(str(result["summary_text"]).split("\n\nFindings")[0])
    written = result.get("output_path")
    lines.append(f"Handover: the table is at {written or path}.")
    return "\n".join(lines)


def record(proposals: list[Proposal]) -> list[dict[str, Any]]:
    return [p.as_dict() for p in proposals]

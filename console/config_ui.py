"""A local web UI for editing the `llms:` and `agents:` blocks of this repo's three agent configs.

    python cli.py configure          # opens http://127.0.0.1:<port> in a browser

Three `agentic_configurations.yaml` files describe every agent this project runs, and each one is
picked separately in the UI because each drives a different part of the run — see `CONFIGS` below for
what a given choice actually changes. Picking one opens its editor: the `llms:` block first (add,
remove, or edit `model`, `api_key`, `max_tokens`, `type`, `temperature`), then the `agents:` block
(same three operations, with `prompt_path` chosen through a file browser served off this machine
rather than typed by hand).

Only `llms:` and `agents:` are rewritten. Everything else in the file — the header comments, `dbs:`,
`embeddings:`, `tools:`, `orchestrators:`, `pipeline:` — is copied back out byte for byte, and inside
the two edited blocks an entry that came back unchanged is re-emitted as its own original lines, so
the comment above `judger_agent_1` (and every other one) survives an edit somewhere else in the file.
An entry that *did* change keeps the comment on its name line and any comment block directly under it;
the rest of its body is re-emitted from the submitted values.

Nothing here talks to a model or imports `agent_builder` — it reads and writes YAML, and the field
lists it validates against are the ones `utils.configs` parses (`LLMConfigs`, `AgentConfigs`) and the
agent classes `builder.factory.AGENT_TYPES` registers. The server binds to 127.0.0.1 only, and the
file browser refuses any path outside this repository.
"""

from __future__ import annotations

import http.server
import json
import re
import socket
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent

# What each config drives, so the picker can say what a choice actually changes before it is made.
# `agents` lists the entries `console/run.py` really runs from that file — a config holds more agent
# entries than the pipeline uses (the judgers and the two `benchmarks/`-prompted agents are configured
# but deliberately not in `STAGES`), and saying so up front is the difference between editing the
# thing that runs and editing the thing that doesn't.
CONFIGS: dict[str, dict[str, Any]] = {
    "console": {
        "path": "console/agentic_configurations.yaml",
        "title": "Console",
        "tagline": "The conversation, the requirements gate, and the script written at the end.",
        "changes": [
            "`chat_agent` — the front door: every turn you type at `cli.py generate` goes here.",
            "`requirement_agent` — the gate that decides whether a run may start at all.",
            "`code_generator_agent` — reads all eight stages' outputs and writes `scripts/main.py`.",
            "Its `llms:` entries back only those three; the pipeline stages read their own configs.",
        ],
        "runs": ["chat_agent", "requirement_agent", "code_generator_agent"],
    },
    "data_engineer": {
        "path": "data_engineer/agentic_configurations.yaml",
        "title": "Data engineer",
        "tagline": "Reading, profiling, cleaning and quality-checking the dataset.",
        "changes": [
            "`data_reader_agent` — reads the source and reports its profile.",
            "`data_analyzer_agent` — the profile, the findings and the analysis steps.",
            "`data_preprocessor_agent` — the cleaning sequence actually applied.",
            "`data_quality_agent` — the post-cleaning check the run is gated on.",
            "Also holds the judgers and `rag_data_engineer_decider_agent`, which `console/run.py` "
            "leaves out of `STAGES` — editing those changes nothing about a normal run.",
        ],
        "runs": [
            "data_reader_agent",
            "data_analyzer_agent",
            "data_preprocessor_agent",
            "data_quality_agent",
        ],
    },
    "feature_engineering": {
        "path": "feature_engineering/agentic_configurations.yaml",
        "title": "Feature engineering",
        "tagline": "Framing the problem, imputing, building features and selecting them.",
        "changes": [
            "`problem_analyzer_agent` — names the problem type and the target.",
            "`missing_value_agent` — the imputation strategy per column.",
            "`feature_prep_agent` — the features built, without ever reading the target.",
            "`feature_selection_agent` — the scorers run and the features kept.",
            "Switching `problem_analyzer_agent`'s type to `rag` is what makes it retrieve from "
            "`ds_knowledge_db` before answering.",
        ],
        "runs": [
            "problem_analyzer_agent",
            "missing_value_agent",
            "feature_prep_agent",
            "feature_selection_agent",
        ],
    },
}

# `agents.<name>.type` -> the class that runs that role, straight out of `builder.factory.AGENT_TYPES`.
# Listed here rather than imported so opening the UI costs no provider SDK import.
AGENT_TYPES = [
    "generator",
    "worker",
    "judger",
    "classifier",
    "planner",
    "thinker",
    "rag",
    "rag_builder",
    "retriever",
]

# `<provider>/<model-id>` — the prefix has to be one `agent_builder.LLM_CALLERS` implements.
LLM_PROVIDERS = [
    "claude",
    "anthropic",
    "openai",
    "google",
    "gemini",
    "grok",
    "xai",
    "ollama",
    "mistral",
    "huggingface",
    "hf",
    "huggingface_local",
    "hf_local",
]

# Providers that run the model in-process: a public Hub repo needs no token, so `api_key` may be empty.
KEYLESS_PROVIDERS = {"huggingface_local", "hf_local"}

LLM_TYPES = ["generator", "judger", "retriever", "tool caller", "tool generator"]

# The fields each editor renders itself, in the order they are written back. Anything else an entry
# carries — `responsiblity_prompt`, `text_vector`, the `dependencies:` typo two console agents have —
# is preserved through the "other fields" YAML box rather than silently dropped.
LLM_FIELDS = ["model", "api_key", "max_tokens", "type", "temperature"]
AGENT_FIELDS = [
    "type",
    "llm",
    "substitute_llm",
    "db_vector",
    "db_text",
    "prompt",
    "prompt_path",
    "dependency_agent",
    "tools",
    "thresholds",
]


class _Dumper(yaml.SafeDumper):
    """`safe_dump`, writing the style the three configs are already written in.

    Three departures from PyYAML's defaults, all cosmetic and all about a rewritten entry sitting
    unremarkably next to the untouched ones around it: sequences are indented under their key
    (`tools:` then `  - "x"`, not a flush-left `- x`), string *values* are double-quoted, and keys
    are left plain — the registered `str` representer would otherwise quote both sides of every
    `"model": "..."`.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
        return super().increase_indent(flow, False)

    def represent_mapping(self, tag: str, mapping: Any, flow_style: bool | None = None) -> Any:
        """PyYAML's own, with the one change that a string key is emitted plain rather than quoted."""
        pairs: list[tuple[Any, Any]] = []
        node = yaml.nodes.MappingNode(tag, pairs, flow_style=flow_style)
        if self.alias_key is not None:
            self.represented_objects[self.alias_key] = node

        for key, value in mapping.items() if hasattr(mapping, "items") else mapping:
            node_key = (
                self.represent_scalar(
                    "tag:yaml.org,2002:str", key
                )  # plain, not the str representer
                if isinstance(key, str)
                else self.represent_data(key)
            )
            pairs.append((node_key, self.represent_data(value)))

        if flow_style is None:
            node.flow_style = (
                self.default_flow_style if self.default_flow_style is not None else False
            )
        return node


_Dumper.add_representer(
    str, lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
)


SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")
ENTRY_RE = re.compile(r"^(\s+)([A-Za-z_][A-Za-z0-9_.-]*):")
EDITABLE = ("llms", "agents")


class ConfigError(ValueError):
    """Something the submitted config says that cannot be written — reported back to the browser."""


# --------------------------------------------------------------------------------------------------
# The YAML document: split into top-level sections, and the two edited ones into named entries.
# --------------------------------------------------------------------------------------------------


@dataclass
class Entry:
    """One `llms:` or `agents:` entry, as both its original lines and its parsed value."""

    name: str
    lead: list[str] = field(
        default_factory=list
    )  # blank/comment lines directly above the name line
    header: str = ""  # the `  name:  # comment` line itself
    body: list[str] = field(default_factory=list)  # everything under it, up to the next entry
    value: Any = None  # what those lines parse to


@dataclass
class Section:
    """One top-level block. `entries` is filled only for the two blocks this UI rewrites."""

    key: str | None  # None for the preamble above the first block
    lines: list[str]
    entries: list[Entry] | None = None
    indent: int = 2
    tail: list[str] = field(
        default_factory=list
    )  # trailing comments, which lead the *next* section


class ConfigDocument:
    """Read one `agentic_configurations.yaml`, edit its `llms:`/`agents:` blocks, write it back."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8")
        self.data: dict[str, Any] = yaml.safe_load(self.text) or {}
        self.sections = self._split_sections(self.text.splitlines())

    # -- reading ------------------------------------------------------------------------------------

    @staticmethod
    def _split_sections(lines: list[str]) -> list[Section]:
        starts = [i for i, line in enumerate(lines) if SECTION_RE.match(line)]
        if not starts:
            return [Section(key=None, lines=list(lines))]

        sections: list[Section] = []
        if starts[0] > 0:
            sections.append(Section(key=None, lines=lines[: starts[0]]))

        for n, start in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            key = SECTION_RE.match(lines[start]).group(1)  # type: ignore[union-attr]
            section = Section(key=key, lines=lines[start:end])
            if key in EDITABLE:
                ConfigDocument._split_entries(section)
            sections.append(section)
        return sections

    @staticmethod
    def _split_entries(section: Section) -> None:
        """Fill `section.entries` from its lines, keeping each entry's own comments attached to it."""
        body = section.lines[1:]  # everything under the `llms:` / `agents:` line
        indents = [
            len(line) - len(line.lstrip())
            for line in body
            if line.strip() and not line.strip().startswith("#") and ENTRY_RE.match(line)
        ]
        section.indent = min(indents) if indents else 2

        starts = [
            i
            for i, line in enumerate(body)
            if (match := ENTRY_RE.match(line)) and len(match.group(1)) == section.indent
        ]

        entries: list[Entry] = []
        for n, start in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else len(body)

            lead_from = start
            while lead_from > (starts[n - 1] if n else -1) + 1:
                previous = body[lead_from - 1].strip()
                if previous and not previous.startswith("#"):
                    break
                lead_from -= 1

            entry = Entry(
                name=ENTRY_RE.match(body[start]).group(2),  # type: ignore[union-attr]
                lead=body[lead_from:start],
                header=body[start],
                body=body[start + 1 : end],
            )
            # The previous entry claimed everything up to `lead_from`; trim what this one took back.
            if entries:
                overlap = len(entries[-1].body) - (start - lead_from)
                entries[-1].body = entries[-1].body[: max(overlap, 0)]
            entry.value = ConfigDocument._parse_entry(entry, section.indent)
            entries.append(entry)

        if entries:
            # Trailing blank/comment lines under the last entry are the block's own closing note (a
            # non-last entry's are already claimed as the next one's lead), so they belong to the
            # section — removing that entry must not take them with it.
            body_lines = entries[-1].body
            while body_lines and (
                not body_lines[-1].strip() or body_lines[-1].strip().startswith("#")
            ):
                body_lines = body_lines[:-1]
            entries[-1].body = body_lines
            claimed = starts[-1] + 1 + len(entries[-1].body)
            section.tail = body[claimed:]
        else:
            section.tail = body
        section.entries = entries

    @staticmethod
    def _parse_entry(entry: Entry, indent: int) -> Any:
        raw = "\n".join(
            line[indent:] if len(line) > indent else line.lstrip()
            for line in [entry.header, *entry.body]
        )
        parsed = yaml.safe_load(raw) or {}
        return parsed.get(entry.name)

    # -- what the browser sees ----------------------------------------------------------------------

    def block(self, key: str) -> dict[str, Any]:
        section = self._section(key)
        if not section or section.entries is None:
            return {}
        return {entry.name: entry.value for entry in section.entries}

    def _section(self, key: str) -> Section | None:
        return next((section for section in self.sections if section.key == key), None)

    def tool_names(self) -> list[str]:
        tools = self.data.get("tools") or []
        if isinstance(tools, dict):
            return sorted(tools)
        return [tool["name"] for tool in tools if isinstance(tool, dict) and tool.get("name")]

    def db_names(self, category: str) -> list[str]:
        dbs = self.data.get("dbs") or []
        if isinstance(dbs, dict):
            dbs = [{"name": name, **cfg} for name, cfg in dbs.items()]
        return [db["name"] for db in dbs if db.get("type") == category and db.get("name")]

    def referenced_agents(self) -> dict[str, list[str]]:
        """Where outside `agents:` an agent name is used, so removing one can be warned about."""
        used: dict[str, list[str]] = {}

        def note(name: Any, where: str) -> None:
            if isinstance(name, str):
                used.setdefault(name, []).append(where)

        for orchestrator, cfg in (self.data.get("orchestrators") or {}).items():
            for sub in (cfg or {}).get("sub_agents") or []:
                note(sub, f"orchestrators.{orchestrator}")

        for flow in self.data.get("pipeline") or []:
            for step in flow.get("steps") or []:
                processor = step.get("processor") or {}
                if processor.get("type") == "agent":
                    note(processor.get("name"), f"pipeline.{flow.get('name')}.{step.get('name')}")
                for judger in step.get("needs_judger") or []:
                    note(judger, f"pipeline.{flow.get('name')}.{step.get('name')}")
        return used

    # -- writing ------------------------------------------------------------------------------------

    def render(self, blocks: dict[str, dict[str, Any]]) -> str:
        out: list[str] = []
        for section in self.sections:
            if section.key in blocks and section.entries is not None:
                out.extend(self._render_section(section, blocks[section.key]))
            else:
                out.extend(section.lines)
        return "\n".join(out).rstrip("\n") + "\n"

    def _render_section(self, section: Section, values: dict[str, Any]) -> list[str]:
        originals = {entry.name: entry for entry in (section.entries or [])}
        out = [section.lines[0]]

        for position, (name, value) in enumerate(values.items()):
            original = originals.get(name)
            if original is not None and original.value == value:
                out.extend([*original.lead, original.header, *original.body])
                continue

            if original is not None:
                out.extend(original.lead)
                out.append(self._header_line(name, original.header, section.indent))
                out.extend(self._kept_comments(original, section.indent))
            else:
                if out[-1].strip():
                    out.append("")
                out.append(f"{' ' * section.indent}{name}:")
            out.extend(self._dump_value(value, section.indent + 2))

        out.extend(section.tail)
        return out

    @staticmethod
    def _header_line(name: str, original_header: str, indent: int) -> str:
        """`  name:` — carrying the inline comment the original name line had, if any."""
        _, _, comment = original_header.partition("#")
        line = f"{' ' * indent}{name}:"
        return f"{line}  # {comment.strip()}" if comment.strip() else line

    @staticmethod
    def _kept_comments(entry: Entry, indent: int) -> list[str]:
        """The comment block directly under the name line, which documents the entry as a whole."""
        kept: list[str] = []
        for line in entry.body:
            if line.strip().startswith("#"):
                kept.append(line)
                continue
            if line.strip():
                break
        return kept

    @staticmethod
    def _dump_value(value: Any, indent: int) -> list[str]:
        text = yaml.dump(
            value,
            Dumper=_Dumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
        return [f"{' ' * indent}{line}" if line.strip() else line for line in text.splitlines()]

    def save(self, blocks: dict[str, dict[str, Any]]) -> None:
        rendered = self.render(blocks)
        yaml.safe_load(rendered)  # never write a file that no longer parses
        self.path.write_text(rendered, encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# Turning what the browser submits into the two blocks, and refusing what cannot be built from.
# --------------------------------------------------------------------------------------------------

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _parse_mapping(text: str, label: str) -> dict[str, Any]:
    """Parse one of the free-form YAML boxes — `thresholds`, and the per-entry `other fields`."""
    if not (text or "").strip():
        return {}
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{label}: not valid YAML — {error}") from error
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{label}: expected `key: value` lines, got {type(parsed).__name__}.")
    return parsed


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalise_llm(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One LLM card -> the `llms:` entry it writes. Every required field is checked here."""
    name = _clean(payload.get("name"))
    if not name:
        raise ConfigError("An LLM has no name.")
    if not NAME_RE.match(name):
        raise ConfigError(f"{name}: a name must start with a letter and hold no spaces.")

    value: dict[str, Any] = {}
    for required in ("model", "type"):
        given = _clean(payload.get(required))
        if not given:
            raise ConfigError(f"{name}: `{required}` is required.")
        value[required] = given

    if "/" not in value["model"]:
        raise ConfigError(
            f"{name}: `model` must be `<provider>/<model-id>`, e.g. `gemini/gemini-3.1-flash-lite`."
        )
    provider = value["model"].split("/", 1)[0]
    if provider not in LLM_PROVIDERS:
        raise ConfigError(
            f"{name}: `{provider}` is not a provider agent-builder has a caller for — "
            f"one of {', '.join(sorted(set(LLM_PROVIDERS)))}."
        )

    api_key = _clean(payload.get("api_key"))
    if api_key:
        value["api_key"] = api_key
    elif provider not in KEYLESS_PROVIDERS:
        raise ConfigError(f"{name}: `api_key` is required.")

    max_tokens = _clean(payload.get("max_tokens"))
    if not max_tokens:
        raise ConfigError(f"{name}: `max_tokens` is required.")
    try:
        value["max_tokens"] = int(max_tokens)
    except ValueError as error:
        raise ConfigError(f"{name}: `max_tokens` must be a whole number.") from error
    if value["max_tokens"] <= 0:
        raise ConfigError(f"{name}: `max_tokens` must be greater than zero.")

    temperature = _clean(payload.get("temperature"))
    if temperature:
        try:
            value["temperature"] = float(temperature)
        except ValueError as error:
            raise ConfigError(f"{name}: `temperature` must be a number, or left empty.") from error

    ordered = {key: value[key] for key in LLM_FIELDS if key in value}
    ordered.update(_parse_mapping(payload.get("extra", ""), f"{name}: other fields"))
    return name, ordered


def normalise_agent(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """One agent card -> the `agents:` entry it writes."""
    name = _clean(payload.get("name"))
    if not name:
        raise ConfigError("An agent has no name.")
    if not NAME_RE.match(name):
        raise ConfigError(f"{name}: a name must start with a letter and hold no spaces.")

    agent_type = _clean(payload.get("type"))
    if not agent_type:
        raise ConfigError(f"{name}: `type` is required.")
    if agent_type not in AGENT_TYPES:
        raise ConfigError(
            f"{name}: `{agent_type}` is not an agent type — one of {', '.join(AGENT_TYPES)}."
        )

    llm = _clean(payload.get("llm"))
    if not llm:
        raise ConfigError(f"{name}: `llm` is required — pick one of the LLMs above.")

    prompt = _clean(payload.get("prompt"))
    prompt_path = _clean(payload.get("prompt_path"))
    if not prompt and not prompt_path:
        raise ConfigError(
            f"{name}: needs a prompt — either `prompt_path` (one .md file) or `prompt` (a directory)."
        )

    value: dict[str, Any] = {"type": agent_type, "llm": llm}
    for optional in ("substitute_llm", "db_vector", "db_text"):
        if given := _clean(payload.get(optional)):
            value[optional] = given
    if prompt:
        value["prompt"] = prompt
    if prompt_path:
        value["prompt_path"] = prompt_path

    dependencies = [_clean(item) for item in payload.get("dependency_agent") or [] if _clean(item)]
    if dependencies:  # a single dependency is written as the plain string the configs already use
        value["dependency_agent"] = dependencies[0] if len(dependencies) == 1 else dependencies

    if tools := [_clean(item) for item in payload.get("tools") or [] if _clean(item)]:
        value["tools"] = tools

    if thresholds := _parse_mapping(payload.get("thresholds", ""), f"{name}: thresholds"):
        value["thresholds"] = thresholds

    ordered = {key: value[key] for key in AGENT_FIELDS if key in value}
    ordered.update(_parse_mapping(payload.get("extra", ""), f"{name}: other fields"))
    return name, ordered


def build_blocks(
    document: ConfigDocument, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate the whole submission as one config, and report what it breaks but can still write."""
    llms: dict[str, Any] = {}
    for card in payload.get("llms") or []:
        name, value = normalise_llm(card)
        if name in llms:
            raise ConfigError(f"Two LLMs are both named `{name}`.")
        llms[name] = value

    agents: dict[str, Any] = {}
    for card in payload.get("agents") or []:
        name, value = normalise_agent(card)
        if name in agents:
            raise ConfigError(f"Two agents are both named `{name}`.")
        agents[name] = value

    if not llms:
        raise ConfigError("A config needs at least one LLM — every agent references one by name.")

    for name, agent in agents.items():
        for slot in ("llm", "substitute_llm"):
            if (referenced := agent.get(slot)) and referenced not in llms:
                raise ConfigError(
                    f"{name}: `{slot}` names `{referenced}`, which is not an LLM here."
                )

    warnings: list[str] = []
    known_tools = set(document.tool_names())
    used_elsewhere = document.referenced_agents()
    # `console`'s code_generator_agent depends on the two pipeline packages' agents by name, so a
    # dependency this file does not define is only dangling when no other config defines it either.
    elsewhere = {
        agent
        for name, meta in CONFIGS.items()
        if (ROOT / meta["path"]) != document.path
        for agent in ConfigDocument(ROOT / meta["path"]).block("agents")
    }

    for name, agent in agents.items():
        dependencies = agent.get("dependency_agent")
        dependencies = [dependencies] if isinstance(dependencies, str) else dependencies or []
        for dependency in dependencies:
            if dependency not in agents and dependency not in elsewhere:
                warnings.append(f"{name}: depends on `{dependency}`, which no config defines.")
        for tool in agent.get("tools") or []:
            if tool not in known_tools:
                warnings.append(f"{name}: `{tool}` is not in this file's `tools:` block.")
        for slot in ("prompt", "prompt_path"):
            if given := agent.get(slot):
                if not (document.path.parent / given).exists() and not (ROOT / given).exists():
                    warnings.append(
                        f"{name}: `{slot}` points at `{given}`, which does not exist yet."
                    )

    for removed in set(document.block("agents")) - set(agents):
        if where := used_elsewhere.get(removed):
            warnings.append(f"`{removed}` is removed but still referenced by {', '.join(where)}.")

    for removed in set(document.block("llms")) - set(llms):
        warnings.append(f"LLM `{removed}` was removed.")

    return {"llms": llms, "agents": agents}, warnings


# --------------------------------------------------------------------------------------------------
# The server. Localhost only, and the file browser never leaves this repository.
# --------------------------------------------------------------------------------------------------


def _document(name: str) -> ConfigDocument:
    if name not in CONFIGS:
        raise ConfigError(f"`{name}` is not one of {', '.join(CONFIGS)}.")
    return ConfigDocument(ROOT / CONFIGS[name]["path"])


def config_summaries() -> list[dict[str, Any]]:
    """The picker's three cards: what each config is, and what editing it changes."""
    summaries = []
    for name, meta in CONFIGS.items():
        document = _document(name)
        summaries.append(
            {
                "name": name,
                "title": meta["title"],
                "tagline": meta["tagline"],
                "path": meta["path"],
                "changes": meta["changes"],
                "runs": meta["runs"],
                "llm_count": len(document.block("llms")),
                "agent_count": len(document.block("agents")),
            }
        )
    return summaries


def config_detail(name: str) -> dict[str, Any]:
    document = _document(name)
    meta = CONFIGS[name]
    agents = document.block("agents")
    return {
        "name": name,
        "title": meta["title"],
        "tagline": meta["tagline"],
        "path": meta["path"],
        "changes": meta["changes"],
        "runs": meta["runs"],
        "config_dir": str(document.path.parent.relative_to(ROOT)),
        "llms": document.block("llms"),
        "agents": agents,
        "available": {
            "agent_types": AGENT_TYPES,
            "llm_types": LLM_TYPES,
            "providers": sorted(set(LLM_PROVIDERS)),
            "tools": document.tool_names(),
            "vector_dbs": document.db_names("vector"),
            "text_dbs": document.db_names("text"),
            "llm_fields": LLM_FIELDS,
            "agent_fields": AGENT_FIELDS,
        },
        "referenced_agents": document.referenced_agents(),
    }


def save_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    document = _document(name)
    blocks, warnings = build_blocks(document, payload)
    document.save(blocks)
    return {
        "saved": CONFIGS[name]["path"],
        "llm_count": len(blocks["llms"]),
        "agent_count": len(blocks["agents"]),
        "warnings": warnings,
    }


def browse(name: str, relative: str) -> dict[str, Any]:
    """List one directory for the `prompt_path` picker, rooted at (and locked to) the repo."""
    document = _document(name)
    config_dir = document.path.parent

    target = (ROOT / relative).resolve() if relative else config_dir
    if not str(target).startswith(str(ROOT)):
        raise ConfigError("That path is outside this repository.")
    if not target.is_dir():
        target = config_dir

    def entry(item: Path) -> dict[str, Any]:
        try:
            for_config = str(item.relative_to(config_dir))
        except ValueError:  # outside the config's own package — a repo-relative path still resolves
            for_config = str(item.relative_to(ROOT))
        return {
            "name": item.name,
            "directory": item.is_dir(),
            "path": str(item.relative_to(ROOT)),
            "value": for_config,
        }

    items = sorted(
        (
            item
            for item in target.iterdir()
            if not item.name.startswith(".")
            and item.name not in ("__pycache__", "node_modules")
            and (item.is_dir() or item.suffix == ".md")
        ),
        key=lambda item: (not item.is_dir(), item.name.lower()),
    )
    parent = target.parent
    return {
        "here": str(target.relative_to(ROOT)) or ".",
        "parent": str(parent.relative_to(ROOT)) if target != ROOT else None,
        "config_dir": str(config_dir.relative_to(ROOT)),
        "entries": [entry(item) for item in items],
    }


class Handler(http.server.BaseHTTPRequestHandler):
    """The whole API, plus the single page that drives it."""

    server_version = "feature-store-agent-config-ui"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        return  # the page polls nothing; a request log would only bury the "open this URL" line

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _handle(self, work: Callable[[], Any]) -> None:
        try:
            self._json(work())
        except ConfigError as error:
            self._json({"error": str(error)}, status=400)
        except (
            Exception
        ) as error:  # a bad path, an unreadable file — say which, don't kill the server
            self._json({"error": f"{type(error).__name__}: {error}"}, status=500)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        url = urlparse(self.path)
        query = {key: value[0] for key, value in parse_qs(url.query).items()}

        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/configs":
            self._handle(config_summaries)
        elif url.path == "/api/config":
            self._handle(lambda: config_detail(query.get("name", "")))
        elif url.path == "/api/browse":
            self._handle(lambda: browse(query.get("name", ""), query.get("path", "")))
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        url = urlparse(self.path)
        query = {key: value[0] for key, value in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)

        if url.path != "/api/config":
            self._json({"error": "not found"}, status=404)
            return

        def work() -> Any:
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as error:
                raise ConfigError(f"malformed request: {error}") from error
            return save_config(query.get("name", ""), payload)

        self._handle(work)


def _free_port(preferred: int) -> int:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def serve(port: int = 8787, open_browser: bool = True) -> None:
    """Run the editor until Ctrl-C. Binds 127.0.0.1 only — nothing here is reachable off this machine."""
    port = _free_port(port)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"

    print(f"Agent config editor: {url}")
    print("Editing:")
    for name, meta in CONFIGS.items():
        print(f"  {name:<20} {meta['path']}")
    print("Ctrl-C to stop.")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------------------------------
# The page. One file, no build step, no CDN — it is served off this machine and edits local files.
# --------------------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent configuration</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #14171c; --muted: #5d6472; --line: #e2e5ea;
    --accent: #2f5bd0; --accent-ink: #ffffff; --danger: #b3261e; --warn: #8a5a00;
    --warn-bg: #fff7e6; --ok: #1f6b3a; --code: #eef0f4;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --panel: #171a20; --ink: #e7e9ee; --muted: #99a1b0; --line: #262b34;
      --accent: #6c8fff; --accent-ink: #0f1115; --danger: #ff6b60; --warn: #e0b25f;
      --warn-bg: #2a2113; --ok: #6bd08d; --code: #1e222a;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
  a { color: var(--accent); }

  header.bar { position: sticky; top: 0; z-index: 20; display: flex; gap: 12px; align-items: center;
               padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--line); }
  header.bar h1 { font-size: 15px; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
  header.bar .path { color: var(--muted); font-size: 12.5px; }
  header.bar .spacer { flex: 1; }

  main { max-width: 1020px; margin: 0 auto; padding: 24px 20px 80px; }
  .lede { color: var(--muted); margin: 0 0 22px; max-width: 62ch; }

  button { font: inherit; cursor: pointer; border-radius: 7px; border: 1px solid var(--line);
           background: var(--panel); color: var(--ink); padding: 7px 12px; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.ghost { background: transparent; }
  button.danger { color: var(--danger); }
  button.small { padding: 4px 9px; font-size: 12.5px; }

  .chooser { max-width: 420px; }
  .chooser select { font-size: 14.5px; padding: 9px 10px; }

  .detail { margin-top: 18px; background: var(--panel); border: 1px solid var(--line);
            border-radius: 12px; overflow: hidden; }
  .detail .top { padding: 16px 18px 14px; }
  .detail h2 { margin: 0 0 4px; font-size: 15.5px; }
  .detail .tagline { color: var(--muted); margin: 0 0 12px; }
  .detail .what { font-size: 12px; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .04em; color: var(--muted); margin: 0 0 6px; }
  .detail ul { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }
  .detail li { margin-bottom: 5px; }
  .detail .foot { padding: 11px 18px; border-top: 1px solid var(--line); display: flex;
                  gap: 14px; align-items: center; color: var(--muted); font-size: 12.5px; }
  .detail .foot .spacer { flex: 1; }
  .placeholder { margin-top: 18px; padding: 26px 18px; text-align: center; color: var(--muted);
                 border: 1px dashed var(--line); border-radius: 12px; }

  section.block { margin-top: 30px; }
  section.block > h2 { font-size: 15px; margin: 0 0 4px; display: flex; align-items: center; gap: 10px; }
  section.block > p { color: var(--muted); margin: 0 0 14px; max-width: 70ch; }

  .entry { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
           margin-bottom: 12px; overflow: hidden; }
  .entry > .head { display: flex; align-items: center; gap: 10px; padding: 10px 14px; }
  .entry > .head .title { font-weight: 600; }
  .entry > .head .meta { color: var(--muted); font-size: 12.5px; }
  .entry > .head .spacer { flex: 1; }
  .entry > .head .caret { color: var(--muted); width: 14px; }
  .entry.open > .head { border-bottom: 1px solid var(--line); }
  .entry .fields { padding: 14px; display: grid; gap: 12px;
                   grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); }
  .entry .fields.closed { display: none; }
  .field.wide { grid-column: 1 / -1; }
  label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 4px; }
  label .req { color: var(--danger); }
  label .hint { font-weight: 400; color: var(--muted); }
  input, select, textarea { width: 100%; font: inherit; padding: 7px 9px; border-radius: 7px;
                            border: 1px solid var(--line); background: var(--bg); color: var(--ink); }
  textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
             min-height: 62px; resize: vertical; }
  input:focus, select:focus, textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .with-button { display: flex; gap: 8px; }
  .with-button input { flex: 1; }

  .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px; border: 1px solid var(--line);
           border-radius: 7px; background: var(--bg); max-height: 148px; overflow: auto; }
  .chips label { display: inline-flex; align-items: center; gap: 6px; margin: 0; font-weight: 400;
                 font-size: 12.5px; padding: 3px 8px; border-radius: 20px; border: 1px solid var(--line);
                 background: var(--panel); cursor: pointer; }
  .chips input { width: auto; }
  .chips .none { color: var(--muted); padding: 3px 6px; }

  .notice { border-radius: 9px; padding: 10px 13px; margin-bottom: 14px; font-size: 13px; }
  .notice.error { background: color-mix(in srgb, var(--danger) 12%, transparent);
                  border: 1px solid var(--danger); }
  .notice.warn { background: var(--warn-bg); border: 1px solid var(--warn); color: var(--warn); }
  .notice.ok { background: color-mix(in srgb, var(--ok) 12%, transparent); border: 1px solid var(--ok); }
  .notice ul { margin: 6px 0 0; padding-left: 18px; }

  dialog { border: 1px solid var(--line); border-radius: 12px; padding: 0; background: var(--panel);
           color: var(--ink); width: min(560px, 92vw); }
  dialog::backdrop { background: rgba(0,0,0,.45); }
  dialog .dhead { padding: 14px 16px; border-bottom: 1px solid var(--line); }
  dialog .dhead h3 { margin: 0 0 3px; font-size: 14.5px; }
  dialog .dhead .where { color: var(--muted); font-size: 12.5px; }
  dialog .list { max-height: 46vh; overflow: auto; }
  dialog .row { display: flex; gap: 9px; align-items: center; width: 100%; text-align: left;
                border: 0; border-bottom: 1px solid var(--line); border-radius: 0; padding: 9px 16px;
                background: transparent; }
  dialog .row:hover { background: var(--code); border-color: var(--line); }
  dialog .row .kind { width: 16px; color: var(--muted); }
  dialog .dfoot { padding: 11px 16px; display: flex; gap: 8px; justify-content: flex-end;
                  border-top: 1px solid var(--line); }
</style>
</head>
<body>
<header class="bar">
  <button id="back" class="ghost small" hidden>&larr; Configs</button>
  <h1 id="heading">Agent configuration</h1>
  <span class="path mono" id="subheading"></span>
  <span class="spacer"></span>
  <button id="save" class="primary" hidden>Save to YAML</button>
</header>
<main>
  <div id="messages"></div>
  <div id="view"></div>
</main>

<dialog id="browser">
  <div class="dhead">
    <h3>Pick a prompt file</h3>
    <div class="where mono" id="browser-where"></div>
  </div>
  <div class="list" id="browser-list"></div>
  <div class="dfoot">
    <button class="ghost" id="browser-cancel">Cancel</button>
  </div>
</dialog>

<script>
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, kids = []) => {
  const node = Object.assign(document.createElement(tag), props);
  (Array.isArray(kids) ? kids : [kids]).forEach(k => node.append(k));
  return node;
};

// The fields each editor renders itself; anything else an entry carries goes to its "other fields"
// box, so a key this UI has no widget for is preserved rather than dropped on save.
const LLM_KNOWN = ["model", "model_name", "api_key", "max_tokens", "type", "temperature"];
const AGENT_KNOWN = ["type", "llm", "substitute_llm", "db_vector", "db_text", "prompt",
                     "prompt_path", "dependency_agent", "tools", "thresholds"];

let state = { view: "picker", detail: null, llms: [], agents: [] };
let dirty = false;  // whether anything has been typed since the last load or save

/* ---- a small YAML writer, for the free-form boxes ------------------------------------------- */
function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) {
    if (!value.length) return "";
    return value.map(item => (item && typeof item === "object")
      ? pad + "-\n" + toYaml(item, indent + 2)
      : pad + "- " + scalar(item)).join("\n");
  }
  if (typeof value === "object") {
    return Object.entries(value).map(([key, inner]) => {
      if (inner && typeof inner === "object") {
        const body = toYaml(inner, indent + 2);
        return body ? pad + key + ":\n" + body : pad + key + ": {}";
      }
      return pad + key + ": " + scalar(inner);
    }).join("\n");
  }
  return pad + scalar(value);
}
function scalar(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string" && (value === "" || /[:#{}\[\],&*?|>%@`"']/.test(value)))
    return JSON.stringify(value);
  return String(value);
}
function rest(value, known) {
  const extra = {};
  Object.entries(value || {}).forEach(([k, v]) => { if (!known.includes(k)) extra[k] = v; });
  return toYaml(extra);
}

/* ---- loading ---------------------------------------------------------------------------------- */
async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function say(kind, text, items = []) {
  const box = el("div", { className: "notice " + kind });
  box.append(el("strong", { textContent: text }));
  if (items.length) box.append(el("ul", {}, items.map(i => el("li", { textContent: i }))));
  $("#messages").replaceChildren(box);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
const clearSay = () => $("#messages").replaceChildren();

async function showPicker() {
  state = { view: "picker", detail: null, llms: [], agents: [] };
  dirty = false;
  $("#back").hidden = true; $("#save").hidden = true;
  $("#heading").textContent = "Agent configuration";
  $("#subheading").textContent = "";
  location.hash = "";
  const configs = await api("/api/configs");

  const description = el("div", { id: "description" });
  const select = el("select");
  ["", ...configs.map(config => config.name)].forEach(name => {
    const config = configs.find(item => item.name === name);
    select.append(el("option", { value: name,
      textContent: config ? config.title : "— choose a configuration —" }));
  });
  // Every change re-renders the panel under it, so the description always describes what is picked.
  select.onchange = () => describe(configs.find(config => config.name === select.value));

  $("#view").replaceChildren(
    el("p", { className: "lede", textContent:
      "Three YAML files describe every agent this project runs. Pick the one whose part of the run " +
      "you want to change — what that choice touches is described below it." }),
    el("div", { className: "chooser" }, [
      wrap("Configuration", { hint: "each drives a different part of the run" }, select),
    ]),
    description,
  );
  describe(null);
}

function describe(config) {
  if (!config) {
    $("#description").replaceChildren(el("div", { className: "placeholder",
      textContent: "Pick a configuration above to see what editing it changes." }));
    return;
  }
  $("#description").replaceChildren(el("div", { className: "detail" }, [
    el("div", { className: "top" }, [
      el("h2", { textContent: config.title }),
      el("p", { className: "tagline", textContent: config.tagline }),
      el("p", { className: "what", textContent: "What editing this changes" }),
      el("ul", {}, config.changes.map(change => {
        const li = el("li");
        li.innerHTML = change.replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]))
                             .replace(/`([^`]+)`/g, "<code>$1</code>");
        return li;
      })),
    ]),
    el("div", { className: "foot" }, [
      el("span", { className: "mono", textContent: config.path }),
      el("span", { textContent: config.llm_count + " llms · " + config.agent_count + " agents" }),
      el("span", { className: "spacer" }),
      el("button", { className: "primary", textContent: "Edit " + config.title + " →",
                     onclick: () => openConfig(config.name) }),
    ]),
  ]));
}

async function openConfig(name) {
  clearSay();
  const detail = await api("/api/config?name=" + encodeURIComponent(name));
  state = {
    view: "editor",
    detail,
    llms: Object.entries(detail.llms).map(([n, v]) => ({
      name: n, open: false,
      model: v.model || v.model_name || "",
      api_key: v.api_key ?? "",
      max_tokens: v.max_tokens ?? "",
      type: v.type || "generator",
      temperature: v.temperature ?? "",
      extra: rest(v, LLM_KNOWN),
    })),
    agents: Object.entries(detail.agents).map(([n, v]) => ({
      name: n, open: false,
      type: v.type || "generator",
      llm: v.llm || "",
      substitute_llm: v.substitute_llm || "",
      db_vector: v.db_vector || "",
      db_text: v.db_text || "",
      prompt: v.prompt || "",
      prompt_path: v.prompt_path || "",
      dependency_agent: typeof v.dependency_agent === "string"
        ? [v.dependency_agent] : (v.dependency_agent || []),
      tools: v.tools || [],
      thresholds: toYaml(v.thresholds || {}),
      extra: rest(v, AGENT_KNOWN),
    })),
  };
  location.hash = name;
  dirty = false;
  $("#back").hidden = false; $("#save").hidden = false;
  $("#heading").textContent = detail.title;
  $("#subheading").textContent = detail.path;
  renderEditor();
}

/* ---- the editor ------------------------------------------------------------------------------- */
function renderEditor() {
  const { detail } = state;
  $("#view").replaceChildren(
    el("p", { className: "lede", textContent: detail.tagline +
      " The pipeline runs " + detail.runs.join(", ") + " from this file." }),
    llmSection(),
    agentSection(),
  );
}

function entryCard(card, index, kind, subtitle, fields) {
  const head = el("div", { className: "head" }, [
    el("span", { className: "caret", textContent: card.open ? "▾" : "▸" }),
    el("span", { className: "title mono", textContent: card.name || "(unnamed)" }),
    el("span", { className: "meta", textContent: subtitle }),
    el("span", { className: "spacer" }),
    el("button", { className: "small danger ghost", textContent: "Remove",
      onclick: (event) => {
        event.stopPropagation();
        if (!confirm("Remove " + (card.name || "this entry") + " from the config?")) return;
        harvest(); state[kind].splice(index, 1); renderEditor();
      } }),
  ]);
  head.onclick = () => { harvest(); state[kind][index].open = !card.open; renderEditor(); };
  const body = el("div", { className: "fields" + (card.open ? "" : " closed") }, fields);
  return el("div", { className: "entry" + (card.open ? " open" : "") }, [head, body]);
}

function textField(card, key, label, opts = {}) {
  const input = el("input", { value: card[key] ?? "", placeholder: opts.placeholder || "" });
  input.dataset.key = key;
  if (opts.list) input.setAttribute("list", opts.list);
  return wrap(label, opts, input);
}

function selectField(card, key, label, options, opts = {}) {
  const select = el("select");
  select.dataset.key = key;
  const all = opts.blank ? ["", ...options] : [...options];  // never the caller's own array
  if (card[key] && !all.includes(card[key])) all.push(card[key]);
  all.forEach(option => select.append(el("option", {
    value: option, textContent: option || (opts.blankLabel || "— none —"),
    selected: option === (card[key] || ""),
  })));
  return wrap(label, opts, select);
}

function areaField(card, key, label, opts = {}) {
  const area = el("textarea", { value: card[key] ?? "", placeholder: opts.placeholder || "" });
  area.dataset.key = key;
  return wrap(label, { ...opts, wide: true }, area);
}

function wrap(label, opts, control) {
  const caption = el("label", {}, [document.createTextNode(label)]);
  if (opts.required) caption.append(el("span", { className: "req", textContent: " *" }));
  if (opts.hint) caption.append(el("span", { className: "hint", textContent: "  " + opts.hint }));
  return el("div", { className: "field" + (opts.wide ? " wide" : "") }, [caption, control]);
}

function chipsField(card, key, label, options, opts = {}) {
  const box = el("div", { className: "chips" });
  box.dataset.key = key;
  box.dataset.multi = "1";
  if (!options.length) box.append(el("span", { className: "none", textContent: opts.empty || "none available" }));
  options.forEach(option => {
    const input = el("input", { type: "checkbox", value: option, checked: (card[key] || []).includes(option) });
    box.append(el("label", {}, [input, document.createTextNode(option)]));
  });
  return wrap(label, { ...opts, wide: true }, box);
}

function llmSection() {
  const { detail } = state;
  const datalist = el("datalist", { id: "provider-models" },
    detail.available.providers.map(p => el("option", { value: p + "/" })));

  const cards = state.llms.map((card, index) => entryCard(card, index, "llms",
    (card.model || "no model") + " · " + (card.max_tokens || "?") + " tokens", [
      textField(card, "name", "Name", { required: true, hint: "referenced by agents" }),
      textField(card, "model", "model", { required: true, list: "provider-models",
        placeholder: "gemini/gemini-3.1-flash-lite", hint: "provider/model-id" }),
      textField(card, "api_key", "api_key", { required: true, placeholder: "GEMINI",
        hint: "env var name, or the key itself — optional for hf_local" }),
      textField(card, "max_tokens", "max_tokens", { required: true, placeholder: "16000",
        hint: "thinking counts against it" }),
      selectField(card, "type", "type", detail.available.llm_types, { required: true }),
      textField(card, "temperature", "temperature", { hint: "optional — Claude 5 rejects it" }),
      areaField(card, "extra", "Other fields (YAML)", {
        hint: "anything this form has no box for — mcp_servers, tools, …" }),
    ]));

  const section = el("section", { className: "block" }, [
    el("h2", {}, [document.createTextNode("LLMs"),
      el("span", { className: "meta", style: "font-weight:400;color:var(--muted)",
        textContent: state.llms.length + " configured" })]),
    el("p", { textContent:
      "Every agent below names one of these. model, api_key, max_tokens and type are required — a " +
      "new entry starts with all four filled in, and the provider in model must be one agent-builder " +
      "has a caller for." }),
    datalist,
    ...cards,
    el("button", { textContent: "+ Add LLM", onclick: () => {
      harvest();
      state.llms.push({ name: uniqueName("llm", state.llms), open: true,
        model: "gemini/gemini-3.1-flash-lite", api_key: "GEMINI", max_tokens: 16000,
        type: "generator", temperature: "", extra: "" });
      renderEditor();
    } }),
  ]);
  return section;
}

function agentSection() {
  const { detail } = state;
  const llmNames = state.llms.map(llm => llm.name).filter(Boolean);

  const cards = state.agents.map((card, index) => {
    const others = state.agents.map(a => a.name).filter(n => n && n !== card.name);
    const pathRow = el("div", { className: "with-button" });
    const pathInput = el("input", { value: card.prompt_path || "",
      placeholder: "prompts/data_reader.md" });
    pathInput.dataset.key = "prompt_path";
    pathRow.append(pathInput, el("button", { type: "button", className: "small",
      textContent: "Browse…", onclick: () => pickPrompt(pathInput) }));

    return entryCard(card, index, "agents",
      card.type + " · " + (card.llm || "no llm") + (detail.runs.includes(card.name) ? " · in the pipeline" : ""), [
        textField(card, "name", "Name", { required: true }),
        selectField(card, "type", "type", detail.available.agent_types, { required: true }),
        selectField(card, "llm", "llm", llmNames, { required: true, blank: true,
          blankLabel: "— pick an LLM —" }),
        selectField(card, "substitute_llm", "substitute_llm", llmNames, { blank: true,
          hint: "used when the primary call raises" }),
        selectField(card, "db_vector", "db_vector", detail.available.vector_dbs, { blank: true,
          hint: "rag / retriever types only" }),
        selectField(card, "db_text", "db_text", detail.available.text_dbs, { blank: true }),
        wrap("prompt_path", { required: true, wide: true,
          hint: "a single .md file, relative to " + detail.config_dir }, pathRow),
        textField(card, "prompt", "prompt", { hint: "a directory of .md files, instead of the above" }),
        chipsField(card, "dependency_agent", "dependency_agent", others,
          { hint: "their output is what {agent_output} renders", empty: "no other agents yet" }),
        chipsField(card, "tools", "tools", detail.available.tools,
          { hint: "from this file's tools: block", empty: "this config declares no tools" }),
        areaField(card, "thresholds", "thresholds (YAML)", { hint: "judger bars, e.g. min_rows: 1000" }),
        areaField(card, "extra", "Other fields (YAML)", {
          hint: "responsiblity_prompt, mcp_servers, anything else" }),
      ]);
  });

  return el("section", { className: "block" }, [
    el("h2", {}, [document.createTextNode("Agents"),
      el("span", { style: "font-weight:400;color:var(--muted)",
        textContent: state.agents.length + " configured" })]),
    el("p", { textContent:
      "type, llm and a prompt are required. prompt_path is picked from this machine with Browse — " +
      "the path is written relative to " + detail.config_dir + ", which is where agent-builder " +
      "resolves it from." }),
    ...cards,
    el("button", { textContent: "+ Add agent", onclick: () => {
      harvest();
      state.agents.push({ name: uniqueName("agent", state.agents), open: true, type: "generator",
        llm: llmNames[0] || "", substitute_llm: "", db_vector: "", db_text: "", prompt: "",
        prompt_path: "", dependency_agent: [], tools: [], thresholds: "", extra: "" });
      renderEditor();
    } }),
  ]);
}

function uniqueName(stem, cards) {
  let n = 1, name;
  do { name = "new_" + stem + "_" + n++; } while (cards.some(card => card.name === name));
  return name;
}

/* ---- read the DOM back into state, so a re-render never loses a keystroke -------------------- */
function harvest() {
  ["llms", "agents"].forEach(kind => {
    const section = kind === "llms" ? 0 : 1;
    const entries = $("#view").querySelectorAll("section.block")[section];
    if (!entries) return;
    entries.querySelectorAll(":scope > .entry").forEach((node, index) => {
      const card = state[kind][index];
      if (!card) return;
      node.querySelectorAll("[data-key]").forEach(control => {
        const key = control.dataset.key;
        if (control.dataset.multi) {
          card[key] = [...control.querySelectorAll("input:checked")].map(box => box.value);
        } else {
          card[key] = control.value;
        }
      });
    });
  });
}

/* ---- the file picker -------------------------------------------------------------------------- */
let pickerTarget = null;
async function pickPrompt(input) {
  pickerTarget = input;
  // Open where the current value points — it is written relative to the config directory, which is
  // also where the server resolves it from — rather than starting over at the top every time.
  const current = (input.value || "").replace(/[^/]*$/, "");
  await loadBrowser(current ? state.detail.config_dir + "/" + current : "");
  if (!$("#browser").open) $("#browser").showModal();
}
async function loadBrowser(path) {
  const listing = await api("/api/browse?name=" + encodeURIComponent(state.detail.name) +
                            "&path=" + encodeURIComponent(path || ""));
  $("#browser-where").textContent = listing.here;
  const rows = [];
  if (listing.parent !== null)
    rows.push(row("↑", "..", () => loadBrowser(listing.parent)));
  listing.entries.forEach(entry => rows.push(entry.directory
    ? row("📁", entry.name, () => loadBrowser(entry.path))
    : row("📄", entry.name, () => {
        pickerTarget.value = entry.value;
        dirty = true;  // a programmatic value change fires no input event
        harvest();
        $("#browser").close();
        renderEditor();
      })));
  if (!rows.length) rows.push(el("div", { className: "row", textContent: "nothing here" }));
  $("#browser-list").replaceChildren(...rows);
}
function row(kind, name, onclick) {
  return el("button", { className: "row", onclick }, [
    el("span", { className: "kind", textContent: kind }),
    el("span", { className: "mono", textContent: name }),
  ]);
}

/* ---- saving ----------------------------------------------------------------------------------- */
async function save() {
  harvest();
  clearSay();
  try {
    const result = await api("/api/config?name=" + encodeURIComponent(state.detail.name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ llms: state.llms, agents: state.agents }),
    });
    await openConfig(state.detail.name);  // re-read, so the page shows what the file now says
    say(result.warnings.length ? "warn" : "ok",
        "Wrote " + result.saved + " — " + result.llm_count + " llms, " + result.agent_count +
        " agents." + (result.warnings.length ? " It still parses; these are worth a look:" : ""),
        result.warnings);
  } catch (error) {
    say("error", "Nothing was written — " + error.message);
  }
}

/* ---- wiring ----------------------------------------------------------------------------------- */
$("#back").onclick = () => {
  if (dirty && !confirm("Leave without saving? Your edits are only in this page.")) return;
  showPicker();
};
$("#save").onclick = () => save();
$("#browser-cancel").onclick = () => $("#browser").close();
$("#view").addEventListener("input", () => { dirty = true; });
$("#view").addEventListener("change", () => { dirty = true; });
window.addEventListener("beforeunload", event => {
  if (dirty) { event.preventDefault(); event.returnValue = ""; }
});

const start = location.hash.slice(1);
(start ? openConfig(start).catch(() => showPicker()) : showPicker())
  .catch(error => say("error", error.message));
</script>
</body>
</html>
"""

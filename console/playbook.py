"""The default tool calls of each pipeline agent: what the console runs when the model does not.

Every worker agent in `console/run.py`'s `STAGES` is meant to call its tools itself. A large model
does. The 0.5B model this project is benchmarked with (`benchmarks/RESULTS.md`) spends minutes
writing a first turn, rarely writes a `TOOL_CALL:` line that parses, and when it does the arguments
are the ones a tool does not take. A stage that makes no call leaves the next one nothing to read,
and a run that never calls a tool never writes a feature table. So an agent that made no successful
call in its turn is given its playbook: the calls a careful person would make at that stage, in
order, with the arguments filled in from what the earlier stages actually found.

Nothing here guesses at data. Each step reads the previous tools' answers (the analyzer's recipe,
the missing-value report, the screening flags) and passes them on; the only decisions made here are
the ones the tools' own docs say to make from those answers. The text each playbook returns is built
from the tools' results alone, so the next agent's `{agent_output}` holds what ran, not what a model
said ran.

    state = State(problem="predict churn (the column churned)", source="data/customers.csv")
    text = PLAYBOOKS["data_analyzer_agent"](state, call)       # `call(tool_name, arguments) -> result`

`call` is whatever runs a tool for the caller (the console's, which also records the call for
`scripts/main.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Call = Callable[[str, dict[str, Any]], dict[str, Any]]

TIME_NAME = re.compile(
    r"(^|_)(date|time|timestamp|ts|month|week|day|year|period|idx|index|dt)(_|$)", re.IGNORECASE
)
# up to this many levels a category is also given as a code, since a tree can still isolate one
# level of it
MAX_LEVELS_CODED = 60
# a categorical with more levels than this is frequency-encoded instead
MAX_LEVELS_ONE_HOT = 10
# ratios are generated pairwise: past this many measurements they are noise
MAX_RATIO_COLUMNS = 6
# a feature store serves a short list: the cut the selection stage makes
TOP_K = 20


@dataclass
class State:
    """What the stages have learned so far, handed from one playbook to the next."""

    problem: str
    source: str
    path: str = ""
    target: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    ids: list[str] = field(default_factory=list)
    entity: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    skewed: list[str] = field(default_factory=list)
    recipe: dict[str, Any] = field(default_factory=dict)
    task: str | None = None
    levels: dict[str, int] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    leaks: list[str] = field(default_factory=list)
    rows_per_entity: float = 1.0
    facts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = self.path or self.source


def stem_path(path: str, suffix: str) -> str:
    """`dir/name.csv` -> `dir/name_<suffix>.csv`: where a stage writes beside the table it read (the tools' own convention)."""
    p = Path(path)
    base = re.sub(r"_(prepared|imputed|features|selected)$", "", p.stem)
    return str(p.with_name(f"{base}_{suffix}.csv"))


def find_target(problem: str, columns: list[str]) -> str | None:
    """The column the problem is about: a column name the statement mentions, best-marked first (`the column X`, `(X)`, `X column`)."""
    mentioned = [c for c in columns if re.search(rf"(?<![\w]){re.escape(c)}(?![\w])", problem)]
    if not mentioned:
        return None

    def rank(column: str) -> tuple[int, int]:
        marked = bool(
            re.search(
                rf"columns?\s+[`'\"(]?{re.escape(column)}\b|\({re.escape(column)}\)|\b{re.escape(column)}\s+column",
                problem,
            )
        )
        return (0 if marked else 1, -len(column))

    return sorted(mentioned, key=rank)[0]


def names(items: Any) -> list[str]:
    return [str(i) for i in items] if isinstance(items, (list, tuple)) else []


def time_like(columns: list[str], numeric: list[str]) -> list[str]:
    """Columns that order the rows in time but that no date parser would call a date: `signup_month`, `reading_idx`, `snapshot_month`."""
    return [c for c in columns if TIME_NAME.search(c) and c in numeric]


def fmt_list(items: list[str], most: int = 12) -> str:
    shown = ", ".join(items[:most])
    return (
        shown + (f" (+{len(items) - most} more)" if len(items) > most else "") if items else "none"
    )


# --------------------------------------------------------------------------------------------------------------------------------------
# data engineering
# --------------------------------------------------------------------------------------------------------------------------------------
def data_reader(state: State, call: Call) -> str:
    result = call("data_reader_tool", {"data_source": state.source})
    if result.get("status") != "ok":
        return f"Could not read {state.source}: {result.get('message', 'unknown error')}."
    state.columns = [c["name"] for c in result.get("columns", [])]
    state.rows = int(result.get("row_count", 0))
    state.target = state.target or find_target(state.problem, state.columns)
    dtypes: dict[str, list[str]] = {}
    for column in result.get("columns", []):
        dtypes.setdefault(column["dtype"], []).append(column["name"])
    kinds = "; ".join(f"{kind}: {fmt_list(cols, 8)}" for kind, cols in dtypes.items())
    return (
        f"Read {state.source}: {state.rows:,} rows x {len(state.columns)} columns ({result.get('format', 'delimited')}).\n"
        f"Column types: {kinds}.\nTarget column: {state.target or 'not found in the problem statement'}.\nHandover: the data is at {state.path}."
    )


def data_analyzer(state: State, call: Call) -> str:
    result = call(
        "data_analyzer_tool",
        {
            "data_source": state.path,
            "target": state.target,
            "detail": "full",
            "output_path": stem_path(state.path, "analysis").replace(".csv", ".json"),
        },
    )
    if result.get("status") != "ok":
        return f"Could not analyse {state.path}: {result.get('message', 'unknown error')}."
    absorb(state, result)
    findings = [f for f in result.get("findings", []) if f.get("level") in ("risk", "warning")][:10]
    lines = [f"- {f.get('level')} [{f.get('analysis')}] {f.get('message')}" for f in findings]
    return (
        result.get("summary_text", "").split("\n\nFindings")[0]
        + "\n\nWorst findings:\n"
        + ("\n".join(lines) or "- none")
        + f"\nEntity columns: {fmt_list(state.ids)}; date columns: {fmt_list(state.dates)}.\nHandover: analysed {state.path}."
    )


def learn(state: State, tool: str, result: dict[str, Any]) -> None:
    """Take a tool's result into the state when the tool ran outside its stage's playbook: a model's own call.

    The reader's columns and the analyzer's findings are what later stages read; any tool that wrote a table moves the working path to it.
    """
    if not isinstance(result, dict) or result.get("status") != "ok":
        return
    if tool == "data_reader_tool":
        state.columns = [c["name"] for c in result.get("columns", [])]
        state.rows = int(result.get("row_count", 0))
        state.target = state.target or find_target(state.problem, state.columns)
    elif tool == "data_analyzer_tool":
        absorb(state, result)
    written = result.get("output_path")
    if written and Path(str(written)).is_file():
        state.path = str(written)


def details_of(result: dict[str, Any], analysis: str) -> dict[str, Any]:
    """The numbers behind one analyzer (`detail="full"`), or {} when it did not run."""
    for entry in result.get("analyses") or []:
        if entry.get("analysis") == analysis:
            return entry.get("details") or {}
    return {}


def absorb(state: State, result: dict[str, Any]) -> None:
    """Take what an analyzer run learned: the roles of the columns, which numbers are skewed, how many levels a category has, what kind of target it is."""
    recipe = result.get("recipe") or {}
    columns = recipe.get("columns") or {}
    state.recipe = recipe
    state.ids = names(columns.get("id_columns")) or state.ids
    state.dates = names(columns.get("date_columns")) or state.dates
    state.numeric = names(columns.get("aggregation_columns")) or state.numeric
    state.categorical = names(columns.get("categorical_columns")) or state.categorical
    state.target = state.target or columns.get("target")
    described = details_of(result, "describe").get("columns") or {}
    state.levels = {
        c: int(info.get("distinct") or 0) for c, info in described.items()
    } or state.levels
    state.kinds = details_of(result, "describe").get("kinds") or state.kinds
    # Leakage: a column that alone ranks the target (almost) perfectly, or one named like it is
    # measured later that also carries signal.
    summary = details_of(result, "target_summary")
    associated = {
        a["column"]: float(a.get("association") or 0) for a in summary.get("associations") or []
    }
    flagged = [s["column"] for s in summary.get("leakage_suspects") or []]
    flagged += [c for c in summary.get("after_the_fact_names") or [] if associated.get(c, 0) >= 0.1]
    state.leaks = list(dict.fromkeys([*state.leaks, *flagged]))
    shaped = details_of(result, "distributions").get("columns") or {}
    state.skewed = [
        c for c, info in shaped.items() if info.get("suggested_transform")
    ] or state.skewed
    kind = str(details_of(result, "target_summary").get("kind") or "")
    state.task = {
        "binary": "binary",
        "multiclass": "multiclass",
        "categorical": "multiclass",
        "numeric": "regression",
        "continuous": "regression",
    }.get(kind, state.task)
    grain = details_of(result, "grain")
    state.entity = names(grain.get("entity_columns")) or state.entity
    state.rows_per_entity = float(
        (grain.get("rows_per_entity") or {}).get("mean") or state.rows_per_entity
    )


def data_preprocessor(state: State, call: Call) -> str:
    leaks = [c for c in state.leaks if c in state.columns and c != state.target]
    gone = {*leaks}
    # Types first, then the placeholders and split columns that would otherwise be read as values,
    # then rows and columns that carry nothing.
    core: list[Any] = [
        "strip_whitespace",
        "coerce_types",
        "ordinal_scales",
        "replace_sentinels",
        "merge_complementary",
        "drop_duplicates",
        "drop_constant_columns",
    ]
    # what the analyzer found to give the answer away goes before anything is built from it
    if leaks:
        core.insert(0, {"step": "drop_columns", "columns": leaks})
    # The analyzer's own clipping advice, as it gave it, for the columns that are still there.
    # Missing values are left for the stage that owns them: it flags a hole before it fills it.
    clip = []
    for step in (state.recipe.get("cleaning") or {}).get("steps", []):
        if isinstance(step, dict) and step.get("step") == "clip_outliers":
            columns = [c for c in names(step.get("columns")) if c != state.target and c not in gone]
            if columns:
                clip.append({**step, "columns": columns})
    out = stem_path(state.path, "prepared")
    result = call(
        "data_prep_tool", {"data_source": state.path, "steps": [*core, *clip], "output_path": out}
    )
    note = ""
    # the optional advice is what to give up first, not the whole stage
    if result.get("status") != "ok" and clip:
        note = f" (the clipping advice failed - {result.get('message', 'unknown error')} - so it was left out)"
        result = call(
            "data_prep_tool", {"data_source": state.path, "steps": core, "output_path": out}
        )
    if result.get("status") != "ok":
        return f"Preparation failed on {state.path}: {result.get('message', 'unknown error')}."
    before = state.rows
    state.path = result.get("output_path") or state.path
    state.rows = int(result.get("row_count") or state.rows)
    lines = []
    for entry in result.get("steps", []):
        detail = {
            k: v
            for k, v in entry.items()
            if k not in ("step", "columns", "rows") and v not in ({}, [], None, 0)
        }
        lines.append(f"- {entry.get('step')}: {str(detail)[:240] if detail else 'changed nothing'}")
    dropped_note = (
        f" Dropped as leakage before anything was built from it: {fmt_list(leaks)}."
        if leaks
        else ""
    )
    return (
        f"Prepared {state.path}: {before:,} -> {state.rows:,} rows.{dropped_note}{note}\nSteps run:\n"
        + "\n".join(lines)
        + "\nMissing values were left as they are: the missing-value stage flags them before it fills them.\n"
        + f"Handover: the prepared table is at {state.path}."
    )


def data_quality(state: State, call: Call) -> str:
    result = call(
        "data_analyzer_tool", {"data_source": state.path, "target": state.target, "detail": "full"}
    )
    if result.get("status") != "ok":
        return f"Could not check {state.path}: {result.get('message', 'unknown error')}."
    absorb(state, result)
    findings = result.get("findings", [])
    risky = [f for f in findings if f.get("level") in ("risk", "warning")]
    duplicates = int(
        (
            details_of(result, "duplicates").get("duplicate_rows")
            or details_of(result, "duplicates").get("duplicates")
            or 0
        )
    )
    return (
        result.get("summary_text", "").split("\n\nFindings")[0]
        + f"\n\nAfter preparation: {len(risky)} warning(s) or risk(s), {duplicates} duplicate row(s) left.\n"
        f"Handover: {state.path} is {'fit' if not duplicates else 'not yet fit'} to train on; what remains is for feature engineering to handle."
    )


# --------------------------------------------------------------------------------------------------------------------------------------
# feature engineering
# --------------------------------------------------------------------------------------------------------------------------------------
def problem_analyzer(state: State, call: Call) -> str:
    task = state.task or "unknown"
    metric = {
        "binary": "ROC AUC (and precision at a fixed alert rate)",
        "regression": "R-squared and mean absolute error",
        "multiclass": "macro F1",
    }.get(task, "a metric that fits the target")
    return (
        f"Problem: {state.problem}\nTarget: {state.target or 'not identified'} ({task}).\nEvaluation: {metric}, on held-out rows split by time or by entity where the table has either.\n"
        f"Entity columns: {fmt_list(state.ids)}. Time columns: {fmt_list(state.dates or time_like(state.columns, state.numeric))}.\n"
        "Leakage watch: any column that only exists after the outcome (a reason, a score, a status, a cost, a next-period value) must not become a feature.\n"
        "Escalation: none\nHandover: the target and the keys above; the data is at "
        + state.path
        + "."
    )


def missing_value(state: State, call: Call) -> str:
    report = call(
        "missing_report_tool",
        {
            "data": state.path,
            "id_columns": state.entity or state.ids,
            "date_columns": state.dates,
            "target": state.target,
        },
    )
    if report.get("status") != "ok":
        return f"Could not report missing values on {state.path}: {report.get('message', 'unknown error')}."
    holes = [
        c
        for c, info in (report.get("missing") or {}).items()
        if isinstance(info, dict) and info.get("missing")
    ]
    holes = [c for c in holes if c != state.target]
    if not holes:
        return f"{state.path} has no missing values outside the target.\nHandover: unchanged, at {state.path}."
    out = stem_path(state.path, "imputed")
    result = call(
        "impute_missing_tool",
        {
            "data": state.path,
            "method": "auto",
            "columns": holes,
            "id_columns": state.entity or state.ids,
            "date_columns": state.dates,
            "indicators": True,
            "output_path": out,
            "target": state.target,
        },
    )
    if result.get("status") != "ok":
        return f"Imputation failed on {state.path}: {result.get('message', 'unknown error')}."
    state.path = result.get("output_path") or state.path
    notes = " ".join(report.get("notes", [])[:3])
    return (
        f"{len(holes)} column(s) had holes: {fmt_list(holes)}. Filled {result.get('filled', 0):,} value(s), {result.get('remaining', 0):,} left; "
        f"flag columns added: {fmt_list(names(result.get('indicators_added')))}.\n{notes}\nHandover: the imputed table, with is_missing flags kept, is at {state.path}."
    )


def feature_plan(state: State) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The feature-prep steps this table calls for, read from the analyzer's roles — dates, history per entity, skew, categories, text, ratios — as
    `(step names, options per step)`: `feature_prep_tool` takes names in `steps` and their arguments in `options`, never a step as a mapping.
    """
    # noqa: E731
    unique = lambda c: bool(state.rows) and state.levels.get(c, 0) >= 0.9 * state.rows
    # An id with a different value on nearly every row names the row; one with a few dozen values (a
    # campaign, a country code) is also a category.
    gone = {
        *state.leaks,
        *[c for c in state.ids if state.levels.get(c, 0) > MAX_LEVELS_CODED or unique(c)],
    }
    kinds = state.kinds
    numeric = [
        c
        for c in (state.numeric or [c for c, k in kinds.items() if k == "numeric"])
        if c != state.target and c not in gone and not c.endswith("_is_missing")
    ]
    time_cols = state.dates or time_like(state.columns, numeric)
    steps: list[str] = []
    options: dict[str, dict[str, Any]] = {}

    def add(step: str, **arguments: Any) -> None:
        steps.append(step)
        if arguments:
            options[step] = arguments

    if state.dates:
        add("date_parts", date_columns=state.dates)
        add("cyclical_dates")
    history = [c for c in state.entity if c not in state.leaks]
    if history and state.rows_per_entity > 1.2 and time_cols:
        # What an entity did before this row, never after it: lags, trailing windows, running totals
        # and the gaps between events.
        measures = [c for c in numeric if c not in time_cols and c not in history][:6]
        add(
            "lags",
            id_columns=history,
            date_columns=time_cols,
            aggregation_columns=measures,
            lags=[1, 7],
        )
        add(
            "rolling_windows",
            id_columns=history,
            date_columns=time_cols,
            aggregation_columns=measures,
            windows=[7],
        )
        add("cumulative", id_columns=history, date_columns=time_cols, aggregation_columns=measures)
        add("event_gaps", id_columns=history, date_columns=time_cols)
    from feature_engineering.tools.feature_prep import coordinate_columns

    if len(coordinate_columns(state.columns)) >= 2:
        add("geo_distance")
    skewed = [c for c in state.skewed if c in numeric]
    if skewed:
        add("math_transformations", columns=skewed, operations=["log1p"])
    measures = [c for c in numeric if c not in time_cols]
    if 2 <= len(measures) <= MAX_RATIO_COLUMNS:
        add("ratios", columns=measures, max_features=12)

    # Text and categories, by the analyzer's own reading of each column. A column with a different
    # value on nearly every row is an identifier (a hash, a serial, an address): encoding it would
    # build a feature out of an id before screening has had the chance to drop it.
    wide = [
        c
        for c, k in kinds.items()
        if k in ("categorical", "text")
        and c not in gone
        and c != state.target
        and c not in time_cols
        and not unique(c)
    ]
    few = [c for c in wide if 0 < state.levels.get(c, 99) <= MAX_LEVELS_ONE_HOT]
    middling = [c for c in wide if MAX_LEVELS_ONE_HOT < state.levels.get(c, 99) <= MAX_LEVELS_CODED]
    many = [c for c in wide if c not in few]
    if any(k == "boolean" and c not in gone and c != state.target for c, k in kinds.items()):
        add("auto_boolean_flags")
    if few:
        add("one_hot_encoding", categorical_columns=few, max_categories=MAX_LEVELS_ONE_HOT)
    # how common a campaign is says little about which campaign it is: give a tree its identity as a
    # code as well
    if middling:
        add("ordinal_encoding", categorical_columns=middling)
    if many:
        add("frequency_encoding", categorical_columns=many)
    if any(kinds.get(c) == "text" for c in wide):
        add("auto_text_shape")
    return steps, options


def target_lags(state: State) -> dict[str, Any] | None:
    """The target's own past, for a numeric quantity followed through time per entity (daily units sold, hourly load): lags of 1 and 7 events.

    Only lags, and only 1 or more: a trailing window or a running total "up to and including this
    row" would include the row's own target.
    """
    time_cols = state.dates or time_like(state.columns, state.numeric)
    if (
        state.task == "regression"
        and state.target
        and state.entity
        and state.rows_per_entity > 3
        and time_cols
    ):
        return {
            "id_columns": state.entity,
            "date_columns": time_cols,
            "aggregation_columns": [state.target],
            "lags": [1, 7],
        }
    return None


def feature_prep(state: State, call: Call) -> str:
    steps, options = feature_plan(state)
    if not steps:
        return f"No feature step applies to {state.path}.\nHandover: unchanged, at {state.path}."
    out = stem_path(state.path, "features")
    result = call(
        "feature_prep_tool",
        {
            "data": state.path,
            "id_columns": state.entity or state.ids,
            "date_columns": state.dates,
            "steps": steps,
            "options": options,
            "output_path": out,
            "target": state.target,
        },
    )
    if result.get("status") != "ok":
        return (
            f"Feature preparation failed on {state.path}: {result.get('message', 'unknown error')}."
        )
    state.path = result.get("output_path") or state.path
    applied = [
        f"{s.get('step')}: +{s.get('features_added', 0) if isinstance(s.get('features_added'), int) else len(s.get('features_added') or [])}"
        for s in result.get("steps_applied", [])
    ]
    built = int(result.get("feature_count", 0))
    lag_arguments = target_lags(state)
    if lag_arguments:
        again = call(
            "feature_prep_tool",
            {
                "data": state.path,
                "id_columns": lag_arguments["id_columns"],
                "date_columns": lag_arguments["date_columns"],
                "steps": ["lags"],
                "options": {"lags": lag_arguments},
                "output_path": out,
            },
        )
        if again.get("status") == "ok":
            state.path = again.get("output_path") or state.path
            built += int(again.get("feature_count", 0))
            applied.append(
                f"lags of {state.target} (1 and 7 events back): +{again.get('feature_count', 0)}"
            )
    return (
        f"Built {built} feature(s) over {result.get('rows', 0):,} rows ({result.get('grain', 'one row per input row')}).\nSteps: "
        f"{'; '.join(applied) or 'none'}.\nHandover: the feature table is at {state.path}."
    )


def feature_selection(state: State, call: Call) -> str:
    keys = list(
        dict.fromkeys(
            [*state.ids, *state.entity, *state.dates, *time_like(state.columns, state.numeric)]
        )
    )
    keys = [c for c in keys if c not in state.leaks]
    screen = call("feature_screening_tool", {"data": state.path})
    if not state.target:
        return (
            f"Screened {state.path}: {len(names(screen.get('usable')))} usable column(s). No target column was found in the problem statement, so nothing could be ranked.\n"
            "Escalation: none\nHandover: " + state.path
        )
    out = stem_path(state.path, "selected")
    result = call(
        "feature_selection_tool",
        {
            "data": state.path,
            "target": state.target,
            "exclude": keys,
            "keep_columns": keys,
            "method": "auto",
            "top_k": TOP_K,
            "drop_noise": True,
            "output_path": out,
        },
    )
    if result.get("status") != "ok":
        return f"Selection failed on {state.path}: {result.get('message', 'unknown error')}."
    state.path = result.get("output_path") or state.path
    selected = names(result.get("selected"))
    dropped = [
        f"{d.get('feature')} ({d.get('reason') or d.get('stage')})"
        for d in result.get("dropped", [])
    ][:12]
    leaks = [s.get("feature") for s in result.get("leakage_suspects", [])]
    return (
        f"Selected {len(selected)} feature(s) for {state.target} by {result.get('method')}: {fmt_list(selected, 20)}.\nDropped: {fmt_list(dropped, 12)}.\n"
        f"Leakage suspects removed: {fmt_list([str(l) for l in leaks])}.\nKept as keys, not features: {fmt_list(keys)}.\nHandover: the selected feature table is at {state.path}."
    )


PLAYBOOKS: dict[str, Callable[[State, Call], str]] = {
    "data_reader_agent": data_reader,
    "data_analyzer_agent": data_analyzer,
    "data_preprocessor_agent": data_preprocessor,
    "data_quality_agent": data_quality,
    "problem_analyzer_agent": problem_analyzer,
    "missing_value_agent": missing_value,
    "feature_prep_agent": feature_prep,
    "feature_selection_agent": feature_selection,
}


STAGE_OF = {
    "data_reader_agent": "data engineering",
    "data_analyzer_agent": "data engineering",
    "data_preprocessor_agent": "data engineering",
    "data_quality_agent": "data engineering",
    "problem_analyzer_agent": "feature engineering",
    "missing_value_agent": "feature engineering",
    "feature_prep_agent": "feature engineering",
    "feature_selection_agent": "feature engineering",
}


def direct_caller(
    record: list[dict[str, Any]] | None = None, agent: Callable[[], str] | None = None
) -> Call:
    """A `call` that runs the tool functions the two configs name, with no agent or model between: for scoring the playbook on its own.

    Arguments a function does not take are dropped, as the toolbox does. Each call is appended to
    `record` in the shape `run.json` uses.
    """
    import importlib
    import inspect

    import yaml

    root = Path(__file__).resolve().parent.parent
    registry: dict[str, tuple[str, str]] = {}
    for package in ("data_engineer", "feature_engineering"):
        config = yaml.safe_load((root / package / "agentic_configurations.yaml").read_text())
        for tool in config.get("tools", []):
            module, _, function = str(tool["caller"]).rpartition(".")
            registry.setdefault(tool["name"], (module, function))

    failures: list[dict[str, Any]] = []

    def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        module, function = registry[name]
        fn = getattr(importlib.import_module(module), function)
        accepted = set(inspect.signature(fn).parameters)
        used = {k: v for k, v in arguments.items() if k in accepted and v is not None}
        result = fn(**used)
        who = agent() if agent else ""
        if isinstance(result, dict) and result.get("status") == "error":
            failures.append(
                {
                    "stage": STAGE_OF.get(who, ""),
                    "agent": who,
                    "tool": name,
                    "arguments": used,
                    "message": str(result.get("message", "")),
                }
            )
        elif record is not None:
            record.append(
                {
                    "stage": STAGE_OF.get(who, ""),
                    "agent": who,
                    "tool": name,
                    "function": f"{module}.{function}",
                    "arguments": used,
                    "result": result,
                }
            )
        return result

    call.failures = failures  # type: ignore[attr-defined]
    return call


def run_chain(
    problem: str, source: str, call: Call, current: Callable[[str], None] | None = None
) -> tuple[State, dict[str, str]]:
    """Run all eight playbooks in pipeline order and return the final state and each agent's text."""
    state = State(problem=problem, source=source)
    outputs: dict[str, str] = {}
    for name, playbook in PLAYBOOKS.items():
        if current:
            current(name)
        outputs[name] = playbook(state, call)
    return state, outputs

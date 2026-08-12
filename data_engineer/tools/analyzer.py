"""Analyzers that read a dataset and say what is in it, before anything changes it.

Nothing here edits a row. Each analyzer looks at the data and returns an `Analysis`: a sentence
summarising what it saw, the numbers behind that sentence, and the findings worth acting on. Running
several of them collects the lot into one `Report`, which is the thing a later step summarises —
`report.summary()` renders it as text for a model to read, `report.to_dict()` as structures for a
tool response, and `write_report` puts it on disk so a run can be collected up afterwards.

    from data_engineer.tools.analyzer import analyze, describe, recipe

    report = analyze("customers.csv", target="churned")

    report.summary()            # the text a summarising agent reads
    report["describe"].details  # the numbers behind one analysis
    report.findings             # everything worth acting on, across every analysis
    recipe(report)              # what to cook: cleaning steps, then feature prep arguments

`recipe` is the handoff. It turns the findings into the arguments the next two stages take — a
`steps` list for `data_engineer.tools.data_prep.prepare`, and the column roles, steps and options for
`feature_engineer.tools.feature_prep.feature_prep_caller` — with a reason attached to every one, so a
model chaining the stages is choosing from an argued plan rather than guessing.

The analyzers read values as they find them: a number written as text is still counted as a number
here, and reported as text-typed so that `coerce_types` gets recommended rather than silently needed.
That is what makes this safe to run first, on a dataset nothing has prepared yet.

Ranking features against a target for *selection* is not this module's job — `target_summary` says how
balanced a target is and flags a feature that predicts it suspiciously well, and
`feature_engineer.tools.feature_selection` does the ranking.

Register the callers in `agentic_configurations.yaml` the way the other tools are registered:

    - name: "data_analyzer_tool"
      description: "Analyse a dataset before touching it: describe every column, missing values,
        duplicates, outliers, distributions, correlations, target balance, time span and grain,
        with a recommended cleaning and feature-prep plan."
      caller: "data_engineer.tools.analyzer.analyzer_caller"
      args:
        - name: "data_source"
          type: "str"
          description: "Path to the dataset to analyse."
          required: false
        - name: "analyzers"
          type: "list"
          description: "Which analyzers to run. Call data_analyzer_steps_tool for the catalogue."
          required: false
        - name: "target"
          type: "str"
          description: "The column being predicted, for the target analysis."
          required: false
        - name: "id_columns"
          type: "list"
          description: "Entity columns. Inferred from the names and the values when omitted."
          required: false
        - name: "date_columns"
          type: "list"
          description: "Timestamp columns. Inferred from the values when omitted."
          required: false
        - name: "detail"
          type: "str"
          description: "How much to return: 'summary' or 'full'."
          enum: ["summary", "full"]
          required: false
        - name: "output_path"
          type: "str"
          description: "Where to write the full report as JSON."
          required: false
"""

import inspect
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Sequence

from data_engineer.tools.dataset import BOOLEANS, MISSING_VALUES, Dataset, read_dataset
from uilts.logger import logger


MISSING = frozenset(value.lower() for value in MISSING_VALUES)

# Two levels written any of these ways are a flag, not a measurement: describing the mean of a 0/1
# column is fine, but treating it as continuous — binning it, clipping its outliers — is not.
BOOLEAN_TOKENS = frozenset({*BOOLEANS, "0", "1", "yes", "no", "y", "n", "t", "f"})

# A column named like a key is one, whatever its values look like: `customer_id` holding integers is
# still an entity, and averaging it would be meaningless.
ID_NAME = re.compile(r"(^|_)(id|key|code|uuid|guid)$")

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%b %d %Y",
    "%d %b %Y",
)

DAY = 86400.0  # seconds, the unit every date-derived number below is reported in

TYPED_SHARE = 0.8  # how much of a column has to parse as a number or a date for that to be its kind
TEXT_RATIO = 0.9  # distinct values per row above which a string column stops being a category
TEXT_LENGTH = 20  # ... and how long those strings have to be to count as free text rather than keys

SKEWED = 1.0  # |skew| a numeric column needs before a log transform is worth recommending
HEAVY = 2.0  # ... and before bucketing it is worth recommending as well
REDUNDANT = 0.95  # |r| above which two columns are carrying the same information
LEAKAGE = 0.98  # ... and above which a feature is predicting the target too well to be honest

LEVELS = ("info", "warning", "risk")  # what a finding can be, in the order a summary should read them


@dataclass
class Finding:
    """One thing worth acting on, and how much it matters.

    Args:
        level: "info" for something to know, "warning" for something to fix, "risk" for something
            that will produce a wrong model if it is left alone.
        message: What was found, in a sentence, naming the columns it is about.
    """

    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


@dataclass
class Analysis:
    """What one analyzer saw.

    Args:
        name: The analyzer that produced it.
        summary: One sentence, for the report a later step summarises.
        details: The numbers behind that sentence, JSON-safe.
        findings: What is worth acting on, if anything.
    """

    name: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        """Render as JSON-safe structures; `detail=False` keeps the sentence and the findings only."""
        rendered: dict[str, Any] = {"analysis": self.name, "summary": self.summary}
        if detail:
            rendered["details"] = self.details
        rendered["findings"] = [finding.to_dict() for finding in self.findings]
        return rendered


@dataclass
class Report:
    """Every analysis run over one dataset, collected in the order they ran.

    The dataset itself is kept so that `recipe` can go back to it for anything the analyses did not
    record; it is not part of what `to_dict` renders.

    Args:
        data: The dataset that was analysed.
        analyses: One entry per analyzer.
    """

    data: Dataset
    analyses: list[Analysis] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.analyses)

    def __contains__(self, name: str) -> bool:
        return any(analysis.name == name for analysis in self.analyses)

    def __getitem__(self, name: str) -> Analysis:
        for analysis in self.analyses:
            if analysis.name == name:
                return analysis

        raise KeyError(f"no analysis named {name!r}; this report has {[a.name for a in self.analyses]}.")

    @property
    def source(self) -> str:
        """Where the rows came from."""
        return self.data.source

    @property
    def findings(self) -> list[dict[str, str]]:
        """Every finding, across every analysis, worst first."""
        found = [
            {"analysis": analysis.name, **finding.to_dict()}
            for analysis in self.analyses
            for finding in analysis.findings
        ]
        return sorted(found, key=lambda finding: -LEVELS.index(finding["level"]))

    def add(self, analysis: Analysis) -> Analysis:
        """Collect one analysis, replacing an earlier run of the same analyzer."""
        self.analyses = [held for held in self.analyses if held.name != analysis.name]
        self.analyses.append(analysis)
        return analysis

    def details(self, name: str, default: Any = None) -> Any:
        """One analysis's numbers, or `default` when that analyzer did not run."""
        return self[name].details if name in self else default

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        """Render as JSON-safe structures — what a tool hands back to an agent."""
        return {
            "source": self.source,
            "rows": len(self.data.rows),
            "columns": len(self.data.columns),
            "analyses": [analysis.to_dict(detail) for analysis in self.analyses],
            "findings": self.findings,
        }

    def summary(self) -> str:
        """Render as the text a summarising step reads: a line per analysis, then the findings."""
        shape = f"{len(self.data.rows):,} rows, {len(self.data.columns)} columns"
        lines = [f"Analysis of {self.source} — {shape}.", ""]
        lines += [f"{analysis.name}: {analysis.summary}" for analysis in self.analyses]

        found = self.findings
        if found:
            lines += ["", "Findings"]
            lines += [f"- {item['level']} [{item['analysis']}] {item['message']}" for item in found]

        return "\n".join(lines)


# ---- what a column is --------------------------------------------------------------------------------


def column_kinds(source: Any) -> dict[str, str]:
    """Say what each column holds: "numeric", "date", "boolean", "categorical", "text", "constant",
    "empty".

    Read from the values rather than from a declared type, because a source that carries no types —
    a CSV, a JSON export — is the normal case here. A column is numeric or date when `TYPED_SHARE` of
    the values it does have parse that way, so a handful of dirty values does not change what a
    column is; they are counted separately by `describe`.
    """
    data = _dataset(source)
    return {column: _kind(_present(data, column)) for column in data.columns}


def column_roles(
    source: Any,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Sort the columns into the parts feature prep asks for: who, when, what was measured.

    The names are the ones `feature_engineer.tools.feature_prep` takes, so this output is the column
    half of a feature prep call. A column named as an id, or holding a distinct value in every row,
    is an entity; dates come from the values; measurements are the numeric columns that are neither;
    everything else with few enough levels is categorical.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        id_columns: Entity columns, when the caller knows them. Inferred otherwise.
        date_columns: Timestamp columns, when the caller knows them. Inferred otherwise.
        target: The column being predicted, kept out of every other role.

    Returns:
        `id_columns`, `date_columns`, `aggregation_columns`, `categorical_columns`, `text_columns`,
        `constant_columns`, `empty_columns` and the `target` it was told about.
    """
    data = _dataset(source)
    kinds = column_kinds(data)
    rows = len(data.rows)

    dates = _columns(data, date_columns) or [
        column for column, kind in kinds.items() if kind == "date" and column != target
    ]
    ids = _columns(data, id_columns) or [
        column
        for column, kind in kinds.items()
        if column not in dates
        and column != target
        and kind not in ("empty", "constant")
        and (
            ID_NAME.search(column.strip().lower())
            or (rows > 1 and kind in ("categorical", "text") and _distinct(data, column) == rows)
        )
    ]

    spoken_for = {*ids, *dates, *([target] if target else [])}
    return {
        "id_columns": ids,
        "date_columns": dates,
        "aggregation_columns": [
            column
            for column, kind in kinds.items()
            if kind in ("numeric", "boolean") and column not in spoken_for
        ],
        "categorical_columns": [
            column
            for column, kind in kinds.items()
            if kind in ("categorical", "boolean") and column not in spoken_for
        ],
        "text_columns": [
            column for column, kind in kinds.items() if kind == "text" and column not in spoken_for
        ],
        "constant_columns": [column for column, kind in kinds.items() if kind == "constant"],
        "empty_columns": [column for column, kind in kinds.items() if kind == "empty"],
        "target": target,
    }


# ---- analyzers ---------------------------------------------------------------------------------------


def describe(source: Any, columns: Sequence[str] | None = None, top: int = 5) -> Analysis:
    """Describe every column: how much of it there is, and what it looks like.

    Numeric columns get the five-number summary with a mean, a standard deviation and a skew;
    dates get their range and how many distinct days they cover; anything else gets its levels and
    the most common of them. Every column, whatever its kind, gets its count, its missing values and
    how many distinct values it holds — which is what the rest of the analyzers key off.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        columns: Restrict the description to these columns.
        top: How many of the most common values to list for a categorical column.

    Returns:
        A profile per column, and the kind each was read as.
    """
    data = _dataset(source)
    wanted = _columns(data, columns) or list(data.columns)
    kinds = column_kinds(data)
    profiles = {column: _profile(data, column, kinds[column], top) for column in wanted}

    counts: dict[str, int] = {}
    for column in wanted:
        counts[kinds[column]] = counts.get(kinds[column], 0) + 1

    findings: list[Finding] = []
    as_text = [
        column
        for column in wanted
        if kinds[column] in ("numeric", "date", "boolean") and profiles[column].get("stored_as")
    ]
    if as_text:
        findings.append(
            Finding(
                "warning",
                f"{_listed(as_text)}: numbers or dates written as text — coerce_types before "
                f"anything measures them.",
            )
        )

    empty = [column for column in wanted if kinds[column] == "empty"]
    constant = [column for column in wanted if kinds[column] == "constant"]
    if empty:
        findings.append(Finding("warning", f"{_listed(empty)}: no values at all."))
    if constant:
        findings.append(
            Finding("info", f"{_listed(constant)}: no variation, so nothing to explain with.")
        )

    shape = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    return Analysis(
        name="describe",
        summary=f"{len(wanted)} columns over {len(data.rows):,} rows: {shape}.",
        details={"row_count": len(data.rows), "kinds": kinds, "columns": profiles},
        findings=findings,
    )


def missing_values(source: Any, max_missing_rate: float = 0.5, patterns: int = 3) -> Analysis:
    """Count the holes: per column, per row, and in the combinations they come in.

    The combinations are the part a per-column rate hides. Columns that go missing together are
    usually one fact that was never captured rather than several independent ones, which is the
    difference between imputing three columns and dropping one source of rows.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        max_missing_rate: The share of missing values above which a column is called out.
        patterns: How many of the most common missing-together combinations to report.

    Returns:
        The rate per column, how many rows are complete, and the common missing patterns.
    """
    data = _dataset(source)
    rows = len(data.rows)

    rates = {
        column: {"missing": missing, "rate": _rounded(_rate(missing, rows))}
        for column in data.columns
        if (missing := sum(1 for row in data.rows if _is_missing(row.get(column))))
    }
    complete = sum(1 for row in data.rows if not any(_is_missing(value) for value in row.values()))

    combinations: dict[tuple[str, ...], int] = {}
    for row in data.rows:
        holes = tuple(column for column in data.columns if _is_missing(row.get(column)))
        if holes:
            combinations[holes] = combinations.get(holes, 0) + 1

    common = sorted(combinations.items(), key=lambda item: -item[1])[: max(int(patterns), 0)]
    over = {column: held["rate"] for column, held in rates.items() if held["rate"] > max_missing_rate}

    findings: list[Finding] = []
    if over:
        findings.append(
            Finding(
                "warning",
                f"{_listed(sorted(over))}: more than {max_missing_rate:.0%} missing, so a feature "
                f"built on that carries the imputation rather than the data.",
            )
        )
    partial = sorted(
        (column for column, held in rates.items() if column not in over),
        key=lambda column: -rates[column]["rate"],
    )
    if partial:
        worst = partial[0]
        findings.append(
            Finding(
                "info",
                f"{len(partial)} column(s) hold some missing values, worst {worst} at "
                f"{rates[worst]['rate']:.1%}; fill_missing decides them, missing_indicators keeps "
                f"the fact that they were missing.",
            )
        )

    return Analysis(
        name="missing_values",
        summary=(
            f"{len(rates)} of {len(data.columns)} columns hold missing values; "
            f"{_rate(complete, rows):.1%} of rows are complete."
        ),
        details={
            "columns": rates,
            "complete_rows": complete,
            "complete_rate": _rounded(_rate(complete, rows)),
            "max_missing_rate": max_missing_rate,
            "patterns": [
                {"columns": list(columns), "rows": count, "share": _rounded(_rate(count, rows))}
                for columns, count in common
            ],
        },
        findings=findings,
    )


def duplicates(
    source: Any,
    subset: Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    examples: int = 2,
) -> Analysis:
    """Count rows that repeat, and keys that repeat, which are two different problems.

    A repeated row is an export artefact and is dropped. A repeated key is the shape of the data:
    several rows per customer means an event log, which is aggregated per entity rather than
    de-duplicated. `grain` says which of the two this is; this says how often it happens.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        subset: Columns that identify a duplicate row. Defaults to every column.
        keys: Columns that should hold one row each. Defaults to the inferred id columns.
        examples: How many duplicated keys to show.

    Returns:
        The duplicate row count and rate, and the repeat counts for each key column.
    """
    data = _dataset(source)
    rows = len(data.rows)
    wanted = _columns(data, subset) or list(data.columns)

    seen: set[tuple[Any, ...]] = set()
    repeated = 0
    for row in data.rows:
        key = tuple(_hashable(row.get(column)) for column in wanted)
        if key in seen:
            repeated += 1
        seen.add(key)

    candidates = _columns(data, keys) or column_roles(data)["id_columns"]
    key_report: dict[str, Any] = {}
    for column in candidates:
        counts: dict[Any, int] = {}
        for row in data.rows:
            value = _hashable(row.get(column))
            counts[value] = counts.get(value, 0) + 1

        duplicated = {value: count for value, count in counts.items() if count > 1}
        worst = sorted(duplicated.items(), key=lambda item: -item[1])[: max(int(examples), 0)]
        key_report[column] = {
            "distinct": len(counts),
            "duplicated": len(duplicated),
            "max_repeats": max(counts.values(), default=0),
            "examples": [{"value": _plain(value), "rows": count} for value, count in worst],
        }

    findings: list[Finding] = []
    if repeated:
        findings.append(
            Finding(
                "warning",
                f"{repeated:,} row(s) ({_rate(repeated, rows):.1%}) repeat across {len(wanted)} "
                f"columns; drop_duplicates removes them before anything counts or aggregates.",
            )
        )
    for column, held in key_report.items():
        if held["duplicated"]:
            findings.append(
                Finding(
                    "info",
                    f"{column} repeats: {held['duplicated']:,} value(s) appear more than once, up to "
                    f"{held['max_repeats']} times. The data is not one row per {column}.",
                )
            )

    return Analysis(
        name="duplicates",
        summary=(
            f"{repeated:,} duplicate row(s) ({_rate(repeated, rows):.1%}); "
            f"{len(key_report)} key column(s) checked."
        ),
        details={
            "duplicate_rows": repeated,
            "duplicate_rate": _rounded(_rate(repeated, rows)),
            "unique_rows": len(seen),
            "subset": wanted,
            "keys": key_report,
        },
        findings=findings,
    )


def outliers(
    source: Any,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
) -> Analysis:
    """Find the values far enough out to set a scale on their own, and say how many there are.

    Nothing is clipped here — `data_prep.clip_outliers` does that — because how many outliers a
    column has decides whether clipping is the right answer: a handful is a data error, and a tenth
    of the column is a distribution that wants a transform instead.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        columns: Numeric columns to check. Defaults to every numeric column.
        method: "iqr" for Tukey's fences, or "zscore" for `factor` standard deviations either side.
        factor: How far beyond the spread a value has to be to count.

    Returns:
        The bounds, the counts either side, and the share of each column that falls outside them.
    """
    if method not in ("iqr", "zscore"):
        raise ValueError(f"unknown method {method!r}; expected 'iqr' or 'zscore'.")

    data = _dataset(source)
    kinds = column_kinds(data)
    wanted = [
        column
        for column in (_columns(data, columns) or list(data.columns))
        if kinds[column] == "numeric"
    ]

    report: dict[str, Any] = {}
    skipped: list[str] = []
    for column in wanted:
        numbers = _numbers(_present(data, column))
        limits = None if len(set(numbers)) <= 2 else _bounds(numbers, method, factor)
        if not limits:
            skipped.append(column)  # a flag, or a column with no spread: fences say nothing about it
            continue

        low, high = limits
        below = sum(1 for number in numbers if number < low)
        above = sum(1 for number in numbers if number > high)
        report[column] = {
            "lower": _rounded(low),
            "upper": _rounded(high),
            "below": below,
            "above": above,
            "outliers": below + above,
            "rate": _rounded(_rate(below + above, len(numbers))),
            "min": _rounded(min(numbers)),
            "max": _rounded(max(numbers)),
        }

    worst = sorted(report, key=lambda column: -report[column]["rate"])
    heavy = [column for column in worst if report[column]["rate"] > 0.05]

    findings: list[Finding] = []
    if heavy:
        findings.append(
            Finding(
                "warning",
                f"{_listed(heavy)}: more than 5% outlying values — a transform (log, bucket) fits "
                f"that better than clipping, which would move a twentieth of the column.",
            )
        )
    light = [column for column in worst if column not in heavy and report[column]["outliers"]]
    if light:
        findings.append(
            Finding("info", f"{_listed(light)}: a few extreme values; clip_outliers holds them in.")
        )

    total = sum(held["outliers"] for held in report.values())
    return Analysis(
        name="outliers",
        summary=(
            f"{total:,} outlying value(s) across {len(report)} numeric column(s), "
            f"by {method} at {factor}."
        ),
        details={"method": method, "factor": factor, "columns": report, "skipped": skipped},
        findings=findings,
    )


def distributions(source: Any, columns: Sequence[str] | None = None, bins: int = 10) -> Analysis:
    """Look at the shape of each numeric column, and say what would straighten it out.

    Skew is the number that decides a transform: a column with a long right tail — spend, tenure,
    counts of anything — carries most of its rows in a fraction of its range, and a model reads the
    tail rather than the body until it is logged or bucketed.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        columns: Numeric columns to look at. Defaults to every numeric column.
        bins: How many equal-width bins the histogram gets.

    Returns:
        Per column: its skew and kurtosis, whether it reads as binary, discrete or continuous, a
        histogram, and the transform worth trying on it.
    """
    data = _dataset(source)
    kinds = column_kinds(data)
    wanted = [
        column
        for column in (_columns(data, columns) or list(data.columns))
        if kinds[column] in ("numeric", "boolean")
    ]

    report: dict[str, Any] = {}
    for column in wanted:
        numbers = _numbers(_present(data, column))
        if not numbers:
            continue

        skew = _skewness(numbers)
        shape = _shape(numbers)
        report[column] = {
            "shape": shape,
            "skew": _rounded(skew),
            "kurtosis": _rounded(_kurtosis(numbers)),
            "zeros": sum(1 for number in numbers if number == 0),
            "negatives": sum(1 for number in numbers if number < 0),
            "distinct": len(set(numbers)),
            "suggested_transform": _suggested_transform(numbers, skew, shape),
            "histogram": _histogram(numbers, bins),
        }

    # A flag is lopsided whenever its two levels are uneven, and no transform changes that, so the
    # skewed list holds only the columns something could actually be done about.
    skewed = [
        column
        for column in report
        if report[column]["shape"] != "binary"
        and report[column]["skew"] is not None
        and abs(report[column]["skew"]) >= SKEWED
    ]
    findings: list[Finding] = []
    if skewed:
        findings.append(
            Finding(
                "info",
                f"{_listed(skewed)}: skewed — feature prep's math_transformations (log1p) or "
                f"binning gives a model the body of the column rather than its tail.",
            )
        )

    return Analysis(
        name="distributions",
        summary=(
            f"{len(report)} numeric column(s) profiled; {len(skewed)} skewed beyond {SKEWED:g}."
        ),
        details={"columns": report, "skewed": skewed},
        findings=findings,
    )


def correlations(
    source: Any,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    top: int = 10,
    redundant_above: float = REDUNDANT,
) -> Analysis:
    """Measure how much the numeric columns already say about each other.

    Two columns that correlate above `redundant_above` are one column twice: keeping both doubles the
    weight the pair carries in a linear model and splits it in a tree. Knowing which pairs those are
    before features are built stops the same redundancy being multiplied into every derived feature.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        columns: Numeric columns to correlate. Defaults to every numeric column.
        method: "pearson" for a straight-line relationship, "spearman" for a monotonic one.
        top: How many of the strongest pairs to report.
        redundant_above: |r| at which a pair is called redundant.

    Returns:
        The strongest pairs, the redundant ones, and the columns there was nothing to measure on.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown method {method!r}; expected 'pearson' or 'spearman'.")

    data = _dataset(source)
    kinds = column_kinds(data)
    wanted = [
        column
        for column in (_columns(data, columns) or list(data.columns))
        if kinds[column] in ("numeric", "boolean")
    ]

    values = {column: [_as_number(row.get(column)) for row in data.rows] for column in wanted}
    pairs: list[dict[str, Any]] = []
    for position, left in enumerate(wanted):
        for right in wanted[position + 1 :]:
            coefficient = _correlation(values[left], values[right], method)
            if coefficient is not None:
                pairs.append({"columns": [left, right], "r": _rounded(coefficient)})

    pairs.sort(key=lambda pair: -abs(pair["r"]))
    redundant = [pair for pair in pairs if abs(pair["r"]) >= redundant_above]

    findings: list[Finding] = []
    if redundant:
        named = _listed([f"{pair['columns'][0]}/{pair['columns'][1]}" for pair in redundant])
        findings.append(
            Finding(
                "warning",
                f"{named}: the same information twice (|r| ≥ {redundant_above:g}) — keep one of "
                f"each pair, or the redundancy is multiplied through every feature built on them.",
            )
        )

    return Analysis(
        name="correlations",
        summary=(
            f"{len(pairs)} numeric pair(s) measured by {method}; "
            f"{len(redundant)} above |r| {redundant_above:g}."
        ),
        details={
            "method": method,
            "columns": wanted,
            "strongest": pairs[: max(int(top), 0)],
            "redundant": redundant,
        },
        findings=findings,
    )


def target_summary(source: Any, target: str | None = None, top: int = 10) -> Analysis:
    """Look at the column being predicted: how balanced it is, and what already predicts it.

    Association is measured, not ranked for selection — `feature_engineer.tools.feature_selection`
    does the ranking. What this is for is the two things that make a model wrong rather than weak: a
    target too rare to learn, and a feature that predicts it almost perfectly, which is nearly always
    a column that was computed from the answer.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        target: The column being predicted.
        top: How many associated columns to report.

    Returns:
        The target's kind and balance, the rows missing it, and the columns most associated with it.
    """
    data = _dataset(source)
    if not target:
        return Analysis("target_summary", "skipped: no target column was given.")
    if target not in data.columns:
        raise ValueError(f"no column named {target!r}; it has {data.columns}.")

    values = data.column(target)
    present = [value for value in values if not _is_missing(value)]
    missing = len(values) - len(present)
    distinct = {_hashable(value) for value in present}
    kind = "binary" if len(distinct) == 2 else _kind(present)

    findings: list[Finding] = []
    if missing:
        findings.append(
            Finding(
                "risk",
                f"{missing:,} row(s) have no {target}; they cannot be trained on and cannot be "
                f"imputed either — drop_missing_rows on the target.",
            )
        )

    details: dict[str, Any] = {"target": target, "kind": kind, "missing": missing}
    numbers: list[float | None]
    if kind == "binary":
        positive = _positive_class(present)
        counts = _counts(present)
        rate = _rate(counts.get(_hashable(positive), 0), len(present))
        details.update(
            {
                "positive_class": _plain(positive),
                "positive_rate": _rounded(rate),
                "classes": [
                    {"value": _plain(value), "rows": count, "share": _rounded(_rate(count, len(present)))}
                    for value, count in sorted(counts.items(), key=lambda item: -item[1])
                ],
            }
        )
        numbers = [
            None if _is_missing(value) else float(_hashable(value) == _hashable(positive))
            for value in values
        ]
        if rate and rate < 0.02:
            findings.append(
                Finding(
                    "risk",
                    f"{target} is positive in only {rate:.2%} of rows; at that rate this is an "
                    f"anomaly-detection problem, not a balanced classification one.",
                )
            )
        elif rate < 0.2:
            findings.append(
                Finding(
                    "warning",
                    f"{target} is positive in {rate:.1%} of rows; the classifier needs class weights "
                    f"or a tuned threshold, and accuracy will not measure it.",
                )
            )
    elif kind in ("numeric", "boolean"):
        details.update(_numeric_profile(_numbers(present)))
        numbers = [_as_number(value) for value in values]
    else:
        counts = _counts(present)
        details["classes"] = [
            {"value": _plain(value), "rows": count, "share": _rounded(_rate(count, len(present)))}
            for value, count in sorted(counts.items(), key=lambda item: -item[1])[:20]
        ]
        details["distinct"] = len(distinct)
        numbers = []
        findings.append(
            Finding(
                "info",
                f"{target} holds {len(distinct)} levels, so this is a multiclass problem; the "
                f"per-column association below is not computed for it.",
            )
        )

    associations = _associations(data, target, numbers) if numbers else []
    details["associations"] = associations[: max(int(top), 0)]

    leaking = [item for item in associations if item["association"] >= LEAKAGE]
    if leaking:
        findings.append(
            Finding(
                "risk",
                f"{_listed([item['column'] for item in leaking])}: an almost perfect predictor of "
                f"{target}. A column computed from the answer leaks it — check before building on it.",
            )
        )

    balance = details.get("positive_rate")
    return Analysis(
        name="target_summary",
        summary=(
            f"{target} is {kind}"
            + (f", positive in {balance:.1%} of rows" if balance is not None else "")
            + f"; {missing:,} row(s) missing it."
        ),
        details=details,
        findings=findings,
    )


def timeline(
    source: Any,
    date_columns: Sequence[str] | None = None,
    reference_date: Any = None,
    periods: int = 12,
) -> Analysis:
    """Look at the time in the data: what it spans, how it is spread, and whether it runs past now.

    The span is what windows are chosen against — trailing 7/30/90-day features need at least that
    much history — and a date beyond the reference is either a bad parse or a field that was filled
    in from the future, which is a leak in a time-ordered problem.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        date_columns: Timestamp columns. Inferred from the values when omitted.
        reference_date: The "now" recency is measured against. Defaults to today.
        periods: How many recent months to report row counts for.

    Returns:
        Per date column: its range, its span in days, the days it covers, its largest gap, how many
        rows are dated after the reference, and the rows per recent month.
    """
    data = _dataset(source)
    wanted = _columns(data, date_columns) or column_roles(data)["date_columns"]
    reference = _as_datetime(reference_date) or datetime.now()

    report: dict[str, Any] = {}
    findings: list[Finding] = []
    for column in wanted:
        moments = sorted(
            moment for moment in (_as_datetime(value) for value in data.column(column)) if moment
        )
        if not moments:
            continue

        gaps = [
            (later - earlier).total_seconds() / DAY
            for earlier, later in zip(moments, moments[1:])
        ]
        ahead = sum(1 for moment in moments if moment > reference)
        months: dict[str, int] = {}
        for moment in moments:
            months[f"{moment.year:04d}-{moment.month:02d}"] = (
                months.get(f"{moment.year:04d}-{moment.month:02d}", 0) + 1
            )

        report[column] = {
            "min": moments[0].isoformat(),
            "max": moments[-1].isoformat(),
            "span_days": _rounded((moments[-1] - moments[0]).total_seconds() / DAY),
            "distinct_days": len({moment.date() for moment in moments}),
            "unparsed": sum(
                1 for value in data.column(column) if not _is_missing(value) and not _as_datetime(value)
            ),
            "largest_gap_days": _rounded(max(gaps, default=0.0)),
            "after_reference": ahead,
            "rows_per_month": dict(sorted(months.items())[-max(int(periods), 0) :]),
        }

        if ahead:
            findings.append(
                Finding(
                    "risk",
                    f"{ahead:,} row(s) have {column} after {reference.date().isoformat()}; either the "
                    f"parse is wrong or the field was filled in after the fact, which leaks the future.",
                )
            )
        if report[column]["unparsed"]:
            findings.append(
                Finding(
                    "warning",
                    f"{report[column]['unparsed']:,} value(s) in {column} do not parse as dates; "
                    f"coerce_types will read them as missing.",
                )
            )

    widest = max((held["span_days"] for held in report.values()), default=0.0)
    if report and widest < 30:
        findings.append(
            Finding(
                "info",
                f"the data spans {widest:.0f} day(s), which is too short for 30- or 90-day trailing "
                f"windows; keep rolling windows inside the span.",
            )
        )

    return Analysis(
        name="timeline",
        summary=(
            f"{len(report)} date column(s); the widest spans {widest:,.0f} days."
            if report
            else "no date columns were found."
        ),
        details={"reference_date": reference.isoformat(), "columns": report},
        findings=findings,
    )


def grain(
    source: Any,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
) -> Analysis:
    """Say what one row is: one entity, or one event in an entity's history.

    This is the single decision feature prep needs most. One row per entity means the features are
    the columns transformed; several rows per entity over time means the features are aggregates,
    lags, windows and gaps, and the output table is a different shape from the input.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        id_columns: Entity columns. Inferred when omitted.
        date_columns: Timestamp columns. Inferred when omitted.

    Returns:
        The entity count, the rows per entity, the span each entity covers, and the verdict.
    """
    data = _dataset(source)
    roles = column_roles(data, id_columns, date_columns)
    ids, dates = roles["id_columns"], roles["date_columns"]
    rows = len(data.rows)

    if not ids:
        return Analysis(
            name="grain",
            summary="no id columns were found, so every row stands alone.",
            details={"rows": rows, "id_columns": [], "verdict": "unknown"},
            findings=[
                Finding(
                    "warning",
                    "no entity column was found or given; per-entity features have nothing to group "
                    "by, and every aggregate would be taken over the whole table.",
                )
            ],
        )

    groups: dict[tuple[Any, ...], list[int]] = {}
    for position, row in enumerate(data.rows):
        groups.setdefault(tuple(_hashable(row.get(column)) for column in ids), []).append(position)

    sizes = sorted(len(members) for members in groups.values())
    repeated = sum(1 for size in sizes if size > 1)
    verdict = "event log" if repeated else "one row per entity"

    spans: list[float] = []
    if dates:
        moments = [_as_datetime(row.get(dates[0])) for row in data.rows]
        for members in groups.values():
            dated = [moments[position] for position in members if moments[position]]
            if len(dated) > 1:
                spans.append((max(dated) - min(dated)).total_seconds() / DAY)

    findings = [
        Finding(
            "info",
            f"the data is an event log: {repeated:,} of {len(groups):,} entities have more than one "
            f"row, up to {sizes[-1]:,}. Per-entity features (group_aggregates, event_gaps, lags, "
            f"rolling_windows) are what this shape is for."
            if repeated
            else f"the data is already one row per {', '.join(ids)}; per-entity aggregation would "
            f"add nothing, so feature prep should work the measurements themselves.",
        )
    ]
    if dates and not repeated:
        findings.append(
            Finding(
                "info",
                "there is a date column but one row per entity, so time features are calendar parts "
                "and recency rather than windows over history.",
            )
        )

    return Analysis(
        name="grain",
        summary=(
            f"{verdict}: {len(groups):,} entities over {rows:,} rows "
            f"({statistics.fmean(sizes):.2f} rows each on average)."
        ),
        details={
            "rows": rows,
            "id_columns": ids,
            "date_columns": dates,
            "entities": len(groups),
            "verdict": verdict,
            "rows_per_entity": {
                "mean": _rounded(statistics.fmean(sizes)),
                "median": _rounded(statistics.median(sizes)),
                "max": sizes[-1],
                "entities_with_repeats": repeated,
            },
            "entity_span_days": {
                "mean": _rounded(statistics.fmean(spans)),
                "median": _rounded(statistics.median(spans)),
                "max": _rounded(max(spans)),
            }
            if spans
            else None,
        },
        findings=findings,
    )


# ---- running them ------------------------------------------------------------------------------------


ANALYZERS: dict[str, Callable[..., Analysis]] = {
    "describe": describe,
    "missing_values": missing_values,
    "duplicates": duplicates,
    "outliers": outliers,
    "distributions": distributions,
    "correlations": correlations,
    "target_summary": target_summary,
    "timeline": timeline,
    "grain": grain,
}

# The order a report reads best in: what the columns are, then what is wrong with them, then what
# they say about each other and about the target, then what shape the table is.
DEFAULT_ANALYZERS = (
    "describe",
    "missing_values",
    "duplicates",
    "outliers",
    "distributions",
    "correlations",
    "target_summary",
    "timeline",
    "grain",
)


def analyze(source: Any, analyzers: Sequence[str] | None = None, **options: Any) -> Report:
    """Run the analyzers over one dataset and collect what they found into a `Report`.

    An analyzer that raises is recorded as having failed and the rest still run, because a report
    with eight analyses and one failure is worth more than no report at all.

    Args:
        source: The dataset, a path to one, a `Prepared`, or the rows themselves.
        analyzers: Which to run, in order. Defaults to `DEFAULT_ANALYZERS`, minus the target
            analysis when no target is given.
        **options: Passed to whichever analyzers take them — `target`, `id_columns`, `date_columns`,
            `columns`, `method`, `factor`, `max_missing_rate`, `bins`, `top`, `reference_date`.
            An analyzer that does not take one is not given it.

    Returns:
        The report, in the order the analyzers ran.
    """
    data = _dataset(source)
    wanted = (
        list(analyzers)
        if analyzers is not None
        else [
            name
            for name in DEFAULT_ANALYZERS
            if name != "target_summary" or options.get("target")
        ]
    )

    unknown = [name for name in wanted if name not in ANALYZERS]
    if unknown:
        raise ValueError(f"unknown analyzer(s) {unknown}; expected some of {sorted(ANALYZERS)}.")

    report = Report(data=data)
    for name in wanted:
        analyzer = ANALYZERS[name]
        try:
            report.add(analyzer(data, **_accepted(analyzer, options)))
        except Exception as error:  # one bad argument shouldn't cost the whole report
            logger.exception(f"Analyzer {name} failed.")
            report.add(
                Analysis(
                    name=name,
                    summary=f"failed: {error}",
                    findings=[Finding("warning", f"{name} did not run: {error}")],
                )
            )

    logger.info(f"Analysed {data.source}: {len(report)} analyses, {len(report.findings)} findings.")
    return report


def write_report(report: Report, path: str | Path, include_recipe: bool = True) -> str:
    """Write a report out as JSON, and return where it was written.

    This is how a run is collected: each stage writes its report, and the summarising step at the end
    reads the files rather than holding every analysis in context as it goes.

    Args:
        report: The report to write.
        path: Where to write it. Parent directories are created.
        include_recipe: Also write the cleaning and feature-prep plan derived from it.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = report.to_dict()
    document["summary_text"] = report.summary()
    if include_recipe:
        document["recipe"] = recipe(report)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, default=str, indent=2)

    logger.info(f"Wrote the analysis of {report.source} to {path}.")
    return str(path)


# ---- what to cook ------------------------------------------------------------------------------------


def recipe(source: Report | Any, target: str | None = None) -> dict[str, Any]:
    """Turn a report into the arguments the next two stages take, with a reason for each one.

    Two blocks come out. `cleaning` is a `steps` list for `data_engineer.tools.data_prep.prepare`:
    what has to be true of the data before a feature is built on it. `feature_prep` is the column
    roles, steps and options for `feature_engineer.tools.feature_prep.feature_prep_caller`: what is
    worth building given the shape the data turned out to have.

    Nothing is run — this is the plan, and a model reading `reasons` can drop a step it disagrees
    with rather than accepting the whole list.

    Args:
        source: A `Report`, or a dataset to analyse first.
        target: The column being predicted, when analysing here rather than passing a report.

    Returns:
        `columns`, `cleaning`, `feature_prep`, `reasons` and the `warnings` behind them.
    """
    report = source if isinstance(source, Report) else analyze(source, target=target)
    data = report.data

    described = report.details("describe", {})
    kinds: dict[str, str] = described.get("kinds") or column_kinds(data)
    profiles: dict[str, Any] = described.get("columns", {})
    target = target or (report.details("target_summary", {}) or {}).get("target")

    roles = column_roles(data, target=target)
    grained = report.details("grain", {}) or {}
    dated = report.details("timeline", {}) or {}
    missed = report.details("missing_values", {}) or {}
    # Only the measurements: a transform of the target changes the question, and a skewed id is an id.
    skewed = [
        column
        for column in ((report.details("distributions", {}) or {}).get("skewed") or [])
        if column in roles["aggregation_columns"]
    ]
    outlying = report.details("outliers", {}) or {}

    reasons: list[dict[str, str]] = []
    cleaning: list[Any] = []

    def step(plan: list[Any], entry: Any, why: str) -> None:
        plan.append(entry)
        reasons.append({"step": entry if isinstance(entry, str) else entry["step"], "why": why})

    untidy = [column for column in data.columns if column != _tidy(column)]
    if untidy:
        step(cleaning, "normalize_columns", f"{_listed(untidy)} are not usable as feature names.")

    if any(kinds[column] in ("categorical", "text") for column in kinds):
        step(cleaning, "strip_whitespace", "text columns split into levels when padding differs.")

    as_text = [column for column in profiles if profiles[column].get("stored_as")]
    if as_text:
        step(cleaning, "coerce_types", f"{_listed(as_text)} hold numbers or dates written as text.")

    if (report.details("duplicates", {}) or {}).get("duplicate_rows"):
        step(cleaning, "drop_duplicates", "rows repeat, and every aggregate would double-count them.")

    if roles["constant_columns"] or roles["empty_columns"]:
        step(
            cleaning,
            "drop_constant_columns",
            f"{_listed(roles['constant_columns'] + roles['empty_columns'])} never vary.",
        )

    heavy = [
        column
        for column, held in (missed.get("columns") or {}).items()
        if held["rate"] > 0.5 and column != target
    ]
    if heavy:
        step(
            cleaning,
            {"step": "drop_missing_columns", "max_missing_rate": 0.5},
            f"{_listed(heavy)} are more imputation than data.",
        )

    if target and (report.details("target_summary", {}) or {}).get("missing"):
        step(
            cleaning,
            {"step": "drop_missing_rows", "columns": [target]},
            f"rows without {target} cannot be trained on, and imputing a target invents the answer.",
        )

    if missed.get("columns"):
        step(
            cleaning,
            {"step": "fill_missing", "strategy": "auto"},
            "holes remain after the drops above; auto takes a median or a mode per column.",
        )

    clipping = [
        column
        for column, held in (outlying.get("columns") or {}).items()
        if 0 < held["rate"] <= 0.05 and column not in skewed
    ]
    if clipping:
        step(
            cleaning,
            {"step": "clip_outliers", "columns": clipping, "method": "iqr"},
            f"{_listed(clipping)} hold a few extreme values that would set the scale.",
        )

    # ---- features
    event_log = grained.get("verdict") == "event log"
    steps: list[str] = []
    options: dict[str, Any] = {
        "categorical_columns": roles["categorical_columns"],
    }

    if missed.get("columns"):
        step(steps, "missing_indicators", "whether a field was filled in is itself predictive.")

    if roles["date_columns"]:
        step(steps, "date_parts", "calendar position — month, weekday, weekend — is signal.")
        step(steps, "recency", "how long ago is what a date means to a model.")
        if len(roles["date_columns"]) > 1:
            step(steps, "date_differences", "the gap between two dates is tenure or lead time.")

    if event_log:
        step(steps, "group_aggregates", "several rows per entity: what it normally does is context.")
        step(steps, "group_relative", "one row means little except against the entity's own history.")
        if roles["date_columns"]:
            step(steps, "event_gaps", "how the entity's events are spaced changes before it churns.")
            step(steps, "cumulative", "everything the entity has done up to this row, without leaking.")
            windows = _windows(dated, grained)
            if windows:
                step(steps, "rolling_windows", f"history is long enough for {windows}-day windows.")
                options["rolling_windows"] = {"windows": windows}
            step(steps, "trend", "the direction a measurement is moving carries more than its level.")

    if roles["categorical_columns"]:
        step(steps, "frequency_encoding", "one column per category, at any cardinality, no leakage.")
        wide = [
            column
            for column in roles["categorical_columns"]
            if (profiles.get(column, {}).get("distinct") or 0) > 10
        ]
        if wide:
            step(steps, "rare_category_grouping", f"{_listed(wide)}: levels rare enough to overfit.")
            step(steps, "ordinal_encoding", "too many levels for one column each.")
        else:
            step(steps, "one_hot_encoding", "few enough levels for a column each.")
            options["one_hot_encoding"] = {"max_categories": 10}

    if skewed:
        step(steps, "math_transformations", f"{_listed(skewed)}: skewed; log1p straightens them.")
        options["math_transformations"] = {"operations": ["log1p"]}
        if any(abs(profiles.get(column, {}).get("skew") or 0) >= HEAVY for column in skewed):
            step(steps, "binning", "a heavy tail reads better as ordered bands than as a number.")

    if len(roles["aggregation_columns"]) >= 3:
        step(steps, "row_statistics", "several comparable measurements per row summarise usefully.")
    if 2 <= len(roles["aggregation_columns"]) <= 6:
        step(steps, "ratios", "a ratio carries what neither of its two columns carries alone.")

    warnings = [
        finding for finding in report.findings if finding["level"] in ("warning", "risk")
    ]
    return {
        "columns": {
            "id_columns": roles["id_columns"],
            "date_columns": roles["date_columns"],
            "aggregation_columns": roles["aggregation_columns"],
            "categorical_columns": roles["categorical_columns"],
            "target": target,
        },
        "cleaning": {"caller": "data_engineer.tools.data_prep.prepare", "steps": cleaning},
        "feature_prep": {
            "caller": "feature_engineer.tools.feature_prep.feature_prep_caller",
            "level": "entity" if event_log else "row",
            "id_columns": roles["id_columns"],
            "date_columns": roles["date_columns"],
            "aggregation_columns": roles["aggregation_columns"],
            "steps": steps,
            "options": options,
        },
        "reasons": reasons,
        "warnings": warnings,
    }


# ---- agent-facing callers ----------------------------------------------------------------------------


def analyzer_caller(
    data_source: Any = None,
    analyzers: Any = None,
    target: str | None = None,
    id_columns: Any = None,
    date_columns: Any = None,
    columns: Any = None,
    limit: int | None = None,
    detail: str = "summary",
    options: dict[str, Any] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Analyse a dataset and report what is in it, with a plan for what to do about it.

    Args:
        data_source: Path to the dataset to analyse, or the rows themselves.
        analyzers: Which analyzers to run, in order. Defaults to all of them.
        target: The column being predicted.
        id_columns: Entity columns. Inferred from the names and the values when omitted.
        date_columns: Timestamp columns. Inferred from the values when omitted.
        columns: Restrict the column-by-column analyses to these.
        limit: Read at most this many rows, for a first look at a large source.
        detail: "summary" returns each analysis's sentence and findings; "full" returns the numbers
            behind them as well. The full report is written to `output_path` either way.
        options: Further analyzer arguments: `max_missing_rate`, `method`, `factor`, `bins`, `top`,
            `reference_date`.
        output_path: Where to write the full report as JSON, for the summarising step to collect.

    Returns:
        The analyses, every finding worst-first, the recipe for the next two stages, and the report
        as text. On failure, `{"status": "error", "message": ...}`.
    """
    try:
        data = (
            read_dataset(data_source, limit=limit)
            if isinstance(data_source, (str, Path))
            else _dataset(data_source)
        )
        if not data.rows:
            return {"status": "error", "message": "the dataset is empty"}

        report = analyze(
            data,
            analyzers=_names(analyzers) or None,
            target=target,
            id_columns=_names(id_columns) or None,
            date_columns=_names(date_columns) or None,
            columns=_names(columns) or None,
            **(options or {}),
        )

        written = write_report(report, output_path) if output_path else None
        return {
            "status": "ok",
            **report.to_dict(detail=detail == "full"),
            "recipe": recipe(report, target),
            "summary_text": report.summary(),
            "report_path": written,
            "truncated": data.truncated,
        }
    except Exception as error:
        logger.exception("analyzer_caller failed.")
        return {"status": "error", "message": str(error)}


def list_analyzers_caller(analyzer: str | None = None) -> dict[str, Any]:
    """List every analyzer, what it looks at, and what it takes.

    Args:
        analyzer: Restrict the listing to one analyzer.

    Returns:
        One entry per analyzer: its name, its purpose, and its arguments with their defaults.
    """
    names = [analyzer] if analyzer else list(ANALYZERS)
    unknown = [name for name in names if name not in ANALYZERS]
    if unknown:
        return {
            "status": "error",
            "message": f"no such analyzer {unknown[0]!r}",
            "analyzers": list(ANALYZERS),
        }

    return {
        "status": "ok",
        "default_analyzers": list(DEFAULT_ANALYZERS),
        "analyzers": [_described(name) for name in names],
    }


def _described(name: str) -> dict[str, Any]:
    """One analyzer's catalogue entry: its first line, and the arguments it declares."""
    analyzer = ANALYZERS[name]
    documentation = (analyzer.__doc__ or "").strip().split("\n\n")[0]
    return {
        "analyzer": name,
        "description": " ".join(documentation.split()),
        "arguments": [
            {
                "name": parameter.name,
                "default": None if parameter.default is inspect.Parameter.empty else parameter.default,
            }
            for parameter in inspect.signature(analyzer).parameters.values()
            if parameter.name != "source"
        ],
    }


# ---- internals: the data ------------------------------------------------------------------------------


def _dataset(source: Any) -> Dataset:
    """Accept whatever the caller has: a `Dataset`, a path, a `Prepared`, or the rows themselves."""
    if isinstance(source, Dataset):
        return source
    if isinstance(source, (str, Path)):
        return read_dataset(source)

    prepared = getattr(source, "data", None)  # a `Prepared` from data_prep, part-way through cleaning
    if isinstance(prepared, Dataset):
        return prepared

    if isinstance(source, (list, tuple)):
        rows = [dict(row) for row in source]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        return Dataset(
            rows=[{column: row.get(column) for column in columns} for row in rows],
            columns=columns,
            source="rows",
            format="rows",
        )

    raise TypeError(f"cannot analyse a {type(source).__name__}; pass a Dataset, a path, or rows.")


def _columns(data: Dataset, columns: Sequence[str] | None) -> list[str]:
    """Resolve a `columns` argument, refusing one the dataset does not have."""
    if not columns:
        return []

    unknown = [column for column in columns if column not in data.columns]
    if unknown:
        raise ValueError(f"no column(s) {unknown}; it has {data.columns}.")

    return list(columns)


def _present(data: Dataset, column: str) -> list[Any]:
    """One column's values, without the holes."""
    return [value for value in data.column(column) if not _is_missing(value)]


def _distinct(data: Dataset, column: str) -> int:
    """How many different values a column holds, ignoring the holes."""
    return len({_hashable(value) for value in _present(data, column)})


def _accepted(analyzer: Callable[..., Any], options: dict[str, Any]) -> dict[str, Any]:
    """Keep only the options this analyzer takes, so one call can carry every analyzer's arguments."""
    parameters = inspect.signature(analyzer).parameters
    return {
        name: value
        for name, value in options.items()
        if name in parameters and name != "source" and value is not None
    }


def _names(value: Any) -> list[str]:
    """Accept a list, a single name, or a comma-separated string — models send all three."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value)]


# ---- internals: values --------------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Whether a value is a hole. Text is included: this runs before anything has read the strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING

    return False


def _as_number(value: Any) -> float | None:
    """Read a value as a number, including one written as text, or None when it is not one."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return None  # a date is orderable, not measurable; `_as_datetime` reads those

    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    """Read a value as a naive datetime: ISO-8601, then the formats an export writes, then an epoch."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None  # a bare number is a measurement here; an epoch has to arrive as text

    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    for pattern in DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue

    if text.isdigit() and len(text) in (10, 13):  # epoch seconds or milliseconds
        try:
            seconds = int(text) / (1000.0 if len(text) == 13 else 1.0)
            return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None

    return None


def _numbers(values: Sequence[Any]) -> list[float]:
    """The values that are numbers, as numbers."""
    return [number for number in (_as_number(value) for value in values) if number is not None]


def _hashable(value: Any) -> Any:
    """A value usable as a key — a JSON source carries lists and dicts, which are not."""
    if isinstance(value, (list, dict, set)):
        return repr(value)

    return value


def _counts(values: Sequence[Any]) -> dict[Any, int]:
    """How often each value occurs."""
    counts: dict[Any, int] = {}
    for value in values:
        key = _hashable(value)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _rate(part: int, whole: int) -> float:
    """A share, with an empty whole reading as zero rather than raising."""
    return part / whole if whole else 0.0


def _rounded(value: Any, digits: int = 6) -> Any:
    """Round a float for reporting; anything else is passed through."""
    if isinstance(value, float) and math.isfinite(value):
        return round(value, digits)

    return value


def _plain(value: Any) -> Any:
    """Render a value into something `json.dumps` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return _rounded(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()

    return str(value)


def _listed(names: Sequence[str], most: int = 5) -> str:
    """Name the columns a finding is about, without letting a wide table fill the sentence."""
    shown = list(names)[:most]
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f" and {rest} other(s)" if rest > 0 else "")


def _tidy(name: str) -> str:
    """What `data_prep.normalize_columns` would rename a column to — used to tell whether it would."""
    characters = [character if character.isalnum() else "_" for character in str(name).strip()]
    tidied = "".join(characters).strip("_")
    while "__" in tidied:
        tidied = tidied.replace("__", "_")

    return tidied.lower()


# ---- internals: what a column is ----------------------------------------------------------------------


def _kind(present: Sequence[Any]) -> str:
    """Read one column's kind off the values it holds. See `column_kinds` for the rules."""
    if not present:
        return "empty"

    distinct = {_hashable(value) for value in present}
    if len(distinct) == 1:
        return "constant"
    if len(distinct) <= 2 and {str(value).strip().lower() for value in present} <= BOOLEAN_TOKENS:
        return "boolean"

    if sum(1 for value in present if _as_number(value) is not None) >= TYPED_SHARE * len(present):
        return "numeric"
    if sum(1 for value in present if _as_datetime(value) is not None) >= TYPED_SHARE * len(present):
        return "date"

    length = statistics.fmean(len(str(value)) for value in present)
    ratio = len(distinct) / len(present)
    return "text" if ratio > TEXT_RATIO and length > TEXT_LENGTH else "categorical"


def _profile(data: Dataset, column: str, kind: str, top: int) -> dict[str, Any]:
    """Describe one column: what every kind shares, then what its own kind adds."""
    values = data.column(column)
    present = [value for value in values if not _is_missing(value)]
    missing = len(values) - len(present)

    profile: dict[str, Any] = {
        "kind": kind,
        "count": len(present),
        "missing": missing,
        "missing_rate": _rounded(_rate(missing, len(values))),
        "distinct": len({_hashable(value) for value in present}),
    }

    texts = sum(1 for value in present if isinstance(value, str))
    if texts and kind in ("numeric", "date", "boolean"):
        # The values are what they claim to be, but written as strings: worth reporting, because
        # every statistic below had to read them, and the next step should not have to.
        profile["stored_as"] = "text" if texts == len(present) else "mixed"

    if kind == "constant":
        profile["value"] = _plain(present[0])
    elif kind in ("numeric", "boolean"):
        profile.update(_numeric_profile(_numbers(present)))
        unreadable = len(present) - len(_numbers(present))
        if unreadable:
            profile["unreadable"] = unreadable
    elif kind == "date":
        profile.update(_date_profile(present))
    else:
        profile.update(_category_profile(present, top))

    return profile


def _numeric_profile(numbers: Sequence[float]) -> dict[str, Any]:
    """The five-number summary, plus what says whether the column is worth transforming."""
    if not numbers:
        return {}

    ordered = sorted(numbers)
    quartiles = (
        statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    )
    return {
        "mean": _rounded(statistics.fmean(numbers)),
        "std": _rounded(statistics.stdev(numbers) if len(numbers) > 1 else 0.0),
        "min": _rounded(ordered[0]),
        "p25": _rounded(quartiles[0]),
        "median": _rounded(quartiles[1]),
        "p75": _rounded(quartiles[2]),
        "max": _rounded(ordered[-1]),
        "sum": _rounded(sum(numbers)),
        "zeros": sum(1 for number in numbers if number == 0),
        "negatives": sum(1 for number in numbers if number < 0),
        "skew": _rounded(_skewness(numbers)),
    }


def _date_profile(present: Sequence[Any]) -> dict[str, Any]:
    """When a date column starts, when it ends, and how much of the span it actually covers."""
    moments = sorted(moment for moment in (_as_datetime(value) for value in present) if moment)
    if not moments:
        return {}

    return {
        "min": moments[0].isoformat(),
        "max": moments[-1].isoformat(),
        "span_days": _rounded((moments[-1] - moments[0]).total_seconds() / DAY),
        "distinct_days": len({moment.date() for moment in moments}),
        "unreadable": len(present) - len(moments),
    }


def _category_profile(present: Sequence[Any], top: int) -> dict[str, Any]:
    """The levels a column holds, and how much of it the most common ones account for."""
    counts = _counts(present)
    common = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[: max(int(top), 0)]
    return {
        "top": [
            {"value": _plain(value), "count": count, "share": _rounded(_rate(count, len(present)))}
            for value, count in common
        ],
        "longest": max((len(str(value)) for value in present), default=0),
    }


# ---- internals: statistics ----------------------------------------------------------------------------


def _skewness(numbers: Sequence[float]) -> float | None:
    """How lopsided a column is: positive for a long right tail, and undefined without a spread."""
    if len(numbers) < 3:
        return None

    middle, spread = statistics.fmean(numbers), statistics.pstdev(numbers)
    if not spread:
        return None

    return sum(((number - middle) / spread) ** 3 for number in numbers) / len(numbers)


def _kurtosis(numbers: Sequence[float]) -> float | None:
    """Excess kurtosis: how much of the column lives in its tails, against a normal distribution."""
    if len(numbers) < 4:
        return None

    middle, spread = statistics.fmean(numbers), statistics.pstdev(numbers)
    if not spread:
        return None

    return sum(((number - middle) / spread) ** 4 for number in numbers) / len(numbers) - 3


def _bounds(numbers: Sequence[float], method: str, factor: float) -> tuple[float, float] | None:
    """The range a column is expected to sit in, or None when there is too little spread to say."""
    if len(numbers) < 2:
        return None

    if method == "iqr":
        first, _, third = statistics.quantiles(numbers, n=4, method="inclusive")
        spread = third - first
        return (first - factor * spread, third + factor * spread) if spread else None

    spread = statistics.pstdev(numbers)
    if not spread:
        return None

    middle = statistics.fmean(numbers)
    return middle - factor * spread, middle + factor * spread


def _shape(numbers: Sequence[float]) -> str:
    """Whether a numeric column is a flag, a count, or a measurement — they want different features."""
    distinct = len(set(numbers))
    if distinct <= 2:
        return "binary"
    if distinct <= 20 and all(float(number).is_integer() for number in numbers):
        return "discrete"

    return "continuous"


def _suggested_transform(numbers: Sequence[float], skew: float | None, shape: str) -> str | None:
    """What would straighten this column out, in the names feature prep knows the transforms by."""
    if shape == "binary" or skew is None or abs(skew) < SKEWED:
        return None
    if abs(skew) >= HEAVY:
        return "binning"

    return "log1p" if min(numbers) >= -1 else "zscore"


def _histogram(numbers: Sequence[float], bins: int) -> dict[str, Any]:
    """Equal-width bins, so the shape of the column survives being reported as a dozen numbers."""
    count = max(int(bins), 1)
    low, high = min(numbers), max(numbers)
    if low == high:
        return {"edges": [_rounded(low), _rounded(high)], "counts": [len(numbers)]}

    width = (high - low) / count
    counts = [0] * count
    for number in numbers:
        counts[min(int((number - low) / width), count - 1)] += 1

    return {"edges": [_rounded(low + width * step) for step in range(count + 1)], "counts": counts}


def _ranks(values: Sequence[float]) -> list[float]:
    """Rank positions, ties sharing their average rank — what Spearman correlates instead of values."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0
        for tied in range(position, end + 1):
            ranks[order[tied]] = shared
        position = end + 1

    return ranks


def _correlation(
    first: Sequence[float | None],
    second: Sequence[float | None],
    method: str = "pearson",
) -> float | None:
    """Correlate two columns over the rows where both are present, or None when it is undefined."""
    pairs = [
        (left, right)
        for left, right in zip(first, second)
        if left is not None and right is not None
    ]
    if len(pairs) < 3:
        return None

    left_values = [left for left, _ in pairs]
    right_values = [right for _, right in pairs]
    if method == "spearman":
        left_values, right_values = _ranks(left_values), _ranks(right_values)

    left_mean, right_mean = statistics.fmean(left_values), statistics.fmean(right_values)
    left_spread = sum((value - left_mean) ** 2 for value in left_values)
    right_spread = sum((value - right_mean) ** 2 for value in right_values)
    if not left_spread or not right_spread:
        return None

    covariance = sum(
        (left - left_mean) * (right - right_mean) for left, right in zip(left_values, right_values)
    )
    return covariance / math.sqrt(left_spread * right_spread)


def _correlation_ratio(groups: dict[Any, list[float]]) -> float | None:
    """How much of a column's variance the groups explain — a correlation for a categorical column.

    The measure a category needs: r asks whether a straight line fits, which a category has no
    ordering for, while this asks whether knowing the level tells you the value.
    """
    values = [value for group in groups.values() for value in group]
    if len(values) < 3 or len(groups) < 2:
        return None

    middle = statistics.fmean(values)
    total = sum((value - middle) ** 2 for value in values)
    if not total:
        return None

    between = sum(
        len(group) * (statistics.fmean(group) - middle) ** 2 for group in groups.values() if group
    )
    return math.sqrt(min(between / total, 1.0))


def _positive_class(present: Sequence[Any]) -> Any:
    """Which of two levels is the one being predicted: the one that says so, else the rarer one."""
    counts = _counts(present)
    for value in counts:
        if str(value).strip().lower() in ("1", "true", "yes", "y", "t"):
            return value

    return min(counts, key=lambda value: counts[value])


def _associations(data: Dataset, target: str, numbers: Sequence[float | None]) -> list[dict[str, Any]]:
    """How strongly each column relates to the target, strongest first.

    A numeric column gets a correlation; a categorical one gets a correlation ratio. Both land on
    0..1, so one list can be read in one order — which is all this is for. Ranking features for
    selection is `feature_engineer.tools.feature_selection`'s job, and takes more than this.

    Ids, dates and free text are left out, and so is any category with nearly a level per row: a
    grouping that splits the rows finely enough explains the target perfectly whatever the target
    is, which would report every key column as a leak.
    """
    kinds = column_kinds(data)
    roles = column_roles(data, target=target)
    ignored = {*roles["id_columns"], *roles["date_columns"], *roles["text_columns"]}
    found: list[dict[str, Any]] = []

    for column in data.columns:
        if column == target or column in ignored or kinds[column] in ("empty", "constant"):
            continue
        if kinds[column] == "categorical" and _distinct(data, column) > 0.5 * len(data.rows):
            continue

        if kinds[column] in ("numeric", "boolean"):
            coefficient = _correlation([_as_number(row.get(column)) for row in data.rows], numbers)
            measure = "correlation"
            strength = None if coefficient is None else abs(coefficient)
        else:
            groups: dict[Any, list[float]] = {}
            for row, value in zip(data.rows, numbers):
                if value is not None and not _is_missing(row.get(column)):
                    groups.setdefault(_hashable(row.get(column)), []).append(value)
            coefficient = _correlation_ratio(groups)
            measure = "correlation_ratio"
            strength = coefficient

        if strength is not None:
            found.append(
                {
                    "column": column,
                    "measure": measure,
                    "value": _rounded(coefficient),
                    "association": _rounded(strength),
                }
            )

    return sorted(found, key=lambda item: -item["association"])


def _windows(dated: dict[str, Any], grained: dict[str, Any]) -> list[int]:
    """Trailing window widths the history is actually long enough to fill.

    A window rolls over one entity's own events, so it is the typical entity's span that has to
    support it, not the dataset's — a table covering two years of customers who each appear for a
    fortnight has no 90-day window in it. A window wider than half the history it rolls over is a
    column that equals the entity's total, which `group_aggregates` has already built.
    """
    spans = [held.get("span_days") or 0 for held in (dated.get("columns") or {}).values()]
    entity = (grained.get("entity_span_days") or {}).get("median") or 0
    span = entity or max(spans, default=0)

    return [width for width in (7, 30, 90) if width <= span / 2]

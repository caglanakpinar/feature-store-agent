"""Prep steps that put a dataset in the state feature engineering needs it in.

What a feature builder needs is not what a source hands over: columns named consistently, values in the
types they represent rather than the strings they were written as, one row per thing, missing values
decided on rather than left as holes, outliers that will not swamp a scale, and skewed columns
transformed into something a model can use. Each of those is a step below, and each reports what it
changed — the count of rows dropped, the value a column was filled with, the bounds it was clipped to —
because a prep step that quietly halves a dataset is worse than one that refuses to run.

    from data_engineer.tools.dataset import read_dataset
    from data_engineer.tools.data_prep import fill_missing, prepare, transform

    data = read_dataset("customers.csv")

    prepared = prepare(data, ["normalize_columns", "strip_whitespace", "coerce_types",
                              "drop_duplicates", {"step": "fill_missing", "strategy": "auto"}])
    prepared = transform(prepared, [{"column": "charges", "operation": "log"}])

    prepared.data      # the Dataset, ready for features
    prepared.steps     # what every step did, in order

Every step takes a `Dataset` or the `Prepared` a previous step returned, and returns a `Prepared`, so
steps chain in any order and a single step is just as usable on its own. `prepare` runs a list of them
by name, which is the form an agent's `preprocessing_steps` argument arrives in.

Nothing here mutates what it is given: a step builds new rows and leaves the dataset it read alone, so
a caller still holds the state before the step ran.
"""

import csv
import inspect
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from data_engineer.tools.dataset import BOOLEANS, DECIMAL, INTEGER, MISSING_VALUES, Dataset
from uilts.logger import logger


# What `coerce_types` casts to when a schema names a type. "auto" is not here: it is the inference
# below, which reads a column as a number only when every value in it is one. The readers are wrapped
# in lambdas rather than named directly because they are defined further down the module.
CASTS: dict[str, Callable[[Any], Any]] = {
    "int": lambda value: int(float(value)),  # via float, so "3.0" and 3.0 both land on 3
    "float": float,
    "str": str,
    "bool": lambda value: _as_bool(value),
    "date": lambda value: _as_datetime(value).date(),
    "datetime": lambda value: _as_datetime(value),
}

# Date formats tried after ISO-8601, in the order they are worth trying: the unambiguous one first,
# then day-first, which is what a European export writes.
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S")

MISSING = frozenset(value.lower() for value in MISSING_VALUES)


@dataclass
class Prepared:
    """A dataset part-way through preparation, and the record of what got it there.

    Args:
        data: The dataset as the last step left it.
        steps: One entry per step run, in order: what it was, what it changed, and the shape it left.
    """

    data: Dataset
    steps: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def columns(self) -> list[str]:
        """The columns as they now stand."""
        return self.data.columns

    def to_dict(self, limit: int | None = 10) -> dict[str, Any]:
        """Render as JSON-safe structures — what a tool hands back to an agent.

        The step log is the point of it: an agent that is told a dataset is clean can check what
        cleaning it, and how much of it went, rather than taking the claim.
        """
        return {**self.data.to_dict(limit=limit), "steps": self.steps}


# ---- naming and types --------------------------------------------------------------------------------


def normalize_columns(
    source: Dataset | Prepared,
    lowercase: bool = True,
    replacements: dict[str, str] | None = None,
) -> Prepared:
    """Rewrite column names into the form a feature name can be built from.

    Spaces and punctuation become underscores and the case is levelled, so `"Monthly Charges ($)"`
    becomes `monthly_charges`. A rename that collides with another column is suffixed rather than
    allowed to overwrite it.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        lowercase: Level the case as well as the punctuation.
        replacements: Explicit renames, applied before the rest, e.g. `{"cust_id": "customer_id"}`.

    Returns:
        The dataset with its columns renamed, and the renames it made.
    """
    prepared = _prepared(source)
    data = prepared.data

    renames: dict[str, str] = {}
    taken: set[str] = set()
    for column in data.columns:
        name = (replacements or {}).get(column, column)
        name = _normalized(name, lowercase) or "column"
        while name in taken:  # two source columns can normalise onto the same name
            name = _numbered(name)
        taken.add(name)
        renames[column] = name

    changed = {old: new for old, new in renames.items() if old != new}
    rows = [{renames[column]: row.get(column) for column in data.columns} for row in data.rows]

    return _step(
        prepared,
        "normalize_columns",
        _with(data, rows, [renames[column] for column in data.columns]),
        {"renamed": changed},
    )


def strip_whitespace(
    source: Dataset | Prepared,
    columns: Sequence[str] | None = None,
    empty_as_missing: bool = True,
) -> Prepared:
    """Trim the whitespace around text values, and read what is left of an empty one as missing.

    Padding is invisible and splits a category in two — `"Yes"` and `"Yes "` encode as two levels —
    which is why this runs before anything that groups, counts or encodes.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        columns: Columns to trim. Defaults to every column holding text.
        empty_as_missing: Read a value that trims to nothing (or to "NA", "null", …) as missing.

    Returns:
        The dataset with its text trimmed, and how many values that changed.
    """
    prepared = _prepared(source)
    data = prepared.data
    wanted = _columns(data, columns)

    stripped = 0
    emptied = 0
    rows = []
    for row in data.rows:
        new = dict(row)
        for column in wanted:
            value = new[column]
            if not isinstance(value, str):
                continue

            text = value.strip()
            if empty_as_missing and text.lower() in MISSING:
                new[column] = None
                emptied += 1
            elif text != value:
                new[column] = text
                stripped += 1
        rows.append(new)

    return _step(
        prepared,
        "strip_whitespace",
        _with(data, rows),
        {"stripped": stripped, "emptied": emptied, "columns": wanted},
    )


def coerce_types(
    source: Dataset | Prepared,
    schema: dict[str, str] | None = None,
    columns: Sequence[str] | None = None,
) -> Prepared:
    """Read values as the types they represent, so a number stops being a string.

    With a `schema` each named column is cast to the type it names — "int", "float", "str", "bool",
    "date", "datetime" — and a value that will not cast becomes missing rather than failing the step,
    with the count reported. Without one, every text column is inspected and converted only when every
    value in it converts, so a column of ids with one letter in it stays text instead of being half
    destroyed.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        schema: Column -> type name. Inferred per column when not given.
        columns: Columns to inspect when inferring. Defaults to every column.

    Returns:
        The dataset with its values cast, the type each column was read as, and what would not cast.
    """
    prepared = _prepared(source)
    data = prepared.data

    if schema:
        unknown = [name for name in schema.values() if name not in CASTS]
        if unknown:
            raise ValueError(f"unknown type(s) {unknown}; expected one of {sorted(CASTS)}.")
        missing_columns = [column for column in schema if column not in data.columns]
        if missing_columns:
            raise ValueError(f"no column(s) {missing_columns}; it has {data.columns}.")
        plan = dict(schema)
    else:
        plan = {column: kind for column in _columns(data, columns) if (kind := _inferred(data, column))}

    rows = [dict(row) for row in data.rows]
    failures: dict[str, int] = {}
    for column, kind in plan.items():
        cast = CASTS[kind]
        for row in rows:
            value = row[column]
            if _is_missing(value):
                row[column] = None
                continue
            try:
                row[column] = cast(value)
            except (TypeError, ValueError):
                row[column] = None
                failures[column] = failures.get(column, 0) + 1

    for column, count in failures.items():
        logger.warning(f"coerce_types: {count} value(s) in {column!r} would not cast; read as missing.")

    return _step(prepared, "coerce_types", _with(data, rows), {"cast": plan, "uncastable": failures})


# ---- rows --------------------------------------------------------------------------------------------


def drop_duplicates(
    source: Dataset | Prepared,
    subset: Sequence[str] | None = None,
    keep: str = "first",
) -> Prepared:
    """Drop rows that repeat, keeping one of each.

    `subset` is what makes this a data-engineering step rather than a cosmetic one: duplicates are
    usually duplicated keys — the same customer exported twice with a different timestamp — not rows
    identical in every column, and a feature built over the second copy double-counts it.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        subset: Columns that identify a row. Defaults to every column.
        keep: Which copy survives, "first" or "last".

    Returns:
        The dataset with one row per key, and how many went.
    """
    if keep not in ("first", "last"):
        raise ValueError(f"keep is {keep!r}; expected 'first' or 'last'.")

    prepared = _prepared(source)
    data = prepared.data
    wanted = _columns(data, subset)

    kept: dict[Any, dict[str, Any]] = {}
    for row in data.rows:
        key = tuple(_hashable(row.get(column)) for column in wanted)
        if key not in kept or keep == "last":
            kept[key] = row

    dropped = len(data.rows) - len(kept)
    if dropped:
        logger.info(f"drop_duplicates: dropped {dropped} row(s) duplicated on {wanted}.")

    return _step(
        prepared,
        "drop_duplicates",
        _with(data, list(kept.values())),
        {"dropped": dropped, "subset": wanted, "keep": keep},
    )


def drop_missing_rows(
    source: Dataset | Prepared,
    columns: Sequence[str] | None = None,
    how: str = "any",
) -> Prepared:
    """Drop rows missing values in the columns that must have them.

    The column that must have one is usually the target: a row with no label cannot be trained on and
    cannot be imputed either, since imputing a target invents the answer.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        columns: Columns that must be present. Defaults to every column.
        how: Drop a row when "any" of those is missing, or only when "all" of them are.

    Returns:
        The dataset without those rows, and how many went.
    """
    if how not in ("any", "all"):
        raise ValueError(f"how is {how!r}; expected 'any' or 'all'.")

    prepared = _prepared(source)
    data = prepared.data
    wanted = _columns(data, columns)

    test = any if how == "any" else all
    rows = [row for row in data.rows if not test(_is_missing(row.get(column)) for column in wanted)]
    dropped = len(data.rows) - len(rows)
    if dropped:
        logger.info(f"drop_missing_rows: dropped {dropped} row(s) missing {how} of {wanted}.")

    return _step(
        prepared,
        "drop_missing_rows",
        _with(data, rows),
        {"dropped": dropped, "columns": wanted, "how": how},
    )


def fill_missing(
    source: Dataset | Prepared,
    strategy: str = "auto",
    columns: Sequence[str] | None = None,
    value: Any = None,
) -> Prepared:
    """Fill the holes, and say what each one was filled with.

    "auto" is the strategy worth defaulting to: a numeric column takes its median, which a skewed
    distribution does not drag the way it drags a mean, and anything else takes its most common value.
    A column with nothing to compute from — every value missing — is left alone and reported, since
    filling it would invent a constant that then looks like a feature.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        strategy: "auto", "median", "mean", "mode", "zero", or "constant" with `value`.
        columns: Columns to fill. Defaults to every column holding a missing value.
        value: What to fill with, for the "constant" strategy.

    Returns:
        The dataset with its holes filled, and the value each column was filled with.
    """
    strategies = ("auto", "median", "mean", "mode", "zero", "constant")
    if strategy not in strategies:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(strategies)}.")

    prepared = _prepared(source)
    data = prepared.data
    wanted = _columns(data, columns)

    filled: dict[str, Any] = {}
    counts: dict[str, int] = {}
    skipped: list[str] = []
    rows = [dict(row) for row in data.rows]
    for column in wanted:
        present = [row[column] for row in rows if not _is_missing(row[column])]
        holes = len(rows) - len(present)
        if not holes:
            continue

        replacement = _replacement(column, present, strategy, value)
        if replacement is None and strategy != "constant":
            skipped.append(column)  # nothing to compute a fill from
            continue

        for row in rows:
            if _is_missing(row[column]):
                row[column] = replacement
        filled[column] = replacement
        counts[column] = holes

    if skipped:
        logger.warning(f"fill_missing: {skipped} had no value to fill from; left as they were.")

    return _step(
        prepared,
        "fill_missing",
        _with(data, rows),
        {"strategy": strategy, "filled_with": filled, "filled": counts, "skipped": skipped},
    )


# ---- columns -----------------------------------------------------------------------------------------


def drop_missing_columns(source: Dataset | Prepared, max_missing_rate: float = 0.5) -> Prepared:
    """Drop columns missing more of their values than a feature can be built from.

    A column that is mostly holes carries the imputation more than it carries the data, and a model
    learns the imputation.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        max_missing_rate: The share of missing values a column is allowed, between 0 and 1.

    Returns:
        The dataset without those columns, and the rate each dropped one was missing at.
    """
    if not 0 <= max_missing_rate <= 1:
        raise ValueError(f"max_missing_rate is {max_missing_rate}; expected a share between 0 and 1.")

    prepared = _prepared(source)
    data = prepared.data

    rates = {column: _missing_rate(data, column) for column in data.columns}
    dropped = {column: rate for column, rate in rates.items() if rate > max_missing_rate}
    kept = [column for column in data.columns if column not in dropped]
    if dropped:
        logger.info(f"drop_missing_columns: dropped {sorted(dropped)} over {max_missing_rate:.0%}.")

    rows = [{column: row[column] for column in kept} for row in data.rows]
    return _step(
        prepared,
        "drop_missing_columns",
        _with(data, rows, kept),
        {"dropped": {column: round(rate, 4) for column, rate in dropped.items()},
         "max_missing_rate": max_missing_rate},
    )


def drop_constant_columns(source: Dataset | Prepared, drop_empty: bool = True) -> Prepared:
    """Drop columns that never vary, because a feature that never varies explains nothing.

    Varying is judged on the values a column does have: one holding a single value and otherwise
    missing counts as constant, since imputing it — which is the next step here — would fill the holes
    with that same value and leave it wholly constant anyway.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        drop_empty: Also drop a column with no values at all, not only one with a single value.

    Returns:
        The dataset without those columns, and the value each dropped one was stuck on.
    """
    prepared = _prepared(source)
    data = prepared.data

    dropped: dict[str, Any] = {}
    for column in data.columns:
        distinct = {_hashable(row[column]) for row in data.rows if not _is_missing(row[column])}
        if len(distinct) == 1:
            dropped[column] = next(value for value in data.column(column) if not _is_missing(value))
        elif not distinct and drop_empty:
            dropped[column] = None

    kept = [column for column in data.columns if column not in dropped]
    if dropped:
        logger.info(f"drop_constant_columns: dropped {sorted(dropped)}, which never vary.")

    rows = [{column: row[column] for column in kept} for row in data.rows]
    return _step(prepared, "drop_constant_columns", _with(data, rows, kept), {"dropped": dropped})


def clip_outliers(
    source: Dataset | Prepared,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
    action: str = "clip",
) -> Prepared:
    """Pull extreme numeric values back to a bound, or drop the rows holding them.

    Clipping is the default because dropping loses the rest of the row with it, and an outlier in one
    column is rarely a reason to throw away every other feature of that customer. The bounds are
    reported so a later step can tell a clipped value from a real one.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        columns: Numeric columns to treat. Defaults to every numeric column.
        method: "iqr" for Tukey's fences (`factor` times the interquartile range beyond the quartiles),
            or "zscore" for `factor` standard deviations either side of the mean.
        factor: How far beyond the spread a value has to be to count as an outlier.
        action: "clip" a value back to the bound, or "drop" the row it is in.

    Returns:
        The dataset with its extremes handled, and the bounds each column was held to.
    """
    if method not in ("iqr", "zscore"):
        raise ValueError(f"unknown method {method!r}; expected 'iqr' or 'zscore'.")
    if action not in ("clip", "drop"):
        raise ValueError(f"unknown action {action!r}; expected 'clip' or 'drop'.")

    prepared = _prepared(source)
    data = prepared.data
    wanted = [column for column in _columns(data, columns) if _is_numeric(data, column)]
    if columns:
        not_numeric = [column for column in columns if column not in wanted]
        if not_numeric:
            logger.warning(f"clip_outliers: {not_numeric} are not numeric; skipping them.")

    bounds: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for column in wanted:
        values = [value for value in data.column(column) if _is_number(value)]
        # A flag or an indicator is numeric but not continuous, and fences say nothing about two
        # levels: on a mostly-zero 0/1 column the quartiles sit at zero, and clipping to them would
        # erase every 1 in it. Two distinct values or fewer are left exactly as they are.
        limits = None if len({*values}) <= 2 else _bounds(values, method, factor)
        if limits:
            bounds[column] = {"lower": limits[0], "upper": limits[1]}
        else:
            skipped.append(column)

    clipped: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in data.rows:
        new = dict(row)
        outlying = False
        for column, limit in bounds.items():
            value = new[column]
            if not _is_number(value) or limit["lower"] <= value <= limit["upper"]:
                continue

            outlying = True
            clipped[column] = clipped.get(column, 0) + 1
            new[column] = min(max(value, limit["lower"]), limit["upper"])
        if action == "drop" and outlying:
            continue
        rows.append(row if action == "drop" else new)

    return _step(
        prepared,
        "clip_outliers",
        _with(data, rows),
        {"method": method, "factor": factor, "action": action, "bounds": bounds, "skipped": skipped,
         "outliers": clipped, "dropped": len(data.rows) - len(rows) if action == "drop" else 0},
    )


# ---- derived columns ---------------------------------------------------------------------------------


def transform(
    source: Dataset | Prepared,
    transformations: Sequence[dict[str, Any]],
) -> Prepared:
    """Add transformed copies of numeric columns, as new columns beside the originals.

    The original is kept on purpose: a transformation is a hypothesis about the shape of a
    relationship — log for something that acts multiplicatively, buckets for something that acts in
    bands — and feature selection is what decides which of the two survives, not this step.

    A value the transformation is not defined for (the log of zero, the square root of a negative) is
    written as missing rather than as an error, and counted.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        transformations: One mapping per column to transform: `{"column": ..., "operation": ...}`,
            with an optional `as` naming the new column and `bins` for "bucket". Operations are
            "log", "log1p", "sqrt", "square", "abs", "reciprocal", "zscore", "minmax", "bucket".

    Returns:
        The dataset with the new columns appended, and what each one was derived from.
    """
    prepared = _prepared(source)
    data = prepared.data
    rows = [dict(row) for row in data.rows]
    columns = list(data.columns)
    added: list[dict[str, Any]] = []

    for wanted in transformations:
        column = wanted.get("column")
        operation = wanted.get("operation")
        if operation not in TRANSFORMATIONS:
            raise ValueError(
                f"unknown operation {operation!r}; expected one of {sorted(TRANSFORMATIONS)}."
            )
        if column not in data.columns:
            raise ValueError(f"no column named {column!r}; it has {data.columns}.")
        if not _is_numeric(data, column):
            raise ValueError(f"{column!r} is not numeric, so it cannot be {operation}-transformed.")

        name = wanted.get("as") or f"{column}_{operation}"
        while name in columns:  # never overwrite a column the dataset already had
            name = _numbered(name)

        values = [row[column] for row in rows]
        derived, detail = TRANSFORMATIONS[operation](values, wanted)
        for row, value in zip(rows, derived):
            row[name] = value

        columns.append(name)
        undefined = sum(1 for value in derived if value is None) - sum(
            1 for value in values if _is_missing(value)
        )
        added.append({"column": name, "from": column, "operation": operation,
                      "undefined": max(undefined, 0), **detail})

    return _step(prepared, "transform", _with(data, rows, columns), {"added": added})


# ---- running a list of steps ---------------------------------------------------------------------------


STEPS: dict[str, Callable[..., Prepared]] = {
    "normalize_columns": normalize_columns,
    "strip_whitespace": strip_whitespace,
    "coerce_types": coerce_types,
    "drop_duplicates": drop_duplicates,
    "drop_missing_rows": drop_missing_rows,
    "fill_missing": fill_missing,
    "drop_missing_columns": drop_missing_columns,
    "drop_constant_columns": drop_constant_columns,
    "clip_outliers": clip_outliers,
    "transform": transform,
}

# The order the steps are worth running in, and what `prepare` runs when it is given none. Naming and
# types come first because every later step reads the values: a median cannot be taken over strings,
# and a duplicate is only visible once its whitespace is gone.
DEFAULT_STEPS = (
    "normalize_columns",
    "strip_whitespace",
    "coerce_types",
    "drop_duplicates",
    "drop_constant_columns",
    "fill_missing",
)


def prepare(
    source: Dataset | Prepared,
    steps: Sequence[str | dict[str, Any]] | None = None,
) -> Prepared:
    """Run a list of steps over a dataset, in order, and return it with the log of what they did.

    A step is named as a string, or as a mapping that names it and carries its arguments —
    `{"step": "fill_missing", "strategy": "median"}` — which is the shape a `preprocessing_steps`
    argument arrives in from an agent. An argument the step does not take is dropped with a warning
    rather than failing the run, since one caller may pass the arguments for several steps.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        steps: The steps to run. Defaults to `DEFAULT_STEPS`.

    Returns:
        The prepared dataset, and one log entry per step.
    """
    prepared = _prepared(source)
    for entry in steps if steps is not None else DEFAULT_STEPS:
        arguments = dict(entry) if isinstance(entry, dict) else {"step": entry}
        name = arguments.pop("step", None) or arguments.pop("name", None)
        step = STEPS.get(str(name))
        if not step:
            raise ValueError(f"unknown step {name!r}; expected one of {sorted(STEPS)}.")

        prepared = step(prepared, **_accepted(step, arguments))

    return prepared


def write_csv(source: Dataset | Prepared, path: str | Path, delimiter: str = ",") -> str:
    """Write a prepared dataset out as a CSV, and return where it was written.

    This is what an `output_path` argument ends in: the next tool in a chain reads a file, not an
    object, so a prepared dataset has to land somewhere before it can be handed on.

    Args:
        source: The dataset, or the `Prepared` a previous step returned.
        path: Where to write it. Parent directories are created.
        delimiter: Field separator to write with.

    Returns:
        The path written to.
    """
    data = _prepared(source).data
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=data.columns, delimiter=delimiter)
        writer.writeheader()
        for row in data.rows:
            writer.writerow({column: _written(row.get(column)) for column in data.columns})

    logger.info(f"Wrote {len(data.rows)} row(s) and {len(data.columns)} column(s) to {path}.")
    return str(path)


# ---- inspection --------------------------------------------------------------------------------------


def numeric_columns(source: Dataset | Prepared) -> list[str]:
    """The columns holding numbers — what the steps above default to when they need numeric ones."""
    data = _prepared(source).data
    return [column for column in data.columns if _is_numeric(data, column)]


# ---- internals ---------------------------------------------------------------------------------------


def _prepared(source: Dataset | Prepared) -> Prepared:
    """Accept a `Dataset` or a `Prepared`, so steps chain and a first step needs no ceremony."""
    if isinstance(source, Prepared):
        return source
    if isinstance(source, Dataset):
        return Prepared(data=source)

    raise TypeError(f"expected a Dataset or a Prepared, not a {type(source).__name__}.")


def _step(
    prepared: Prepared,
    name: str,
    data: Dataset,
    changes: dict[str, Any],
) -> Prepared:
    """Return the dataset a step produced, with that step appended to the log.

    The shape after the step goes in every entry, so the log answers "what did cleaning cost me"
    without having to replay it.
    """
    entry = {"step": name, **changes, "rows": len(data.rows), "columns": len(data.columns)}
    return Prepared(data=data, steps=[*prepared.steps, entry])


def _with(data: Dataset, rows: list[dict[str, Any]], columns: Sequence[str] | None = None) -> Dataset:
    """Build the dataset a step produced, keeping where it came from and how it was read."""
    return Dataset(
        rows=rows,
        columns=list(columns) if columns is not None else list(data.columns),
        source=data.source,
        format=data.format,
        truncated=data.truncated,
    )


def _columns(data: Dataset, columns: Sequence[str] | None) -> list[str]:
    """Resolve a `columns` argument, refusing one the dataset does not have."""
    if columns is None:
        return list(data.columns)

    unknown = [column for column in columns if column not in data.columns]
    if unknown:
        raise ValueError(f"no column(s) {unknown}; it has {data.columns}.")

    return list(columns)


def _is_missing(value: Any) -> bool:
    """Whether a value is a hole: None, or the NaN a float column can carry one as."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_number(value: Any) -> bool:
    """Whether a value is a number to compute with — booleans are not, though Python says they are."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_missing(value)


def _is_numeric(data: Dataset, column: str) -> bool:
    """Whether a column holds numbers: at least one, and nothing that isn't."""
    present = [value for value in data.column(column) if not _is_missing(value)]
    return bool(present) and all(_is_number(value) for value in present)


def _missing_rate(data: Dataset, column: str) -> float:
    """The share of a column's values that are missing; an empty dataset is missing nothing."""
    if not data.rows:
        return 0.0

    return sum(1 for row in data.rows if _is_missing(row.get(column))) / len(data.rows)


def _hashable(value: Any) -> Any:
    """A value usable as a dict key — JSON sources carry lists and dicts, which are not."""
    if isinstance(value, (list, dict, set)):
        return repr(value)

    return value


def _normalized(name: str, lowercase: bool) -> str:
    """Rewrite one column name: word characters kept, everything else an underscore."""
    characters = [character if character.isalnum() else "_" for character in str(name).strip()]
    name = ''.join(characters).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    if name and name[0].isdigit():
        name = f"column_{name}"  # a name that starts with a digit is not usable as an identifier

    return name.lower() if lowercase else name


def _numbered(name: str) -> str:
    """Suffix a name that is already taken: `charges` -> `charges_2` -> `charges_3`."""
    stem, _, tail = name.rpartition("_")
    if stem and tail.isdigit():
        return f"{stem}_{int(tail) + 1}"

    return f"{name}_2"


def _inferred(data: Dataset, column: str) -> str | None:
    """The type a text column can be read as, or None to leave it as it is.

    Every value has to convert: a column of ids with one letter in it is text, and converting the rest
    of it would leave a column that is numbers where the data was clean and holes where it wasn't.
    """
    present = [value for value in data.column(column) if not _is_missing(value)]
    texts = [value for value in present if isinstance(value, str)]
    if not texts or len(texts) != len(present):  # already typed, or a mix this cannot speak for
        return None

    stripped = [text.strip() for text in texts]
    if all(text.lower() in BOOLEANS for text in stripped):
        return "bool"
    if all(INTEGER.match(text) for text in stripped):
        return "int"
    if all(DECIMAL.match(text) for text in stripped):
        return "float"

    return None


def _as_bool(value: Any) -> bool:
    """Read a value as a boolean, including the strings a CSV writes one as."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    text = str(value).strip().lower()
    if text in BOOLEANS:
        return BOOLEANS[text]
    if text in ("yes", "y", "1", "t"):
        return True
    if text in ("no", "n", "0", "f"):
        return False

    raise ValueError(f"{value!r} is not a boolean.")


def _as_datetime(value: Any) -> datetime:
    """Read a value as a datetime: ISO-8601 first, then the formats an export tends to write."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    for pattern in DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue

    raise ValueError(f"{value!r} is not a date this can read.")


def _replacement(column: str, present: list[Any], strategy: str, value: Any) -> Any:
    """Work out what one column's holes should be filled with. See `fill_missing` for the rules."""
    if strategy == "constant":
        return value
    if strategy == "zero":
        return 0
    if not present:
        return None

    numbers = [item for item in present if _is_number(item)]
    numeric = len(numbers) == len(present)
    if strategy == "auto":
        strategy = "median" if numeric else "mode"

    if strategy in ("median", "mean"):
        if not numeric:
            logger.info(f"fill_missing: {column!r} is not numeric; filling it with its mode instead.")
            return _mode(present)
        return statistics.median(numbers) if strategy == "median" else statistics.fmean(numbers)

    return _mode(present)


def _mode(values: list[Any]) -> Any:
    """The most common value, taking the first of a tie so the fill is deterministic."""
    counts: dict[Any, int] = {}
    for value in values:
        key = _hashable(value)
        counts[key] = counts.get(key, 0) + 1

    return max(counts, key=lambda key: counts[key])


def _bounds(values: list[float], method: str, factor: float) -> tuple[float, float] | None:
    """The range a column's values are held to, or None when there is too little to say.

    No spread means no bounds, under either method: a column whose middle half is a single value has
    nothing to measure distance from, and fences drawn on it would clip everything that isn't it.
    """
    if len(values) < 2:
        return None

    if method == "iqr":
        first, _, third = statistics.quantiles(values, n=4, method="inclusive")
        spread = third - first
        return (first - factor * spread, third + factor * spread) if spread else None

    spread = statistics.pstdev(values)
    if not spread:
        return None

    middle = statistics.fmean(values)
    return middle - factor * spread, middle + factor * spread


def _defined(values: list[Any], function: Callable[[float], float]) -> list[Any]:
    """Apply a function value by value, writing missing where it is not defined."""
    derived: list[Any] = []
    for value in values:
        if not _is_number(value):
            derived.append(None)
            continue
        try:
            derived.append(function(float(value)))
        except (ValueError, ZeroDivisionError, OverflowError):
            derived.append(None)

    return derived


def _standardised(values: list[Any], wanted: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Centre on the mean and scale by the standard deviation."""
    numbers = [value for value in values if _is_number(value)]
    if len(numbers) < 2:
        return [None] * len(values), {"skipped": "too few values"}

    middle, spread = statistics.fmean(numbers), statistics.pstdev(numbers)
    if not spread:
        return [None] * len(values), {"skipped": "no variance"}

    derived = _defined(values, lambda value: (value - middle) / spread)
    return derived, {"mean": middle, "stdev": spread}


def _scaled(values: list[Any], wanted: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Scale onto 0..1, which needs the range to be known before anything can be scaled onto it."""
    numbers = [value for value in values if _is_number(value)]
    if not numbers:
        return [None] * len(values), {"skipped": "no values"}

    low, high = min(numbers), max(numbers)
    if low == high:
        return [None] * len(values), {"skipped": "no range"}

    return _defined(values, lambda value: (value - low) / (high - low)), {"min": low, "max": high}


def _bucketed(values: list[Any], wanted: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Cut into equal-frequency bands, numbered from zero.

    Quantile edges rather than equal-width ones, so a skewed column produces bands that are actually
    populated instead of one band holding almost every row.
    """
    bins = int(wanted.get("bins", 4))
    if bins < 2:
        raise ValueError(f"bins is {bins}; a bucket transformation needs at least 2.")

    numbers = sorted(value for value in values if _is_number(value))
    if len(numbers) < bins:
        return [None] * len(values), {"skipped": "fewer values than bins", "bins": bins}

    edges = statistics.quantiles(numbers, n=bins, method="inclusive")
    derived = [
        None if not _is_number(value) else sum(1 for edge in edges if value > edge) for value in values
    ]
    return derived, {"bins": bins, "edges": edges}


# operation -> what it does to a column's values, and what it wants recorded about how it did it.
TRANSFORMATIONS: dict[str, Callable[[list[Any], dict[str, Any]], tuple[list[Any], dict[str, Any]]]] = {
    "log": lambda values, wanted: (_defined(values, math.log), {}),
    "log1p": lambda values, wanted: (_defined(values, math.log1p), {}),
    "sqrt": lambda values, wanted: (_defined(values, math.sqrt), {}),
    "square": lambda values, wanted: (_defined(values, lambda value: value**2), {}),
    "abs": lambda values, wanted: (_defined(values, abs), {}),
    "reciprocal": lambda values, wanted: (_defined(values, lambda value: 1 / value), {}),
    "zscore": _standardised,
    "minmax": _scaled,
    "bucket": _bucketed,
}


def _accepted(step: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep only the arguments `step` takes, so one caller can pass several steps' arguments."""
    parameters = inspect.signature(step).parameters
    accepted = {name: value for name, value in arguments.items() if name in parameters}
    for name in arguments.keys() - accepted.keys():
        logger.warning(f"{step.__name__} does not take {name!r}; dropping it.")

    return accepted


def _written(value: Any) -> Any:
    """Render a value for a CSV cell: missing as empty, dates as ISO-8601, the rest as they are."""
    if _is_missing(value):
        return ''
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value

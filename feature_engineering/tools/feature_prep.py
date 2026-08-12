"""Feature preparation steps the feature-engineering agent runs over a dataset.

Every step here works from the same four things an agent knows about a table:

    id_columns           who the row is about        (customer_id, account_id, ...)
    date_columns         when it happened            (event_date, signup_date, ...)
    aggregation_columns  what was measured           (amount, quantity, tickets, ...)
    data                 where the rows come from    (a CSV/JSON path, or the rows themselves)

None of the steps look at a target column, so none of them can leak the label — target-aware ranking
lives in `feature_selection.py`, deliberately apart from this module.

Three functions are meant to be called by an agent; the rest are the steps behind them and are
importable on their own:

    feature_prep_caller             run named steps, in order, over a dataset
    feature_generation_caller       run a whole generic bundle: row-level, entity-level or numeric
    list_feature_prep_steps_caller  the catalogue — every step, what it does, what it takes

They take JSON arguments, return JSON-serialisable summaries, and never hand a full dataset back to
the model: the rows are written to `output_path` and the reply carries the new column names, counts
and a small sample. Errors come back as `{"status": "error", ...}` rather than raising, so a
model that passes a bad column name can read what went wrong and try again.

Register them in `agentic_configurations.yaml` the same way the other tools are registered:

    - name: "feature_prep_tool"
      description: "Build features from id, date and measurement columns: date parts, per-entity
        aggregates, lags, rolling windows, encodings and numeric transforms."
      caller: "feature_engineer.tools.feature_prep.feature_prep_caller"
      args:
        - name: "data"
          type: "str"
          description: "Path to the CSV/JSON to build features from."
          required: false
        - name: "id_columns"
          type: "list"
          description: "Entity columns to group by. Inferred from the column names when omitted."
          required: false
        - name: "date_columns"
          type: "list"
          description: "Timestamp columns. Inferred from the values when omitted."
          required: false
        - name: "aggregation_columns"
          type: "list"
          description: "Numeric columns to aggregate and transform. Inferred when omitted."
          required: false
        - name: "steps"
          type: "list"
          description: "Steps to run, in order. Call feature_prep_steps_tool for the catalogue."
          required: false
        - name: "options"
          type: "dict"
          description: "Per-step arguments: {step_name: {argument: value}}, or flat for all steps."
          required: false
        - name: "output_path"
          type: "str"
          description: "Where to write the prepared dataset."
          required: false
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timedelta, timezone
from statistics import fmean, median, stdev
from typing import Any

logger = logging.getLogger(__name__)

Row = dict[str, Any]
Table = list[Row]

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y%m%d",
    "%b %d %Y",
    "%d %b %Y",
)

DATE_PARTS = (
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "dayofweek",
    "dayofyear",
    "hour",
    "minute",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "days_in_month",
)

AGGREGATION_FUNCTIONS = (
    "count",
    "count_non_missing",
    "nunique",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "var",
    "range",
    "first",
    "last",
)

MATH_OPERATIONS = ("log1p", "sqrt", "square", "cube_root", "reciprocal", "abs", "sign", "exp_decay")

SCALING_METHODS = ("zscore", "minmax", "robust", "rank")

DAY = 86400.0  # seconds, the unit every date-derived feature is reported in


# --------------------------------------------------------------------------------------------------
# values: the small decisions every step shares — what counts as missing, a number, a date, a key
# --------------------------------------------------------------------------------------------------
def is_missing(value: Any) -> bool:
    """True for None, NaN, and the strings a CSV uses to mean "nothing here"."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_TOKENS
    return False


def to_number(value: Any) -> float | None:
    """Read `value` as a float, or None when it isn't one. Booleans count as 0.0/1.0."""
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def to_datetime(value: Any) -> datetime | None:
    """Read `value` as a naive UTC datetime, or None when it isn't a date.

    ISO 8601 is tried first, then `DATE_FORMATS`, then a 10- or 13-digit epoch. Anything timezone
    aware is converted to UTC and stripped, so dates from mixed sources stay subtractable.
    """
    if is_missing(value):
        return None
    if isinstance(value, datetime):
        return _naive_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    text = str(value).strip()
    try:
        return _naive_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass

    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    if text.isdigit() and len(text) in (10, 13):  # epoch seconds or milliseconds
        seconds = int(text) / (1000.0 if len(text) == 13 else 1.0)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None

    return None


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _group_key(value: Any) -> str:
    """Normalise a value used as a grouping key, so 1 and "1" land in the same entity."""
    return "" if is_missing(value) else str(value).strip()


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for number in (to_number(value) for value in values) if number is not None]


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / float(denominator)  # type: ignore[arg-type]


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """The value at `fraction` through `sorted_values`, interpolating between neighbours."""
    if not sorted_values:
        raise ValueError("no values to take a quantile of")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _percentile_ranks(values: Sequence[float | None]) -> list[float | None]:
    """Rank every value in 0..1, ties sharing their average rank. None stays None."""
    present = [(value, index) for index, value in enumerate(values) if value is not None]
    ranks: list[float | None] = [None] * len(values)
    if len(present) < 2:
        return [0.0 if value is not None else None for value in values]

    present.sort(key=lambda pair: pair[0])
    position = 0
    while position < len(present):
        end = position
        while end + 1 < len(present) and present[end + 1][0] == present[position][0]:
            end += 1
        average_rank = (position + end) / 2.0
        for tied in range(position, end + 1):
            ranks[present[tied][1]] = average_rank / (len(present) - 1)
        position = end + 1
    return ranks


def _sanitize(name: str) -> str:
    """Turn a column or category into something safe to paste into a feature name."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return cleaned or "value"


def _as_columns(value: Any) -> list[str]:
    """Accept a list, a single name, or a comma-separated string — models send all three."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _as_numbers(value: Any, default: Sequence[float]) -> list[float]:
    """Accept a list, a single number, or a comma-separated string of numbers."""
    if value is None:
        return list(default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str):
        value = [part for part in value.replace(",", " ").split() if part]
    parsed = [to_number(item) for item in value] if isinstance(value, (list, tuple)) else []
    numbers = [number for number in parsed if number is not None]
    return numbers or list(default)


# --------------------------------------------------------------------------------------------------
# data in and out: a path, a JSON blob, or rows already in memory — always the same list of dicts
# --------------------------------------------------------------------------------------------------
def load_data(data: Any) -> Table:
    """Read `data` into a list of row dicts.

    Accepts a `data_engineer.tools.dataset.Dataset` as the stage before this one produces it, a path
    to a `.csv`, `.json` or `.jsonl` file, raw CSV or JSON text, a list of row mappings, a
    column-oriented `{column: values}` mapping, or any object exposing `to_dict("records")` (pandas)
    or `to_dicts()` (polars). Values are left as they were read; steps coerce them as they go.
    """
    if data is None:
        raise ValueError("no data given; pass a path to a CSV/JSON file or the rows themselves")

    if isinstance(data, str):
        return _load_text(data)

    if hasattr(data, "rows") and hasattr(data, "columns"):  # a Dataset from the data_engineer tools
        return [dict(row) for row in data.rows]
    if hasattr(data, "to_dict") and hasattr(data, "columns"):  # a pandas DataFrame, unimported
        return [dict(row) for row in data.to_dict("records")]
    if hasattr(data, "to_dicts"):  # a polars DataFrame
        return [dict(row) for row in data.to_dicts()]

    if isinstance(data, dict):
        return _rows_from_columns(data)

    if isinstance(data, (list, tuple)):
        rows = list(data)
        if not rows:
            return []
        if all(isinstance(row, dict) for row in rows):
            return [dict(row) for row in rows]
        raise ValueError("a list of rows must hold mappings, one per row")

    raise ValueError(f"cannot read data of type {type(data).__name__}")


def _load_text(data: str) -> Table:
    """Read a path, or the text of a CSV/JSON document passed inline."""
    text = data.strip()
    if os.path.exists(data):
        if not data.lower().endswith((".csv", ".json", ".jsonl")):
            return _read_next_door(data)  # parquet, excel, compressed — the data_engineer readers

        with open(data, newline="", encoding="utf-8") as handle:
            if data.lower().endswith(".jsonl"):
                return [json.loads(line) for line in handle if line.strip()]
            if data.lower().endswith(".json"):
                return _rows_from_json(json.load(handle))
            return [dict(row) for row in csv.DictReader(handle)]

    if text.startswith(("[", "{")):
        return _rows_from_json(json.loads(text))
    if "\n" in text and "," in text:  # a small CSV pasted straight into the argument
        return [dict(row) for row in csv.DictReader(text.splitlines())]

    raise FileNotFoundError(f"no such file: {data!r}")


def _read_next_door(path: str) -> Table:
    """Hand a format this module doesn't read to the data_engineer readers, if they're importable.

    Keeping CSV and JSON native means feature prep works on its own, with nothing but the standard
    library; anything else — Parquet, Excel, a compressed file — is already read next door, so this
    borrows that reader rather than growing a second one.
    """
    try:
        from data_engineer.tools.dataset import read_dataset
    except ImportError as error:
        raise ValueError(
            f"cannot read {path!r}: feature prep reads CSV, JSON and JSON Lines on its own, and "
            f"data_engineer.tools.dataset is not importable for the rest ({error})"
        ) from error

    return [dict(row) for row in read_dataset(path).rows]


def _rows_from_json(document: Any) -> Table:
    if isinstance(document, list):
        return [dict(row) for row in document]
    if isinstance(document, dict):
        for key in ("rows", "data", "records"):
            if isinstance(document.get(key), list):
                return [dict(row) for row in document[key]]
        return _rows_from_columns(document)
    raise ValueError("JSON data must be a list of rows or a mapping of columns")


def _rows_from_columns(columns: dict[str, Any]) -> Table:
    """Transpose a `{column: values}` mapping into rows."""
    if not all(isinstance(values, (list, tuple)) for values in columns.values()):
        raise ValueError("a column-oriented mapping must hold a list of values per column")
    length = max((len(values) for values in columns.values()), default=0)
    return [
        {
            column: (values[index] if index < len(values) else None)
            for column, values in columns.items()
        }
        for index in range(length)
    ]


def column_names(rows: Table) -> list[str]:
    """Every column present, in the order it first appears."""
    names: dict[str, None] = {}
    for row in rows:
        names.update(dict.fromkeys(row))
    return list(names)


def write_data(rows: Table, path: str) -> str:
    """Write rows to `path` as CSV or JSON, creating the directory if it isn't there."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, default=str, indent=2)
        return path

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _for_csv(value) for key, value in row.items()} for row in rows)
    return path


def _for_csv(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def _default_output_path(data: Any, suffix: str) -> str | None:
    """Derive `<source>__<suffix>.csv` next to the input, so steps can be chained by path."""
    if not isinstance(data, str) or not os.path.exists(data):
        return None
    stem, extension = os.path.splitext(data)
    return f"{stem}__{suffix}{extension if extension.lower() in ('.csv', '.json') else '.csv'}"


# --------------------------------------------------------------------------------------------------
# column inference: what an agent leaves out, worked out from the data
# --------------------------------------------------------------------------------------------------
def column_kinds(rows: Table, sample: int = 400) -> dict[str, str]:
    """Classify every column as "numeric", "date", "boolean", "categorical" or "empty"."""
    kinds: dict[str, str] = {}
    inspected = rows[:sample]

    for column in column_names(rows):
        values = [row.get(column) for row in inspected]
        present = [value for value in values if not is_missing(value)]
        if not present:
            kinds[column] = "empty"
            continue

        distinct = {_group_key(value).lower() for value in present}
        if distinct <= {"0", "1", "true", "false", "yes", "no"}:
            kinds[column] = "boolean"
        elif sum(to_number(value) is not None for value in present) >= 0.8 * len(present):
            kinds[column] = "numeric"
        elif sum(to_datetime(value) is not None for value in present) >= 0.8 * len(present):
            kinds[column] = "date"
        else:
            kinds[column] = "categorical"
    return kinds


def infer_columns(
    rows: Table,
    id_columns: Sequence[str] | None = None,
    date_columns: Sequence[str] | None = None,
    aggregation_columns: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Fill in whichever of the three column groups the caller left out.

    Ids are taken from the naming convention (`id`, `*_id`, `*_key`, `*_code`) and otherwise from a
    categorical column repeating across rows; dates from the values that parse as dates;
    measurements from every numeric column that is neither an id nor a date.
    """
    kinds = column_kinds(rows)
    ids = [column for column in _as_columns(id_columns) if column in kinds]
    dates = [column for column in _as_columns(date_columns) if column in kinds]
    aggregations = [column for column in _as_columns(aggregation_columns) if column in kinds]

    if not dates:
        dates = [column for column, kind in kinds.items() if kind == "date"]

    if not ids:
        ids = [
            column
            for column, kind in kinds.items()
            if column not in dates
            and kind != "date"
            and re.search(r"(^|_)(id|key|code|uuid)$", column.strip().lower())
        ]
    if not ids:  # nothing named like an id: the categorical column that repeats the most
        repeated = [
            (len(rows) / max(len({_group_key(row.get(column)) for row in rows}), 1), column)
            for column, kind in kinds.items()
            if kind == "categorical" and column not in dates
        ]
        ids = [max(repeated)[1]] if repeated and max(repeated)[0] > 1.5 else []

    if not aggregations:
        aggregations = [
            column
            for column, kind in kinds.items()
            if kind == "numeric" and column not in ids and column not in dates
        ]

    return {
        "id_columns": ids,
        "date_columns": dates,
        "aggregation_columns": aggregations,
        "categorical_columns": [
            column
            for column, kind in kinds.items()
            if kind in ("categorical", "boolean") and column not in ids and column not in dates
        ],
    }


def _grouped(rows: Table, id_columns: Sequence[str]) -> dict[tuple[str, ...], list[int]]:
    """Row indexes per entity. With no id columns every row lands in one global group."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(_group_key(row.get(column)) for column in id_columns)
        groups.setdefault(key, []).append(index)
    return groups


def _id_key(id_columns: Sequence[str]) -> str:
    return "_".join(_sanitize(column) for column in id_columns) or "all"


def _parsed_dates(rows: Table, date_column: str) -> list[datetime | None]:
    return [to_datetime(row.get(date_column)) for row in rows]


def _ordered_by_date(indexes: Sequence[int], dates: Sequence[datetime | None]) -> list[int]:
    """Group members oldest first, rows without a date keeping their original order at the end."""
    dated = sorted((index for index in indexes if dates[index] is not None), key=lambda i: dates[i])
    return dated + [index for index in indexes if dates[index] is None]


def _assign(rows: Table, index: int, name: str, value: Any) -> None:
    rows[index][name] = value


def _fill(rows: Table, names: Sequence[str], default: Any = None) -> None:
    """Give every row the new columns, so a row a step skipped isn't simply missing them."""
    for row in rows:
        for name in names:
            row.setdefault(name, default)


# --------------------------------------------------------------------------------------------------
# steps — date and time
# --------------------------------------------------------------------------------------------------
def add_date_part_features(
    rows: Table, date_columns: Sequence[str] = (), parts: Sequence[str] | None = None
) -> tuple[Table, list[str]]:
    """Split each date column into its calendar parts: year, month, day of week, is_weekend, ...

    Args:
        date_columns: The timestamp columns to split.
        parts: Which parts to build. Defaults to all of `DATE_PARTS`.
    """
    wanted = [part for part in _as_columns(parts) or list(DATE_PARTS) if part in DATE_PARTS]
    added: list[str] = []

    for column in date_columns:
        dates = _parsed_dates(rows, column)
        names = [f"{column}_{part}" for part in wanted]
        for index, moment in enumerate(dates):
            for part, name in zip(wanted, names):
                _assign(rows, index, name, _date_part(moment, part))
        added.extend(names)

    _fill(rows, added)
    return rows, added


def _date_part(moment: datetime | None, part: str) -> Any:
    if moment is None:
        return None
    if part == "year":
        return moment.year
    if part == "quarter":
        return (moment.month - 1) // 3 + 1
    if part == "month":
        return moment.month
    if part == "week":
        return moment.isocalendar()[1]
    if part == "day":
        return moment.day
    if part == "dayofweek":
        return moment.weekday()
    if part == "dayofyear":
        return moment.timetuple().tm_yday
    if part == "hour":
        return moment.hour
    if part == "minute":
        return moment.minute
    if part == "is_weekend":
        return int(moment.weekday() >= 5)
    if part == "is_month_start":
        return int(moment.day == 1)
    if part == "is_month_end":
        return int(moment.day == _days_in_month(moment))
    if part == "is_quarter_start":
        return int(moment.day == 1 and moment.month in (1, 4, 7, 10))
    if part == "days_in_month":
        return _days_in_month(moment)
    return None


def _days_in_month(moment: datetime) -> int:
    following = datetime(moment.year + (moment.month == 12), moment.month % 12 + 1, 1)
    return (following - timedelta(days=1)).day


def add_cyclical_date_features(
    rows: Table, date_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """Encode month, day of week, day of year and hour as sine/cosine pairs.

    December sits next to January on a circle but 11 columns away on a number line; a linear model
    reads the pair correctly and a tree model is no worse off.
    """
    cycles = (("month", 12), ("dayofweek", 7), ("dayofyear", 365), ("hour", 24))
    added: list[str] = []

    for column in date_columns:
        dates = _parsed_dates(rows, column)
        for part, period in cycles:
            sine, cosine = f"{column}_{part}_sin", f"{column}_{part}_cos"
            for index, moment in enumerate(dates):
                value = _date_part(moment, part)
                angle = None if value is None else 2 * math.pi * float(value) / period
                _assign(rows, index, sine, None if angle is None else math.sin(angle))
                _assign(rows, index, cosine, None if angle is None else math.cos(angle))
            added.extend([sine, cosine])

    _fill(rows, added)
    return rows, added


def add_recency_features(
    rows: Table, date_columns: Sequence[str] = (), reference_date: Any = None
) -> tuple[Table, list[str]]:
    """Measure how long ago each date was, in days, weeks and months.

    Args:
        reference_date: The "now" to measure against. Defaults to the latest date in the column,
            which keeps the feature stable when the dataset is replayed later.
    """
    added: list[str] = []

    for column in date_columns:
        dates = _parsed_dates(rows, column)
        present = [moment for moment in dates if moment is not None]
        if not present:
            continue

        reference = to_datetime(reference_date) or max(present)
        names = [f"{column}_days_since", f"{column}_weeks_since", f"{column}_months_since"]
        for index, moment in enumerate(dates):
            days = None if moment is None else (reference - moment).total_seconds() / DAY
            _assign(rows, index, names[0], days)
            _assign(rows, index, names[1], None if days is None else days / 7.0)
            _assign(rows, index, names[2], None if days is None else days / 30.44)
        added.extend(names)

    _fill(rows, added)
    return rows, added


def add_date_difference_features(
    rows: Table, date_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """The gap in days between every pair of date columns — tenure, time-to-event, lead time."""
    added: list[str] = []
    parsed = {column: _parsed_dates(rows, column) for column in date_columns}

    for position, earlier in enumerate(date_columns):
        for later in date_columns[position + 1 :]:
            name = f"{earlier}_to_{later}_days"
            for index in range(len(rows)):
                start, end = parsed[earlier][index], parsed[later][index]
                gap = None if start is None or end is None else (end - start).total_seconds() / DAY
                _assign(rows, index, name, gap)
            added.append(name)

    _fill(rows, added)
    return rows, added


# --------------------------------------------------------------------------------------------------
# steps — per entity
# --------------------------------------------------------------------------------------------------
def aggregate(values: Sequence[Any], function: str) -> Any:
    """Apply one named aggregation from `AGGREGATION_FUNCTIONS` to a group's values."""
    if function == "count":
        return len(values)
    if function == "count_non_missing":
        return sum(1 for value in values if not is_missing(value))
    if function == "nunique":
        return len({_group_key(value) for value in values if not is_missing(value)})
    if function in ("first", "last"):
        present = [value for value in values if not is_missing(value)]
        if not present:
            return None
        return present[0] if function == "first" else present[-1]

    numbers = _numbers(values)
    if not numbers:
        return None
    if function == "sum":
        return sum(numbers)
    if function == "mean":
        return fmean(numbers)
    if function == "median":
        return median(numbers)
    if function == "min":
        return min(numbers)
    if function == "max":
        return max(numbers)
    if function == "range":
        return max(numbers) - min(numbers)
    if function == "std":
        return stdev(numbers) if len(numbers) > 1 else 0.0
    if function == "var":
        return stdev(numbers) ** 2 if len(numbers) > 1 else 0.0
    raise ValueError(f"unknown aggregation {function!r}; expected one of {AGGREGATION_FUNCTIONS}")


def add_group_aggregate_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    functions: Sequence[str] = ("count", "sum", "mean", "max", "std"),
) -> tuple[Table, list[str]]:
    """Aggregate each measurement per entity and write the result back onto every one of its rows.

    This is the workhorse: `amount_mean_by_customer_id` on a transaction row tells a model what the
    customer normally spends, which is the context a single transaction can't carry on its own.
    """
    wanted = [function for function in _as_columns(functions) if function in AGGREGATION_FUNCTIONS]
    groups = _grouped(rows, id_columns)
    suffix = _id_key(id_columns)
    added: list[str] = []

    count_name = f"row_count_by_{suffix}"
    for indexes in groups.values():
        for index in indexes:
            _assign(rows, index, count_name, len(indexes))
    added.append(count_name)

    for column in aggregation_columns:
        for function in wanted:
            if function == "count":
                continue  # already covered by row_count_by_*
            name = f"{column}_{function}_by_{suffix}"
            for indexes in groups.values():
                value = aggregate([rows[index].get(column) for index in indexes], function)
                for index in indexes:
                    _assign(rows, index, name, value)
            added.append(name)

    _fill(rows, added)
    return rows, added


def add_group_relative_features(
    rows: Table, id_columns: Sequence[str] = (), aggregation_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """Place each value against its entity's own distribution: ratio, deviation, z-score, share.

    An amount of 500 means one thing for a customer who averages 20 and another for one who averages
    5,000 — these four say which.
    """
    groups = _grouped(rows, id_columns)
    suffix = _id_key(id_columns)
    added: list[str] = []

    for column in aggregation_columns:
        names = [
            f"{column}_ratio_to_{suffix}_mean",
            f"{column}_minus_{suffix}_mean",
            f"{column}_zscore_in_{suffix}",
            f"{column}_share_of_{suffix}_sum",
        ]
        for indexes in groups.values():
            numbers = _numbers(rows[index].get(column) for index in indexes)
            if not numbers:
                continue
            average = fmean(numbers)
            spread = stdev(numbers) if len(numbers) > 1 else 0.0
            total = sum(numbers)
            for index in indexes:
                value = to_number(rows[index].get(column))
                _assign(rows, index, names[0], _safe_divide(value, average))
                _assign(rows, index, names[1], None if value is None else value - average)
                _assign(
                    rows,
                    index,
                    names[2],
                    None if value is None or not spread else (value - average) / spread,
                )
                _assign(rows, index, names[3], _safe_divide(value, total))
        added.extend(names)

    _fill(rows, added)
    return rows, added


def add_lag_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    lags: Sequence[int] = (1,),
) -> tuple[Table, list[str]]:
    """The same measurement `k` events earlier for this entity, plus its change since then.

    Args:
        lags: How many events back to look. `(1, 2)` gives the previous and the one before it.
    """
    if not date_columns:
        logger.warning("add_lag_features needs a date column to order events by; skipping.")
        return rows, []

    dates = _parsed_dates(rows, date_columns[0])
    steps = [int(lag) for lag in _as_numbers(lags, (1,)) if lag >= 1]
    added: list[str] = []

    for column in aggregation_columns:
        for lag in steps:
            names = [f"{column}_lag_{lag}", f"{column}_diff_{lag}", f"{column}_pct_change_{lag}"]
            for indexes in _grouped(rows, id_columns).values():
                ordered = _ordered_by_date(indexes, dates)
                for position, index in enumerate(ordered):
                    previous = (
                        to_number(rows[ordered[position - lag]].get(column))
                        if position >= lag
                        else None
                    )
                    current = to_number(rows[index].get(column))
                    _assign(rows, index, names[0], previous)
                    _assign(
                        rows,
                        index,
                        names[1],
                        None if previous is None or current is None else current - previous,
                    )
                    _assign(
                        rows,
                        index,
                        names[2],
                        (
                            None
                            if current is None
                            else _safe_divide(current - (previous or 0.0), previous)
                        ),
                    )
            added.extend(names)

    _fill(rows, added)
    return rows, added


def add_rolling_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    windows: Sequence[int] = (7, 30, 90),
) -> tuple[Table, list[str]]:
    """Trailing time windows per entity: what happened in the `w` days up to and including this row.

    The window is half-open on the left — `(t - w days, t]` — so a row never sees its own future.

    Args:
        windows: Window widths in days.
    """
    if not date_columns:
        logger.warning(
            "add_rolling_features needs a date column to measure windows over; skipping."
        )
        return rows, []

    dates = _parsed_dates(rows, date_columns[0])
    widths = [int(window) for window in _as_numbers(windows, (7, 30, 90)) if window > 0]
    groups = _grouped(rows, id_columns)
    added: list[str] = []

    for width in widths:
        event_name = f"events_last_{width}d"
        value_names = {
            column: (f"{column}_sum_last_{width}d", f"{column}_mean_last_{width}d")
            for column in aggregation_columns
        }
        for indexes in groups.values():
            dated = sorted(
                (index for index in indexes if dates[index] is not None), key=lambda i: dates[i]
            )
            start = 0
            totals = {column: 0.0 for column in aggregation_columns}
            counts = {column: 0 for column in aggregation_columns}

            for position, index in enumerate(dated):
                for column in aggregation_columns:  # the row entering the window
                    value = to_number(rows[index].get(column))
                    if value is not None:
                        totals[column] += value
                        counts[column] += 1

                while (dates[index] - dates[dated[start]]).total_seconds() >= width * DAY:
                    for column in aggregation_columns:  # the row leaving it
                        value = to_number(rows[dated[start]].get(column))
                        if value is not None:
                            totals[column] -= value
                            counts[column] -= 1
                    start += 1

                _assign(rows, index, event_name, position - start + 1)
                for column, (sum_name, mean_name) in value_names.items():
                    _assign(rows, index, sum_name, totals[column])
                    _assign(rows, index, mean_name, _safe_divide(totals[column], counts[column]))

        added.append(event_name)
        added.extend(name for pair in value_names.values() for name in pair)

    _fill(rows, added)
    return rows, added


def add_cumulative_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
) -> tuple[Table, list[str]]:
    """Everything this entity has done up to this row: running count, sum, mean and max."""
    if not date_columns:
        logger.warning("add_cumulative_features needs a date column to order events by; skipping.")
        return rows, []

    dates = _parsed_dates(rows, date_columns[0])
    groups = _grouped(rows, id_columns)
    added = ["event_number"]

    for indexes in groups.values():
        for position, index in enumerate(_ordered_by_date(indexes, dates)):
            _assign(rows, index, "event_number", position + 1)

    for column in aggregation_columns:
        names = [f"{column}_cumsum", f"{column}_cummean", f"{column}_cummax"]
        for indexes in groups.values():
            total, seen, highest = 0.0, 0, None
            for index in _ordered_by_date(indexes, dates):
                value = to_number(rows[index].get(column))
                if value is not None:
                    total += value
                    seen += 1
                    highest = value if highest is None else max(highest, value)
                _assign(rows, index, names[0], total)
                _assign(rows, index, names[1], _safe_divide(total, seen))
                _assign(rows, index, names[2], highest)
        added.extend(names)

    _fill(rows, added)
    return rows, added


def add_event_gap_features(
    rows: Table, id_columns: Sequence[str] = (), date_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """How this entity's events are spaced: days since the previous one, against its own average.

    A gap that is three times the customer's usual gap is the signal; the raw gap alone isn't.
    """
    if not date_columns:
        logger.warning("add_event_gap_features needs a date column; skipping.")
        return rows, []

    dates = _parsed_dates(rows, date_columns[0])
    suffix = _id_key(id_columns)
    added = [
        "days_since_previous_event",
        "days_since_first_event",
        f"mean_gap_days_by_{suffix}",
        "gap_ratio_to_mean_gap",
    ]

    for indexes in _grouped(rows, id_columns).values():
        ordered = [index for index in _ordered_by_date(indexes, dates) if dates[index] is not None]
        gaps = [
            (dates[ordered[position]] - dates[ordered[position - 1]]).total_seconds() / DAY
            for position in range(1, len(ordered))
        ]
        average_gap = fmean(gaps) if gaps else None

        for position, index in enumerate(ordered):
            gap = gaps[position - 1] if position else None
            _assign(rows, index, added[0], gap)
            _assign(rows, index, added[1], (dates[index] - dates[ordered[0]]).total_seconds() / DAY)
            _assign(rows, index, added[2], average_gap)
            _assign(rows, index, added[3], _safe_divide(gap, average_gap))

    _fill(rows, added)
    return rows, added


def add_trend_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
) -> tuple[Table, list[str]]:
    """The slope of each measurement against time for this entity — is it rising or falling?

    A least-squares slope in units per day, broadcast onto every row of the entity.
    """
    if not date_columns:
        logger.warning("add_trend_features needs a date column; skipping.")
        return rows, []

    dates = _parsed_dates(rows, date_columns[0])
    suffix = _id_key(id_columns)
    added: list[str] = []

    for column in aggregation_columns:
        name = f"{column}_trend_by_{suffix}"
        for indexes in _grouped(rows, id_columns).values():
            points = [
                (dates[index], to_number(rows[index].get(column)))
                for index in indexes
                if dates[index] is not None and to_number(rows[index].get(column)) is not None
            ]
            slope = _slope_per_day(points)
            for index in indexes:
                _assign(rows, index, name, slope)
        added.append(name)

    _fill(rows, added)
    return rows, added


def _slope_per_day(points: Sequence[tuple[datetime, float]]) -> float | None:
    """Least-squares slope of value against days elapsed, or None when it isn't defined."""
    if len(points) < 2:
        return None
    origin = min(moment for moment, _ in points)
    days = [(moment - origin).total_seconds() / DAY for moment, _ in points]
    values = [value for _, value in points]
    mean_days, mean_values = fmean(days), fmean(values)
    variance = sum((day - mean_days) ** 2 for day in days)
    if not variance:
        return None
    covariance = sum((day - mean_days) * (value - mean_values) for day, value in zip(days, values))
    return covariance / variance


# --------------------------------------------------------------------------------------------------
# steps — numeric transforms
# --------------------------------------------------------------------------------------------------
def add_math_transformations(
    rows: Table,
    columns: Sequence[str] = (),
    operations: Sequence[str] = ("log1p", "sqrt", "square"),
) -> tuple[Table, list[str]]:
    """Reshape skewed measurements: log1p, sqrt, square, cube_root, reciprocal, abs, sign, decay.

    Operations that are undefined for a value (log of a negative, reciprocal of zero) leave None
    there rather than inventing a number.
    """
    wanted = [name for name in _as_columns(operations) if name in MATH_OPERATIONS]
    added: list[str] = []

    for column in columns:
        for operation in wanted:
            name = f"{column}_{operation}"
            for index, row in enumerate(rows):
                _assign(rows, index, name, _transform(to_number(row.get(column)), operation))
            added.append(name)

    _fill(rows, added)
    return rows, added


def _transform(value: float | None, operation: str) -> float | None:
    if value is None:
        return None
    if operation == "log1p":
        return math.log1p(value) if value > -1 else None
    if operation == "sqrt":
        return math.sqrt(value) if value >= 0 else None
    if operation == "square":
        return value * value
    if operation == "cube_root":
        return math.copysign(abs(value) ** (1 / 3), value)
    if operation == "reciprocal":
        return 1.0 / value if value else None
    if operation == "abs":
        return abs(value)
    if operation == "sign":
        return float((value > 0) - (value < 0))
    if operation == "exp_decay":
        return math.exp(-value) if -700 < value else None
    return None


def add_scaling_features(
    rows: Table, columns: Sequence[str] = (), method: str = "zscore"
) -> tuple[Table, list[str]]:
    """Put measurements on a comparable scale.

    Args:
        method: "zscore" (mean 0, sd 1), "minmax" (0..1), "robust" (median and IQR, so outliers
            don't set the scale) or "rank" (percentile position, which ignores the shape entirely).
    """
    if method not in SCALING_METHODS:
        raise ValueError(f"unknown scaling method {method!r}; expected one of {SCALING_METHODS}")
    added: list[str] = []

    for column in columns:
        values = [to_number(row.get(column)) for row in rows]
        numbers = [value for value in values if value is not None]
        if not numbers:
            continue

        name = f"{column}_{method}"
        scaled = _scale(values, numbers, method)
        for index, value in enumerate(scaled):
            _assign(rows, index, name, value)
        added.append(name)

    _fill(rows, added)
    return rows, added


def _scale(
    values: Sequence[float | None], numbers: Sequence[float], method: str
) -> list[float | None]:
    if method == "rank":
        return _percentile_ranks(values)

    if method == "zscore":
        centre = fmean(numbers)
        spread = stdev(numbers) if len(numbers) > 1 else 0.0
    elif method == "minmax":
        centre = min(numbers)
        spread = max(numbers) - centre
    else:  # robust
        ordered = sorted(numbers)
        centre = median(ordered)
        spread = _quantile(ordered, 0.75) - _quantile(ordered, 0.25)

    if not spread:
        return [None if value is None else 0.0 for value in values]
    return [None if value is None else (value - centre) / spread for value in values]


def add_binning_features(
    rows: Table, columns: Sequence[str] = (), bins: int = 4, strategy: str = "quantile"
) -> tuple[Table, list[str]]:
    """Cut each measurement into ordered buckets — the tenure_bucket move, done generically.

    Args:
        bins: How many buckets.
        strategy: "quantile" for equal-sized buckets, "uniform" for equal-width ones.
    """
    count = max(int(bins), 2)
    added: list[str] = []

    for column in columns:
        values = [to_number(row.get(column)) for row in rows]
        numbers = sorted(value for value in values if value is not None)
        if len(numbers) < count:
            continue

        if strategy == "uniform":
            low, high = numbers[0], numbers[-1]
            edges = [low + (high - low) * step / count for step in range(1, count)]
        else:
            edges = [_quantile(numbers, step / count) for step in range(1, count)]

        name = f"{column}_bin"
        for index, value in enumerate(values):
            bucket = None if value is None else sum(value > edge for edge in edges)
            _assign(rows, index, name, bucket)
        added.append(name)

    _fill(rows, added)
    return rows, added


def add_outlier_features(
    rows: Table, columns: Sequence[str] = (), lower: float = 0.01, upper: float = 0.99
) -> tuple[Table, list[str]]:
    """Clip each measurement to its `lower`..`upper` quantiles, and flag what had to be clipped.

    The flag is often the more useful of the two: an extreme value is itself information, and this
    keeps that information while stopping the magnitude from dominating a model.
    """
    added: list[str] = []

    for column in columns:
        values = [to_number(row.get(column)) for row in rows]
        numbers = sorted(value for value in values if value is not None)
        if len(numbers) < 3:
            continue

        floor, ceiling = _quantile(numbers, lower), _quantile(numbers, upper)
        names = [f"{column}_winsorized", f"{column}_is_outlier"]
        for index, value in enumerate(values):
            clipped = None if value is None else min(max(value, floor), ceiling)
            _assign(rows, index, names[0], clipped)
            _assign(rows, index, names[1], None if value is None else int(value != clipped))
        added.extend(names)

    _fill(rows, added)
    return rows, added


def add_ratio_features(
    rows: Table, columns: Sequence[str] = (), max_features: int = 12
) -> tuple[Table, list[str]]:
    """Divide each measurement by every other one — the "per" features, generated exhaustively.

    Ratios carry what neither column carries alone (spend per order, tickets per year of tenure).
    `max_features` keeps a wide table from turning into a combinatorial one.
    """
    added: list[str] = []

    for numerator, denominator in _pairs(columns, max_features):
        name = f"{numerator}_per_{denominator}"
        for index, row in enumerate(rows):
            _assign(
                rows,
                index,
                name,
                _safe_divide(to_number(row.get(numerator)), to_number(row.get(denominator))),
            )
        added.append(name)

    _fill(rows, added)
    return rows, added


def add_interaction_features(
    rows: Table,
    columns: Sequence[str] = (),
    operations: Sequence[str] = ("product", "difference"),
    max_features: int = 12,
) -> tuple[Table, list[str]]:
    """Multiply and subtract measurement pairs, for the effects a linear model can't reach alone.

    Args:
        operations: Any of "product", "difference", "sum".
    """
    wanted = [name for name in _as_columns(operations) if name in ("product", "difference", "sum")]
    added: list[str] = []

    for left, right in _pairs(columns, max_features, ordered=False):
        for operation in wanted:
            name = f"{left}_{operation[:4]}_{right}"
            for index, row in enumerate(rows):
                first, second = to_number(row.get(left)), to_number(row.get(right))
                if first is None or second is None:
                    _assign(rows, index, name, None)
                elif operation == "product":
                    _assign(rows, index, name, first * second)
                elif operation == "difference":
                    _assign(rows, index, name, first - second)
                else:
                    _assign(rows, index, name, first + second)
            added.append(name)

    _fill(rows, added)
    return rows, added


def _pairs(columns: Sequence[str], limit: int, ordered: bool = True) -> list[tuple[str, str]]:
    """Column pairs, both directions when `ordered`, capped at `limit`."""
    pairs = [
        (left, right)
        for position, left in enumerate(columns)
        for right in (columns if ordered else columns[position + 1 :])
        if left != right
    ]
    return pairs[: max(int(limit), 0)]


def add_row_statistics(rows: Table, columns: Sequence[str] = ()) -> tuple[Table, list[str]]:
    """Summarise each row across its measurements: total, mean, max, spread, how many are non-zero.

    Useful the moment a row holds several comparable amounts — a basket, a set of channel counts, a
    month's worth of columns.
    """
    if len(columns) < 2:
        return rows, []

    added = ["row_total", "row_mean", "row_max", "row_min", "row_std", "row_nonzero_count"]
    for index, row in enumerate(rows):
        numbers = _numbers(row.get(column) for column in columns)
        _assign(rows, index, added[0], sum(numbers) if numbers else None)
        _assign(rows, index, added[1], fmean(numbers) if numbers else None)
        _assign(rows, index, added[2], max(numbers) if numbers else None)
        _assign(rows, index, added[3], min(numbers) if numbers else None)
        _assign(rows, index, added[4], stdev(numbers) if len(numbers) > 1 else 0.0)
        _assign(rows, index, added[5], sum(1 for number in numbers if number))
    return rows, added


# --------------------------------------------------------------------------------------------------
# steps — categorical and missingness
# --------------------------------------------------------------------------------------------------
def add_frequency_encoding(
    rows: Table, categorical_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """Replace a category with how often it occurs — one column, any cardinality, no leakage."""
    added: list[str] = []

    for column in categorical_columns:
        counts: dict[str, int] = {}
        for row in rows:
            counts[_group_key(row.get(column))] = counts.get(_group_key(row.get(column)), 0) + 1

        names = [f"{column}_frequency", f"{column}_frequency_ratio"]
        for index, row in enumerate(rows):
            count = counts.get(_group_key(row.get(column)), 0)
            _assign(rows, index, names[0], count)
            _assign(rows, index, names[1], count / len(rows) if rows else None)
        added.extend(names)

    _fill(rows, added)
    return rows, added


def add_one_hot_features(
    rows: Table, categorical_columns: Sequence[str] = (), max_categories: int = 10
) -> tuple[Table, list[str]]:
    """A 0/1 column per category, for the `max_categories` most common ones."""
    added: list[str] = []

    for column in categorical_columns:
        counts: dict[str, int] = {}
        for row in rows:
            key = _group_key(row.get(column))
            if key:
                counts[key] = counts.get(key, 0) + 1

        common = sorted(counts, key=lambda key: (-counts[key], key))[: max(int(max_categories), 1)]
        names = {category: f"{column}_is_{_sanitize(category)}" for category in common}
        for index, row in enumerate(rows):
            key = _group_key(row.get(column))
            for category, name in names.items():
                _assign(rows, index, name, int(key == category))
        added.extend(names.values())

    _fill(rows, added, 0)
    return rows, added


def add_ordinal_encoding(
    rows: Table, categorical_columns: Sequence[str] = ()
) -> tuple[Table, list[str]]:
    """Number the categories, most common first, so the codes are stable across runs of the data."""
    added: list[str] = []

    for column in categorical_columns:
        counts: dict[str, int] = {}
        for row in rows:
            key = _group_key(row.get(column))
            if key:
                counts[key] = counts.get(key, 0) + 1

        codes = {
            category: code
            for code, category in enumerate(sorted(counts, key=lambda key: (-counts[key], key)))
        }
        name = f"{column}_ordinal"
        for index, row in enumerate(rows):
            _assign(rows, index, name, codes.get(_group_key(row.get(column))))
        added.append(name)

    _fill(rows, added)
    return rows, added


def add_rare_category_grouping(
    rows: Table, categorical_columns: Sequence[str] = (), min_frequency: float = 0.01
) -> tuple[Table, list[str]]:
    """Fold categories rarer than `min_frequency` into a single "other" level.

    A category seen four times in fifty thousand rows is noise a model will happily overfit to.
    """
    added: list[str] = []
    floor = max(1, int(min_frequency * len(rows)))

    for column in categorical_columns:
        counts: dict[str, int] = {}
        for row in rows:
            key = _group_key(row.get(column))
            counts[key] = counts.get(key, 0) + 1

        name = f"{column}_grouped"
        for index, row in enumerate(rows):
            key = _group_key(row.get(column))
            _assign(rows, index, name, key if counts.get(key, 0) >= floor else "other")
        added.append(name)

    _fill(rows, added)
    return rows, added


def add_missing_indicators(rows: Table, columns: Sequence[str] = ()) -> tuple[Table, list[str]]:
    """Flag what was missing before anything imputes it, and count the gaps per row.

    Whether a field was filled in is often predictive in its own right, and imputation destroys it.
    """
    inspected = list(columns) or column_names(rows)
    added: list[str] = []

    for column in inspected:
        if not any(is_missing(row.get(column)) for row in rows):
            continue  # a column with nothing missing would only add a column of zeros
        name = f"{column}_is_missing"
        for index, row in enumerate(rows):
            _assign(rows, index, name, int(is_missing(row.get(column))))
        added.append(name)

    for index, row in enumerate(rows):
        _assign(rows, index, "row_missing_count", sum(is_missing(row.get(c)) for c in inspected))
    added.append("row_missing_count")

    _fill(rows, added, 0)
    return rows, added


# --------------------------------------------------------------------------------------------------
# entity tables: many rows per entity in, one row per entity out
# --------------------------------------------------------------------------------------------------
def build_entity_feature_table(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    functions: Sequence[str] = ("count", "sum", "mean", "median", "min", "max", "std", "last"),
    windows: Sequence[int] = (),
    reference_date: Any = None,
) -> tuple[Table, list[str]]:
    """Collapse an event log into one feature row per entity — the shape a feature store serves.

    Each measurement contributes its aggregates and its trend; each date column contributes the
    first and last time it was seen, the span between them, how recent it is and how many
    distinct days it covers; `windows` adds trailing "in the last w days" totals measured from
    `reference_date`.

    Args:
        windows: Trailing window widths in days, e.g. `(7, 30, 90)`.
        reference_date: The "now" the recency and windows are measured from. Defaults to the latest
            date in the data.
    """
    wanted = [function for function in _as_columns(functions) if function in AGGREGATION_FUNCTIONS]
    widths = [int(width) for width in _as_numbers(windows, ()) if width > 0]
    primary = date_columns[0] if date_columns else None
    dates = _parsed_dates(rows, primary) if primary else []
    latest = to_datetime(reference_date) or max(
        (moment for moment in dates if moment is not None), default=None
    )

    entities: Table = []
    features: list[str] = []

    for key, indexes in _grouped(rows, id_columns).items():
        entity: Row = dict(zip(id_columns, key))
        entity["event_count"] = len(indexes)

        for column in aggregation_columns:
            values = [rows[index].get(column) for index in indexes]
            for function in wanted:
                entity[f"{column}_{function}"] = aggregate(values, function)
            if primary:
                points = [
                    (dates[index], to_number(rows[index].get(column)))
                    for index in indexes
                    if dates[index] is not None and to_number(rows[index].get(column)) is not None
                ]
                entity[f"{column}_trend"] = _slope_per_day(points)

        for column in date_columns:
            entity.update(_entity_date_features(rows, indexes, column, latest))

        for width in widths:
            entity.update(
                _entity_window_features(rows, indexes, dates, aggregation_columns, width, latest)
            )

        entities.append(entity)
        features = [name for name in entity if name not in id_columns]

    return entities, features


def _entity_date_features(
    rows: Table, indexes: Sequence[int], column: str, reference: datetime | None
) -> Row:
    """First seen, last seen, span, recency, active days and average gap, for one entity."""
    moments = sorted(
        moment for moment in (to_datetime(rows[index].get(column)) for index in indexes) if moment
    )
    if not moments:
        return {
            f"{column}_first": None,
            f"{column}_last": None,
            f"{column}_span_days": None,
            f"{column}_recency_days": None,
            f"{column}_active_days": 0,
            f"{column}_mean_gap_days": None,
        }

    span = (moments[-1] - moments[0]).total_seconds() / DAY
    return {
        f"{column}_first": moments[0].isoformat(),
        f"{column}_last": moments[-1].isoformat(),
        f"{column}_span_days": span,
        f"{column}_recency_days": (
            None if reference is None else (reference - moments[-1]).total_seconds() / DAY
        ),
        f"{column}_active_days": len({moment.date() for moment in moments}),
        f"{column}_mean_gap_days": span / (len(moments) - 1) if len(moments) > 1 else None,
    }


def _entity_window_features(
    rows: Table,
    indexes: Sequence[int],
    dates: Sequence[datetime | None],
    aggregation_columns: Sequence[str],
    width: int,
    reference: datetime | None,
) -> Row:
    """Totals over the `width` days before `reference`, for one entity."""
    features: Row = {f"events_last_{width}d": 0}
    for column in aggregation_columns:
        features[f"{column}_sum_last_{width}d"] = 0.0

    if reference is None:
        return features

    cutoff = reference - timedelta(days=width)
    for index in indexes:
        moment = dates[index] if index < len(dates) else None
        if moment is None or not (cutoff < moment <= reference):
            continue
        features[f"events_last_{width}d"] += 1
        for column in aggregation_columns:
            value = to_number(rows[index].get(column))
            if value is not None:
                features[f"{column}_sum_last_{width}d"] += value
    return features


def build_rfm_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    reference_date: Any = None,
    buckets: int = 5,
) -> tuple[Table, list[str]]:
    """Recency, frequency and monetary value per entity, scored into `buckets` and segmented.

    The oldest generic feature block there is, and still the one that carries most of the signal in
    customer problems. Scores run 1..`buckets`, high being good — recency is reversed so that a
    customer seen yesterday scores highest. The segment label is a plain reading of the R and F
    scores, not a clustering.
    """
    if not date_columns:
        logger.warning("build_rfm_features needs a date column to measure recency from; skipping.")
        return [], []

    dates = _parsed_dates(rows, date_columns[0])
    monetary = aggregation_columns[0] if aggregation_columns else None
    latest = to_datetime(reference_date) or max(
        (moment for moment in dates if moment is not None), default=None
    )
    if latest is None:
        logger.warning("build_rfm_features found no readable dates; skipping.")
        return [], []

    entities: Table = []
    for key, indexes in _grouped(rows, id_columns).items():
        moments = [dates[index] for index in indexes if dates[index] is not None]
        amounts = _numbers(rows[index].get(monetary) for index in indexes) if monetary else []
        entity: Row = dict(zip(id_columns, key))
        entity["recency_days"] = (latest - max(moments)).total_seconds() / DAY if moments else None
        entity["frequency"] = len(indexes)
        entity["monetary_sum"] = sum(amounts) if amounts else 0.0
        entity["monetary_mean"] = fmean(amounts) if amounts else 0.0
        entity["monetary_max"] = max(amounts) if amounts else 0.0
        entities.append(entity)

    count = max(int(buckets), 2)
    recency_scores = _bucket_scores(
        [entity["recency_days"] for entity in entities], count, reverse=True
    )
    frequency_scores = _bucket_scores([entity["frequency"] for entity in entities], count)
    monetary_scores = _bucket_scores([entity["monetary_sum"] for entity in entities], count)

    for entity, recency, frequency, money in zip(
        entities, recency_scores, frequency_scores, monetary_scores
    ):
        entity["r_score"] = recency
        entity["f_score"] = frequency
        entity["m_score"] = money
        entity["rfm_score"] = (
            None if None in (recency, frequency, money) else recency + frequency + money
        )
        entity["rfm_segment"] = _rfm_segment(recency, frequency, count)

    features = [name for name in (entities[0] if entities else {}) if name not in id_columns]
    return entities, features


def _bucket_scores(values: Sequence[Any], buckets: int, reverse: bool = False) -> list[int | None]:
    """Score values 1..buckets by percentile, `reverse` making the smallest value the best."""
    ranks = _percentile_ranks([to_number(value) for value in values])
    scores: list[int | None] = []
    for rank in ranks:
        if rank is None:
            scores.append(None)
            continue
        position = 1 - rank if reverse else rank
        scores.append(min(buckets, int(position * buckets) + 1))
    return scores


def _rfm_segment(recency: int | None, frequency: int | None, buckets: int) -> str:
    if recency is None or frequency is None:
        return "unknown"
    high = buckets - 1
    if recency >= high and frequency >= high:
        return "champion"
    if frequency >= high:
        return "at_risk"
    if recency >= high:
        return "new_or_promising"
    if recency <= 2 and frequency <= 2:
        return "lost"
    return "needs_attention"


# --------------------------------------------------------------------------------------------------
# generic generators: whole bundles, for an agent that knows the columns but not which steps to pick
# --------------------------------------------------------------------------------------------------
def generate_generic_row_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
) -> tuple[Table, list[str]]:
    """Enrich every row with the features that are worth having on almost any event table.

    Missingness flags, calendar and cyclical date parts, recency, gaps between date columns, the
    entity's own aggregates and where this row sits inside them, the previous event and the change
    since it, trailing 7/30/90-day windows, event spacing, and frequency-encoded categories.

    The row count doesn't change, so the result still joins back to whatever the rows came from.
    """
    added: list[str] = []
    for step in (
        lambda: add_missing_indicators(rows, aggregation_columns),
        lambda: add_date_part_features(rows, date_columns),
        lambda: add_cyclical_date_features(rows, date_columns),
        lambda: add_recency_features(rows, date_columns),
        lambda: add_date_difference_features(rows, date_columns),
        lambda: add_group_aggregate_features(rows, id_columns, aggregation_columns),
        lambda: add_group_relative_features(rows, id_columns, aggregation_columns),
        lambda: add_lag_features(rows, id_columns, date_columns, aggregation_columns),
        lambda: add_rolling_features(rows, id_columns, date_columns, aggregation_columns),
        lambda: add_event_gap_features(rows, id_columns, date_columns),
        lambda: add_frequency_encoding(rows, categorical_columns),
        lambda: add_row_statistics(rows, aggregation_columns),
    ):
        _, names = step()
        added.extend(names)
    return rows, added


def generate_generic_entity_features(
    rows: Table,
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    aggregation_columns: Sequence[str] = (),
    reference_date: Any = None,
    windows: Sequence[int] = (7, 30, 90),
) -> tuple[Table, list[str]]:
    """Turn an event log into one feature row per entity: full aggregates, trends, windows and RFM.

    This is the table a feature store actually serves — keyed by entity, with no row-level detail
    left in it. RFM is merged in on the same key, and ratios between the entity totals are derived
    on top, since per-entity rates (spend per active day, events per day of tenure) usually carry
    more than the totals do.
    """
    entities, features = build_entity_feature_table(
        rows,
        id_columns,
        date_columns,
        aggregation_columns,
        windows=windows,
        reference_date=reference_date,
    )
    if not entities:
        return [], []

    rfm_rows, rfm_features = build_rfm_features(
        rows, id_columns, date_columns, aggregation_columns, reference_date=reference_date
    )
    if rfm_rows:
        by_key = {
            tuple(_group_key(entity.get(column)) for column in id_columns): entity
            for entity in rfm_rows
        }
        for entity in entities:
            key = tuple(_group_key(entity.get(column)) for column in id_columns)
            entity.update({name: by_key.get(key, {}).get(name) for name in rfm_features})
        features.extend(name for name in rfm_features if name not in features)

    rates = _entity_rate_features(entities, date_columns, aggregation_columns)
    features.extend(rates)
    return entities, features


def _entity_rate_features(
    entities: Table, date_columns: Sequence[str], aggregation_columns: Sequence[str]
) -> list[str]:
    """Per-day and per-event rates, which compare across entities where raw totals don't."""
    if not date_columns:
        return []

    primary = date_columns[0]
    added: list[str] = []
    for entity in entities:
        span = to_number(entity.get(f"{primary}_span_days"))
        active = to_number(entity.get(f"{primary}_active_days"))
        events = to_number(entity.get("event_count"))
        entity["events_per_day"] = _safe_divide(events, span)
        entity["events_per_active_day"] = _safe_divide(events, active)
        for column in aggregation_columns:
            total = to_number(entity.get(f"{column}_sum"))
            entity[f"{column}_per_day"] = _safe_divide(total, span)
            entity[f"{column}_per_event"] = _safe_divide(total, events)

    added = ["events_per_day", "events_per_active_day"]
    for column in aggregation_columns:
        added.extend([f"{column}_per_day", f"{column}_per_event"])
    return added


def generate_generic_numeric_features(
    rows: Table, columns: Sequence[str] = (), max_features: int = 12
) -> tuple[Table, list[str]]:
    """Work the measurements themselves: reshape, scale, bin, clip, and cross them pairwise.

    Nothing here needs an id or a date, so it works on a plain flat table — the case where every row
    is already one entity.
    """
    added: list[str] = []
    for step in (
        lambda: add_math_transformations(rows, columns),
        lambda: add_scaling_features(rows, columns, "zscore"),
        lambda: add_binning_features(rows, columns),
        lambda: add_outlier_features(rows, columns),
        lambda: add_ratio_features(rows, columns, max_features),
        lambda: add_interaction_features(rows, columns, max_features=max_features),
        lambda: add_row_statistics(rows, columns),
    ):
        _, names = step()
        added.extend(names)
    return rows, added


# --------------------------------------------------------------------------------------------------
# the catalogue, and the dispatcher that runs it
# --------------------------------------------------------------------------------------------------
FEATURE_PREP_STEPS: dict[str, Callable[..., tuple[Table, list[str]]]] = {
    "date_parts": add_date_part_features,
    "cyclical_dates": add_cyclical_date_features,
    "recency": add_recency_features,
    "date_differences": add_date_difference_features,
    "group_aggregates": add_group_aggregate_features,
    "group_relative": add_group_relative_features,
    "lags": add_lag_features,
    "rolling_windows": add_rolling_features,
    "cumulative": add_cumulative_features,
    "event_gaps": add_event_gap_features,
    "trend": add_trend_features,
    "math_transformations": add_math_transformations,
    "scaling": add_scaling_features,
    "binning": add_binning_features,
    "outliers": add_outlier_features,
    "ratios": add_ratio_features,
    "interactions": add_interaction_features,
    "row_statistics": add_row_statistics,
    "frequency_encoding": add_frequency_encoding,
    "one_hot_encoding": add_one_hot_features,
    "ordinal_encoding": add_ordinal_encoding,
    "rare_category_grouping": add_rare_category_grouping,
    "missing_indicators": add_missing_indicators,
    "entity_table": build_entity_feature_table,
    "rfm": build_rfm_features,
    "generic_row_features": generate_generic_row_features,
    "generic_entity_features": generate_generic_entity_features,
    "generic_numeric_features": generate_generic_numeric_features,
}

RESHAPING_STEPS = (
    "entity_table",
    "rfm",
    "generic_entity_features",
)  # one row per entity, not per row

DEFAULT_STEPS = (
    "missing_indicators",
    "date_parts",
    "recency",
    "group_aggregates",
    "group_relative",
    "event_gaps",
    "frequency_encoding",
    "row_statistics",
)


def _run_step(name: str, rows: Table, pool: dict[str, Any]) -> tuple[Table, list[str]]:
    """Call one step with the arguments it declares, drawn from the shared pool."""
    function = FEATURE_PREP_STEPS[name]
    parameters = inspect.signature(function).parameters
    arguments = {
        key: value
        for key, value in pool.items()
        if key in parameters and key not in ("rows", "data")
    }
    return function(rows, **arguments)


def run_steps(
    rows: Table,
    steps: Sequence[str],
    columns: dict[str, list[str]],
    options: dict[str, Any] | None = None,
    reference_date: Any = None,
) -> tuple[Table, list[dict[str, Any]]]:
    """Run named steps in order, threading the output of each into the next.

    Args:
        columns: The resolved `id_columns`, `date_columns`, `aggregation_columns` and
            `categorical_columns`.
        options: Either `{step_name: {argument: value}}`, or a flat `{argument: value}` applied to
            every step that declares that argument. Both forms can be mixed in one mapping: an entry
            counts as per-step only when it names a step *and* holds a mapping, so a flat
            `lags: [1, 2]` stays a shared argument even though `lags` is also a step name.
    """
    options = dict(options or {})
    per_step = {
        name: value
        for name, value in options.items()
        if name in FEATURE_PREP_STEPS and isinstance(value, dict)
    }
    shared = {name: value for name, value in options.items() if name not in per_step}
    report: list[dict[str, Any]] = []

    for name in steps:
        if name not in FEATURE_PREP_STEPS:
            report.append(
                {"step": name, "status": "unknown step", "features_added": 0, "columns": []}
            )
            continue

        pool: dict[str, Any] = {
            **columns,
            "columns": columns["aggregation_columns"],
            "reference_date": reference_date,
            **shared,
            **(per_step.get(name) or {}),
        }
        try:
            produced, added = _run_step(name, rows, pool)
        except Exception as error:  # a bad argument from a model shouldn't end the whole run
            logger.exception(f"Feature prep step {name} failed.")
            report.append(
                {"step": name, "status": f"failed: {error}", "features_added": 0, "columns": []}
            )
            continue

        if rows and not produced:  # a grain-changing step that found nothing to key on
            report.append(
                {
                    "step": name,
                    "status": "produced no rows; the data was left as it was",
                    "features_added": 0,
                    "columns": [],
                }
            )
            continue

        rows = produced
        report.append(
            {
                "step": name,
                "status": "ok" if added else "no features added",
                "features_added": len(added),
                "columns": added,
            }
        )
    return rows, report


# --------------------------------------------------------------------------------------------------
# agent-facing callers
# --------------------------------------------------------------------------------------------------
def feature_prep_caller(
    data: Any = None,
    id_columns: Any = None,
    date_columns: Any = None,
    aggregation_columns: Any = None,
    steps: Any = None,
    options: dict[str, Any] | None = None,
    reference_date: Any = None,
    output_path: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Run feature preparation steps over a dataset and report what was built.

    Args:
        data: Path to the CSV/JSON to build features from, or the rows themselves.
        id_columns: Entity columns to group by. Inferred from the column names when omitted.
        date_columns: Timestamp columns. Inferred from the values when omitted.
        aggregation_columns: Numeric columns to aggregate and transform. Inferred when omitted.
        steps: Step names, run in this order. Defaults to `DEFAULT_STEPS`; call
            `list_feature_prep_steps_caller` for the catalogue.
        options: Per-step arguments, `{step: {argument: value}}`, or flat for every step.
        reference_date: The "now" that recency and window features are measured from.
        output_path: Where to write the prepared dataset. Defaults to a file beside the input.
        limit: How many sample rows to return.

    Returns:
        The resolved columns, one entry per step with the features it added, where the data was
        written, and a small sample. On failure, `{"status": "error", "message": ...}`.
    """
    try:
        rows = load_data(data)
        if not rows:
            return {"status": "error", "message": "the dataset is empty"}

        columns = infer_columns(rows, id_columns, date_columns, aggregation_columns)
        wanted = _as_columns(steps) or list(DEFAULT_STEPS)
        before = set(column_names(rows))

        rows, report = run_steps(rows, wanted, columns, options, reference_date)
        added = [name for name in column_names(rows) if name not in before]
        reshaped = any(step["step"] in RESHAPING_STEPS for step in report)

        destination = output_path or _default_output_path(
            data, "entity_features" if reshaped else "features"
        )
        written = write_data(rows, destination) if destination and rows else None

        return {
            "status": "ok",
            "rows": len(rows),
            "columns": len(column_names(rows)),
            "grain": "one row per entity" if reshaped else "one row per input row",
            "resolved_columns": columns,
            "steps_applied": report,
            "features_added": added,
            "feature_count": len(added),
            "output_path": written,
            "sample": _sample(rows, limit),
            "notes": _notes(report, columns, written, destination),
        }
    except Exception as error:
        logger.exception("feature_prep_caller failed.")
        return {"status": "error", "message": str(error)}


def feature_generation_caller(
    data: Any = None,
    id_columns: Any = None,
    date_columns: Any = None,
    aggregation_columns: Any = None,
    level: str = "row",
    reference_date: Any = None,
    output_path: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Generate a whole bundle of generic features in one call, without naming individual steps.

    Args:
        data: Path to the CSV/JSON to build features from, or the rows themselves.
        id_columns: Entity columns to group by. Inferred when omitted.
        date_columns: Timestamp columns. Inferred when omitted.
        aggregation_columns: Numeric columns to build from. Inferred when omitted.
        level: "row" enriches every row in place; "entity" collapses the log into one row per entity
            with aggregates, trends, windows and RFM; "numeric" works the measurements alone, for a
            table that is already one row per entity; "all" runs row then numeric.
        reference_date: The "now" that recency and window features are measured from.
        output_path: Where to write the generated dataset. Defaults to a file beside the input.
        limit: How many sample rows to return.

    Returns:
        The same envelope `feature_prep_caller` returns.
    """
    bundles = {
        "row": ["generic_row_features"],
        "entity": ["generic_entity_features"],
        "numeric": ["generic_numeric_features"],
        "all": ["generic_row_features", "generic_numeric_features"],
    }
    if level not in bundles:
        return {
            "status": "error",
            "message": f"unknown level {level!r}; expected one of {sorted(bundles)}",
        }

    return feature_prep_caller(
        data=data,
        id_columns=id_columns,
        date_columns=date_columns,
        aggregation_columns=aggregation_columns,
        steps=bundles[level],
        reference_date=reference_date,
        output_path=output_path or _default_output_path(data, f"{level}_features"),
        limit=limit,
    )


def list_feature_prep_steps_caller(step: str | None = None) -> dict[str, Any]:
    """List every feature preparation step, what it does and what it takes.

    Args:
        step: Restrict the listing to one step.

    Returns:
        One entry per step: its name, its purpose, whether it changes the grain of the data, and its
        arguments with their defaults.
    """
    names = [step] if step else list(FEATURE_PREP_STEPS)
    unknown = [name for name in names if name not in FEATURE_PREP_STEPS]
    if unknown:
        return {
            "status": "error",
            "message": f"no such step {unknown[0]!r}",
            "steps": list(FEATURE_PREP_STEPS),
        }

    return {
        "status": "ok",
        "default_steps": list(DEFAULT_STEPS),
        "aggregation_functions": list(AGGREGATION_FUNCTIONS),
        "steps": [_describe_step(name) for name in names],
    }


def _describe_step(name: str) -> dict[str, Any]:
    function = FEATURE_PREP_STEPS[name]
    documentation = (function.__doc__ or "").strip().split("\n\n")[0]
    arguments = [
        {
            "name": parameter.name,
            "default": _describe_default(parameter.default),
        }
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name not in ("rows", "data")  # the table itself, however the step names it
    ]
    return {
        "step": name,
        "description": " ".join(documentation.split()),
        "grain": "one row per entity" if name in RESHAPING_STEPS else "unchanged",
        "arguments": arguments,
    }


def _describe_default(default: Any) -> Any:
    if default is inspect.Parameter.empty:
        return None
    if isinstance(default, tuple):
        return list(default)
    return default


def _sample(rows: Table, limit: int, max_columns: int = 60) -> list[Row]:
    """A few rows, narrowed to `max_columns`, so a wide table doesn't fill the model's context."""
    taken = rows[: max(int(limit), 0)]
    return [
        {key: _for_csv(value) for key, value in list(row.items())[:max_columns]} for row in taken
    ]


def _notes(
    report: Sequence[dict[str, Any]],
    columns: dict[str, list[str]],
    written: str | None,
    destination: str | None,
) -> list[str]:
    """The things worth telling the model without making it read the whole report."""
    notes: list[str] = []
    if not columns["id_columns"]:
        notes.append(
            "No id columns were found or given; per-entity steps aggregated over all rows."
        )
    if not columns["date_columns"]:
        notes.append("No date columns were found or given; time-based steps were skipped.")
    if not columns["aggregation_columns"]:
        notes.append(
            "No numeric columns were found or given; measurement steps had nothing to build."
        )

    quiet = [step["step"] for step in report if not step["features_added"]]
    if quiet:
        notes.append(f"These steps added nothing: {', '.join(quiet)}.")

    notes.extend(_grain_notes(report))
    if not written and not destination:
        notes.append("Nothing was written: pass output_path to keep the result for the next tool.")
    return notes


def _grain_notes(report: Sequence[dict[str, Any]]) -> list[str]:
    """Warn when a grain-changing step threw away the row-level work that ran before it.

    `entity_table`, `rfm` and `generic_entity_features` each build a fresh table keyed by entity, so
    features added to the original rows earlier in the same call are not carried into it — a model
    chaining steps needs telling, or it will report features the output doesn't have.
    """
    reshaping = [
        position for position, step in enumerate(report) if step["step"] in RESHAPING_STEPS
    ]
    if not reshaping:
        return []

    notes = []
    dropped = [step["step"] for step in report[: reshaping[0]] if step["features_added"]]
    if dropped:
        notes.append(
            f"{report[reshaping[0]]['step']} rebuilt the table as one row per entity, so the "
            f"row-level features from {', '.join(dropped)} are not in the output; run those in "
            f"their own call if you need them."
        )
    if len(reshaping) > 1:
        notes.append(
            "More than one grain-changing step ran in the same call, and each rebuilds the table "
            "from the rows it was given. Run entity_table, rfm and generic_entity_features in "
            "separate calls and join their outputs on the id columns."
        )
    return notes


# --------------------------------------------------------------------------------------------------
# the steps that register themselves into FEATURE_PREP_STEPS on import
# --------------------------------------------------------------------------------------------------
# Importing the module is the registration: each of these reads the helpers above and appends its own
# steps to the catalogue — `auto_*` for the zero-configuration features, `impute_*` for the
# missing-value fills. The imports sit at the bottom, after everything they read, so the modules
# resolve whichever one the caller imports first.
from feature_engineering.tools import auto_features as _auto_features  # noqa: E402,F401
from feature_engineering.tools import missing as _missing  # noqa: E402,F401

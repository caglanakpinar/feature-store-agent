"""Deciding what a hole in the data means, and filling it so a model can still learn from the row.

The stage before this one reads a dataset correctly and the stage after it builds features. In
between sits the question every column with a gap in it asks: was the value never measured,
measured and lost, or measured as "none"? Nothing here can answer that — only the source can — but
everything here makes the answer explicit, and writes down what it did, instead of leaving a `None`
for a later step to trip over.

Two things separate these imputers from `data_engineer.tools.data_prep.fill_missing`, which puts one
number in a column and is the right tool while the data is still being cleaned:

    the fill can come from context   the entity's own history, the rows either side of it in time,
                                     the rows that look most like it — not only the column's middle
    the fact is kept                 a `{column}_is_missing` flag is written *before* the hole is
                                     filled, because whether a field was filled in is often the
                                     stronger feature, and imputing is exactly what destroys it

Every step takes the rows and returns them with the holes filled, plus the names of the flag columns
it added — the same `(rows, added)` shape every step in `feature_prep.py` returns, so they compose
with those and register into the same catalogue under `impute_*` names:

    from feature_enginering.tools.missing import impute_auto, missing_profile

    profile = missing_profile(rows)                    # what is missing, and what it looks like
    rows, added = impute_auto(rows, id_columns=["customer_id"], date_columns=["event_date"])

    feature_prep_caller(data="events.csv", steps=["impute_auto", "group_aggregates"])

The methods, and what each one reads to decide a fill:

    statistic       the column          one value for the whole column: median, mean, mode, ...
    group           the entity          the entity's own median, the column's when it has none
    forward         the past            the last value this entity actually had
    interpolate     both neighbours     a straight line between the readings either side of the hole
    knn             similar rows        the k rows closest across the other numeric columns
    regression      the other columns   a least-squares fit on them, iterated when asked
    random_sample   the distribution    a value drawn from the ones the column really holds
    category        nothing at all      "missing" as a level of its own, which invents no value
    plan            a previous run      the fills a training run learned, replayed on new rows

The one that matters most is not in that list: **which of them is safe at prediction time.**
`forward` reads only the past, so a row filled by it could have been filled the same way the moment
it arrived. Everything else reads the whole column — a row is filled partly from its own future, and
a model scored on a time-split holdout will flatter itself. On a time-ordered training set, use
`forward`, or learn the fills on the training rows alone (`learn_fill_values`) and replay them with
`impute_plan`. `missing_report_caller` says which case a dataset is in.

Nothing here reads a target column, so no fill can leak a label. Two kinds of column are left out of
every method's default set: a date column, because inventing a timestamp invents when something
happened, and a column with nothing present at all, because filling it produces a constant that then
looks like a feature. Both are reported rather than silently skipped, and naming either one in
`columns` fills it anyway — the rule is a default, not a veto.

Steps overwrite the values in the rows they are handed. The callers write to a new file
(`<source>__imputed.csv` by default) and leave the input untouched.

Run through `feature_prep_caller`, a step's `columns` argument is filled from the aggregation
columns that call resolved, so only the measurements are imputed. Widen it there when that is not
what you want, or use `impute_missing_caller`, whose `columns` defaults to the whole table:

    feature_prep_caller(data="events.csv", steps=["impute_auto"],
                        options={"impute_auto": {"columns": ["amount", "plan"]}})

Register the two agent-facing callers in `agentic_configurations.yaml` alongside the other tools:

    - name: "missing_report_tool"
      description: "Report what is missing in a dataset — the rate per column, which columns go
        missing together, whether the holes look random, and the imputation each column needs."
      caller: "feature_enginering.tools.missing.missing_report_caller"
      args:
        - name: "data"
          type: "str"
          description: "Path to the CSV/JSON to inspect."
          required: false
        - name: "columns"
          type: "list"
          description: "Columns to inspect. Defaults to every column."
          required: false
        - name: "id_columns"
          type: "list"
          description: "Entity columns. Inferred from the column names when omitted."
          required: false
        - name: "date_columns"
          type: "list"
          description: "Timestamp columns. Inferred from the values when omitted."
          required: false

    - name: "impute_missing_tool"
      description: "Fill missing values by one of: auto, statistic, group, forward, interpolate,
        knn, regression, random_sample, category, plan. Flags every hole before filling it."
      caller: "feature_enginering.tools.missing.impute_missing_caller"
      args:
        - name: "data"
          type: "str"
          description: "Path to the CSV/JSON to impute."
          required: false
        - name: "method"
          type: "str"
          description: "Which imputer to run. 'auto' picks one per column and explains why."
          enum: ["auto", "statistic", "group", "forward", "interpolate", "knn", "regression",
                 "random_sample", "category", "plan"]
          required: false
        - name: "columns"
          type: "list"
          description: "Columns to fill. Defaults to every column holding a hole."
          required: false
        - name: "statistic"
          type: "str"
          description: "The value rule for the statistic and group methods."
          enum: ["median", "mean", "mode", "min", "max", "zero", "constant"]
          required: false
        - name: "id_columns"
          type: "list"
          description: "Entity columns the group, forward and interpolate methods fill within."
          required: false
        - name: "date_columns"
          type: "list"
          description: "Timestamp columns that order the forward and interpolate methods."
          required: false
        - name: "indicators"
          type: "bool"
          description: "Write a {column}_is_missing flag before filling. On by default."
          required: false
        - name: "options"
          type: "dict"
          description: "Extra arguments for the chosen method: neighbours, rounds, limit, label,
            direction, seed, values."
          required: false
        - name: "output_path"
          type: "str"
          description: "Where to write the imputed dataset."
          required: false
"""

from __future__ import annotations

import heapq
import inspect
import logging
import math
import random
from collections.abc import Callable, Sequence
from datetime import datetime
from statistics import fmean, median, pstdev
from typing import Any

# The same-package internals every step here shares with feature prep. What counts as missing has to
# be the one definition, or this module fills holes that module cannot see, and the other way
# round.
from feature_enginering.tools.feature_prep import (
    FEATURE_PREP_STEPS,
    Table,
    _as_columns,
    _default_output_path,
    _group_key,
    _grouped,
    _numbers,
    _ordered_by_date,
    _parsed_dates,
    _sample,
    column_kinds,
    column_names,
    infer_columns,
    is_missing,
    load_data,
    to_number,
    write_data,
)

logger = logging.getLogger(__name__)

# The flag written before a hole is filled. Deliberately the name that
# `feature_prep.add_missing_indicators` already uses, so running that step first and an imputer
# second produces one flag between them, not two.
FLAG = "_is_missing"

STATISTICS = ("median", "mean", "mode", "min", "max", "zero", "constant")

MOSTLY_MISSING = 0.6  # past this share of holes, the flag carries more than the filled values do
NOT_RANDOM_GAP = 0.2  # standardised gap above which missingness looks tied to another column
EVIDENCE_COLUMNS = 40  # how many complete columns that evidence is looked for in, per column
REPEATED_ENTITY = 1.5  # rows per entity above which a table is a log rather than one row per thing

DEFAULT_NEIGHBOURS = 5
MAX_DONORS = 2000  # knn donors kept, evenly spaced — the cost is holes x donors x predictors
RIDGE = 1e-6  # ridge added to the normal equations, so a collinear fit resolves instead of failing


# --------------------------------------------------------------------------------------------------
# what is missing: the rates, the patterns, and whether the holes look random
# --------------------------------------------------------------------------------------------------
def missing_counts(rows: Table, columns: Sequence[str] = ()) -> dict[str, int]:
    """How many values each column is missing, listing only the columns missing any."""
    wanted = _as_columns(columns) or column_names(rows)
    counts = {}
    for column in wanted:
        holes = sum(1 for row in rows if is_missing(row.get(column)))
        if holes:
            counts[column] = holes
    return counts


def missing_patterns(
    rows: Table, columns: Sequence[str] = (), top: int = 3
) -> list[dict[str, Any]]:
    """The combinations of columns that go missing together, commonest first.

    This is the part a per-column rate hides. Three columns each missing a tenth of their values is
    one story when the holes are in different rows and quite another when they are the same rows —
    the second means one thing failed, once, and the fill for all three should come from the same
    decision.
    """
    wanted = _as_columns(columns) or column_names(rows)
    tally: dict[tuple[str, ...], int] = {}
    for row in rows:
        holes = tuple(column for column in wanted if is_missing(row.get(column)))
        if holes:
            tally[holes] = tally.get(holes, 0) + 1

    common = sorted(tally.items(), key=lambda item: (-item[1], item[0]))[: max(int(top), 0)]
    return [
        {"columns": list(pattern), "rows": count, "share": count / len(rows) if rows else 0.0}
        for pattern, count in common
    ]


def missing_profile(rows: Table, columns: Sequence[str] = (), patterns: int = 3) -> dict[str, Any]:
    """Describe every hole in the dataset: how many, in which columns, and in which rows together.

    `related_to` is the piece worth reading twice. For each column with holes it names the complete
    numeric column whose values differ most between the rows where this one is missing and the rows
    where it is present, on a standard-deviation scale. A gap there means the missingness is not
    random — the rows with holes in them came from somewhere the complete rows didn't — and that in
    turn means the flag is a feature and the fill is a guess made in a corner of the data.
    """
    wanted = _as_columns(columns) or column_names(rows)
    kinds = column_kinds(rows)
    counts = missing_counts(rows, wanted)
    total = len(rows)

    # Only columns that are themselves complete can serve as evidence: a gap measured against a
    # column with holes of its own would measure the two missingness patterns against each other.
    evidence = [
        column for column in wanted if kinds.get(column) == "numeric" and column not in counts
    ][:EVIDENCE_COLUMNS]

    described: dict[str, dict[str, Any]] = {}
    for column, holes in counts.items():
        related, gap = _missingness_evidence(rows, column, evidence)
        described[column] = {
            "kind": kinds.get(column, "empty"),
            "missing": holes,
            "rate": holes / total if total else 0.0,
            "present": total - holes,
            "distinct": len(
                {_group_key(row.get(column)) for row in rows if not is_missing(row.get(column))}
            ),
            "related_to": related,
            "related_gap": gap,
        }

    complete = sum(1 for row in rows if not any(is_missing(row.get(c)) for c in wanted))
    return {
        "rows": total,
        "columns_checked": len(wanted),
        "columns_with_holes": len(counts),
        "complete_rows": complete,
        "complete_rate": complete / total if total else 0.0,
        "columns": described,
        "patterns": missing_patterns(rows, wanted, patterns),
    }


def _missingness_evidence(
    rows: Table, column: str, evidence: Sequence[str]
) -> tuple[str | None, float | None]:
    """The complete column whose mean moves most between this column's missing and present rows."""
    holes = [row for row in rows if is_missing(row.get(column))]
    present = [row for row in rows if not is_missing(row.get(column))]
    if len(holes) < 2 or len(present) < 2:
        return None, None

    worst: tuple[str | None, float] = (None, 0.0)
    for other in evidence:
        missing_side = _numbers(row.get(other) for row in holes)
        present_side = _numbers(row.get(other) for row in present)
        if len(missing_side) < 2 or len(present_side) < 2:
            continue

        spread = pstdev(missing_side + present_side)
        if not spread:
            continue

        gap = (fmean(missing_side) - fmean(present_side)) / spread
        # Two standard errors as well as a real size, so a handful of missing rows cannot produce a
        # finding: with thirty holes against two hundred rows, a gap of a third of a deviation is
        # what randomness looks like, and reporting that as evidence is worse than saying nothing.
        error = math.sqrt(1 / len(missing_side) + 1 / len(present_side))
        if abs(gap) >= max(NOT_RANDOM_GAP, 2 * error) and abs(gap) > abs(worst[1]):
            worst = (other, gap)

    if worst[0] is None:
        return None, None
    return worst[0], round(worst[1], 3)


# --------------------------------------------------------------------------------------------------
# choosing: the rules `impute_auto` follows, written down where they can be read before they are run
# --------------------------------------------------------------------------------------------------
def suggest_imputations(
    rows: Table,
    columns: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    max_rate: float = MOSTLY_MISSING,
) -> dict[str, dict[str, Any]]:
    """Pick a method for every column with holes, and say why — the plan `impute_auto` executes.

    The rules, in the order they are tried:

        a date column               left alone: a timestamp is when something happened, and filling
                                    one invents the event
        nothing present at all      left alone: there is nothing to fill from, and a constant column
                                    is not data
        a category or a boolean     "missing" as a level of its own, which guesses at nothing
        a measurement, per entity,  forward fill, the only method that reads the past alone, so what
        with a date to order by     it produces at training time it can produce at serving time
        a measurement, per entity   the entity's own median, falling back to the column's
        anything else numeric       the column's median

    Only four methods appear here, and none of them is `knn`, `regression` or `interpolate`. Those
    three can be better and they can be much worse, and which it is depends on things this function
    cannot see — so they stay a choice an agent makes and defends, not a default it inherits.

    Args:
        max_rate: The share of holes above which a column is called out as mostly missing. It is
            still filled: the flag is what carries it, and the fill keeps the column usable.
    """
    kinds = column_kinds(rows)
    keys = [column for column in _as_columns(id_columns) if column in kinds]
    dates = [column for column in _as_columns(date_columns) if column in kinds]
    total = len(rows)
    repeated = bool(keys) and total >= REPEATED_ENTITY * max(len(_grouped(rows, keys)), 1)

    plan: dict[str, dict[str, Any]] = {}
    for column, holes in missing_counts(rows, columns).items():
        kind = kinds.get(column, "empty")
        rate = holes / total if total else 0.0

        if kind == "date" or column in dates:
            strategy, reason = "leave", "a date: filling it would invent when something happened"
        elif kind == "empty":
            strategy, reason = "leave", "nothing present to fill from; drop the column instead"
        elif kind in ("categorical", "boolean"):
            strategy, reason = "category", "a category: the hole becomes a level of its own"
        elif repeated and dates:
            strategy, reason = "forward", "a measurement per entity over time: carry the last one"
        elif repeated:
            strategy, reason = "group", "several rows per entity: the entity's own middle"
        else:
            strategy, reason = "statistic", "one row per thing: the column's median"

        if rate > max_rate and strategy != "leave":
            reason += f"; {rate:.0%} missing, so the flag carries more than the fill does"

        plan[column] = {
            "strategy": strategy,
            "reason": reason,
            "kind": kind,
            "missing": holes,
            "rate": rate,
        }
    return plan


# --------------------------------------------------------------------------------------------------
# the shared pieces: which columns, the flag, and the one value a column is filled with
# --------------------------------------------------------------------------------------------------
def _wanted(rows: Table, columns: Sequence[str] = ()) -> list[str]:
    """The columns asked for that both exist and have a hole in them.

    Given no columns this is every column with a hole *except* the two kinds no method should fill
    on its own initiative: a date column, where a fill invents when something happened, and a column
    with nothing in it at all, where a fill is a constant rather than data. Naming either of them
    explicitly fills it — the rule is a default, not a veto.
    """
    kinds = column_kinds(rows)
    asked = _as_columns(columns)
    unknown = [column for column in asked if column not in kinds]
    if unknown:
        logger.warning(f"No column(s) {unknown} in the data; they cannot be imputed.")

    candidates = (
        [column for column in asked if column in kinds]
        if asked
        else [column for column, kind in kinds.items() if kind not in ("date", "empty")]
    )
    holed = missing_counts(rows, candidates)
    return [column for column in candidates if column in holed]


def _flag_missing(rows: Table, columns: Sequence[str]) -> list[str]:
    """Write `{column}_is_missing` for each column, before anything fills it.

    A flag that is already there is left as it is, so an earlier `missing_indicators` step and a
    later imputer produce one column between them rather than one each — and the earlier one, which
    was written when the holes were still there, is the one that survives.
    """
    existing = set(column_names(rows))
    added: list[str] = []

    for column in columns:
        name = f"{column}{FLAG}"
        if name in existing:
            continue

        for row in rows:
            row[name] = int(is_missing(row.get(column)))
        added.append(name)
    return added


def _statistic(values: Sequence[Any], statistic: str, value: Any = None) -> Any:
    """The value a column is filled with, or None when it has nothing to compute one from."""
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; expected one of {STATISTICS}")
    if statistic == "constant":
        return value
    if statistic == "zero":
        return 0.0

    present = [item for item in values if not is_missing(item)]
    if not present:
        return None
    if statistic == "mode":
        return _mode(present)

    numbers = _numbers(present)
    if (
        not numbers
    ):  # a numeric rule asked of a text column: its commonest value is the honest answer
        logger.debug(f"No numbers to take a {statistic} of; using the most common value instead.")
        return _mode(present)

    if statistic == "mean":
        return fmean(numbers)
    if statistic == "median":
        return median(numbers)
    if statistic == "min":
        return min(numbers)
    return max(numbers)


def _mode(present: Sequence[Any]) -> Any:
    """The commonest value, counted on the normalised key but returned as it was written."""
    counts: dict[str, int] = {}
    originals: dict[str, Any] = {}
    for value in present:
        key = _group_key(value)
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, value)

    best = max(counts, key=lambda key: (counts[key], -list(originals).index(key)))
    return originals[best]


def _sequences(
    rows: Table, id_columns: Sequence[str], date_columns: Sequence[str]
) -> tuple[list[list[int]], list[datetime | None]]:
    """Each entity's row indexes, oldest first, and the dates they were ordered by.

    With no id columns every row belongs to one sequence, which is right for a single time series
    and wrong for a table of unrelated things — which is why nothing picks a sequence-following
    method on its own unless an id column says the rows repeat.
    """
    dates = _parsed_dates(rows, _as_columns(date_columns)[0]) if _as_columns(date_columns) else []
    groups = _grouped(rows, _as_columns(id_columns))
    if not dates:
        return [list(indexes) for indexes in groups.values()], []
    return [_ordered_by_date(indexes, dates) for indexes in groups.values()], dates


# --------------------------------------------------------------------------------------------------
# imputers — one value for the column, or one per entity
# --------------------------------------------------------------------------------------------------
def learn_fill_values(
    rows: Table, columns: Sequence[str] = (), statistic: str = "median", value: Any = None
) -> dict[str, Any]:
    """Work out what each column would be filled with, without filling anything.

    This is the half of `impute_statistic` worth keeping. Learn the fills on the training rows, hand
    the mapping to `impute_plan`, and every later batch — the holdout, tomorrow's scoring run — is
    filled with the values the model was fitted around, instead of with its own median.
    """
    fills: dict[str, Any] = {}
    for column in _wanted(rows, columns):
        replacement = _statistic([row.get(column) for row in rows], statistic, value)
        if replacement is None:
            logger.warning(f"{column!r} has no value to fill from; leaving its holes as they are.")
            continue
        fills[column] = replacement
    return fills


def impute_plan(
    rows: Table, values: dict[str, Any] | None = None, indicators: bool = True
) -> tuple[Table, list[str]]:
    """Fill from fills that were learned somewhere else — the serving half of `learn_fill_values`.

    Args:
        values: `{column: value}`, as `learn_fill_values` returns and the caller reports back.
        indicators: Write `{column}_is_missing` before filling.
    """
    fills = dict(values or {})
    if not fills:
        logger.warning("impute_plan was given no values; nothing was filled.")
        return rows, []

    added = _flag_missing(rows, _wanted(rows, list(fills))) if indicators else []
    for row in rows:
        for column, value in fills.items():
            if is_missing(row.get(column)):
                row[column] = value
    return rows, added


def impute_statistic(
    rows: Table,
    columns: Sequence[str] = (),
    statistic: str = "median",
    value: Any = None,
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Fill every hole in a column with one value computed from that column.

    The median is the default because it is the one that does not move: a mean is dragged by the
    same long tail that made the column worth imputing carefully, and a column filled with a dragged
    mean has a spike in it where no data is.

    Args:
        columns: Columns to fill. Defaults to every column with a hole.
        statistic: "median", "mean", "mode", "min", "max", "zero", or "constant" with `value`.
        value: What to fill with, for the "constant" statistic.
        indicators: Write `{column}_is_missing` before filling.
    """
    wanted = _wanted(rows, columns)
    added = _flag_missing(rows, wanted) if indicators else []

    fills = learn_fill_values(rows, wanted, statistic, value)
    for row in rows:
        for column, replacement in fills.items():
            if is_missing(row.get(column)):
                row[column] = replacement
    return rows, added


def impute_group(
    rows: Table,
    columns: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    statistic: str = "median",
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Fill from the entity's own values, and from the whole column only where it has none.

    A customer with three readings and a gap has already told you what their fourth looks like, and
    it is not the population median. This is the fill to reach for the moment the rows repeat over
    something, and the reason the id columns are worth getting right before imputing at all.

    Args:
        id_columns: The entity to fill within. Without one this is `impute_statistic`.
        statistic: The value rule, applied inside the entity and again as the fallback.
    """
    wanted = _wanted(rows, columns)
    keys = _as_columns(id_columns)
    if not keys:
        logger.warning("impute_group has no id columns to group by; filling from the whole column.")
        return impute_statistic(rows, wanted, statistic, indicators=indicators)

    added = _flag_missing(rows, wanted) if indicators else []
    groups = _grouped(rows, keys)
    fallbacks = learn_fill_values(rows, wanted, statistic)

    for column in wanted:
        borrowed = 0
        for indexes in groups.values():
            holes = [index for index in indexes if is_missing(rows[index].get(column))]
            if not holes:
                continue

            replacement = _statistic([rows[index].get(column) for index in indexes], statistic)
            if replacement is None:  # the entity has nothing anywhere; the column decides instead
                replacement = fallbacks.get(column)
                borrowed += len(holes)
            if replacement is None:
                continue

            for index in holes:
                rows[index][column] = replacement

        if borrowed:
            logger.info(
                f"impute_group: {borrowed} row(s) had no {column!r} anywhere in their entity; "
                f"filled from the column's {statistic}."
            )
    return rows, added


# --------------------------------------------------------------------------------------------------
# imputers — along a sequence: the past, and the two readings either side
# --------------------------------------------------------------------------------------------------
def impute_forward(
    rows: Table,
    columns: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    direction: str = "forward",
    limit: int | None = None,
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Carry the entity's last known value forward over the holes that follow it.

    The one method here that a serving pipeline can reproduce: filling a row with what this entity
    was last seen to be is something you can do the moment the row arrives, which is not true of
    anything computed across the whole column. It is also the honest reading of a slow-moving
    measurement — a subscription tier that was "premium" in March is premium in April unless
    something says otherwise.

    A hole before the entity's first reading has no past to carry, and stays a hole unless
    `direction` says to reach backwards for it. Reaching backwards fills a row with a value from its
    own future, which is fine for a table built at one point in time and is leakage on a
    time-ordered training set — so it is off by default.

    Args:
        direction: "forward", "backward", or "both".
        limit: How many consecutive rows one value may be carried over. None carries it as far as
            it goes, which on a sparse column means one old reading fills the rest of the history.
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError(f"direction is {direction!r}; expected 'forward', 'backward' or 'both'.")

    wanted = _wanted(rows, columns)
    added = _flag_missing(rows, wanted) if indicators else []
    sequences, _ = _sequences(rows, id_columns, date_columns)
    if not _as_columns(date_columns):
        logger.info("impute_forward has no date column; carrying values in the row order given.")

    for column in wanted:
        for order in sequences:
            if direction in ("forward", "both"):
                _carry(rows, column, order, limit)
            if direction in ("backward", "both"):
                _carry(rows, column, list(reversed(order)), limit)
    return rows, added


def _carry(rows: Table, column: str, order: Sequence[int], limit: int | None) -> int:
    """Carry the last present value along `order`, at most `limit` rows past where it was seen."""
    filled = 0
    carried: Any = None
    distance = 0

    for index in order:
        value = rows[index].get(column)
        if not is_missing(value):
            carried, distance = value, 0
            continue
        if carried is None:
            continue  # nothing seen yet in this direction

        distance += 1
        if limit is not None and distance > limit:
            continue
        rows[index][column] = carried
        filled += 1
    return filled


def impute_interpolate(
    rows: Table,
    columns: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    method: str = "time",
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Draw a straight line between the readings either side of a hole and read the value off it.

    For a measurement that moves continuously — a meter, a balance, a temperature — this is the fill
    that keeps the shape of the series, where carrying the last value forward leaves it in steps.
    For a measurement that jumps, it invents a smoothness the data does not have.

    Only holes with a reading on both sides are filled: there is no extrapolation past the entity's
    first and last readings, so a leading gap stays a gap. Follow this with `impute_forward` when
    those matter.

    Args:
        method: "time" spaces the line by the date gaps, so a hole three days after one reading and
            one day before the next lands three quarters of the way along. "linear" spaces it by row
            position, which is what "time" falls back to when there are no dates to weigh it by.
    """
    if method not in ("time", "linear"):
        raise ValueError(f"method is {method!r}; expected 'time' or 'linear'.")

    kinds = column_kinds(rows)
    wanted = _wanted(rows, columns)
    skipped = [column for column in wanted if kinds.get(column) != "numeric"]
    if skipped:
        logger.info(f"impute_interpolate: {skipped} are not numeric; there is no line to draw.")

    numeric = [column for column in wanted if kinds.get(column) == "numeric"]
    added = _flag_missing(rows, numeric) if indicators else []
    sequences, dates = _sequences(rows, id_columns, date_columns)

    for column in numeric:
        for order in sequences:
            anchors = [
                position
                for position, index in enumerate(order)
                if to_number(rows[index].get(column)) is not None
            ]
            for left, right in zip(anchors, anchors[1:]):
                if right - left < 2:
                    continue

                start = to_number(rows[order[left]].get(column))
                end = to_number(rows[order[right]].get(column))
                if start is None or end is None:  # anchors are numbers by construction
                    continue
                for position in range(left + 1, right):
                    index = order[position]
                    if not is_missing(rows[index].get(column)):
                        continue  # present but unreadable: not a hole, and not ours to overwrite
                    fraction = _fraction(order, dates, left, position, right, method)
                    rows[index][column] = start + (end - start) * fraction
    return rows, added


def _fraction(
    order: Sequence[int],
    dates: Sequence[datetime | None],
    left: int,
    position: int,
    right: int,
    method: str,
) -> float:
    """How far along the line from the left anchor to the right one this hole sits."""
    if method == "time" and dates:
        start, hole, end = (dates[order[point]] for point in (left, position, right))
        if start and hole and end and end > start:
            return (hole - start).total_seconds() / (end - start).total_seconds()
    return (position - left) / (right - left)


# --------------------------------------------------------------------------------------------------
# imputers — from the other columns: similar rows, and a fit on the rest of the table
# --------------------------------------------------------------------------------------------------
def impute_knn(
    rows: Table,
    columns: Sequence[str] = (),
    feature_columns: Sequence[str] = (),
    neighbours: int = DEFAULT_NEIGHBOURS,
    max_donors: int = MAX_DONORS,
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Fill a hole with what the rows most like this one hold in that column.

    Every numeric column is standardised first, so distance is not decided by whichever column
    happens to be measured in thousands, and two rows are compared only on the columns they both
    have — a donor with its own holes elsewhere is still a donor. Numeric targets take the
    distance-weighted mean of the neighbours; a category takes their weighted vote.

    Worth the cost when the columns genuinely predict each other and the holes are few. It reads the
    whole table, so like every method except `impute_forward` it is not point-in-time safe, and it
    pulls filled values toward the middle of whatever neighbourhood it found.

    Args:
        feature_columns: The columns distance is measured across. Defaults to every numeric column
            except the one being filled.
        neighbours: How many rows vote.
        max_donors: How many complete rows are considered, taken evenly spaced across the table. The
            work is holes x donors x columns, so this is the knob that keeps a big table finishing.
    """
    kinds = column_kinds(rows)
    wanted = _wanted(rows, columns)
    added = _flag_missing(rows, wanted) if indicators else []

    predictors = [column for column in _as_columns(feature_columns) if column in kinds] or [
        column for column, kind in kinds.items() if kind == "numeric"
    ]
    scaled = _standardised(rows, predictors)
    count = max(int(neighbours), 1)

    for column in wanted:
        numeric = kinds.get(column) == "numeric"
        usable = [
            position
            for position, row in enumerate(rows)
            if scaled[position].keys() - {column} and not is_missing(row.get(column))
        ]
        holes = [position for position, row in enumerate(rows) if is_missing(row.get(column))]
        if not usable:
            logger.warning(f"impute_knn: nothing to compare {column!r} against; leaving its holes.")
            continue

        donors = _stride(usable, max_donors)
        if len(donors) * len(holes) > 5_000_000:
            logger.warning(
                f"impute_knn: {column!r} needs {len(donors) * len(holes):,} comparisons; "
                f"lower max_donors if this is slow."
            )

        stranded = []
        for index in holes:
            near = _nearest(scaled, index, donors, column, count)
            if not near:
                stranded.append(index)
                continue
            rows[index][column] = _combine(
                [rows[donor].get(column) for _, donor in near],
                [1.0 / (distance + 1e-9) for distance, _ in near],
                numeric,
            )

        if stranded:  # no column in common with any donor: the column's own middle is what is left
            replacement = _statistic(
                [row.get(column) for row in rows], "median" if numeric else "mode"
            )
            for index in stranded:
                rows[index][column] = replacement
            logger.info(
                f"impute_knn: {len(stranded)} row(s) had nothing in common with any donor for "
                f"{column!r}; filled them from the column instead."
            )
    return rows, added


def _standardised(rows: Table, predictors: Sequence[str]) -> list[dict[str, float]]:
    """Each row's numeric columns as z-scores, holding only the columns that row actually has."""
    scaled: list[dict[str, float]] = [{} for _ in rows]
    for column in predictors:
        values = [to_number(row.get(column)) for row in rows]
        numbers = [value for value in values if value is not None]
        if len(numbers) < 2:
            continue

        centre = fmean(numbers)
        spread = pstdev(numbers) or 1.0
        for position, value in enumerate(values):
            if value is not None:
                scaled[position][column] = (value - centre) / spread
    return scaled


def _nearest(
    scaled: Sequence[dict[str, float]],
    index: int,
    donors: Sequence[int],
    column: str,
    count: int,
) -> list[tuple[float, int]]:
    """The `count` donors closest to `index`, as `(distance, donor)`, ignoring `column` itself."""
    here = {name: value for name, value in scaled[index].items() if name != column}
    if not here:
        return []

    distances = []
    for donor in donors:
        there = scaled[donor]
        shared = here.keys() & there.keys()
        if not shared:
            continue
        total = sum((here[name] - there[name]) ** 2 for name in shared)
        distances.append((math.sqrt(total / len(shared)), donor))  # per shared column, so partial
    return heapq.nsmallest(count, distances)  # overlaps stay comparable


def _stride(items: Sequence[int], cap: int) -> list[int]:
    """Thin a list to `cap` entries, evenly spaced — deterministic, where a random sample is not."""
    if cap <= 0 or len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[int(position * step)] for position in range(cap)]


def _combine(values: Sequence[Any], weights: Sequence[float], numeric: bool) -> Any:
    """One value out of the neighbours': their weighted mean, or their weighted vote."""
    if numeric:
        pairs = [
            (number, weight)
            for number, weight in (
                (to_number(value), weight) for value, weight in zip(values, weights)
            )
            if number is not None
        ]
        total = sum(weight for _, weight in pairs)
        return sum(number * weight for number, weight in pairs) / total if total else None

    tally: dict[str, float] = {}
    originals: dict[str, Any] = {}
    for value, weight in zip(values, weights):
        key = _group_key(value)
        tally[key] = tally.get(key, 0.0) + weight
        originals.setdefault(key, value)
    return originals[max(tally, key=lambda key: tally[key])]


def impute_regression(
    rows: Table,
    columns: Sequence[str] = (),
    feature_columns: Sequence[str] = (),
    rounds: int = 1,
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Predict each missing measurement from the other numeric columns, by least squares.

    The columns are seeded with their medians so a fit has something to read, then each target is
    regressed on the rest and its holes replaced by the fit's predictions. With `rounds` above one
    the seeds are replaced by the previous round's predictions and the whole thing runs again, which
    is the chained-equations idea in its simplest form: the second round's fit reads imputed values
    rather than medians, and the columns stop pulling each other toward the middle.

    Predictions are clipped to the range the column was actually seen in, because a linear fit
    extrapolates cheerfully into ages of minus four. Where a column cannot be fitted at all — too
    few complete rows, or predictors that are copies of each other — its median stands, and it is
    logged.

    Args:
        feature_columns: The columns to fit on. Defaults to every numeric column but the target.
        rounds: How many times to sweep every target. One is a plain regression on median-seeded
            inputs; two or three is usually where it stops moving.
    """
    kinds = column_kinds(rows)
    wanted = [column for column in _wanted(rows, columns) if kinds.get(column) == "numeric"]
    if not wanted:
        logger.info("impute_regression found no numeric column with holes; nothing to fit.")
        return rows, []

    added = _flag_missing(rows, wanted) if indicators else []
    predictors = [column for column in _as_columns(feature_columns) if column in kinds] or [
        column for column, kind in kinds.items() if kind == "numeric"
    ]

    # One working copy of the numeric table: holes seeded with the median, so every fit sees a full
    # matrix. Only the seeded positions are written back at the end, so a present value is never
    # replaced by a prediction of itself.
    working: dict[str, list[float]] = {}
    observed: dict[str, list[bool]] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for column in dict.fromkeys([*predictors, *wanted]):
        values = [to_number(row.get(column)) for row in rows]
        numbers = [value for value in values if value is not None]
        if not numbers:
            continue
        middle = median(numbers)
        working[column] = [middle if value is None else value for value in values]
        observed[column] = [value is not None for value in values]
        bounds[column] = (min(numbers), max(numbers))

    fitted = [column for column in wanted if column in working]
    for _ in range(max(int(rounds), 1)):
        for column in fitted:
            inputs = [name for name in predictors if name in working and name != column]
            if not inputs:
                continue
            _regress_into(working, observed, bounds, column, inputs)

    for column in fitted:
        for position, row in enumerate(rows):
            if is_missing(row.get(column)):
                row[column] = working[column][position]
    return rows, added


def _regress_into(
    working: dict[str, list[float]],
    observed: dict[str, list[bool]],
    bounds: dict[str, tuple[float, float]],
    column: str,
    inputs: Sequence[str],
) -> None:
    """Fit `column` on `inputs` over the rows that have it, then predict the rows that do not."""
    training = [position for position, seen in enumerate(observed[column]) if seen]
    if len(training) < len(inputs) + 2:
        logger.info(
            f"impute_regression: {column!r} has {len(training)} row(s) to fit {len(inputs)} "
            f"column(s) on; its median stands."
        )
        return

    design = [[1.0, *(working[name][position] for name in inputs)] for position in training]
    coefficients = _least_squares(design, [working[column][position] for position in training])
    if coefficients is None:
        logger.info(f"impute_regression: {column!r} would not resolve to a fit; its median stands.")
        return

    low, high = bounds[column]
    for position, seen in enumerate(observed[column]):
        if seen:
            continue
        prediction = coefficients[0] + sum(
            coefficient * working[name][position]
            for coefficient, name in zip(coefficients[1:], inputs)
        )
        working[column][position] = min(max(prediction, low), high)


def _least_squares(
    design: Sequence[Sequence[float]], targets: Sequence[float], ridge: float = RIDGE
) -> list[float] | None:
    """Solve `(XᵀX + λI)β = Xᵀy`, the normal equations, or None when the fit will not resolve."""
    width = len(design[0])
    normal = [
        [sum(row[left] * row[right] for row in design) for right in range(width)]
        for left in range(width)
    ]
    moment = [
        sum(row[left] * target for row, target in zip(design, targets)) for left in range(width)
    ]

    scale = sum(normal[position][position] for position in range(width)) / width or 1.0
    for position in range(width):
        normal[position][position] += ridge * scale  # keeps collinear columns from dividing by zero
    return _solve(normal, moment)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None when the matrix is singular."""
    size = len(vector)
    augmented = [[*row, vector[position]] for position, row in enumerate(matrix)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            if not factor:
                continue
            for position in range(column, size + 1):
                augmented[row][position] -= factor * augmented[column][position]

    solution = [0.0] * size
    for row in reversed(range(size)):
        total = augmented[row][size] - sum(
            augmented[row][position] * solution[position] for position in range(row + 1, size)
        )
        solution[row] = total / augmented[row][row]

    return solution if all(math.isfinite(value) for value in solution) else None


# --------------------------------------------------------------------------------------------------
# imputers — the two that refuse to invent a number
# --------------------------------------------------------------------------------------------------
def impute_random_sample(
    rows: Table, columns: Sequence[str] = (), seed: int = 0, indicators: bool = True
) -> tuple[Table, list[str]]:
    """Fill each hole with a value drawn from the ones the column actually holds.

    Every other numeric method here puts the same value in every hole, which narrows the column: a
    tenth of the rows land exactly on the median, and any model reading the spread reads it wrong.
    Drawing keeps the distribution's shape at the price of putting a value in a row with no claim
    to it — worth it when the spread of a column is itself a feature, and not otherwise.

    Args:
        seed: Fixed, so a run repeats. A different seed is a different dataset, which is the reason
            this method cannot be replayed at serving time the way a learned fill can.
    """
    wanted = _wanted(rows, columns)
    added = _flag_missing(rows, wanted) if indicators else []
    generator = random.Random(seed)

    for column in wanted:
        present = [row.get(column) for row in rows if not is_missing(row.get(column))]
        if not present:
            logger.warning(f"{column!r} has nothing to draw from; leaving its holes as they are.")
            continue
        for row in rows:
            if is_missing(row.get(column)):
                row[column] = generator.choice(present)
    return rows, added


def impute_category(
    rows: Table, columns: Sequence[str] = (), label: str = "missing", indicators: bool = True
) -> tuple[Table, list[str]]:
    """Make the hole a category of its own, rather than guessing which category it was.

    For anything that isn't a number this is usually the right answer and always the honest one. The
    commonest value is a guess that quietly grows whichever level is already largest; "missing" is a
    level the encodings downstream will count, rank and one-hot exactly like any other, and if it
    predicts something, that is a real finding about the rows that had holes in them.

    Numeric columns are skipped: a number column holding the string "missing" is no longer numeric.

    Args:
        label: The level to write. Renamed when the column already contains it.
    """
    kinds = column_kinds(rows)
    wanted = _wanted(rows, columns)
    skipped = [column for column in wanted if kinds.get(column) == "numeric"]
    if skipped:
        logger.info(f"impute_category: {skipped} are numeric; a label would make them text.")

    fillable = [column for column in wanted if kinds.get(column) != "numeric"]
    added = _flag_missing(rows, fillable) if indicators else []

    for column in fillable:
        level = label
        if any(_group_key(row.get(column)) == label for row in rows):
            level = f"{label}_value"  # the column already uses the word; don't merge two meanings
            logger.info(f"impute_category: {column!r} already holds {label!r}; using {level!r}.")

        for row in rows:
            if is_missing(row.get(column)):
                row[column] = level
    return rows, added


# --------------------------------------------------------------------------------------------------
# the dispatcher: one method per column, chosen by the rules and then run
# --------------------------------------------------------------------------------------------------
def impute_auto(
    rows: Table,
    columns: Sequence[str] = (),
    id_columns: Sequence[str] = (),
    date_columns: Sequence[str] = (),
    indicators: bool = True,
) -> tuple[Table, list[str]]:
    """Run `suggest_imputations`' plan: the right method per column, and a median where one is left.

    The passes run in a fixed order — categories, then carried values, then per-entity middles, then
    the column's own — and a final sweep fills anything the earlier passes could not reach, which in
    practice is the leading hole in an entity's history that had no past to carry. Columns the plan
    left alone stay alone; they are the date columns and the empty ones, and both are reported by
    `missing_report_caller` rather than filled here.
    """
    plan = suggest_imputations(rows, columns, id_columns, date_columns)
    added: list[str] = []

    for strategy in ("category", "forward", "group", "statistic"):
        chosen = [column for column, entry in plan.items() if entry["strategy"] == strategy]
        if not chosen:
            continue

        if strategy == "category":
            _, names = impute_category(rows, chosen, indicators=indicators)
        elif strategy == "forward":
            _, names = impute_forward(rows, chosen, id_columns, date_columns, indicators=indicators)
        elif strategy == "group":
            _, names = impute_group(rows, chosen, id_columns, indicators=indicators)
        else:
            _, names = impute_statistic(rows, chosen, indicators=indicators)
        added.extend(names)

    kinds = column_kinds(rows)
    remaining = [
        column
        for column, entry in plan.items()
        if entry["strategy"] != "leave" and any(is_missing(row.get(column)) for row in rows)
    ]
    if remaining:
        logger.info(f"impute_auto: {remaining} still had holes after their method; filled them.")
        numbers = [column for column in remaining if kinds.get(column) == "numeric"]
        rest = [column for column in remaining if kinds.get(column) != "numeric"]
        for chosen, step in ((numbers, impute_statistic), (rest, impute_category)):
            if chosen:
                _, names = step(rows, chosen, indicators=indicators)
                added.extend(names)

    return rows, added


IMPUTERS: dict[str, Callable[..., tuple[Table, list[str]]]] = {
    "auto": impute_auto,
    "statistic": impute_statistic,
    "group": impute_group,
    "forward": impute_forward,
    "interpolate": impute_interpolate,
    "knn": impute_knn,
    "regression": impute_regression,
    "random_sample": impute_random_sample,
    "category": impute_category,
    "plan": impute_plan,
}

# Registered from here, the same way `auto_features` registers itself: by the time this runs
# `feature_prep` has finished executing, whichever of the two modules the caller imported first.
FEATURE_PREP_STEPS.update({f"impute_{name}": function for name, function in IMPUTERS.items()})


# --------------------------------------------------------------------------------------------------
# agent-facing callers
# --------------------------------------------------------------------------------------------------
def missing_report_caller(
    data: Any = None,
    columns: Any = None,
    id_columns: Any = None,
    date_columns: Any = None,
    patterns: int = 3,
) -> dict[str, Any]:
    """Report what a dataset is missing and what each column needs, without changing anything.

    Args:
        data: Path to the CSV/JSON to inspect, or the rows themselves.
        columns: Columns to inspect. Defaults to every column.
        id_columns: Entity columns. Inferred from the column names when omitted.
        date_columns: Timestamp columns. Inferred from the values when omitted.
        patterns: How many missing-together combinations to report.

    Returns:
        The rate per column, the rows that are complete, the columns that go missing together, and a
        per-column recommendation with the reason behind it. On failure,
        `{"status": "error", "message": ...}`.
    """
    try:
        rows = load_data(data)
        if not rows:
            return {"status": "error", "message": "the dataset is empty"}

        resolved = infer_columns(rows, id_columns, date_columns, None)
        profile = missing_profile(rows, _as_columns(columns), patterns)
        plan = suggest_imputations(
            rows, _as_columns(columns), resolved["id_columns"], resolved["date_columns"]
        )

        return {
            "status": "ok",
            "rows": profile["rows"],
            "columns": profile["columns_checked"],
            "columns_with_holes": profile["columns_with_holes"],
            "complete_rows": profile["complete_rows"],
            "complete_rate": round(profile["complete_rate"], 4),
            "resolved_columns": resolved,
            "missing": profile["columns"],
            "patterns": profile["patterns"],
            "recommended": plan,
            "notes": _report_notes(profile, plan),
        }
    except Exception as error:
        logger.exception("missing_report_caller failed.")
        return {"status": "error", "message": str(error)}


def impute_missing_caller(
    data: Any = None,
    method: str = "auto",
    columns: Any = None,
    statistic: str = "median",
    id_columns: Any = None,
    date_columns: Any = None,
    value: Any = None,
    indicators: bool = True,
    options: dict[str, Any] | None = None,
    output_path: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Fill a dataset's missing values and report, per column, what was filled and with what.

    Args:
        data: Path to the CSV/JSON to impute, or the rows themselves.
        method: "auto", "statistic", "group", "forward", "interpolate", "knn", "regression",
            "random_sample", "category", or "plan". Call `missing_report_tool` first — it recommends
            one per column and says why.
        columns: Columns to fill. Defaults to every column with a hole.
        statistic: The value rule for the statistic and group methods.
        id_columns: Entity columns. Inferred when omitted.
        date_columns: Timestamp columns. Inferred when omitted.
        value: What to fill with, for `statistic="constant"`.
        indicators: Write `{column}_is_missing` before filling. Leave this on unless something
            downstream cannot take the extra columns: it is the only record that a value was ever
            missing, and it is often the better feature of the two.
        options: Extra arguments for the chosen method — `neighbours`, `max_donors`,
            `feature_columns`, `rounds`, `direction`, `limit`, `label`, `seed`, `values`.
        output_path: Where to write the imputed dataset. Defaults to a file beside the input.
        limit: How many sample rows to return.

    Returns:
        Per column: its kind, how many holes it had, how many were filled and how many are left, and
        the method that filled them. Plus `fill_values` where the method learned one value per
        column — replay those on the next batch with `method="plan"` rather than imputing it afresh
        — the flag columns added, where the data was written, and a small sample. On failure,
        `{"status": "error", "message": ...}`.
    """
    try:
        if method not in IMPUTERS:
            return {
                "status": "error",
                "message": f"unknown method {method!r}; expected one of {sorted(IMPUTERS)}",
            }

        rows = load_data(data)
        if not rows:
            return {"status": "error", "message": "the dataset is empty"}

        resolved = infer_columns(rows, id_columns, date_columns, None)
        # The plan covers every column asked about, `wanted` only the ones a method will touch: the
        # difference is the dates and the empty columns, and saying so is half of what this reports.
        plan = suggest_imputations(
            rows, _as_columns(columns), resolved["id_columns"], resolved["date_columns"]
        )
        wanted = _wanted(rows, _as_columns(columns))
        if not wanted:
            return {
                "status": "ok",
                "rows": len(rows),
                "method": method,
                "resolved_columns": resolved,
                "columns": {},
                "filled": 0,
                "remaining": sum(entry["missing"] for entry in plan.values()),
                "notes": [
                    "Nothing was imputed: no column asked for holds a hole a method should fill.",
                    *_left_by_rule(plan),
                ],
            }

        before = missing_counts(rows, wanted)
        kinds = column_kinds(rows)
        fills = _replayable(rows, method, plan, wanted, statistic, value, options)

        pool: dict[str, Any] = {
            "columns": wanted,
            "id_columns": resolved["id_columns"],
            "date_columns": resolved["date_columns"],
            "statistic": statistic,
            "value": value,
            "indicators": indicators,
            **(options or {}),
        }
        rows, added = _call(IMPUTERS[method], rows, pool)
        after = missing_counts(rows, wanted)

        destination = output_path or _default_output_path(data, "imputed")
        written = write_data(rows, destination) if destination else None

        report = {
            column: {
                "kind": kinds.get(column, "empty"),
                "method": plan[column]["strategy"] if method == "auto" else method,
                "missing_before": before.get(column, 0),
                "filled": before.get(column, 0) - after.get(column, 0),
                "remaining": after.get(column, 0),
                "filled_with": fills.get(column),
            }
            for column in wanted
        }
        return {
            "status": "ok",
            "rows": len(rows),
            "method": method,
            "resolved_columns": resolved,
            "columns": report,
            "filled": sum(entry["filled"] for entry in report.values()),
            "remaining": sum(entry["remaining"] for entry in report.values()),
            "fill_values": fills,
            "indicators_added": added,
            "output_path": written,
            "sample": _sample(rows, limit),
            "notes": _impute_notes(method, plan, report, fills, indicators, written, destination),
        }
    except Exception as error:
        logger.exception("impute_missing_caller failed.")
        return {"status": "error", "message": str(error)}


def _call(function: Callable[..., Any], rows: Table, pool: dict[str, Any]) -> Any:
    """Call an imputer with the arguments it declares, drawn from the pool — `_run_step`'s trick."""
    parameters = inspect.signature(function).parameters
    return function(
        rows,
        **{name: value for name, value in pool.items() if name in parameters and name != "rows"},
    )


def _replayable(
    rows: Table,
    method: str,
    plan: dict[str, dict[str, Any]],
    wanted: Sequence[str],
    statistic: str,
    value: Any,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    """The fills that are one value per column, learned before the run — what `impute_plan` replays.

    Only the methods that put a single value in a column have one. `group`, `forward`, `knn` and the
    rest decide per row, so there is nothing to hand the next batch, and the honest report is an
    empty mapping rather than a value that was only true for some of the rows.
    """
    if method == "plan":
        return dict((options or {}).get("values") or {})
    if method == "statistic":
        return learn_fill_values(rows, wanted, statistic, value)
    if method == "category":
        label = str((options or {}).get("label", "missing"))
        return {column: label for column in wanted if plan.get(column, {}).get("kind") != "numeric"}
    if method != "auto":
        return {}

    # An empty list means "no column", where every other function here reads it as "all of them" —
    # so the two column sets are tested before they are passed anywhere.
    chosen = [column for column in wanted if plan.get(column, {}).get("strategy") == "statistic"]
    fills = learn_fill_values(rows, chosen, "median") if chosen else {}
    fills.update(
        {
            column: "missing"
            for column in wanted
            if plan.get(column, {}).get("strategy") == "category"
        }
    )
    return fills


def _left_by_rule(plan: dict[str, dict[str, Any]]) -> list[str]:
    """The one note every caller owes the model: what was not filled, and why it was not."""
    left = [column for column, entry in plan.items() if entry["strategy"] == "leave"]
    if not left:
        return []

    return [
        f"Left alone by rule: {', '.join(left)}. A date column is not filled — a fill invents when "
        f"something happened — and a column with nothing in it has nothing to fill from, so a fill "
        f"would be a constant. Name either in `columns` to fill it anyway, or drop the column."
    ]


def _report_notes(profile: dict[str, Any], plan: dict[str, dict[str, Any]]) -> list[str]:
    """What the profile says that a table of rates does not."""
    described: dict[str, dict[str, Any]] = profile["columns"]
    if not described:
        return ["Nothing is missing; there is nothing to impute."]

    notes: list[str] = _left_by_rule(plan)
    heavy = [column for column, entry in described.items() if entry["rate"] > MOSTLY_MISSING]
    if heavy:
        notes.append(
            f"Mostly holes: {', '.join(heavy)}. Whatever fills them is closer to a constant than "
            f"to data — keep the {FLAG.lstrip('_')} flags and expect those to carry the column."
        )

    related = {
        column: entry for column, entry in described.items() if entry["related_to"] is not None
    }
    if related:
        worst = max(related, key=lambda column: abs(related[column]["related_gap"]))
        notes.append(
            f"The holes do not look random: rows missing {worst!r} differ from the rest by "
            f"{related[worst]['related_gap']:+.2f} standard deviations on "
            f"{related[worst]['related_to']!r}. That makes the flag a feature and the fill a guess "
            f"made in one corner of the data."
        )

    shared = [entry for entry in profile["patterns"] if len(entry["columns"]) > 1]
    if shared:
        notes.append(
            f"{shared[0]['rows']:,} row(s) are missing {', '.join(shared[0]['columns'])} together; "
            f"one thing failed for those rows, so fill them from one decision."
        )
    return notes


def _impute_notes(
    method: str,
    plan: dict[str, dict[str, Any]],
    report: dict[str, dict[str, Any]],
    fills: dict[str, Any],
    indicators: bool,
    written: str | None,
    destination: str | None,
) -> list[str]:
    """The things worth telling the model without making it read every column back."""
    notes: list[str] = []
    filled = sum(entry["filled"] for entry in report.values())

    if filled and method not in ("forward", "plan"):
        replay = (
            "replay the fill_values below on the next batch with method='plan'"
            if fills
            else "this method decides per row, so there is nothing to replay: learn a plan with "
            "method='statistic' on the training rows instead"
        )
        notes.append(
            "These fills were computed across the whole table, so a row was partly filled from its "
            f"own future. That is fine for a snapshot; on a time-split training set, {replay}, or "
            f"use method='forward', which reads the past alone."
        )

    notes.extend(_left_by_rule(plan))

    stubborn = {
        column: entry["remaining"] for column, entry in report.items() if entry["remaining"]
    }
    if stubborn:
        notes.append(
            f"Still missing after the run: {stubborn}. {method} fills only where it has something "
            f"to fill from — a carried value needs a past, an interpolation needs a reading either "
            f"side, a fit needs the other columns. Close the rest with method='statistic'."
        )

    if not indicators:
        notes.append(
            "No flags were written, so the fact that these values were missing is now gone from "
            "the dataset. Run missing_indicators before imputing if anything downstream needs it."
        )

    if not written and not destination:
        notes.append("Nothing was written: pass output_path to keep the result for the next tool.")
    return notes

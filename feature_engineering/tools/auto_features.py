"""Feature steps that need nothing but the data.

Every step in `feature_prep.py` has to be told which columns are ids, which are dates and which are
measurements. The steps here are told nothing. Each takes one argument — the rows — works out what
each column actually holds, and returns the rows with new columns on them plus the names of what it
built:

    from feature_engineer.tools.auto_features import add_all_auto_features, add_calendar_features

    rows, added = add_calendar_features(rows)     # -> ["signup_date__dayofweek", ...]
    rows, added = add_all_auto_features(rows)     # every step below, in order

That single-argument shape is the point of the module. It means a step can be run over a table
nobody has described yet, the eleven of them compose in any order, and an agent that has a CSV and
no schema still has something to call.

What gets built depends on what a column turns out to be:

    text        length, word and character counts, class ratios, Shannon entropy, a shape
                signature, and flags for the formats it matches (email, url, uuid, ...)
    numeric     sign, magnitude, decimal places, leading digit, and where the value sits in its own
                column: distance from the mean and median, z-scores, percentile, IQR position
    date        calendar parts (day of week, month, quarter, week of month, season, ...), time of
                day when the column carries one, and position within the column's own date range
    category    how often the value occurs, how rare it is, and its self-information in bits
    boolean     a 0/1 flag, whatever the source wrote it as

Steps read the table repeatedly, so they need to tell a column that was in the data from a column an
earlier step added — otherwise the second step builds `charges__zscore__minus_mean` and the third
builds features of that. A derived column is therefore always named `{source}__{feature}`, and the
double underscore, together with the fixed vocabulary of feature names in `DERIVED_SUFFIXES`, is
what every step uses to skip its own output. A column that was in the data all along is left alone
even if it happens to contain a double underscore.

Nothing here looks at a target column, so nothing here can leak a label.

The steps register themselves into `feature_prep.FEATURE_PREP_STEPS` under `auto_*` names, so they
need no tool of their own: whenever `feature_prep_caller` is registered in
`agentic_configurations.yaml`, these come with it.

    feature_prep_caller(data="events.csv", steps=["auto_all"])
    feature_prep_caller(data="events.csv", steps=["auto_calendar", "auto_column_statistics"])
    list_feature_prep_steps_caller()          # lists these alongside the configured steps
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import chain
from statistics import fmean, median, pstdev
from typing import Any

# Same-package internals, deliberately shared rather than re-implemented: a value that counts as
# missing here has to be the value that counts as missing there, or the two modules disagree about
# what the data holds.
from feature_engineering.tools.feature_prep import (
    FEATURE_PREP_STEPS,
    Table,
    _group_key,
    _percentile_ranks,
    _quantile,
    _safe_divide,
    column_kinds,
    column_names,
    is_missing,
    to_datetime,
    to_number,
)

logger = logging.getLogger(__name__)

# What separates a source column from the feature derived off it. Two underscores rather than one
# because one is what ordinary column names are already full of.
MARKER = "__"

TEXT_SHAPE_FEATURES = (
    "length",
    "word_count",
    "digit_count",
    "letter_count",
    "upper_count",
    "punctuation_count",
    "whitespace_count",
    "starts_with_digit",
    "is_upper",
)

TEXT_COMPOSITION_FEATURES = (
    "digit_ratio",
    "letter_ratio",
    "upper_ratio",
    "symbol_ratio",
    "unique_char_ratio",
    "entropy",
    "shape",
)

# Formats worth knowing a column is in. Anchored, because a step that reports "is_email" for a
# sentence with an address somewhere in it is reporting something else.
TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE),
    "url": re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE),
    "uuid": re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.IGNORECASE),
    "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    "phone": re.compile(r"^\+?\d[\d\s().-]{6,}$"),
    "numeric_text": re.compile(r"^[+-]?\d+(\.\d+)?$"),
    "hex_code": re.compile(r"^#?[0-9a-f]{6}$", re.IGNORECASE),
    "currency": re.compile(r"^[$£€¥]\s?[\d,]+(\.\d+)?$|^[\d,]+(\.\d+)?\s?[$£€¥]$"),
}

TEXT_PATTERN_FEATURES = tuple(f"is_{name}" for name in TEXT_PATTERNS) + ("domain",)

NUMERIC_SHAPE_FEATURES = (
    "sign",
    "is_zero",
    "is_negative",
    "is_whole",
    "decimal_places",
    "integer_digits",
    "first_digit",
    "magnitude",
    "is_round",
)

COLUMN_STATISTIC_FEATURES = (
    "minus_mean",
    "minus_median",
    "over_median",
    "zscore",
    "mad_zscore",
    "percentile",
    "iqr_position",
    "above_median",
    "is_outlier",
)

CALENDAR_FEATURES = (
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "dayofweek",
    "dayofyear",
    "week_of_month",
    "days_in_month",
    "days_to_month_end",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_end",
    "season",
)

# Only built for a column that carries a time, since a date-only column would take four constants.
TIME_OF_DAY_FEATURES = ("hour", "minute_of_day", "part_of_day", "is_business_hour")

RELATIVE_TIME_FEATURES = (
    "days_since_first",
    "days_until_last",
    "span_position",
    "chronological_rank",
)

CATEGORY_PROFILE_FEATURES = (
    "frequency",
    "frequency_ratio",
    "rarity_rank",
    "is_rare",
    "is_mode",
    "self_information",
)

BOOLEAN_FEATURES = ("flag",)

ROW_PROFILE_FEATURES = (
    "missing_count",
    "missing_ratio",
    "filled_count",
    "numeric_count",
    "zero_count",
    "negative_count",
    "text_length",
    "distinct_values",
)

# The whole vocabulary this module appends after the marker. A column only counts as derived when it
# carries the marker *and* ends in one of these, so a source column that happens to be called
# `order__total` survives every step instead of being quietly skipped.
DERIVED_SUFFIXES = frozenset(
    chain(
        TEXT_SHAPE_FEATURES,
        TEXT_COMPOSITION_FEATURES,
        TEXT_PATTERN_FEATURES,
        NUMERIC_SHAPE_FEATURES,
        COLUMN_STATISTIC_FEATURES,
        CALENDAR_FEATURES,
        TIME_OF_DAY_FEATURES,
        RELATIVE_TIME_FEATURES,
        CATEGORY_PROFILE_FEATURES,
        BOOLEAN_FEATURES,
        ROW_PROFILE_FEATURES,
    )
)

RARE_SHARE = 0.01  # a category under this share of the rows is reported as rare
MAX_CATEGORY_SHARE = 0.5  # past this many distinct values a column is an id, not a category
DAY_SECONDS = 86400.0
TRUE_TOKENS = {"1", "true", "t", "yes", "y"}
SEASONS = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


# --------------------------------------------------------------------------------------------------
# what is in the table: source columns, and what each of them holds
# --------------------------------------------------------------------------------------------------
def is_derived(name: str) -> bool:
    """Whether a column was built by a step here rather than read from the data."""
    stem, marker, suffix = str(name).rpartition(MARKER)
    return bool(stem and marker) and suffix in DERIVED_SUFFIXES


def source_columns(data: Table) -> list[str]:
    """The columns that came with the data, with everything derived here filtered out.

    This is what makes the steps re-runnable: each one re-reads the table it is handed, and without
    this it would keep finding its own output and building features of features.
    """
    return [column for column in column_names(data) if not is_derived(column)]


def source_kinds(data: Table) -> dict[str, str]:
    """Each source column classified as "numeric", "date", "boolean", "categorical" or "empty"."""
    kinds = column_kinds(data)
    return {column: kinds.get(column, "empty") for column in source_columns(data)}


def _of_kind(data: Table, *wanted: str) -> list[str]:
    """The source columns holding one of `wanted`."""
    return [column for column, kind in source_kinds(data).items() if kind in wanted]


def _named(column: str, feature: str) -> str:
    """`charges` + `zscore` -> `charges__zscore`, the one place the marker is applied."""
    return f"{column}{MARKER}{feature}"


def _text(value: Any) -> str | None:
    """A value as text, or None when there is nothing there to measure."""
    return None if is_missing(value) else str(value).strip()


def _blank(data: Table, names: Sequence[str]) -> None:
    """Give every row the new columns, so a row a step could say nothing about still has the key."""
    for row in data:
        for name in names:
            row.setdefault(name, None)


# --------------------------------------------------------------------------------------------------
# steps — text: what a string is shaped like, made of, and formatted as
# --------------------------------------------------------------------------------------------------
def add_text_shape_features(data: Table) -> tuple[Table, list[str]]:
    """Count what every text column is made of: characters, words, digits, letters, capitals.

    Length and word count are the two features that most often carry something on their own — a
    free-text field is long when the person had something to say, and a code field is a fixed length
    until the day it isn't. The counts underneath them are what the ratios in
    `add_text_composition_features` are built from.
    """
    added: list[str] = []

    for column in _of_kind(data, "categorical"):
        names = {feature: _named(column, feature) for feature in TEXT_SHAPE_FEATURES}
        for row in data:
            text = _text(row.get(column))
            if text is None:
                continue

            row[names["length"]] = len(text)
            row[names["word_count"]] = len(text.split())
            row[names["digit_count"]] = sum(1 for character in text if character.isdigit())
            row[names["letter_count"]] = sum(1 for character in text if character.isalpha())
            row[names["upper_count"]] = sum(1 for character in text if character.isupper())
            row[names["punctuation_count"]] = sum(
                1 for character in text if not character.isalnum() and not character.isspace()
            )
            row[names["whitespace_count"]] = sum(1 for character in text if character.isspace())
            row[names["starts_with_digit"]] = int(bool(text) and text[0].isdigit())
            row[names["is_upper"]] = int(text.isupper())
        added.extend(names.values())

    _blank(data, added)
    return data, added


def add_text_composition_features(data: Table) -> tuple[Table, list[str]]:
    """Turn each text column into proportions, entropy, and the shape its values are written in.

    Ratios say what a value is made of without saying how long it is, which is what separates a
    reference like `INV-2024-0091` from a name. Entropy separates a random-looking key from a word.
    And `shape` — `INV-2024-0091` collapsed to `A-9-9` — is the one to reach for when a source
    starts writing its ids differently halfway through a table: the format becomes a value you can
    count.

    `shape` is text, so pass it through `feature_prep.add_frequency_encoding` to get a number.
    """
    added: list[str] = []

    for column in _of_kind(data, "categorical"):
        names = {feature: _named(column, feature) for feature in TEXT_COMPOSITION_FEATURES}
        for row in data:
            text = _text(row.get(column))
            if text is None:
                continue

            size = len(text)
            digits = sum(1 for character in text if character.isdigit())
            letters = sum(1 for character in text if character.isalpha())
            uppers = sum(1 for character in text if character.isupper())
            symbols = sum(
                1 for character in text if not character.isalnum() and not character.isspace()
            )

            row[names["digit_ratio"]] = _safe_divide(digits, size)
            row[names["letter_ratio"]] = _safe_divide(letters, size)
            row[names["upper_ratio"]] = _safe_divide(uppers, letters)
            row[names["symbol_ratio"]] = _safe_divide(symbols, size)
            row[names["unique_char_ratio"]] = _safe_divide(len(set(text)), size)
            row[names["entropy"]] = _entropy(text)
            row[names["shape"]] = _shape(text)
        added.extend(names.values())

    _blank(data, added)
    return data, added


def _entropy(text: str) -> float:
    """Shannon entropy over the characters, in bits: 0.0 for `aaaa`, ~3.6 for a random token."""
    if not text:
        return 0.0

    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1

    total = float(len(text))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _shape(text: str) -> str:
    """Collapse a value to its character classes: `INV-2024-0091` -> `A-9-9`, `Jo Smith` -> `Aa Aa`.

    Runs of the same class collapse to one symbol, so values of the same format land on the same
    shape whatever their length, and a change of format shows up as a new value.
    """
    signature: list[str] = []
    for character in text:
        if character.isdigit():
            token = "9"
        elif character.isupper():
            token = "A"
        elif character.islower():
            token = "a"
        elif character.isspace():
            token = " "
        else:
            token = character
        if not signature or signature[-1] != token:
            signature.append(token)

    return "".join(signature)


def add_text_pattern_features(data: Table) -> tuple[Table, list[str]]:
    """Flag the formats a text column is written in, and pull the domain out when there is one.

    A flag is only added where it varies. A pattern that matches every value describes the column
    rather than the row — `is_email` on a column of nothing but addresses is a constant — and a
    pattern that matches none of them is a column of zeros; a wide table gathers enough of those
    without help.

    The domain is the exception, and is extracted whenever anything in the column is an address or a
    url. It is most useful precisely where the flag is worthless: in a column that is *all*
    addresses, the format says nothing and `gmail.com` versus a company domain says a lot.
    """
    added: list[str] = []

    for column in _of_kind(data, "categorical"):
        texts = [text for row in data if (text := _text(row.get(column))) is not None]
        if not texts:
            continue

        hits = {
            name: sum(1 for text in texts if pattern.match(text))
            for name, pattern in TEXT_PATTERNS.items()
        }
        names = {
            name: _named(column, f"is_{name}") for name, hit in hits.items() if 0 < hit < len(texts)
        }
        addressed = bool(hits["email"] or hits["url"])
        domain = _named(column, "domain")
        if not names and not addressed:
            continue

        for row in data:
            text = _text(row.get(column))
            if text is None:
                continue

            for name, feature in names.items():
                row[feature] = int(bool(TEXT_PATTERNS[name].match(text)))
            if addressed:
                row[domain] = _domain(text)

        added.extend(names.values())
        if addressed:
            added.append(domain)

    _blank(data, added)
    return data, added


def _domain(text: str) -> str | None:
    """The host of a url or the domain of an address, lowercased; None when the value is neither.

    The patterns decide, not the text: a phone number is not a domain just because nothing else
    claimed it, and returning the value unchanged would make the column a copy of its source.
    """
    if TEXT_PATTERNS["email"].match(text):
        return text.rpartition("@")[2].lower() or None
    if not TEXT_PATTERNS["url"].match(text):
        return None

    host = re.sub(r"^https?://", "", text, flags=re.IGNORECASE)
    host = host.split("/")[0].split("?")[0].removeprefix("www.")
    return host.lower() or None


# --------------------------------------------------------------------------------------------------
# steps — numbers: the shape of a value, and where it sits in its own column
# --------------------------------------------------------------------------------------------------
def add_numeric_shape_features(data: Table) -> tuple[Table, list[str]]:
    """Describe how each number is written: its sign, size, precision and leading digit.

    These say things the value itself doesn't. `decimal_places` separates a measured amount from one
    somebody typed; `is_round` finds the estimates in a column of exact figures; `magnitude` is the
    order of magnitude, which is the comparable part of a column spanning six of them; and
    `first_digit` is the leading significant digit, which in genuine accounting data is a 1 about
    thirty percent of the time and in fabricated data usually isn't.
    """
    added: list[str] = []

    for column in _of_kind(data, "numeric"):
        names = {feature: _named(column, feature) for feature in NUMERIC_SHAPE_FEATURES}
        for row in data:
            number = to_number(row.get(column))
            if number is None:
                continue

            whole = number == int(number)
            row[names["sign"]] = (number > 0) - (number < 0)
            row[names["is_zero"]] = int(number == 0)
            row[names["is_negative"]] = int(number < 0)
            row[names["is_whole"]] = int(whole)
            row[names["decimal_places"]] = _decimal_places(number)
            row[names["integer_digits"]] = len(str(abs(int(number))))
            row[names["first_digit"]] = _first_digit(number)
            row[names["magnitude"]] = None if number == 0 else math.floor(math.log10(abs(number)))
            row[names["is_round"]] = int(whole and int(number) % 10 == 0)
        added.extend(names.values())

    _blank(data, added)
    return data, added


def _decimal_places(number: float) -> int:
    """How many digits the value carries after the point, to the ten this can be sure of."""
    fraction = f"{number:.10f}".partition(".")[2]
    return len(fraction.rstrip("0"))


def _first_digit(number: float) -> int:
    """The leading significant digit — 3 for both 0.0034 and 34_000, and 0 only for zero."""
    return int(f"{abs(number):.6e}"[0])


def add_column_statistic_features(data: Table) -> tuple[Table, list[str]]:
    """Restate every number as its distance from the middle of its own column.

    A charge of 90 means nothing until you know the column: it is the distance from the mean, the
    ratio to the median and the percentile that a model can compare across columns, and that stay
    comparable when the next batch arrives on a different scale.

    Both a mean-based and a median-based reading are built, because they disagree exactly where it
    matters. `zscore` uses the mean and standard deviation, which one large value drags;
    `mad_zscore` uses the median and the median absolute deviation, which it doesn't — so a value
    the first calls ordinary and the second calls extreme is a value worth looking at. `is_outlier`
    is the usual 1.5-IQR fence, reported rather than acted on.

    A column with fewer than two values to compute from is skipped rather than filled with zeros.
    """
    added: list[str] = []

    for column in _of_kind(data, "numeric"):
        values = [to_number(row.get(column)) for row in data]
        present = [value for value in values if value is not None]
        if len(present) < 2:
            logger.debug(f"auto_column_statistics: {column!r} has too few values; skipping it.")
            continue

        names = {feature: _named(column, feature) for feature in COLUMN_STATISTIC_FEATURES}
        middle, centre = fmean(present), median(present)
        spread = pstdev(present)
        deviation = median([abs(value - centre) for value in present])
        ordered = sorted(present)
        first, third = _quantile(ordered, 0.25), _quantile(ordered, 0.75)
        reach = third - first
        ranks = _percentile_ranks(values)

        for row, value, rank in zip(data, values, ranks):
            if value is None:
                continue

            row[names["minus_mean"]] = value - middle
            row[names["minus_median"]] = value - centre
            row[names["over_median"]] = _safe_divide(value, centre)
            row[names["zscore"]] = _safe_divide(value - middle, spread)
            row[names["mad_zscore"]] = _safe_divide(0.6745 * (value - centre), deviation)
            row[names["percentile"]] = rank
            row[names["iqr_position"]] = _safe_divide(value - first, reach)
            row[names["above_median"]] = int(value > centre)
            row[names["is_outlier"]] = int(
                bool(reach) and not first - 1.5 * reach <= value <= third + 1.5 * reach
            )
        added.extend(names.values())

    _blank(data, added)
    return data, added


# --------------------------------------------------------------------------------------------------
# steps — dates: the calendar a moment falls on, and where it falls in the column
# --------------------------------------------------------------------------------------------------
def add_calendar_features(data: Table) -> tuple[Table, list[str]]:
    """Split every date column into the calendar and clock parts a model can actually use.

    A timestamp is unusable as a number — it only ever goes up, so a model trained on it learns the
    period it was trained in. Its parts are the opposite: day of week and hour repeat, which is what
    makes weekly and daily rhythm learnable.

    Hour, minute of day, part of day and business hours are only built when the column carries a
    time at all; against a date-only column they would be four columns of zeros.
    """
    added: list[str] = []

    for column in _of_kind(data, "date"):
        moments = [to_datetime(row.get(column)) for row in data]
        timed = any(moment.hour or moment.minute or moment.second for moment in moments if moment)
        features = CALENDAR_FEATURES + (TIME_OF_DAY_FEATURES if timed else ())
        names = {feature: _named(column, feature) for feature in features}
        if not timed:
            logger.debug(f"auto_calendar: {column!r} carries no time; skipping the clock parts.")

        for row, moment in zip(data, moments):
            if moment is None:
                continue

            for feature, name in names.items():
                row[name] = _calendar_part(moment, feature)
        added.extend(names.values())

    _blank(data, added)
    return data, added


def _calendar_part(moment: datetime, part: str) -> Any:
    """One calendar or clock part. See `CALENDAR_FEATURES` and `TIME_OF_DAY_FEATURES`."""
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
    if part == "week_of_month":
        return (moment.day - 1) // 7 + 1
    if part == "days_in_month":
        return _days_in_month(moment)
    if part == "days_to_month_end":
        return _days_in_month(moment) - moment.day
    if part == "is_weekend":
        return int(moment.weekday() >= 5)
    if part == "is_month_start":
        return int(moment.day == 1)
    if part == "is_month_end":
        return int(moment.day == _days_in_month(moment))
    if part == "is_quarter_end":
        return int(moment.month in (3, 6, 9, 12) and moment.day == _days_in_month(moment))
    if part == "season":
        return SEASONS[moment.month]  # 0 winter, 1 spring, 2 summer, 3 autumn
    if part == "hour":
        return moment.hour
    if part == "minute_of_day":
        return moment.hour * 60 + moment.minute
    if part == "part_of_day":
        return moment.hour // 6  # 0 night, 1 morning, 2 afternoon, 3 evening
    if part == "is_business_hour":
        return int(moment.weekday() < 5 and 9 <= moment.hour < 18)
    return None


def _days_in_month(moment: datetime) -> int:
    if moment.month == 12:
        return 31
    return (
        datetime(moment.year, moment.month + 1, 1) - datetime(moment.year, moment.month, 1)
    ).days


def add_relative_time_features(data: Table) -> tuple[Table, list[str]]:
    """Place every date inside the range its own column covers.

    This is recency without a reference date: the newest row in the column is the "now", so
    `days_until_last` is how stale a row is, `days_since_first` is how far into the history it sits,
    and `span_position` is both of those on a 0..1 scale that survives the next batch changing the
    range. `chronological_rank` is the same ordering by position rather than by time, which is what
    you want when the dates cluster.

    Unlike `feature_prep.add_recency_features` nothing here is measured against the wall clock, so
    the values a row gets today are the values it gets when the pipeline is replayed next year.
    """
    added: list[str] = []

    for column in _of_kind(data, "date"):
        moments = [to_datetime(row.get(column)) for row in data]
        present = [moment for moment in moments if moment is not None]
        if len(present) < 2:
            logger.debug(f"auto_relative_time: {column!r} has too few dates; skipping it.")
            continue

        names = {feature: _named(column, feature) for feature in RELATIVE_TIME_FEATURES}
        first, last = min(present), max(present)
        span = (last - first).total_seconds() / DAY_SECONDS
        ranks = _percentile_ranks([m.timestamp() if m else None for m in moments])

        for row, moment, rank in zip(data, moments, ranks):
            if moment is None:
                continue

            since = (moment - first).total_seconds() / DAY_SECONDS
            row[names["days_since_first"]] = since
            row[names["days_until_last"]] = (last - moment).total_seconds() / DAY_SECONDS
            row[names["span_position"]] = _safe_divide(since, span)
            row[names["chronological_rank"]] = rank
        added.extend(names.values())

    _blank(data, added)
    return data, added


# --------------------------------------------------------------------------------------------------
# steps — categories, booleans, and the row as a whole
# --------------------------------------------------------------------------------------------------
def add_category_profile_features(data: Table) -> tuple[Table, list[str]]:
    """Replace a category with how common it is — one set of numbers, at any cardinality.

    One-hot encoding costs a column per category and can't encode a category it never saw; this
    costs six columns whatever the cardinality, and a value that turns up for the first time is
    simply the rarest thing in the column rather than an error. `self_information` is the surprise
    in bits, `-log2(p)`, which is the rarity on a scale that doesn't collapse at the thin end the
    way the raw share does.

    A column with a distinct value in more than half its rows is an identifier, not a category — its
    frequency would be 1 all the way down — so it is skipped and logged.
    """
    added: list[str] = []
    total = len(data)

    for column in _of_kind(data, "categorical", "boolean"):
        counts: dict[str, int] = {}
        for row in data:
            key = _group_key(row.get(column))
            if key:
                counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2 or len(counts) > MAX_CATEGORY_SHARE * total:
            logger.debug(f"auto_category_profile: {column!r} is not a category; skipping it.")
            continue

        ordered = sorted(counts, key=lambda key: (-counts[key], key))
        ranks = {key: rank for rank, key in enumerate(ordered)}
        names = {feature: _named(column, feature) for feature in CATEGORY_PROFILE_FEATURES}

        for row in data:
            key = _group_key(row.get(column))
            count = counts.get(key)
            if not count:
                continue

            share = count / total
            row[names["frequency"]] = count
            row[names["frequency_ratio"]] = share
            row[names["rarity_rank"]] = ranks[key]
            row[names["is_rare"]] = int(share < RARE_SHARE)
            row[names["is_mode"]] = int(key == ordered[0])
            row[names["self_information"]] = -math.log2(share)
        added.extend(names.values())

    _blank(data, added)
    return data, added


def add_boolean_flags(data: Table) -> tuple[Table, list[str]]:
    """Read every boolean column as a 0/1 flag, whatever the source spelled it as.

    `Yes`, `TRUE`, `t` and `1` all arrive meaning the same thing and none of them are a number.
    Missing stays missing rather than becoming a zero, which would be an answer nobody gave.
    """
    added: list[str] = []

    for column in _of_kind(data, "boolean"):
        name = _named(column, "flag")
        for row in data:
            text = _text(row.get(column))
            if text is None:
                continue

            row[name] = int(text.lower() in TRUE_TOKENS)
        added.append(name)

    _blank(data, added)
    return data, added


def add_row_profile_features(data: Table) -> tuple[Table, list[str]]:
    """Summarise each row across all of its source columns: how full it is, and what is in it.

    How much of a row is missing is a feature in its own right, and often a strong one, because
    incompleteness is rarely random — the rows with holes in them came from somewhere, or from
    someone, that the complete rows didn't. The rest are the cheap cross-column counts: zeros,
    negatives, how much text the row carries, how many distinct values it holds.
    """
    columns = source_columns(data)
    if not columns:
        return data, []

    names = {feature: f"row{MARKER}{feature}" for feature in ROW_PROFILE_FEATURES}
    for row in data:
        values = [row.get(column) for column in columns]
        present = [value for value in values if not is_missing(value)]
        numbers = [number for number in map(to_number, present) if number is not None]

        row[names["missing_count"]] = len(values) - len(present)
        row[names["missing_ratio"]] = (len(values) - len(present)) / len(values)
        row[names["filled_count"]] = len(present)
        row[names["numeric_count"]] = len(numbers)
        row[names["zero_count"]] = sum(1 for number in numbers if number == 0)
        row[names["negative_count"]] = sum(1 for number in numbers if number < 0)
        row[names["text_length"]] = sum(len(str(value).strip()) for value in present)
        row[names["distinct_values"]] = len({_group_key(value) for value in present})

    return data, list(names.values())


# --------------------------------------------------------------------------------------------------
# the bundle, the catalogue, and the registration that puts both in front of the agent
# --------------------------------------------------------------------------------------------------
def add_all_auto_features(data: Table) -> tuple[Table, list[str]]:
    """Run every step above, in order, and return everything they built.

    Order is for readability rather than correctness: each step re-reads the table and skips what
    the others added, so any order produces the same columns. Booleans go first because reading them
    as flags is the cheapest thing here, and the row profile goes last so it is counting the columns
    the data arrived with rather than a table that has already doubled in width.

    On a wide table this builds a lot of columns — call the individual steps when that matters, and
    `feature_selection.py` when it doesn't but the model can't take all of them.
    """
    added: list[str] = []
    for step in (
        add_boolean_flags,
        add_calendar_features,
        add_relative_time_features,
        add_numeric_shape_features,
        add_column_statistic_features,
        add_text_shape_features,
        add_text_composition_features,
        add_text_pattern_features,
        add_category_profile_features,
        add_row_profile_features,
    ):
        _, names = step(data)
        added.extend(names)

    logger.info(
        f"auto_all: built {len(added)} feature(s) from {len(source_columns(data))} column(s)."
    )
    return data, added


AUTO_FEATURE_STEPS: dict[str, Callable[[Table], tuple[Table, list[str]]]] = {
    "auto_boolean_flags": add_boolean_flags,
    "auto_calendar": add_calendar_features,
    "auto_relative_time": add_relative_time_features,
    "auto_numeric_shape": add_numeric_shape_features,
    "auto_column_statistics": add_column_statistic_features,
    "auto_text_shape": add_text_shape_features,
    "auto_text_composition": add_text_composition_features,
    "auto_text_patterns": add_text_pattern_features,
    "auto_category_profile": add_category_profile_features,
    "auto_row_profile": add_row_profile_features,
    "auto_all": add_all_auto_features,
}

# Registering from here rather than from `feature_prep` is what keeps the two modules importable in
# either order: by the time this line runs, `feature_prep` has finished executing, whichever of the
# two the caller imported first.
FEATURE_PREP_STEPS.update(AUTO_FEATURE_STEPS)

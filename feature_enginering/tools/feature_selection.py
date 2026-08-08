"""Target-aware feature selection: which of the columns built so far are worth keeping.

`feature_prep.py` and `auto_features.py` build features without ever looking at the label, which is
what keeps them from leaking it. This module is the other half, and it is deliberately separate: every
score here is a comparison against the target, and something that reads the target can leak it, rank
on it, or overfit to it. Nothing here builds a feature. It ranks what exists, says what is redundant,
what is unusable, and what looks too good to be true, and writes out the columns worth carrying.

Three functions are meant to be called by an agent; the rest are the scorers behind them and are
importable on their own:

    feature_selection_caller            screen, rank, prune and cut, in one call
    feature_screening_caller            the health check alone — no target needed
    list_feature_selection_methods_caller   the catalogue: every method, what it takes, what it needs

Selection runs in four stages, and the reply reports each one, because a feature that disappears
between "the data" and "the model" has to be accounted for:

    screen      drop what cannot be a feature at all: empty, constant, mostly missing, an id,
                a duplicate of another column
    rank        score every survivor against the target with one comparable measure
    leakage     flag a score so high the feature is probably the answer in disguise
    prune       drop the weaker of any two features that carry the same information

Methods come in two kinds. The ones implemented here need nothing but the standard library —
correlation, mutual information, ANOVA F, Cramér's V, single-feature AUC, information value,
variance — and they cover every combination of feature and target type, so a selection always runs.
The rest are `scikit-learn`'s, and are worth the dependency for what the standard library cannot do
well: a proper continuous mutual-information estimator, model-based importance, L1 selection,
recursive elimination and permutation importance. scikit-learn is an optional extra:

    poetry install -E selection      # or: pip install scikit-learn

When it isn't installed, those methods report themselves unavailable and name the install; they never
raise, and the stdlib methods carry on. `list_feature_selection_methods_caller` says which is which
on the machine it is running on, so a model can pick a method it can actually use.

Register the callers in `agentic_configurations.yaml` the way the other tools are registered:

    - name: "feature_selection_tool"
      description: "Rank features against the target, drop what is unusable, redundant or leaking,
        and write out the columns worth keeping."
      caller: "feature_enginering.tools.feature_selection.feature_selection_caller"
      args:
        - name: "data"
          type: "str"
          description: "Path to the CSV/JSON holding the features."
          required: false
        - name: "target"
          type: "str"
          description: "The column being predicted."
          required: false
        - name: "method"
          type: "str"
          description: "Ranking method. Call feature_selection_methods_tool for the catalogue."
          required: false
        - name: "top_k"
          type: "int"
          description: "How many features to keep."
          required: false
        - name: "features"
          type: "list"
          description: "Restrict the ranking to these columns. Defaults to everything else."
          required: false
        - name: "output_path"
          type: "str"
          description: "Where to write the selected columns."
          required: false
"""

from __future__ import annotations

import importlib.util
import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

# Same-package internals, deliberately shared rather than re-implemented: what counts as missing, how
# a value becomes a number, and how a dataset is read and written have to mean the same thing in both
# modules, or the two disagree about what the data holds.
from feature_enginering.tools.feature_prep import (
    Table,
    _as_columns,
    _default_output_path,
    _group_key,
    _quantile,
    _sample,
    column_kinds,
    column_names,
    is_missing,
    load_data,
    to_number,
    write_data,
)

logger = logging.getLogger(__name__)

MAX_BINS = 10  # quantile bins a numeric column is discretised into for mutual information and IV
MAX_LEVELS = 50  # categories kept before the rest are grouped as "other", so rare levels don't
# dominate a count-based score
MAX_ROWS = 20000  # rows scored before the table is systematically sampled down to this
MAX_COMPARED = 120  # features compared pairwise for redundancy; beyond this only the ranked leaders

# Screening thresholds. Each is a default, and each is reported alongside what it rejected, so a model
# that disagrees can pass its own rather than having to guess why a column disappeared.
MAX_MISSING_RATE = 0.5  # a column missing more than half its values cannot carry a feature
QUASI_CONSTANT_SHARE = 0.99  # one value in this share of the rows is constant in all but name
ID_LIKE_SHARE = 0.95  # distinct values per row above this is an identifier, not a feature
MAX_CORRELATION = 0.95  # two features this alike carry one feature's worth of information
LEAKAGE_SCORE = 0.98  # a score this high is usually the target wearing a different name

SMOOTHING = 0.5  # added to every cell of a contingency table, so an empty one is not a divide by zero


# --------------------------------------------------------------------------------------------------
# the target: what is being predicted, and what kind of problem that makes it
# --------------------------------------------------------------------------------------------------
def resolve_task(rows: Table, target: str, task: str | None = None) -> str:
    """Work out whether the target makes this binary classification, multiclass, or regression.

    A caller who knows is believed. Otherwise: two distinct values is binary; anything non-numeric is
    classification; a numeric column with few enough distinct values relative to the rows is
    classification too, since an integer 0/1/2 label is still a label.

    Args:
        rows: The dataset.
        target: The column being predicted.
        task: "binary", "multiclass" or "regression" to settle it outright.

    Returns:
        One of "binary", "multiclass", "regression".
    """
    tasks = ("binary", "multiclass", "regression")
    if task:
        if task not in tasks:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(tasks)}")
        return task

    present = [row.get(target) for row in rows if not is_missing(row.get(target))]
    if not present:
        raise ValueError(f"the target {target!r} has no values")

    distinct = {_group_key(value) for value in present}
    if len(distinct) <= 1:
        raise ValueError(f"the target {target!r} never varies; there is nothing to predict")
    if len(distinct) == 2:
        return "binary"

    numeric = sum(to_number(value) is not None for value in present)
    if numeric < 0.8 * len(present):  # text labels: classification whatever the count
        return "multiclass"

    # A numeric target with few distinct values is a label, not a measurement. The threshold scales
    # with the data: 20 classes in a million rows is a label; 20 in 30 rows is a small sample of a
    # continuous column.
    return "multiclass" if len(distinct) <= min(20, max(2, len(present) // 20)) else "regression"


# --------------------------------------------------------------------------------------------------
# the scoring context: every vector a scorer needs, computed once and shared between them
# --------------------------------------------------------------------------------------------------
@dataclass
class Scoring:
    """The prepared vectors every scorer reads, so no two of them derive the same thing twice.

    Args:
        rows: The dataset, already sampled down if it was larger than `max_rows`.
        target: The column being predicted.
        task: "binary", "multiclass" or "regression".
        features: The candidate columns, in the order they will be reported.
        kinds: Each column's kind, from `column_kinds`.
    """

    rows: Table
    target: str
    task: str
    features: list[str]
    kinds: dict[str, str]
    _numeric: dict[str, list[float | None]] = field(default_factory=dict)
    _labels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def classification(self) -> bool:
        return self.task in ("binary", "multiclass")

    def numeric(self, column: str) -> list[float | None]:
        """A column as floats, with None wherever it is missing or is not a number."""
        if column not in self._numeric:
            self._numeric[column] = [to_number(row.get(column)) for row in self.rows]
        return self._numeric[column]

    def labels(self, column: str) -> list[str]:
        """A column as discrete labels: numeric columns binned, categories capped, missing kept.

        Missing is a label of its own rather than a dropped row, because whether a value is there at
        all is often the strongest thing a column has to say — and dropping it would silently score
        different features on different subsets of the data.
        """
        if column not in self._labels:
            self._labels[column] = _discretise(
                [row.get(column) for row in self.rows],
                numeric=self.kinds.get(column) in ("numeric", "boolean"),
            )
        return self._labels[column]

    def target_labels(self) -> list[str]:
        """The target as discrete labels — binned when it is continuous."""
        return self.labels(self.target)

    def target_numbers(self) -> list[float | None]:
        """The target as floats. For binary classification the rarer class is 1."""
        if self.classification and self.task == "binary":
            positive = self.positive_class()
            return [
                None if is_missing(row.get(self.target)) else float(_group_key(row.get(self.target)) == positive)
                for row in self.rows
            ]
        return self.numeric(self.target)

    def positive_class(self) -> str:
        """The class treated as positive: the rarer of the two, which is the one worth predicting."""
        counts = Counter(
            _group_key(row.get(self.target)) for row in self.rows if not is_missing(row.get(self.target))
        )
        return min(counts, key=lambda label: (counts[label], label))

    def paired(self, column: str) -> tuple[list[float], list[float]]:
        """One feature and the target as two float vectors, over the rows where both are present."""
        feature, target = self.numeric(column), self.target_numbers()
        pairs = [
            (value, label)
            for value, label in zip(feature, target)
            if value is not None and label is not None
        ]
        return [value for value, _ in pairs], [label for _, label in pairs]


# --------------------------------------------------------------------------------------------------
# the scorers implemented here: nothing but the standard library, and every feature/target combination
# --------------------------------------------------------------------------------------------------
def _score_correlation(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Pearson correlation between a numeric feature and the target."""
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        values, target = scoring.paired(column)
        correlation = _pearson(values, target)
        if correlation is None:
            scores[column] = _unscored("not numeric, or too few rows to correlate")
            continue
        scores[column] = {"score": abs(correlation), "correlation": correlation, "n": len(values)}
    return scores


def _score_spearman(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Rank correlation: the monotonic relationship, whatever shape it takes."""
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        values, target = scoring.paired(column)
        correlation = _pearson(_ranks(values), _ranks(target)) if len(values) > 1 else None
        if correlation is None:
            scores[column] = _unscored("not numeric, or too few rows to correlate")
            continue
        scores[column] = {"score": abs(correlation), "correlation": correlation, "n": len(values)}
    return scores


def _score_mutual_information(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """How much knowing the feature reduces uncertainty about the target.

    Both sides are discretised — numeric columns into quantile bins, categories capped — so one
    measure covers every combination of types and the scores stay comparable across features. The
    score is normalised by the target's own entropy, making it "the share of what there is to know",
    between 0 and 1, rather than a count of bits that grows with the number of levels.
    """
    target = scoring.target_labels()
    entropy = _entropy(target)
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        bits = _mutual_information(scoring.labels(column), target)
        scores[column] = {
            "score": (bits / entropy) if entropy else 0.0,
            "bits": bits,
            "target_entropy": entropy,
        }
    return scores


def _score_anova_f(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """The F statistic: how far apart the classes are on this feature, against the spread within them."""
    if not scoring.classification:
        return {column: _unscored("needs a classification target") for column in scoring.features}

    target = scoring.target_labels()
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        groups = _groups(scoring.numeric(column), target)
        statistic = _f_statistic(groups)
        if statistic is None:
            scores[column] = _unscored("not numeric, or a class with too few values")
            continue
        scores[column] = {"score": statistic, "groups": len(groups)}
    return scores


def _score_cramers_v(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Cramér's V: the association between two discrete columns, on a 0..1 scale."""
    target = scoring.target_labels()
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        statistic, association = _cramers_v(scoring.labels(column), target)
        scores[column] = {"score": association, "chi_square": statistic}
    return scores


def _score_auc(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Single-feature ROC AUC: how well ordering the rows by this feature alone ranks the positives.

    Reported as the Gini-style `2|AUC - 0.5|`, so 0 is useless and 1 is a perfect ranker either way
    up, with the raw AUC alongside — under 0.5 means the feature ranks the classes backwards, which
    is a finding, not a fault.
    """
    if scoring.task != "binary":
        return {column: _unscored("needs a binary target") for column in scoring.features}

    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        values, target = scoring.paired(column)
        area = _auc(values, target)
        if area is None:
            scores[column] = _unscored("not numeric, or one class has no rows here")
            continue
        scores[column] = {"score": abs(area - 0.5) * 2, "auc": area, "n": len(values)}
    return scores


def _score_information_value(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Information value: the credit-scoring measure of how well a binned feature separates a
    binary target.

    The convention it is read against: under 0.02 is useless, 0.1 weak, 0.3 medium, 0.5 strong, and
    much above that is usually leakage rather than a good feature.
    """
    if scoring.task != "binary":
        return {column: _unscored("needs a binary target") for column in scoring.features}

    positive = scoring.positive_class()
    outcomes = [
        None if is_missing(row.get(scoring.target)) else _group_key(row.get(scoring.target)) == positive
        for row in scoring.rows
    ]
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        value, bins = _information_value(scoring.labels(column), outcomes)
        scores[column] = {"score": value, "bins": bins, "positive_class": positive}
    return scores


def _score_variance(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Variance, which ignores the target: what is left when there is no label to rank against.

    Scale-dependent by nature — a column in cents varies more than the same column in euros — so the
    coefficient of variation is reported beside it, and neither is comparable across units.
    """
    scores: dict[str, dict[str, Any]] = {}
    for column in scoring.features:
        values = [value for value in scoring.numeric(column) if value is not None]
        if len(values) < 2:
            scores[column] = _unscored("not numeric, or too few values")
            continue

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scores[column] = {
            "score": variance,
            "variance": variance,
            "coefficient_of_variation": (math.sqrt(variance) / abs(mean)) if mean else None,
        }
    return scores


# --------------------------------------------------------------------------------------------------
# the scorers scikit-learn does better, imported only when one is asked for
# --------------------------------------------------------------------------------------------------
def _score_sklearn_mutual_information(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """scikit-learn's mutual information, estimated from nearest neighbours rather than from bins.

    Worth the dependency where the binned version is weakest: a continuous feature whose relationship
    with the target is real but does not survive being cut into ten buckets.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    estimate = mutual_info_classif if scoring.classification else mutual_info_regression
    values = estimate(matrix, target, random_state=0)
    return _from_vector(names, values, notes, key="bits")


def _score_f_test(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """scikit-learn's ANOVA F test, with the p-value the standard library cannot give.

    The F statistic alone says a feature separates the classes; the p-value says whether that would
    have happened anyway at this sample size.
    """
    from sklearn.feature_selection import f_classif, f_regression

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    test = f_classif if scoring.classification else f_regression
    statistics, probabilities = test(matrix, target)

    scores = _from_vector(names, statistics, notes, key="f_statistic")
    for name, probability in zip(names, probabilities):
        scores[name]["p_value"] = None if _nan(probability) else float(probability)
    return scores


def _score_random_forest(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Importance from a random forest, which sees interactions no single-feature score can.

    Impurity importance is biased towards high-cardinality columns, so a column that looks strong here
    and weak everywhere else is worth checking with `permutation` before believing.

    Args:
        trees: How many trees to grow.
        max_depth: Depth limit per tree.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    forest = (RandomForestClassifier if scoring.classification else RandomForestRegressor)(
        n_estimators=int(options.get("trees", 200)),
        max_depth=options.get("max_depth"),
        random_state=0,
        n_jobs=-1,
    )
    forest.fit(matrix, target)
    return _from_vector(names, forest.feature_importances_, notes, key="importance")


def _score_l1(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """L1-penalised coefficients, which drive the coefficients of useless features to exactly zero.

    Features are standardised first, without which the penalty falls hardest on whichever column
    happens to be measured in the largest units.

    Args:
        strength: Regularisation strength — `C` for classification, `alpha` for regression.
    """
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    scaled = StandardScaler().fit_transform(matrix)

    if scoring.classification:
        model: Any = LogisticRegression(
            penalty="l1", solver="liblinear", C=float(options.get("strength", 1.0)), random_state=0
        )
        model.fit(scaled, target)
        weights = [max(abs(column) for column in row) for row in zip(*model.coef_)]
    else:
        model = Lasso(alpha=float(options.get("strength", 0.01)), random_state=0)
        model.fit(scaled, target)
        weights = [abs(coefficient) for coefficient in model.coef_]

    scores = _from_vector(names, weights, notes, key="coefficient")
    for name in names:
        if not _unavailable(scores[name]):
            scores[name]["eliminated"] = scores[name]["score"] == 0.0
    return scores


def _score_rfe(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Recursive feature elimination: fit, drop the weakest, refit, and keep going.

    A wrapper rather than a filter — it scores a feature by what the model does without it, so it
    catches a feature that only earns its place alongside another one.

    Args:
        keep: How many features to keep. Defaults to half of them.
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression, LogisticRegression

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    keep = max(1, min(int(options.get("keep", max(1, len(names) // 2))), len(names)))
    estimator = (
        LogisticRegression(max_iter=1000, random_state=0)
        if scoring.classification
        else LinearRegression()
    )
    elimination = RFE(estimator, n_features_to_select=keep).fit(matrix, target)

    # `ranking_` counts up from 1 for the survivors, so it is inverted into a score that, like every
    # other method here, is larger for a better feature.
    highest = max(elimination.ranking_)
    scores = _from_vector(
        names, [(highest - rank + 1) / highest for rank in elimination.ranking_], notes, key=None
    )
    for name, rank, kept in zip(names, elimination.ranking_, elimination.support_):
        if not _unavailable(scores[name]):
            scores[name].update({"rank": int(rank), "kept": bool(kept)})
    return scores


def _score_permutation(scoring: Scoring, **options: Any) -> dict[str, dict[str, Any]]:
    """Permutation importance: how much worse the model gets when one feature is shuffled.

    The most honest of the model-based scores and the slowest — it refits nothing but re-scores the
    model once per feature per repeat, and it measures what the model actually uses rather than what
    the fitting procedure happened to favour.

    Args:
        repeats: How many times each feature is shuffled.
        trees: Size of the forest scored against.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    matrix, names, notes = _matrix(scoring)
    target = _target_vector(scoring)
    stratify = target if scoring.classification and _every_class_twice(target) else None
    train_x, test_x, train_y, test_y = train_test_split(
        matrix, target, test_size=0.25, random_state=0, stratify=stratify
    )
    model = (RandomForestClassifier if scoring.classification else RandomForestRegressor)(
        n_estimators=int(options.get("trees", 100)), random_state=0, n_jobs=-1
    )
    model.fit(train_x, train_y)
    importance = permutation_importance(
        model, test_x, test_y, n_repeats=int(options.get("repeats", 5)), random_state=0, n_jobs=-1
    )

    scores = _from_vector(names, importance.importances_mean, notes, key="importance")
    for name, deviation in zip(names, importance.importances_std):
        if not _unavailable(scores[name]):
            scores[name]["std"] = float(deviation)
    return scores


# --------------------------------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Method:
    """One ranking method: what it measures, what it works on, and what it needs installed.

    Args:
        function: The scorer.
        summary: One line on what the score means.
        tasks: The tasks it can score. Empty means every task.
        scale: What the score's range is, so two methods are not compared as if they shared one.
        requires: The package it needs, or None for the ones implemented here.
    """

    function: Callable[..., dict[str, dict[str, Any]]]
    summary: str
    tasks: tuple[str, ...] = ()
    scale: str = "unbounded, higher is better"
    requires: str | None = None

    @property
    def available(self) -> bool:
        """Whether this method can run on this machine."""
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def supports(self, task: str) -> bool:
        return not self.tasks or task in self.tasks


SELECTION_METHODS: dict[str, Method] = {
    "mutual_information": Method(
        _score_mutual_information,
        "Share of the target's uncertainty the feature resolves, from binned values.",
        scale="0..1",
    ),
    "correlation": Method(
        _score_correlation,
        "Pearson correlation: the strength of a straight-line relationship.",
        scale="0..1 (the sign is reported separately)",
    ),
    "spearman": Method(
        _score_spearman,
        "Rank correlation: any monotonic relationship, straight or not.",
        scale="0..1 (the sign is reported separately)",
    ),
    "auc": Method(
        _score_auc,
        "How well the feature alone ranks the positive class.",
        tasks=("binary",),
        scale="0..1, as 2|AUC-0.5|",
    ),
    "information_value": Method(
        _score_information_value,
        "How well the binned feature separates a binary target (credit-scoring IV).",
        tasks=("binary",),
        scale="0 to unbounded; >0.5 is usually leakage",
    ),
    "anova_f": Method(
        _score_anova_f,
        "F statistic: class separation against the spread within each class.",
        tasks=("binary", "multiclass"),
    ),
    "cramers_v": Method(
        _score_cramers_v,
        "Association between two discrete columns, from their contingency table.",
        scale="0..1",
    ),
    "variance": Method(
        _score_variance,
        "Spread alone, ignoring the target — for when there is no label yet.",
    ),
    "sklearn_mutual_information": Method(
        _score_sklearn_mutual_information,
        "Mutual information estimated from nearest neighbours, not from bins.",
        scale="0 to unbounded, in nats",
        requires="sklearn",
    ),
    "f_test": Method(
        _score_f_test,
        "ANOVA F test with the p-value alongside the statistic.",
        requires="sklearn",
    ),
    "random_forest": Method(
        _score_random_forest,
        "Impurity importance from a random forest, which sees interactions.",
        scale="0..1, summing to 1 across features",
        requires="sklearn",
    ),
    "l1": Method(
        _score_l1,
        "L1-penalised coefficients, which zero out the features that earn nothing.",
        requires="sklearn",
    ),
    "rfe": Method(
        _score_rfe,
        "Recursive elimination: drop the weakest, refit, repeat.",
        scale="0..1, derived from the elimination order",
        requires="sklearn",
    ),
    "permutation": Method(
        _score_permutation,
        "How much worse a fitted model gets when the feature is shuffled.",
        requires="sklearn",
    ),
}

DEFAULT_METHOD = "mutual_information"  # what "auto" resolves to: comparable, and always available


def resolve_method(method: str | None, task: str) -> tuple[str, Method]:
    """Pick the method to rank with, and refuse one that cannot answer for this target.

    "auto" is `mutual_information`: it scores every combination of feature and target type on one
    0..1 scale, needs nothing installed, and so is the one method that always has an answer.
    """
    name = (method or "auto").strip().lower()
    if name in ("auto", "default", ""):
        name = DEFAULT_METHOD

    chosen = SELECTION_METHODS.get(name)
    if not chosen:
        raise ValueError(f"unknown method {name!r}; expected one of {sorted(SELECTION_METHODS)}")
    if not chosen.supports(task):
        raise ValueError(
            f"method {name!r} needs a {' or '.join(chosen.tasks)} target, and this one is {task}"
        )
    if not chosen.available:
        raise ValueError(
            f"method {name!r} needs {chosen.requires} installed: `poetry install -E selection`. "
            f"Methods that need nothing: {sorted(name for name, method in SELECTION_METHODS.items() if method.requires is None)}"
        )
    return name, chosen


# --------------------------------------------------------------------------------------------------
# stage 1: screening — what cannot be a feature at all, target or no target
# --------------------------------------------------------------------------------------------------
def screen_features(
    rows: Table,
    features: Sequence[str] | None = None,
    max_missing_rate: float = MAX_MISSING_RATE,
    quasi_constant_share: float = QUASI_CONSTANT_SHARE,
    id_like_share: float = ID_LIKE_SHARE,
) -> list[dict[str, Any]]:
    """Profile every candidate column and flag the ones that cannot carry a feature.

    This runs without a target, which is what makes it worth calling on its own: everything it finds
    is a property of the column itself, and none of it can be affected by the label.

    The flags, each of which is a reason to drop:

        empty            no values at all
        constant         one distinct value
        quasi_constant   one value in `quasi_constant_share` of the rows
        high_missing     missing in more than `max_missing_rate` of the rows
        id_like          nearly as many distinct values as rows: an identifier
        date_column      a raw timestamp, which is not a feature until something is derived from it
        duplicate        identical, value for value, to a column already listed

    Args:
        rows: The dataset.
        features: Columns to screen. Defaults to every column.
        max_missing_rate: The share of missing values a column is allowed.
        quasi_constant_share: The share one value may hold before the column counts as constant.
        id_like_share: Distinct values per row above which a column is an identifier.

    Returns:
        One entry per column: its kind, missing rate, distinct count, dominant value share, the
        flags it raised, and whether it survived.
    """
    kinds = column_kinds(rows)
    candidates = [name for name in (_as_columns(features) or column_names(rows)) if name in kinds]
    total = len(rows) or 1

    seen: dict[tuple[Any, ...], str] = {}
    screened: list[dict[str, Any]] = []
    for column in candidates:
        values = [row.get(column) for row in rows]
        present = [value for value in values if not is_missing(value)]
        counts = Counter(_group_key(value) for value in present)
        distinct = len(counts)
        missing_rate = 1 - len(present) / total
        dominant = (counts.most_common(1)[0][1] / len(present)) if present else 0.0

        flags: list[str] = []
        if not present:
            flags.append("empty")
        elif distinct == 1:
            flags.append("constant")
        elif dominant >= quasi_constant_share:
            flags.append("quasi_constant")
        if present and missing_rate > max_missing_rate:
            flags.append("high_missing")
        if present and distinct / len(present) >= id_like_share and distinct > 20:
            flags.append("id_like")
        if kinds.get(column) == "date":
            flags.append("date_column")

        signature = tuple(_group_key(value) for value in values)
        duplicate = seen.get(signature)
        if present and duplicate:
            flags.append("duplicate")
        elif present:
            seen[signature] = column

        screened.append(
            {
                "feature": column,
                "kind": kinds.get(column, "empty"),
                "missing_rate": round(missing_rate, 4),
                "distinct": distinct,
                "dominant_share": round(dominant, 4),
                "duplicate_of": duplicate if present and duplicate else None,
                "flags": flags,
                "usable": not flags,
            }
        )
    return screened


# --------------------------------------------------------------------------------------------------
# stage 2: ranking
# --------------------------------------------------------------------------------------------------
def rank_features(
    rows: Table,
    target: str,
    features: Sequence[str] | None = None,
    method: str | None = None,
    task: str | None = None,
    max_rows: int = MAX_ROWS,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every candidate feature against the target, strongest first.

    Args:
        rows: The dataset.
        target: The column being predicted.
        features: Columns to rank. Defaults to every column but the target.
        method: How to score. Defaults to "auto" — see `resolve_method`.
        task: "binary", "multiclass" or "regression". Inferred from the target when omitted.
        max_rows: Rows to score over; a larger table is sampled systematically down to this.
        options: Extra arguments for the method, e.g. `{"trees": 500}`.

    Returns:
        The ranking, one entry per feature, and the context it was produced in: the task, the method,
        the rows scored and the scale the scores are on.
    """
    if target not in column_names(rows):
        raise ValueError(f"no target column named {target!r}")

    resolved_task = resolve_task(rows, target, task)
    name, chosen = resolve_method(method, resolved_task)

    candidates = [
        column
        for column in (_as_columns(features) or column_names(rows))
        if column != target and column in column_names(rows)
    ]
    if not candidates:
        return [], {"task": resolved_task, "method": name, "rows_scored": 0, "scale": chosen.scale}

    sampled = _sampled(rows, max_rows)
    scoring = Scoring(
        rows=sampled,
        target=target,
        task=resolved_task,
        features=candidates,
        kinds=column_kinds(sampled),
    )
    scores = chosen.function(scoring, **(options or {}))

    ranking = [
        {"feature": column, "method": name, **scores.get(column, _unscored("not scored"))}
        for column in candidates
    ]
    ranking.sort(key=lambda entry: (entry.get("score") is not None, entry.get("score") or 0.0), reverse=True)
    for position, entry in enumerate(ranking, start=1):
        entry["rank"] = position if entry.get("score") is not None else None

    return ranking, {
        "task": resolved_task,
        "method": name,
        "rows_scored": len(sampled),
        "rows_total": len(rows),
        "scale": chosen.scale,
        "measures": chosen.summary,
    }


# --------------------------------------------------------------------------------------------------
# stage 3: redundancy
# --------------------------------------------------------------------------------------------------
def redundant_pairs(
    rows: Table,
    features: Sequence[str],
    threshold: float = MAX_CORRELATION,
    max_rows: int = MAX_ROWS,
) -> list[dict[str, Any]]:
    """Find pairs of features that carry the same information.

    One measure would not cover every pair, so the measure follows the types: Pearson between two
    numeric columns, Cramér's V between two categorical ones, and the correlation ratio for a mixed
    pair. All three land on 0..1, so one threshold applies to all of them.

    Args:
        rows: The dataset.
        features: The columns to compare, pairwise.
        threshold: The association at which two features count as redundant.
        max_rows: Rows to compare over.

    Returns:
        One entry per redundant pair: the two columns, how associated they are, and by which measure.
    """
    sampled = _sampled(rows, max_rows)
    kinds = column_kinds(sampled)
    columns = list(features)[:MAX_COMPARED]

    numbers = {column: [to_number(row.get(column)) for row in sampled] for column in columns}
    labels = {column: [_group_key(row.get(column)) for row in sampled] for column in columns}

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(columns, 2):
        association, measure = _association(
            left, right, kinds, numbers, labels
        )
        if association is not None and association >= threshold:
            pairs.append(
                {
                    "feature": left,
                    "duplicate_of": right,
                    "association": round(association, 4),
                    "measure": measure,
                }
            )
    return pairs


# --------------------------------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------------------------------
def select_features(
    rows: Table,
    target: str,
    features: Sequence[str] | None = None,
    method: str | None = None,
    task: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    exclude: Sequence[str] | None = None,
    max_missing_rate: float = MAX_MISSING_RATE,
    max_correlation: float | None = MAX_CORRELATION,
    leakage_score: float | None = LEAKAGE_SCORE,
    max_rows: int = MAX_ROWS,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Screen, rank, flag leakage, prune redundancy and cut — the four stages in one call.

    Every feature that does not survive is accounted for in `dropped`, with the stage and the reason,
    because "the model used 12 of your 80 columns" is only useful with the other 68 explained.

    Args:
        rows: The dataset.
        target: The column being predicted.
        features: Candidate columns. Defaults to every column but the target.
        method: How to rank. Defaults to "auto".
        task: "binary", "multiclass" or "regression". Inferred when omitted.
        top_k: Keep at most this many features.
        min_score: Keep only features scoring at least this.
        exclude: Columns to leave out of the candidates entirely.
        max_missing_rate: The share of missing values a column is allowed before screening drops it.
        max_correlation: The association at which the weaker of two features is pruned. None to keep
            redundant features.
        leakage_score: The score above which a feature is treated as leakage. None to keep them,
            which is the right choice only when the "leak" is a feature you know to be legitimate.
        max_rows: Rows to score over.
        options: Extra arguments for the ranking method.

    Returns:
        The selected columns, the full ranking, everything dropped and why, the leakage suspects and
        the redundant pairs.
    """
    excluded = set(_as_columns(exclude))
    candidates = [
        column
        for column in (_as_columns(features) or column_names(rows))
        if column != target and column not in excluded
    ]
    dropped: list[dict[str, Any]] = []

    screened = screen_features(rows, candidates, max_missing_rate=max_missing_rate)
    for entry in screened:
        if not entry["usable"]:
            dropped.append(
                {
                    "feature": entry["feature"],
                    "stage": "screening",
                    "reason": ", ".join(entry["flags"]),
                    "duplicate_of": entry["duplicate_of"],
                }
            )
    usable = [entry["feature"] for entry in screened if entry["usable"]]

    ranking, context = rank_features(
        rows, target, usable, method=method, task=task, max_rows=max_rows, options=options
    )
    scored = [entry for entry in ranking if entry.get("score") is not None]
    for entry in ranking:
        if entry.get("score") is None:
            dropped.append(
                {
                    "feature": entry["feature"],
                    "stage": "ranking",
                    "reason": entry.get("note") or "the method could not score it",
                }
            )

    leaking: list[dict[str, Any]] = []
    if leakage_score is not None:
        for entry in list(scored):
            if entry["score"] >= leakage_score:
                leaking.append({"feature": entry["feature"], "score": entry["score"]})
                scored.remove(entry)
                dropped.append(
                    {
                        "feature": entry["feature"],
                        "stage": "leakage",
                        "reason": f"scored {entry['score']:.4f}, at or above the leakage bar of {leakage_score}",
                    }
                )

    redundant: list[dict[str, Any]] = []
    if max_correlation is not None and len(scored) > 1:
        order = {entry["feature"]: position for position, entry in enumerate(scored)}
        for pair in redundant_pairs(
            rows, [entry["feature"] for entry in scored], threshold=max_correlation, max_rows=max_rows
        ):
            # The pair is reported weaker-first: whichever of the two ranked lower is the one dropped.
            weaker, stronger = sorted(
                (pair["feature"], pair["duplicate_of"]), key=lambda name: order.get(name, 0), reverse=True
            )
            if weaker not in order:
                continue
            redundant.append({**pair, "feature": weaker, "duplicate_of": stronger})
            scored = [entry for entry in scored if entry["feature"] != weaker]
            order.pop(weaker, None)
            dropped.append(
                {
                    "feature": weaker,
                    "stage": "redundancy",
                    "reason": f"{pair['measure']} {pair['association']} with {stronger}, which ranked higher",
                }
            )

    if min_score is not None:
        for entry in scored:
            if entry["score"] < min_score:
                dropped.append(
                    {
                        "feature": entry["feature"],
                        "stage": "threshold",
                        "reason": f"scored {entry['score']:.4f}, under the minimum of {min_score}",
                    }
                )
        scored = [entry for entry in scored if entry["score"] >= min_score]

    if top_k is not None and len(scored) > max(int(top_k), 0):
        for entry in scored[int(top_k) :]:
            dropped.append(
                {
                    "feature": entry["feature"],
                    "stage": "top_k",
                    "reason": f"ranked {entry['rank']}, outside the top {top_k}",
                }
            )
        scored = scored[: int(top_k)]

    return {
        **context,
        "target": target,
        "selected": [entry["feature"] for entry in scored],
        "selected_count": len(scored),
        "candidates": len(candidates),
        "ranking": ranking,
        "screening": screened,
        "leakage_suspects": leaking,
        "redundant": redundant,
        "dropped": dropped,
    }


# --------------------------------------------------------------------------------------------------
# agent-facing callers
# --------------------------------------------------------------------------------------------------
def feature_selection_caller(
    data: Any = None,
    target: str | None = None,
    method: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    features: Any = None,
    exclude: Any = None,
    task: str | None = None,
    keep_columns: Any = None,
    max_missing_rate: float = MAX_MISSING_RATE,
    max_correlation: float | None = MAX_CORRELATION,
    leakage_score: float | None = LEAKAGE_SCORE,
    options: dict[str, Any] | None = None,
    output_path: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """Rank the features against the target, drop what cannot be used, and write out what is left.

    Args:
        data: Path to the CSV/JSON holding the features, or the rows themselves.
        target: The column being predicted.
        method: Ranking method; "auto" picks one that always works. See the catalogue.
        top_k: How many features to keep.
        min_score: Keep only features scoring at least this.
        features: Restrict the candidates to these columns.
        exclude: Columns to leave out of the candidates entirely, e.g. ids.
        task: "binary", "multiclass" or "regression". Inferred from the target when omitted.
        keep_columns: Columns to carry into the output even though they are not features — the ids
            the rows are keyed by, the date they belong to.
        max_missing_rate: The share of missing values a column is allowed.
        max_correlation: The association at which the weaker of two features is pruned.
        leakage_score: The score at which a feature is treated as leakage rather than a good feature.
        options: Extra arguments for the ranking method, e.g. `{"trees": 500}`.
        output_path: Where to write the selected columns. Defaults to a file beside the input.
        limit: How many sample rows to return.

    Returns:
        The selected features, the full ranking, what was dropped at each stage and why, where the
        result was written, and a small sample. On failure, `{"status": "error", "message": ...}`.
    """
    try:
        rows = load_data(data)
        if not rows:
            return {"status": "error", "message": "the dataset is empty"}
        if not target:
            return {
                "status": "error",
                "message": "no target given; feature selection needs the column being predicted. "
                "Call feature_screening_tool for what can be judged without one.",
            }
        if target not in column_names(rows):
            return {
                "status": "error",
                "message": f"no target column named {target!r}",
                "columns": column_names(rows),
            }

        result = select_features(
            rows,
            target,
            features=_as_columns(features) or None,
            method=method,
            task=task,
            top_k=top_k,
            min_score=min_score,
            exclude=_as_columns(exclude) or None,
            max_missing_rate=max_missing_rate,
            max_correlation=max_correlation,
            leakage_score=leakage_score,
            options=options,
        )

        carried = [column for column in _as_columns(keep_columns) if column in column_names(rows)]
        columns = [*carried, *result["selected"], target]
        destination = output_path or _default_output_path(data, "selected")
        written = (
            write_data([{column: row.get(column) for column in columns} for row in rows], destination)
            if destination
            else None
        )

        return {
            "status": "ok",
            **result,
            "kept_columns": carried,
            "output_path": written,
            "sample": _sample([{column: row.get(column) for column in columns} for row in rows], limit),
            "notes": _notes(result, written, destination, carried),
        }
    except Exception as error:
        logger.exception("feature_selection_caller failed.")
        return {"status": "error", "message": str(error)}


def feature_screening_caller(
    data: Any = None,
    features: Any = None,
    max_missing_rate: float = MAX_MISSING_RATE,
    quasi_constant_share: float = QUASI_CONSTANT_SHARE,
    id_like_share: float = ID_LIKE_SHARE,
    limit: int = 3,
) -> dict[str, Any]:
    """Check which columns can be features at all, without needing a target.

    Everything this reports is a property of the column itself — how much of it is missing, how much
    it varies, whether another column already says the same thing — so it can be run before a target
    exists, and nothing it does can leak a label.

    Args:
        data: Path to the CSV/JSON, or the rows themselves.
        features: Restrict the screen to these columns.
        max_missing_rate: The share of missing values a column is allowed.
        quasi_constant_share: The share one value may hold before the column counts as constant.
        id_like_share: Distinct values per row above which a column is an identifier.
        limit: How many sample rows to return.

    Returns:
        One entry per column with its flags, the usable and unusable lists, and a small sample. On
        failure, `{"status": "error", "message": ...}`.
    """
    try:
        rows = load_data(data)
        if not rows:
            return {"status": "error", "message": "the dataset is empty"}

        screened = screen_features(
            rows,
            _as_columns(features) or None,
            max_missing_rate=max_missing_rate,
            quasi_constant_share=quasi_constant_share,
            id_like_share=id_like_share,
        )
        usable = [entry["feature"] for entry in screened if entry["usable"]]
        rejected = [entry for entry in screened if not entry["usable"]]

        return {
            "status": "ok",
            "rows": len(rows),
            "columns": len(screened),
            "usable": usable,
            "usable_count": len(usable),
            "rejected": [
                {"feature": entry["feature"], "flags": entry["flags"], "duplicate_of": entry["duplicate_of"]}
                for entry in rejected
            ],
            "screening": screened,
            "sample": _sample(rows, limit),
            "notes": _screening_notes(screened),
        }
    except Exception as error:
        logger.exception("feature_screening_caller failed.")
        return {"status": "error", "message": str(error)}


def list_feature_selection_methods_caller(method: str | None = None, task: str | None = None) -> dict[str, Any]:
    """List the ranking methods, what each measures, and which of them can run here.

    Args:
        method: Restrict the listing to one method.
        task: Restrict it to the methods that can score this kind of target.

    Returns:
        One entry per method: what it measures, the scale its scores are on, the tasks it supports,
        what it needs installed, and whether that is installed.
    """
    names = [method] if method else list(SELECTION_METHODS)
    unknown = [name for name in names if name not in SELECTION_METHODS]
    if unknown:
        return {
            "status": "error",
            "message": f"no such method {unknown[0]!r}",
            "methods": list(SELECTION_METHODS),
        }

    listed = [
        {
            "method": name,
            "measures": SELECTION_METHODS[name].summary,
            "scale": SELECTION_METHODS[name].scale,
            "tasks": list(SELECTION_METHODS[name].tasks) or ["binary", "multiclass", "regression"],
            "requires": SELECTION_METHODS[name].requires,
            "available": SELECTION_METHODS[name].available,
        }
        for name in names
        if not task or SELECTION_METHODS[name].supports(task)
    ]
    missing = sorted({entry["requires"] for entry in listed if entry["requires"] and not entry["available"]})

    return {
        "status": "ok",
        "default_method": DEFAULT_METHOD,
        "methods": listed,
        "notes": (
            [f"Install the rest with `poetry install -E selection` ({', '.join(missing)} missing)."]
            if missing
            else []
        ),
    }


# --------------------------------------------------------------------------------------------------
# the measures themselves
# --------------------------------------------------------------------------------------------------
def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or None when either side never varies."""
    if len(left) < 2:
        return None

    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    spread_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    spread_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if not spread_left or not spread_right:
        return None

    return max(-1.0, min(1.0, covariance / (spread_left * spread_right)))


def _ranks(values: Sequence[float]) -> list[float]:
    """Rank the values from 1, with tied values sharing the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def _entropy(labels: Sequence[str]) -> float:
    """Shannon entropy of a discrete column, in bits."""
    counts = Counter(labels)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )


def _mutual_information(left: Sequence[str], right: Sequence[str]) -> float:
    """Mutual information between two discrete columns, in bits."""
    total = len(left)
    if not total:
        return 0.0

    joint = Counter(zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    information = 0.0
    for (a, b), count in joint.items():
        joint_probability = count / total
        expected = (left_counts[a] / total) * (right_counts[b] / total)
        if joint_probability and expected:
            information += joint_probability * math.log2(joint_probability / expected)
    return max(0.0, information)


def _groups(values: Sequence[float | None], labels: Sequence[str]) -> list[list[float]]:
    """The feature's values, split into one list per target class."""
    grouped: dict[str, list[float]] = {}
    for value, label in zip(values, labels):
        if value is not None:
            grouped.setdefault(label, []).append(value)
    return [group for group in grouped.values() if len(group) >= 2]


def _f_statistic(groups: Sequence[Sequence[float]]) -> float | None:
    """The one-way ANOVA F: variance between the groups over variance within them."""
    if len(groups) < 2:
        return None

    total = sum(len(group) for group in groups)
    grand_mean = sum(sum(group) for group in groups) / total
    between = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 for group in groups)
    within = sum(
        sum((value - sum(group) / len(group)) ** 2 for value in group) for group in groups
    )
    if total <= len(groups):
        return None
    if not within:
        return None if not between else float("inf")

    return (between / (len(groups) - 1)) / (within / (total - len(groups)))


def _cramers_v(left: Sequence[str], right: Sequence[str]) -> tuple[float, float]:
    """The chi-square statistic of two discrete columns, and Cramér's V derived from it."""
    total = len(left)
    if not total:
        return 0.0, 0.0

    joint = Counter(zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    statistic = 0.0
    for a, left_count in left_counts.items():
        for b, right_count in right_counts.items():
            expected = left_count * right_count / total
            observed = joint.get((a, b), 0)
            if expected:
                statistic += (observed - expected) ** 2 / expected

    smaller = min(len(left_counts), len(right_counts)) - 1
    if smaller < 1:
        return statistic, 0.0

    return statistic, min(1.0, math.sqrt(statistic / (total * smaller)))


def _auc(values: Sequence[float], outcomes: Sequence[float]) -> float | None:
    """ROC AUC from rank sums, which handles ties exactly and needs no threshold sweep."""
    positives = [index for index, outcome in enumerate(outcomes) if outcome == 1.0]
    negatives = [index for index, outcome in enumerate(outcomes) if outcome != 1.0]
    if not positives or not negatives:
        return None

    ranks = _ranks(values)
    positive_rank_sum = sum(ranks[index] for index in positives)
    count_positive, count_negative = len(positives), len(negatives)
    statistic = positive_rank_sum - count_positive * (count_positive + 1) / 2
    return statistic / (count_positive * count_negative)


def _information_value(labels: Sequence[str], outcomes: Sequence[bool | None]) -> tuple[float, int]:
    """Information value over the binned feature, and how many bins it was computed across."""
    positives = sum(1 for outcome in outcomes if outcome is True)
    negatives = sum(1 for outcome in outcomes if outcome is False)
    if not positives or not negatives:
        return 0.0, 0

    per_bin: dict[str, list[int]] = {}
    for label, outcome in zip(labels, outcomes):
        if outcome is None:
            continue
        counts = per_bin.setdefault(label, [0, 0])
        counts[0 if outcome else 1] += 1

    value = 0.0
    for good, bad in per_bin.values():
        share_positive = (good + SMOOTHING) / (positives + SMOOTHING * len(per_bin))
        share_negative = (bad + SMOOTHING) / (negatives + SMOOTHING * len(per_bin))
        value += (share_positive - share_negative) * math.log(share_positive / share_negative)
    return value, len(per_bin)


def _correlation_ratio(numbers: Sequence[float | None], labels: Sequence[str]) -> float | None:
    """Eta: how much of a numeric column's variance is explained by which category a row is in."""
    groups = _groups(numbers, labels)
    if len(groups) < 2:
        return None

    total = sum(len(group) for group in groups)
    grand_mean = sum(sum(group) for group in groups) / total
    between = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 for group in groups)
    overall = sum((value - grand_mean) ** 2 for group in groups for value in group)
    if not overall:
        return None

    return min(1.0, math.sqrt(between / overall))


def _association(
    left: str,
    right: str,
    kinds: dict[str, str],
    numbers: dict[str, list[float | None]],
    labels: dict[str, list[str]],
) -> tuple[float | None, str]:
    """How alike two columns are, on 0..1, by whichever measure their types call for."""
    numeric = ("numeric", "boolean")
    left_numeric = kinds.get(left) in numeric
    right_numeric = kinds.get(right) in numeric

    if left_numeric and right_numeric:
        pairs = [
            (a, b)
            for a, b in zip(numbers[left], numbers[right])
            if a is not None and b is not None
        ]
        correlation = _pearson([a for a, _ in pairs], [b for _, b in pairs])
        return (abs(correlation) if correlation is not None else None), "correlation"

    if not left_numeric and not right_numeric:
        return _cramers_v(labels[left], labels[right])[1], "cramers_v"

    number, label = (numbers[left], labels[right]) if left_numeric else (numbers[right], labels[left])
    return _correlation_ratio(number, label), "correlation_ratio"


# --------------------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------------------
def _discretise(values: Sequence[Any], numeric: bool, bins: int = MAX_BINS) -> list[str]:
    """Turn a column into discrete labels: numeric values binned by quantile, categories capped.

    Missing becomes its own label rather than a dropped row — whether a value is there at all is
    often the strongest signal a column carries, and dropping it would score different features over
    different subsets of the data.
    """
    if numeric:
        numbers = sorted(
            number for number in (to_number(value) for value in values) if number is not None
        )
        if len(numbers) > bins:
            edges = sorted({_quantile(numbers, index / bins) for index in range(1, bins)})
            return [_bin_label(to_number(value), edges) for value in values]

    labels = ["missing" if is_missing(value) else _group_key(value) for value in values]
    common = {label for label, _ in Counter(labels).most_common(MAX_LEVELS)}
    return [label if label in common else "other" for label in labels]


def _bin_label(number: float | None, edges: Sequence[float]) -> str:
    """Which quantile bin a number falls in, as a label."""
    if number is None:
        return "missing"
    return f"bin_{sum(1 for edge in edges if number > edge)}"


def _sampled(rows: Table, max_rows: int) -> Table:
    """Take every nth row when the table is larger than `max_rows`.

    Systematic rather than random, so two calls over the same data score the same rows and a model
    comparing two methods is comparing the methods rather than the sample.
    """
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    step = math.ceil(len(rows) / max_rows)
    logger.info(f"Scoring every {step} rows of {len(rows)}, the sample cap being {max_rows}.")
    return rows[::step][:max_rows]


def _unscored(note: str) -> dict[str, Any]:
    """What a scorer returns for a feature it cannot measure — never an exception, always a reason."""
    return {"score": None, "note": note}


def _unavailable(entry: dict[str, Any]) -> bool:
    return entry.get("score") is None


def _matrix(scoring: Scoring) -> tuple[list[list[float]], list[str], dict[str, str]]:
    """Encode the candidate features as the numeric matrix scikit-learn needs.

    Numeric columns are imputed with their median, and categorical ones are frequency-encoded — each
    value replaced by how often it occurs — which keeps the matrix one column wide per feature so an
    importance can still be attributed back to the feature it came from. One-hot encoding would
    split that importance across the levels and make the scores incomparable with the other methods.
    """
    matrix: list[list[float]] = [[] for _ in scoring.rows]
    names: list[str] = []
    notes: dict[str, str] = {}

    for column in scoring.features:
        if scoring.kinds.get(column) in ("numeric", "boolean"):
            values = scoring.numeric(column)
            present = [value for value in values if value is not None]
            if not present:
                notes[column] = "no numeric values to encode"
                continue
            fill = _quantile(sorted(present), 0.5)
            encoded = [fill if value is None else value for value in values]
        else:
            labels = [_group_key(row.get(column)) for row in scoring.rows]
            counts = Counter(labels)
            encoded = [float(counts[label]) for label in labels]
            notes[column] = "frequency-encoded"

        names.append(column)
        for row, value in zip(matrix, encoded):
            row.append(float(value))

    if not names:
        raise ValueError("none of the candidate features could be encoded as numbers")

    return matrix, names, notes


def _target_vector(scoring: Scoring) -> list[Any]:
    """The target as scikit-learn wants it: class labels for classification, floats for regression."""
    if scoring.classification:
        return [_group_key(row.get(scoring.target)) for row in scoring.rows]

    numbers = scoring.numeric(scoring.target)
    present = [number for number in numbers if number is not None]
    if not present:
        raise ValueError(f"the target {scoring.target!r} has no numeric values to regress on")

    fill = _quantile(sorted(present), 0.5)
    return [fill if number is None else number for number in numbers]


def _from_vector(
    names: Sequence[str],
    values: Sequence[Any],
    notes: dict[str, str],
    key: str | None,
) -> dict[str, dict[str, Any]]:
    """Turn one score per encoded column back into the per-feature records every scorer returns."""
    scores: dict[str, dict[str, Any]] = {}
    for name, value in zip(names, values):
        score = None if _nan(value) else float(value)
        entry: dict[str, Any] = {"score": score}
        if key and score is not None:
            entry[key] = score
        if name in notes:
            entry["encoding"] = notes[name]
        if score is None:
            entry["note"] = "the estimator returned no score"
        scores[name] = entry

    for column, note in notes.items():  # columns that could not be encoded at all
        scores.setdefault(column, _unscored(note))
    return scores


def _nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _every_class_twice(labels: Sequence[Any]) -> bool:
    """Whether a stratified split is possible: every class needs at least two rows."""
    counts = Counter(labels)
    return bool(counts) and min(counts.values()) >= 2


def _notes(
    result: dict[str, Any],
    written: str | None,
    destination: str | None,
    carried: Sequence[str],
) -> list[str]:
    """The things worth telling the model without making it read the whole report."""
    notes: list[str] = []
    if not result["selected"]:
        notes.append(
            "Nothing was selected. Read `dropped` for the stage each candidate fell at — screening "
            "rejecting everything usually means the table is ids and dates rather than features."
        )
    if result["leakage_suspects"]:
        leaking = ", ".join(entry["feature"] for entry in result["leakage_suspects"])
        notes.append(
            f"Dropped as leakage: {leaking}. A feature that predicts the target almost perfectly is "
            f"usually derived from it, or recorded after it. If one of these is legitimately "
            f"available at prediction time, pass leakage_score=null to keep it."
        )
    if result["redundant"]:
        notes.append(
            f"{len(result['redundant'])} feature(s) were dropped as duplicates of a higher-ranked "
            f"one; see `redundant` for which pairs and how alike they were."
        )
    if result.get("rows_scored", 0) < result.get("rows_total", 0):
        notes.append(
            f"Scored a systematic sample of {result['rows_scored']} of {result['rows_total']} rows. "
            f"Pass a larger max_rows to score all of them."
        )
    if result["task"] == "regression" and result["method"] == "mutual_information":
        notes.append(
            "The target is continuous and was binned into quantiles to score it. For a continuous "
            "estimate rather than a binned one, use sklearn_mutual_information."
        )
    if carried:
        notes.append(f"Carried through without being ranked: {', '.join(carried)}.")
    if not written and not destination:
        notes.append("Nothing was written: pass output_path to keep the result for the next tool.")
    return notes


def _screening_notes(screened: Sequence[dict[str, Any]]) -> list[str]:
    """What the screen found, summarised by the reason rather than by the column."""
    reasons: Counter[str] = Counter(flag for entry in screened for flag in entry["flags"])
    notes = [
        f"{count} column(s) flagged {flag}." for flag, count in sorted(reasons.items())
    ]
    if reasons.get("date_column"):
        notes.append(
            "A raw date is not a feature until something is derived from it: run feature_prep_tool's "
            "date_parts, recency or event_gaps steps first."
        )
    if reasons.get("id_like"):
        notes.append(
            "An id-like column identifies the row rather than describing it. Pass it as keep_columns "
            "to carry it into the output without ranking it."
        )
    return notes

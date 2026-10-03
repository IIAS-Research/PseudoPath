"""Validation scoring shared by the NER adapters."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from numbers import Real
from typing import TYPE_CHECKING

from .._routing import EntityPrediction
from ..training import SelectionMetric

if TYPE_CHECKING:
    from spacy.tokens import Doc


def score_predictions(
    documents: Sequence[Doc],
    predictions: Sequence[Sequence[EntityPrediction]],
    *,
    metric: SelectionMetric,
    labels: Collection[str] | None = None,
) -> float:
    """Filter labels and score exact entity matches or call a custom metric.

    Built-in scores count matching character ranges and labels across all
    documents. A custom metric receives gold rows followed by predicted rows.
    The result must be a finite number.
    """
    if len(documents) != len(predictions):
        raise ValueError("validation prediction count differs from document count")
    gold_rows = tuple(
        tuple(
            EntityPrediction(span.start_char, span.end_char, span.label_)
            for span in document.ents
            if labels is None or span.label_ in labels
        )
        for document in documents
    )
    predicted_rows = tuple(
        tuple(entity for entity in entities if labels is None or entity.label in labels)
        for entities in predictions
    )
    if callable(metric):
        score = metric(gold_rows, predicted_rows)
    else:
        true_positive = predicted = gold = 0
        for expected_row, observed_row in zip(gold_rows, predicted_rows, strict=True):
            expected = set(expected_row)
            observed = set(observed_row)
            true_positive += len(expected & observed)
            predicted += len(observed)
            gold += len(expected)
        if metric == "precision":
            numerator, denominator = true_positive, predicted
        elif metric == "recall":
            numerator, denominator = true_positive, gold
        elif metric == "f1":
            numerator, denominator = 2 * true_positive, gold + predicted
        elif metric == "f2":
            numerator, denominator = 5 * true_positive, 4 * gold + predicted
        else:
            raise ValueError("unknown selection metric")
        score = numerator / denominator if denominator else 0.0
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
        raise ValueError("selection metric must return a finite numeric scalar")
    return float(score)


def labels_in(documents: Sequence[Doc]) -> frozenset[str]:
    return frozenset(span.label_ for document in documents for span in document.ents)

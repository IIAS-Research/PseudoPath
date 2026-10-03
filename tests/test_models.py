from __future__ import annotations

from types import SimpleNamespace

import pytest

from pseudopath._routing import EntityPrediction
from pseudopath.adapters._common import score_predictions


@pytest.mark.parametrize(
    ("metric", "expected"),
    (("f2", 5 / 14), ("f1", 2 / 5), ("precision", 1 / 2), ("recall", 1 / 3)),
)
def test_exact_metrics_filter_both_gold_and_predictions(metric, expected) -> None:
    gold = SimpleNamespace(
        ents=(
            SimpleNamespace(start_char=0, end_char=4, label_="NOM"),
            SimpleNamespace(start_char=5, end_char=8, label_="HOPITAL"),
            SimpleNamespace(start_char=9, end_char=13, label_="HOPITAL"),
        )
    )
    predictions = (
        (
            EntityPrediction(0, 4, "NOM"),
            EntityPrediction(5, 8, "VILLE"),
        ),
    )
    assert score_predictions((gold,), predictions, metric=metric, labels={"NOM"}) == 1.0
    assert score_predictions((gold,), predictions, metric=metric) == pytest.approx(expected)


def test_custom_metric_receives_filtered_offset_entities_and_requires_a_finite_score() -> None:
    gold = SimpleNamespace(
        ents=(
            SimpleNamespace(start_char=0, end_char=4, label_="NOM"),
            SimpleNamespace(start_char=5, end_char=8, label_="HOPITAL"),
        )
    )
    predictions = ((EntityPrediction(0, 4, "NOM"), EntityPrediction(5, 8, "VILLE")), ())

    def custom_metric(expected, observed):
        assert expected == ((EntityPrediction(0, 4, "NOM"),), ())
        assert observed == expected
        return -2.5

    documents = (gold, SimpleNamespace(ents=()))
    assert score_predictions(documents, predictions, metric=custom_metric, labels={"NOM"}) == -2.5
    with pytest.raises(ValueError, match="finite"):
        score_predictions(documents, predictions, metric=lambda gold, predictions: float("nan"))

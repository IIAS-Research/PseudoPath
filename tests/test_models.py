from __future__ import annotations

from types import SimpleNamespace

import pytest

from pseudopath._routing import EntityPrediction
from pseudopath.adapters._common import score_predictions
from pseudopath.adapters.edsnlp import EdsNLPAdapter, _neural_only


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


def test_transformer_pipeline_contains_only_neural_components() -> None:
    class Pipeline:
        pipe_names = ("normalizer", "ner")

    _neural_only(Pipeline())

    class PipelineWithRules:
        pipe_names = ("normalizer", "ner", "clean")

    with pytest.raises(ValueError, match="only normalizer and ner"):
        _neural_only(PipelineWithRules())


def test_transformer_output_requires_ner_span_group() -> None:
    pytest.importorskip("torch")
    document = SimpleNamespace(spans={})
    model = SimpleNamespace(
        pipe_names=("normalizer", "ner"),
        train=lambda _training: None,
        make_doc=lambda _text: document,
        pipe=iter,
    )
    adapter = EdsNLPAdapter()
    with pytest.raises(KeyError, match="pseudo-ml"):
        adapter.predict(model, ["No entity."])
    document.spans["pseudo-ml"] = ()
    assert adapter.predict(model, ["No entity."]) == ((),)


@pytest.mark.parametrize("kind", ["tok2vec", "transformer"])
@pytest.mark.parametrize(
    "settings",
    [{"steps": 1.5}, {"validation_interval": True}, {"seed": 2.5}, {"selection_labels": (123,)}],
)
def test_training_rejects_invalid_setting_types(kind, settings):
    from pseudopath import Tok2VecTraining, TransformerTraining

    cls = Tok2VecTraining if kind == "tok2vec" else TransformerTraining
    with pytest.raises((ValueError, TypeError)):
        cls(**settings)


@pytest.mark.parametrize(
    "kind,settings",
    [
        ("tok2vec", {"batch_size": True}),
        ("tok2vec", {"learning_rate": True}),
        ("transformer", {"max_length": True}),
        ("transformer", {"adam_betas": (0.9,)}),
        ("transformer", {"task_lr": "0.01"}),
    ],
)
def test_training_rejects_invalid_batch_and_numeric_values(kind, settings):
    from pseudopath import Tok2VecTraining, TransformerTraining

    cls = Tok2VecTraining if kind == "tok2vec" else TransformerTraining
    with pytest.raises((ValueError, TypeError)):
        cls(**settings)

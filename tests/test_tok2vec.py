from __future__ import annotations

import edsnlp
import pytest
import spacy
from pseudopath.adapters.spacy import SpacyNERAdapter, _aligned_example
from pseudopath.training import Tok2VecTraining
from spacy.tokens import Doc, Span


def _annotated(text: str):
    nlp = edsnlp.blank("eds")
    document = nlp.make_doc(text)
    entity = document.char_span(0, 4, label="NOM", alignment_mode="strict")
    assert entity is not None
    document.ents = (entity,)
    return document


@pytest.mark.parametrize(
    ("selection", "scores", "greater_is_better", "expected_step"),
    (
        ("best", (0.5, 0.5), True, 2),
        ("last", (0.8, 0.5), True, 3),
        ("best", (-0.8, -0.5), True, 3),
        ("best", (0.8, 0.5), False, 3),
    ),
)
def test_tok2vec_selects_checkpoint_and_roundtrips(
    tmp_path, selection, scores, greater_is_better, expected_step
) -> None:
    values = iter(scores)

    def metric(gold, predictions):
        return next(values)

    model = spacy.blank("eds")
    model.add_pipe("ner")
    adapter = SpacyNERAdapter()
    assert adapter.selected_step is None
    assert adapter.selected_score is None
    adapter.fit(
        model,
        (_annotated("Jean arrive."),),
        (_annotated("Jean reste."),),
        training=Tok2VecTraining(
            steps=3,
            validation_interval=2,
            batch_size=1,
            checkpoint_selection=selection,
            selection_metric=metric,
            greater_is_better=greater_is_better,
        ),
        work_dir=tmp_path / "fit",
    )

    assert adapter.training_summary is not None
    assert [row["step"] for row in adapter.training_summary["checkpoints"]] == [2, 3]
    assert [row["path"] for row in adapter.training_summary["checkpoints"]] == [
        "checkpoints/step-00000002",
        "checkpoints/step-00000003",
    ]
    assert sorted(path.name for path in (tmp_path / "fit" / "checkpoints").iterdir()) == [
        "step-00000002",
        "step-00000003",
    ]
    assert adapter.training_summary["selected_step"] == expected_step
    assert adapter.training_summary["selected_score"] == scores[0 if expected_step == 2 else 1]
    assert adapter.selected_step == adapter.training_summary["selected_step"]
    assert adapter.selected_score == adapter.training_summary["selected_score"]
    assert adapter.training_summary["training"]["selection_metric"] == "custom"

    expected = adapter.predict(model, ("Jean reste.",))
    artifact = tmp_path / "artifact"
    adapter.to_disk(model, artifact)
    restored = spacy.load(artifact)
    assert adapter.predict(restored, ("Jean reste.",)) == expected


def test_tok2vec_aligns_foreign_tokenization_before_training() -> None:
    foreign = spacy.blank("fr")
    source = Doc(
        foreign.vocab,
        words=[" ", "Dr.Du", "pont", " "],
        spaces=[False, False, False, False],
    )
    source.ents = (Span(source, 0, 2, label="LONG"), Span(source, 2, 4, label="SHORT"))
    nlp = spacy.blank("eds")
    nlp.add_pipe("ner")
    example = _aligned_example(nlp, source)
    assert [(span.start_char, span.end_char, span.label_) for span in example.reference.ents] == [
        (1, 10, "LONG")
    ]

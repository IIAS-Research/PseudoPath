from __future__ import annotations

from types import SimpleNamespace

import pytest
from pseudopath._calibration import calibrate_profiles
from pseudopath._router import LineRouter, RouterTraining
from pseudopath._routing import (
    EntityPrediction,
    Line,
    merge_spans,
    pack_selected_lines,
    project_packed_spans,
    route_document,
    split_lines,
)


def _doc(text: str, *entities: EntityPrediction) -> SimpleNamespace:
    spans = tuple(
        SimpleNamespace(start_char=entity.start, end_char=entity.end, label_=entity.label)
        for entity in entities
    )
    return SimpleNamespace(text=text, ents=spans)


def test_packing_projection_and_boundaries() -> None:
    text = "alpha\nbeta\ngamma"
    assert split_lines(text) == (Line(0, 6), Line(6, 11), Line(11, 16))
    packed = pack_selected_lines(text, (Line(0, 6), Line(11, 16)))
    assert packed is not None
    assert packed.text == "alpha\n\ngamma"
    assert project_packed_spans(
        packed,
        (
            EntityPrediction(0, 5, "NOM"),
            EntityPrediction(5, 8, "NOM"),  # Crosses the inserted separator.
            EntityPrediction(7, 12, "NOM"),
        ),
    ) == (
        EntityPrediction(0, 5, "NOM"),
        EntityPrediction(11, 16, "NOM"),
    )


def test_merge_prefers_length_leftmost_then_model() -> None:
    assert merge_spans(
        (EntityPrediction(0, 4, "MODEL"),),
        (EntityPrediction(0, 4, "RULE"),),
    ) == (EntityPrediction(0, 4, "MODEL"),)
    assert merge_spans(
        (EntityPrediction(1, 5, "MODEL"),),
        (EntityPrediction(0, 4, "RULE"),),
    ) == (EntityPrediction(0, 4, "RULE"),)
    assert merge_spans(
        (EntityPrediction(1, 5, "MODEL"),),
        (EntityPrediction(0, 6, "RULE"),),
    ) == (EntityPrediction(0, 6, "RULE"),)


def test_router_fit_continues_weights_and_roundtrips(tmp_path) -> None:
    doc = _doc("Jean\nneutre\n", EntityPrediction(0, 4, "NOM"))
    router = LineRouter(RouterTraining(dimension=1024, epochs=1))
    router.fit((doc,), ((),), target_labels=frozenset({"NOM"}))
    first = router.weights.tobytes()
    assert len(router.scores(doc.text, ())) == 2
    router.fit((doc,), ((),), target_labels=frozenset({"NOM"}))
    assert router.weights.tobytes() != first

    directory = tmp_path / "router"
    router.to_disk(directory)
    loaded = LineRouter.from_disk(directory)
    assert loaded.scores(doc.text, ()) == router.scores(doc.text, ())
    with pytest.raises(ValueError, match="cannot change"):
        router.fit(
            (doc,),
            ((),),
            target_labels=frozenset({"NOM"}),
            training=RouterTraining(dimension=2048, epochs=1),
        )


def test_router_dimension_need_not_be_a_power_of_two() -> None:
    assert RouterTraining(dimension=1025).dimension == 1025
    with pytest.raises(ValueError, match=">= 1024"):
        RouterTraining(dimension=1023)


def test_all_lines_bypasses_router_and_empty_selection_skips_model() -> None:
    model_inputs = []

    def predict(text: str) -> tuple[EntityPrediction, ...]:
        model_inputs.append(text)
        return (EntityPrediction(0, 4, "NOM"),)

    unfitted = LineRouter(RouterTraining(dimension=1024))
    all_lines = route_document("Jean\n", (), predict, router=unfitted, threshold=None)
    assert all_lines.entities == (EntityPrediction(0, 4, "NOM"),)
    assert model_inputs == ["Jean\n"]

    none_selected = route_document(
        "Jean\n", (), predict, router=unfitted, threshold=1.0, scores=(0.2,)
    )
    assert none_selected.entities == ()
    assert model_inputs == ["Jean\n"]


def test_calibration_uses_rules_and_prefers_highest_eligible_threshold() -> None:
    text = "JEAN\nMARIE\n"
    doc = _doc(
        text,
        EntityPrediction(0, 4, "NOM"),
        EntityPrediction(5, 10, "PRENOM"),
    )
    rules = ((EntityPrediction(0, 4, "NOM"),),)
    prediction_seen = False

    class Router:
        def scores(self, _text, _rules):
            assert prediction_seen  # The first call uses the full document.
            return (0.2, 0.8)

    def predict(packed: str) -> tuple[EntityPrediction, ...]:
        nonlocal prediction_seen
        prediction_seen = True
        start = packed.find("MARIE")
        if start < 0:
            return ()
        return (EntityPrediction(start, start + 5, "PRENOM"),)

    labels = frozenset({"NOM", "PRENOM"})
    profiles = calibrate_profiles(
        (doc,),
        rules,
        Router(),
        predict,
        target_labels=labels,
        recall_groups={"target": labels},
    )
    assert profiles == {"prudent": 0.8, "balanced": 0.8, "fast": 0.8}


def test_recall_group_protects_rare_hospital_label() -> None:
    text = "A" * 1000 + "\nB"
    doc = _doc(
        text,
        EntityPrediction(0, 1000, "NOM"),
        EntityPrediction(1001, 1002, "HOPITAL"),
    )

    class Router:
        def scores(self, _text, _rules):
            return (0.9, 0.2)

    def predict(packed: str) -> tuple[EntityPrediction, ...]:
        spans = []
        if packed.startswith("A"):
            spans.append(EntityPrediction(0, 1000, "NOM"))
        if (position := packed.find("B")) >= 0:
            spans.append(EntityPrediction(position, position + 1, "HOPITAL"))
        return tuple(spans)

    labels = frozenset({"NOM", "HOPITAL"})
    global_profiles = calibrate_profiles(
        (doc,),
        ((),),
        Router(),
        predict,
        target_labels=labels,
        recall_groups={"target": labels},
    )
    grouped_profiles = calibrate_profiles(
        (doc,),
        ((),),
        Router(),
        predict,
        target_labels=labels,
        recall_groups={
            "identifiers": frozenset({"NOM"}),
            "hospital": frozenset({"HOPITAL"}),
        },
    )
    assert global_profiles["prudent"] == 0.9
    assert grouped_profiles == {"prudent": 0.2, "balanced": 0.2, "fast": 0.2}

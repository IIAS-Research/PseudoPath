from __future__ import annotations


from pseudopath._routing import (
    EntityPrediction,
    Line,
    merge_spans,
    pack_selected_lines,
    project_packed_spans,
    split_lines,
)


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

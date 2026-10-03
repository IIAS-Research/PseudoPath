"""Tests for entity boundary cleanup and overlap removal."""

import spacy
from spacy.tokens import Span

from pseudopath._entity_cleanup import clean_and_select


def test_label_specific_boundaries_and_empty_spans() -> None:
    nlp = spacy.blank("fr")
    cases = (
        ("(+33) 6 12 34 56 78!!!", "TEL", "(+33) 6 12 34 56 78"),
        ("!!! Alice.", "PRENOM", "Alice."),
        ("!!! Alice.", "NOM", "Alice"),
        ("!!!", "NOM", None),
    )
    for text, label, expected in cases:
        doc = nlp.make_doc(text)
        spans = clean_and_select((Span(doc, 0, len(doc), label=label),))
        assert [span.text for span in spans] == ([] if expected is None else [expected])


def test_selection_prefers_longest_leftmost_and_input_order_on_ties() -> None:
    doc = spacy.blank("fr").make_doc("Alice Bob Claire")
    spans = (
        Span(doc, 0, 2, label="FIRST"),
        Span(doc, 1, 3, label="OTHER"),
        Span(doc, 0, 2, label="SECOND"),
        Span(doc, 2, 3, label="TAIL"),
        Span(doc, 0, 1, label="SHORT"),
    )
    assert [(span.text, span.label_) for span in clean_and_select(spans)] == [
        ("Alice Bob", "FIRST"),
        ("Claire", "TAIL"),
    ]

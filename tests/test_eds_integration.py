"""Tests for fixed rules and training document preparation."""

import random

import edsnlp
from pseudopath._reader import read_training_docs
from pseudopath._rules import _FixedRules


def test_fixed_rules_keep_published_entity_behavior() -> None:
    rules = _FixedRules()
    cases = {
        "Contact : test@example.fr ou 06 12 34 56 78.": [
            ("MAIL", "test@example.fr"),
            ("TEL", "06 12 34 56 78"),
        ],
        "Le patient habite 12 rue de la Paix 75002 PARIS.": [
            ("ADRESSE", "12 rue de la Paix"),
            ("ZIP", "75002"),
        ],
        "Numéro 1 84 05 75 116 006 11": [("SECU", "1 84 05 75 116 006 11")],
    }
    for text, expected in cases.items():
        doc = rules.make_doc(text)
        assert [(item.label, text[item.start : item.end]) for item in rules.predict(doc)] == (
            expected
        )

    doc = rules.make_doc("Luc Martin consulte.")
    doc._.context = {"PRENOM": ["Luc"], "NOM": ["Martin"]}
    assert [(item.label, doc.text[item.start : item.end]) for item in rules.predict(doc)] == [
        ("PRENOM", "Luc"),
        ("NOM", "Martin"),
    ]


def test_training_reader_preserves_gold_spans_when_splitting() -> None:
    nlp = edsnlp.blank("eds")
    doc = nlp.make_doc("Luc Martin consulte.\nIl habite Paris.\nUne ligne ordinaire suit.")
    name = doc.char_span(0, 10, label="NOM", alignment_mode="expand")
    assert name is not None
    doc.ents = (name,)
    doc.spans["pseudo-ml"] = [name]

    random.seed(42)
    parts = read_training_docs(nlp, (doc,), max_length=8)
    assert [part.text for part in parts] == [
        "Luc Martin consulte.",
        "\nIl habite Paris.\n",
        "Une ligne ordinaire suit.",
    ]
    assert [(span.text, span.label_) for span in parts[0].spans["pseudo-ml"]] == [
        ("Luc Martin", "NOM")
    ]
    assert all(not part.spans["pseudo-ml"] for part in parts[1:])


def test_context_short_names_and_email_alias_preserve_all_terms():
    rules = _FixedRules()
    doc = rules.make_doc("Li Ng Wu first@example.fr second@example.fr")
    context = {
        "NOM": ["Li", "Ng", "Wu"],
        "MAIL": ["first@example.fr"],
        "EMAIL": ["second@example.fr"],
    }
    doc._.context = context
    assert [(doc.text[e.start : e.end], e.label) for e in rules.predict(doc)] == [
        ("Li", "NOM"),
        ("Ng", "NOM"),
        ("Wu", "NOM"),
        ("first@example.fr", "MAIL"),
        ("second@example.fr", "MAIL"),
    ]
    # Exercise the alias itself without relying on the built-in email regex.
    doc = rules.make_doc("Martin Dupont")
    doc._.context = {"MAIL": ["Martin"], "EMAIL": ["Dupont"]}
    assert [doc.text[e.start : e.end] for e in rules.predict(doc)] == ["Martin", "Dupont"]
    assert context["MAIL"] == ["first@example.fr"]


def test_invalid_context_is_rejected():
    import pytest

    rules = _FixedRules()
    for context in ({"PRENOM": "Luc"}, [], {"NOM": [123]}, {"": ["Luc"]}):
        doc = rules.make_doc("Luc")
        doc._.context = context
        with pytest.raises(ValueError, match="context"):
            rules.predict(doc)


def test_international_phone_formats():
    rules = _FixedRules()
    for phone in (
        "+33 6 12 34 56 78",
        "0033 6 12 34 56 78",
        "+33 (0)6 12 34 56 78",
        "06 12 34 56 78",
        "06.12.34.56.78",
        "06-12-34-56-78",
    ):
        doc = rules.make_doc("Contact : " + phone + ".")
        assert [(doc.text[e.start : e.end], e.label) for e in rules.predict(doc)] == [
            (phone, "TEL")
        ]

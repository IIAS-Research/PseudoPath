"""Tests for fixed rules and training document preparation."""

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

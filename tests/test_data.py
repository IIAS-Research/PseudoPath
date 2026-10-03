import json

import pytest
from pseudopath.data import read_jsonl


def test_jsonl_preserves_exact_offsets_and_context(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    text = "Maël-Dupont\nconsulte."
    row = {
        "note_text": text,
        "entities": [{"start": 1, "end": 3, "label": "NOM"}],
        "context": {"NOM": ["Dupont"]},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    (document,) = read_jsonl(path, text_key="note_text")
    assert document.text == text
    assert [(span.start_char, span.end_char, span.text, span.label_) for span in document.ents] == [
        (1, 3, "aë", "NOM")
    ]
    assert document._.context == row["context"]


def test_jsonl_reads_negative_documents(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text('{"text": "Contact : 0612345678.", "entities": []}\n', encoding="utf-8")
    (document,) = read_jsonl(path)
    assert document.text == "Contact : 0612345678."
    assert document.char_span(10, 20, label="TEL").text == "0612345678"
    assert not document.ents
    assert document._.context is None


@pytest.mark.parametrize(
    "entities, message",
    [
        ([{"start": 0, "end": 20, "label": "NOM"}], "invalid entity"),
        (
            [
                {"start": 0, "end": 4, "label": "NOM"},
                {"start": 2, "end": 6, "label": "PRENOM"},
            ],
            "must not overlap",
        ),
    ],
)
def test_jsonl_rejects_invalid_annotations(tmp_path, entities, message) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps({"text": "Martin", "entities": entities}), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        read_jsonl(path)
    assert f"{path}:1" in str(error.value)

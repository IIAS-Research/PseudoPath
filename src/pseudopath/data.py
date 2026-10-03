"""Read character-offset annotations without changing their boundaries."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import edsnlp
from spacy.tokens import Doc


def read_jsonl(path: str | Path, *, text_key: str = "text") -> tuple[Doc, ...]:
    """Read JSON Lines with ``text`` and ``entities`` fields.

    Each entity has ``start``, ``end``, and ``label`` fields. Character ranges
    include ``start`` and exclude ``end``. Use ``text_key="note_text"`` for
    datasets with that text field.
    Optional ``context`` maps entity labels to lists of known terms.

    The returned documents preserve exact annotations. Model adapters apply their
    own tokenization for training.
    """
    path = Path(path)
    tokenizer = edsnlp.blank("eds")
    if not Doc.has_extension("context"):
        Doc.set_extension("context", default=None)

    documents = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{location}: document must be an object")
            text = row.get(text_key)
            entities = row.get("entities")
            if not isinstance(text, str) or not text:
                raise ValueError(f"{location}: {text_key} must be a nonempty string")
            if not isinstance(entities, list):
                raise ValueError(f"{location}: entities must be a list")
            for entity in entities:
                if (
                    not isinstance(entity, dict)
                    or type(entity.get("start")) is not int
                    or type(entity.get("end")) is not int
                    or not isinstance(entity.get("label"), str)
                    or not entity["label"]
                    or not 0 <= entity["start"] < entity["end"] <= len(text)
                ):
                    raise ValueError(f"{location}: invalid entity")
            entities = sorted(entities, key=lambda entity: entity["start"])
            if any(left["end"] > right["start"] for left, right in pairwise(entities)):
                raise ValueError(f"{location}: entities must not overlap")

            tokens = tokenizer.make_doc(text)
            boundaries = sorted(
                {
                    0,
                    len(text),
                    *(offset for token in tokens for offset in (token.idx, token.idx + len(token))),
                    *(offset for entity in entities for offset in (entity["start"], entity["end"])),
                }
            )
            document = Doc(
                tokenizer.vocab,
                words=[text[start:end] for start, end in pairwise(boundaries)],
                spaces=[False] * (len(boundaries) - 1),
            )
            document.ents = [
                document.char_span(entity["start"], entity["end"], label=entity["label"])
                for entity in entities
            ]
            context = row.get("context")
            if context is not None and (
                not isinstance(context, dict)
                or any(
                    not isinstance(label, str)
                    or not isinstance(terms, list)
                    or any(not isinstance(term, str) for term in terms)
                    for label, terms in context.items()
                )
            ):
                raise ValueError(f"{location}: context must map labels to lists of strings")
            document._.context = context
            documents.append(document)

    if not documents:
        raise ValueError(f"{path}: no documents")
    return tuple(documents)

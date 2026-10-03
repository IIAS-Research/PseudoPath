"""Prepare annotated documents for EDS-NLP transformer training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from spacy.tokens import Doc, Span


def _copy_chunk(doc: Doc, start: int, end: int) -> Doc:
    """Copy a token slice and clip its annotated spans to the chunk boundaries."""
    chunk = doc[start:end].as_doc(copy_user_data=True)
    # Keep document extensions, including patient context, on the chunk.
    chunk.user_data.update(doc.user_data)
    for name, spans in doc.spans.items():
        projected = []
        for span in spans:
            left = max(span.start, start)
            right = min(span.end, end)
            if left < right:
                projected.append(Span(chunk, left - start, right - start, label=span.label))
        chunk.spans[name] = projected
    return chunk


def _chunk_bounds(doc: Doc, max_length: int) -> Iterator[tuple[int, int]]:
    # Keep sentence boundaries away from annotated tokens.
    for entity in doc.ents:
        for token in entity:
            token.is_sent_start = False

    sentence_ends = (
        (sentence.end for sentence in doc.sents)
        if doc.has_annotation("SENT_START")
        else (len(doc),)
    )
    start = end = 0
    for end in sentence_ends:
        if end - start <= max_length:
            continue
        while end - start > max_length:
            # Bias random chunk lengths toward max_length.
            size = max(1, int(max_length * random.random() ** 0.3))
            next_end = start + size
            yield start, next_end
            start = next_end
        yield start, end
        start = end
    if start < end:
        yield start, end


def read_training_docs(nlp: object, documents: Sequence[Doc], max_length: int) -> tuple[Doc, ...]:
    """Normalize and shuffle documents, then copy their annotations into chunks."""
    import edsnlp

    preparer = edsnlp.Pipeline(nlp.lang, vocab=nlp.vocab, vocab_config=None)
    preparer.add_pipe("eds.normalizer")
    preparer.add_pipe("eds.sentences")
    normalized = list(preparer.pipe(documents))
    random.shuffle(normalized)

    chunks = []
    for doc in normalized:
        if not doc:
            continue
        for start, end in _chunk_bounds(doc, max_length):
            chunk = _copy_chunk(doc, start, end)
            if chunk.text.strip():
                chunks.append(chunk)
    return tuple(chunks)

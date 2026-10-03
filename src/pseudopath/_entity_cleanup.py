"""Clean entity boundaries and select non-overlapping spans."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence

from edsnlp.utils.filter import filter_spans
from spacy.tokens import Span

_ALL_PUNCT = re.escape(string.punctuation)
_NO_DOT = re.escape(string.punctuation.replace(".", ""))
_TEL_PUNCT = re.escape(string.punctuation.replace("+", "").replace("(", ""))
_EMPTY_CHARS = string.punctuation + " \n"

# Adapted from EDS-Pseudo's CleanEntities.
# TEL keeps a leading + or (. PRENOM keeps a trailing dot.
_DEFAULT = re.compile(rf"^[\s{_ALL_PUNCT}]*(.*?)[\s{_ALL_PUNCT}]*$", re.DOTALL)
_TEL = re.compile(rf"^[\s{_TEL_PUNCT}]*(.*?)[\s{_ALL_PUNCT}]*$", re.DOTALL)
_PRENOM = re.compile(rf"^[\s{_ALL_PUNCT}]*(.*?)[\s{_NO_DOT}]*$", re.DOTALL)


def clean_and_select(spans: Sequence[Span]) -> tuple[Span, ...]:
    """Trim punctuation, then keep the longest leftmost non-overlapping spans."""
    cleaned = []
    for span in spans:
        if not span.text.strip(_EMPTY_CHARS):
            continue

        pattern = _TEL if span.label_ == "TEL" else _PRENOM if span.label_ == "PRENOM" else _DEFAULT
        match = pattern.match(span.text)
        start = span[0].idx + match.start(1)
        end = span[0].idx + match.end(1)
        result = span.doc.char_span(start, end, label=span.label_, alignment_mode="expand")
        if result is not None:
            cleaned.append(result)

    return tuple(filter_spans(cleaned))

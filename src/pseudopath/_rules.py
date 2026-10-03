"""Fixed rules for entities that do not need a trained model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import punctuation

import edsnlp
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.utils.filter import filter_spans
from spacy.tokens import Doc, Span

from ._entity_cleanup import clean_and_select
from ._routing import EntityPrediction
from ._rule_patterns import ADDRESS_PATTERNS, SIMPLE_PATTERNS

_PUNCT_TO_SPACE = str.maketrans(punctuation, " " * len(punctuation))


class _FixedRules:
    """Match fixed rule patterns and terms supplied by the document context."""

    def __init__(self) -> None:
        if not Doc.has_extension("context"):
            Doc.set_extension("context", default=None)

        nlp = edsnlp.blank("eds")
        nlp.add_pipe("eds.normalizer", name="normalizer")
        self._nlp = nlp

        self._simple = RegexMatcher(attr="NORM")
        self._simple.build_patterns(SIMPLE_PATTERNS)
        self._addresses = RegexMatcher(attr="NORM")
        self._addresses.build_patterns({"ADRESSE": ADDRESS_PATTERNS})

    def make_doc(self, text: str) -> Doc:
        """Create an EDS-tokenized document without running the rules."""
        return self._nlp.make_doc(text)

    def _address_spans(self, document: Doc) -> list[Span]:
        """Keep supported address matches and their city and postcode spans."""
        accepted = []
        for span, groups in self._addresses.match_with_groupdict_as_spans(document):
            if groups.get("UPPER_STREET") is not None and groups.get("NUMERO") is not None:
                accepted.append((span, groups))
            elif (
                groups.get("ZIP") is not None
                and groups.get("VILLE") is not None
                and groups.get("NUMERO") is not None
            ):
                accepted.append((span, groups))
            elif (
                groups.get("ZIP") is not None
                or groups.get("VILLE") is not None
                or groups.get("TRIGGER") is not None
            ) and (
                groups.get("STREET_PIECE") is not None
                or (
                    groups.get("NUMERO") is not None
                    and groups.get("LOWER_STREET_PIECE") is not None
                )
            ):
                accepted.append((span, groups))

        addresses, cities, zip_codes = [], [], []
        for span, groups in filter_spans(accepted):
            addresses.append(span)
            if "ZIP" in groups:
                zip_codes.append(groups["ZIP"])
            if "VILLE" in groups:
                cities.append(groups["VILLE"])
        return [*addresses, *cities, *zip_codes]

    def _context_spans(self, document: Doc, context: Mapping[str, Sequence[str]]) -> list[Span]:
        """Match known patient terms on the normalized document."""
        if not isinstance(context, Mapping) or any(
            not isinstance(label, str)
            or not label
            or not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) for value in values)
            for label, values in context.items()
        ):
            raise ValueError("context must map nonempty labels to lists of strings")
        terms = {label: list(values) for label, values in context.items()}
        if "EMAIL" in terms:
            terms.setdefault("MAIL", []).extend(terms.pop("EMAIL"))
        if not terms:
            return []

        patterns = {
            label: {
                variant
                for value in values
                if value.translate(_PUNCT_TO_SPACE).strip()
                for variant in (value, value.title(), value.upper())
            }
            for label, values in terms.items()
        }
        matcher = EDSPhraseMatcher(self._nlp.vocab, attr="NORM")
        matcher.build_patterns(nlp=self._nlp, terms=patterns)
        spans = list(matcher(document, as_spans=True))
        for span in spans:
            if span.label_ == "NOM_NAISS":
                span.label_ = "NOM"
        return spans

    def predict(self, source: Doc) -> tuple[EntityPrediction, ...]:
        """Find rule entities on a copy, preserving the source annotations."""
        document = self._nlp(self.make_doc(source.text))
        if document.text != source.text:
            raise RuntimeError("rule normalization changed the document text")
        context = source._.context if source._.context is not None else {}
        spans = [
            *self._simple(document, as_spans=True),
            *self._address_spans(document),
            *self._context_spans(document, context),
        ]
        return tuple(
            EntityPrediction(span.start_char, span.end_char, span.label_)
            for span in clean_and_select(spans)
        )

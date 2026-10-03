"""Line packing and span projection for routed inference."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._router import LineRouter


@dataclass(frozen=True, slots=True, order=True)
class EntityPrediction:
    """An entity label and character range, with an exclusive end offset."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or not self.label:
            raise ValueError("entity predictions need valid offsets and a label")


@dataclass(frozen=True, slots=True)
class Line:
    """A source line's character range, including its trailing newline."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PackedSegment:
    """Map one continuous section of packed text to the source document."""

    packed_start: int
    packed_end: int
    document_start: int


@dataclass(frozen=True, slots=True)
class PackedInput:
    """Selected text and the source mapping for each section."""

    text: str = field(repr=False)
    segments: tuple[PackedSegment, ...]


@dataclass(frozen=True, slots=True)
class RoutedPrediction:
    """Merged entities and character counts for the selected lines and model input."""

    entities: tuple[EntityPrediction, ...]
    selected_characters: int
    model_characters: int


def split_lines(text: str) -> tuple[Line, ...]:
    """Split physical lines, including their newline character."""
    if not text:
        return ()
    lines = []
    start = 0
    for match in re.finditer("\n", text):
        lines.append(Line(start, match.end()))
        start = match.end()
    if start < len(text):
        lines.append(Line(start, len(text)))
    return tuple(lines)


def pack_selected_lines(
    text: str,
    lines: Sequence[Line],
    *,
    separator: str = "\n",
) -> PackedInput | None:
    """Pack selected lines into one text and keep their source offsets.

    Lines must be ordered and disjoint. Adjacent lines form one segment.
    Gaps are replaced by ``separator``. An empty selection returns ``None``.
    """
    if not separator:
        raise ValueError("separator must not be empty")
    if not lines:
        return None

    runs: list[Line] = []
    for line in lines:
        if not 0 <= line.start < line.end <= len(text):
            raise ValueError("selected line lies outside its document")
        if runs and line.start < runs[-1].end:
            raise ValueError("selected lines must be ordered and disjoint")
        if runs and line.start == runs[-1].end:
            runs[-1] = Line(runs[-1].start, line.end)
        else:
            runs.append(line)

    chunks: list[str] = []
    segments: list[PackedSegment] = []
    cursor = 0
    for index, run in enumerate(runs):
        if index:
            chunks.append(separator)
            cursor += len(separator)
        chunk = text[run.start : run.end]
        chunks.append(chunk)
        segments.append(PackedSegment(cursor, cursor + len(chunk), run.start))
        cursor += len(chunk)
    return PackedInput(
        text="".join(chunks),
        segments=tuple(segments),
    )


def project_packed_spans(
    packed: PackedInput,
    spans: Sequence[EntityPrediction],
) -> tuple[EntityPrediction, ...]:
    """Map model spans back to the document.

    Keep spans within one source segment. Drop spans that cross a gap or
    cover an inserted separator.
    """
    projected = []
    for span in spans:
        if span.end > len(packed.text):
            raise ValueError("model prediction lies outside its packed input")
        for segment in packed.segments:
            if segment.packed_start <= span.start < span.end <= segment.packed_end:
                offset = segment.document_start - segment.packed_start
                projected.append(
                    EntityPrediction(span.start + offset, span.end + offset, span.label)
                )
                break
    return tuple(projected)


def merge_spans(
    model_spans: Sequence[EntityPrediction],
    rule_spans: Sequence[EntityPrediction],
) -> tuple[EntityPrediction, ...]:
    """Union overlapping ranges without losing detected characters.

    Label each union using the longest original span, then the earliest.
    Identical boundaries prefer the model label. Adjacent ranges stay separate.
    """
    candidates = [(span, 0) for span in model_spans] + [(span, 1) for span in rule_spans]
    candidates.sort(key=lambda item: (item[0].start, item[0].end))
    components: list[list[tuple[EntityPrediction, int]]] = []
    end = -1
    for candidate in candidates:
        span, _source = candidate
        if not components or span.start >= end:
            components.append([candidate])
            end = span.end
        else:
            components[-1].append(candidate)
            end = max(end, span.end)
    result = []
    for component in components:
        winner, _source = min(
            component,
            key=lambda item: (
                -(item[0].end - item[0].start),
                item[0].start,
                item[1],
                item[0].label,
            ),
        )
        result.append(
            EntityPrediction(
                min(item[0].start for item in component),
                max(item[0].end for item in component),
                winner.label,
            )
        )
    return tuple(result)


def route_document(
    text: str,
    rule_spans: Sequence[EntityPrediction],
    predict_packed: Callable[[str], Sequence[EntityPrediction]],
    *,
    router: LineRouter | None,
    threshold: float | None,
    scores: Sequence[float] | None = None,
) -> RoutedPrediction:
    """Select lines, predict their entities, and merge them with rule results.

    ``threshold=None`` sends the whole document to the model without scoring
    the router. Calibration can pass cached ``scores`` to avoid computing
    them again. If no lines are selected, return rule entities alone.
    """
    lines = split_lines(text)
    if threshold is None:
        selected = lines
    else:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number between 0 and 1")
        if scores is None:
            if router is None:
                raise ValueError("routed prediction requires a fitted router")
            scores = router.scores(text, rule_spans)
        if len(scores) != len(lines):
            raise ValueError("router returned one score per physical line")
        selected = tuple(
            line for line, score in zip(lines, scores, strict=True) if score >= threshold
        )

    packed = pack_selected_lines(text, selected)
    if packed is None:
        return RoutedPrediction(merge_spans((), rule_spans), 0, 0)
    local_spans = predict_packed(packed.text)
    projected = project_packed_spans(packed, local_spans)
    return RoutedPrediction(
        entities=merge_spans(projected, rule_spans),
        selected_characters=sum(line.length for line in selected),
        model_characters=len(packed.text),
    )

"""Choose routing thresholds within the validation recall budgets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from ._router import LineRouter
from ._routing import EntityPrediction, RoutedPrediction, route_document, split_lines

if TYPE_CHECKING:
    from spacy.tokens import Doc

# Recall is measured from 0 to 1. A loss of 0.001 is 0.1 percentage point.
PROFILE_LOSSES = {
    "prudent": 0.001,
    "balanced": 0.005,
    "fast": 0.010,
}


def _selection_key(scores: Sequence[float], threshold: float) -> tuple[int, ...]:
    return tuple(index for index, score in enumerate(scores) if score >= threshold)


def _gold_prefixes(
    documents: Sequence[Doc], target_labels: frozenset[str]
) -> tuple[tuple[tuple[int, ...], ...], int]:
    prefixes = []
    support = 0
    for document in documents:
        text = document.text
        target = bytearray(len(text))
        for span in document.ents:
            if span.label_ in target_labels:
                for position in range(span.start_char, span.end_char):
                    target[position] = text[position].isalnum()
        # Prefix sums count gold characters inside any predicted range.
        prefix = [0]
        for value in target:
            prefix.append(prefix[-1] + value)
        prefixes.append(tuple(prefix))
        support += prefix[-1]
    if not support:
        raise ValueError("character recall is undefined without gold target characters")
    return tuple(prefixes), support


def _character_recall(
    predictions: Sequence[RoutedPrediction],
    prefixes: Sequence[Sequence[int]],
    support: int,
) -> float:
    covered = sum(
        prefix[span.end] - prefix[span.start]
        for state, prefix in zip(predictions, prefixes, strict=True)
        for span in state.entities
    )
    return covered / support


def calibrate_profiles(
    documents: Sequence[Doc],
    rule_spans: Sequence[Sequence[EntityPrediction]],
    router: LineRouter,
    predict_packed: Callable[[str], Sequence[EntityPrediction]],
    *,
    target_labels: frozenset[str],
    recall_groups: Mapping[str, frozenset[str]],
) -> dict[str, float]:
    """Choose the threshold that sends the fewest characters to the model.

    Each profile limits character-recall loss against all-lines prediction
    on the validation corpus, both overall and within each recall group.
    Cache predictions by selected lines, since several thresholds can select
    the same lines in a document.
    """
    if len(documents) != len(rule_spans):
        raise ValueError("documents and rule predictions have different lengths")
    prefixes, support = _gold_prefixes(documents, target_labels)
    if not recall_groups:
        raise ValueError("recall_groups cannot be empty")
    group_prefixes = {
        name: _gold_prefixes(documents, labels) for name, labels in recall_groups.items()
    }

    baseline = tuple(
        route_document(
            document.text,
            rules,
            predict_packed,
            router=None,
            threshold=None,
        )
        for document, rules in zip(documents, rule_spans, strict=True)
    )
    baseline_recall = _character_recall(baseline, prefixes, support)
    baseline_group_recalls = {
        name: _character_recall(baseline, group_prefix, group_support)
        for name, (group_prefix, group_support) in group_prefixes.items()
    }

    scores_by_doc = tuple(
        router.scores(document.text, rules)
        for document, rules in zip(documents, rule_spans, strict=True)
    )
    thresholds = tuple(sorted({0.0, 1.0, *(score for scores in scores_by_doc for score in scores)}))
    cached: list[dict[tuple[int, ...], RoutedPrediction]] = []
    for document, rules, scores, full in zip(
        documents, rule_spans, scores_by_doc, baseline, strict=True
    ):
        states = {tuple(range(len(split_lines(document.text)))): full}
        for threshold in (0.0, 1.0, *scores):
            key = _selection_key(scores, threshold)
            if key not in states:
                states[key] = route_document(
                    document.text,
                    rules,
                    predict_packed,
                    router=router,
                    threshold=threshold,
                    scores=scores,
                )
        cached.append(states)

    points = []
    for threshold in thresholds:
        states = tuple(
            choices[_selection_key(scores, threshold)]
            for choices, scores in zip(cached, scores_by_doc, strict=True)
        )
        points.append(
            (
                threshold,
                _character_recall(states, prefixes, support),
                {
                    name: _character_recall(states, group_prefix, group_support)
                    for name, (group_prefix, group_support) in group_prefixes.items()
                },
                sum(state.model_characters for state in states),
                sum(state.selected_characters for state in states),
            )
        )

    # Break equal model-input costs by source length, then the higher threshold.
    return {
        profile: min(
            (
                point
                for point in points
                if baseline_recall - point[1] <= maximum_loss + 1e-12
                and all(
                    baseline_group_recalls[name] - point[2][name] <= maximum_loss + 1e-12
                    for name in recall_groups
                )
            ),
            key=lambda point: (point[3], point[4], -point[0]),
        )[0]
        for profile, maximum_loss in PROFILE_LOSSES.items()
    }

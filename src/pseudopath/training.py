"""Training settings for the supported NER adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import Literal

from ._routing import EntityPrediction

SelectionMetric = (
    Literal["f2", "f1", "precision", "recall"]
    | Callable[[Sequence[Sequence[EntityPrediction]], Sequence[Sequence[EntityPrediction]]], float]
)

PATIENT_IDENTIFIER_LABELS = (
    "ADRESSE",
    "DATE",
    "DATE_NAISSANCE",
    "IPP",
    "MAIL",
    "NDA",
    "NOM",
    "PRENOM",
    "SECU",
    "TEL",
    "VILLE",
    "ZIP",
)


@dataclass(frozen=True, slots=True)
class TransformerTraining:
    """Learning and checkpoint settings for an EDS-NLP Transformer model.

    ``max_length`` limits training chunks in tokens. Model window and stride
    are configured when creating the model. ``trainable_transformer_layers``
    selects the final encoder layers, or all Transformer weights with ``"all"``.
    """

    steps: int = 4000
    validation_interval: int = 400
    seed: int = 42
    batch_words: int = 2000
    grad_accumulation_tokens: int = 32000
    max_length: int = 384
    task_lr: float = 5e-5
    transformer_lr: float = 5e-6
    trainable_transformer_layers: int | Literal["all"] = 4
    warmup_fraction: float = 0.1
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    grad_clip: float = 5.0
    selection_labels: tuple[str, ...] | None = None
    selection_metric: SelectionMetric = "f2"
    greater_is_better: bool = True
    checkpoint_selection: Literal["best", "last"] = "best"

    def __post_init__(self) -> None:
        _validate_integer_settings(
            self,
            (
                "steps",
                "validation_interval",
                "seed",
                "batch_words",
                "grad_accumulation_tokens",
                "max_length",
            ),
        )
        _validate_labels(self.selection_labels)
        for name in (
            "task_lr",
            "transformer_lr",
            "warmup_fraction",
            "weight_decay",
            "adam_epsilon",
            "grad_clip",
        ):
            _validate_real(name, getattr(self, name))
        if not isinstance(self.adam_betas, (tuple, list)) or len(self.adam_betas) != 2:
            raise ValueError("adam_betas must contain two numeric values")
        for value in self.adam_betas:
            _validate_real("adam_betas", value)
        if self.steps < 1 or self.validation_interval < 1:
            raise ValueError("steps and validation_interval must be positive")
        if self.checkpoint_selection not in {"best", "last"}:
            raise ValueError("checkpoint_selection must be 'best' or 'last'")
        if not callable(self.selection_metric) and self.selection_metric not in (
            "f2",
            "f1",
            "precision",
            "recall",
        ):
            raise ValueError(
                "selection_metric must be 'f2', 'f1', 'precision', 'recall', or callable"
            )
        if not isinstance(self.greater_is_better, bool):
            raise TypeError("greater_is_better must be a bool")
        if min(self.batch_words, self.grad_accumulation_tokens, self.max_length) < 1:
            raise ValueError("batch and sequence sizes must be positive")
        if self.trainable_transformer_layers != "all" and (
            type(self.trainable_transformer_layers) is not int
            or self.trainable_transformer_layers < 0
        ):
            raise ValueError("trainable_transformer_layers must be nonnegative or 'all'")
        if self.trainable_transformer_layers == 0 and self.transformer_lr != 0:
            raise ValueError("transformer_lr must be zero when no transformer layer is trainable")
        if self.trainable_transformer_layers != 0 and self.transformer_lr <= 0:
            raise ValueError("transformer_lr must be positive for trainable transformer layers")
        if self.selection_labels is not None and (
            not self.selection_labels
            or len(set(self.selection_labels)) != len(self.selection_labels)
            or any(not label for label in self.selection_labels)
        ):
            raise ValueError("selection_labels must contain distinct nonempty labels")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 < self.adam_betas[0] < 1 or not 0 < self.adam_betas[1] < 1:
            raise ValueError("adam_betas must be in (0, 1)")
        for name in ("task_lr", "transformer_lr", "weight_decay", "adam_epsilon", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or (
                value <= 0 if name in {"task_lr", "adam_epsilon", "grad_clip"} else value < 0
            ):
                raise ValueError(f"{name} is invalid")


@dataclass(frozen=True, slots=True)
class Tok2VecTraining:
    """Learning and checkpoint settings for a spaCy Tok2Vec NER model.

    ``batch_size`` counts documents. ``selection_labels`` limits validation
    scoring, and ``checkpoint_selection`` keeps the best or final checkpoint.
    """

    steps: int = 4000
    validation_interval: int = 400
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 1e-3
    dropout: float = 0.1
    l2: float = 0.01
    grad_clip: float = 1.0
    selection_labels: tuple[str, ...] | None = None
    selection_metric: SelectionMetric = "f2"
    greater_is_better: bool = True
    checkpoint_selection: Literal["best", "last"] = "best"

    def __post_init__(self) -> None:
        _validate_integer_settings(self, ("steps", "validation_interval", "seed", "batch_size"))
        _validate_labels(self.selection_labels)
        for name in ("learning_rate", "dropout", "l2", "grad_clip"):
            _validate_real(name, getattr(self, name))
        if self.steps < 1 or self.validation_interval < 1:
            raise ValueError("steps and validation_interval must be positive")
        if self.checkpoint_selection not in {"best", "last"}:
            raise ValueError("checkpoint_selection must be 'best' or 'last'")
        if not callable(self.selection_metric) and self.selection_metric not in (
            "f2",
            "f1",
            "precision",
            "recall",
        ):
            raise ValueError(
                "selection_metric must be 'f2', 'f1', 'precision', 'recall', or callable"
            )
        if not isinstance(self.greater_is_better, bool):
            raise TypeError("greater_is_better must be a bool")
        if self.selection_labels is not None and (
            not self.selection_labels
            or len(set(self.selection_labels)) != len(self.selection_labels)
            or any(not label for label in self.selection_labels)
        ):
            raise ValueError("selection_labels must contain distinct nonempty labels")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in ("learning_rate", "dropout", "l2", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or (
                value <= 0 if name in {"learning_rate", "grad_clip"} else value < 0
            ):
                raise ValueError(f"{name} is invalid")
        if self.dropout >= 1:
            raise ValueError("dropout must be below 1")


def _training_metadata(settings: TransformerTraining | Tok2VecTraining) -> dict[str, object]:
    """Record settings and replace callable metrics with the name ``custom``."""
    metadata = {field.name: getattr(settings, field.name) for field in fields(settings)}
    if callable(settings.selection_metric):
        metadata["selection_metric"] = "custom"
    return metadata


def _validate_integer_settings(settings: object, names: tuple[str, ...]) -> None:
    for name in names:
        if type(getattr(settings, name)) is not int:
            raise TypeError(f"{name} must be an integer")


def _validate_labels(labels: tuple[str, ...] | None) -> None:
    if labels is not None and (
        not isinstance(labels, (tuple, list))
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("selection_labels must contain distinct nonempty string labels")


def _validate_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")

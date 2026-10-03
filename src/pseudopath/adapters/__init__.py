"""Connect NER model libraries to PseudoPath."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from .._routing import EntityPrediction
from .spacy import SpacyNERAdapter

if TYPE_CHECKING:
    from spacy.tokens import Doc


ModelT = TypeVar("ModelT")


class NERAdapter(Protocol[ModelT]):
    """Train, predict, and save a model supplied by the caller.

    Training must preserve the source documents' annotations and context.
    Predictions use character offsets in the supplied texts.
    """

    def fit(
        self,
        model: ModelT,
        train_docs: Sequence[Doc],
        validation_docs: Sequence[Doc],
        *,
        training: Any,
        work_dir: Path,
    ) -> ModelT:
        """Train the model with the supplied settings and return the model to use."""
        ...

    def predict(self, model: ModelT, texts: Sequence[str]) -> Sequence[Sequence[EntityPrediction]]:
        """Return one sequence of entities per text, in the same order."""
        ...

    def to_disk(self, model: ModelT, path: Path) -> None:
        """Save the model at the supplied path."""
        ...


__all__ = ["NERAdapter", "SpacyNERAdapter"]

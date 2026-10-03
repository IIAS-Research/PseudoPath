"""Character-offset entity predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class EntityPrediction:
    """An entity label and character range, with an exclusive end offset."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or not self.label:
            raise ValueError("entity predictions need valid offsets and a label")

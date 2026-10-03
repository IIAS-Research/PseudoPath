"""Learn which lines contain identifying characters missed by fixed rules."""

from __future__ import annotations

import json
import math
import random
import re
import sys
import unicodedata
import zlib
from array import array
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from ._routing import EntityPrediction, Line, split_lines

if TYPE_CHECKING:
    from spacy.tokens import Doc

_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_UPPER_RE = re.compile(r"A+")
_LOWER_RE = re.compile(r"a+")
_DIGIT_RE = re.compile(r"0+")


@dataclass(frozen=True, slots=True)
class RouterTraining:
    """Settings for the line router's hashed features and logistic training.

    ``dimension`` sets the weight array size. ``ngrams`` sets character
    feature lengths. These settings must stay fixed when continuing weights.
    """

    dimension: int = 1 << 17
    epochs: int = 6
    learning_rate: float = 0.35
    seed: int = 42
    ngrams: tuple[int, ...] = (3, 4, 5)

    def __post_init__(self) -> None:
        for name in ("dimension", "epochs", "seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"router {name} must be an integer")
        if not isinstance(self.ngrams, tuple) or any(type(size) is not int for size in self.ngrams):
            raise TypeError("router ngrams must be a tuple of integers")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real):
            raise TypeError("router learning_rate must be numeric")
        if self.dimension < 1024:
            raise ValueError("router dimension must be >= 1024")
        if self.epochs < 1:
            raise ValueError("router epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("router learning rate must be finite and positive")
        if not self.ngrams or any(size < 1 for size in self.ngrams):
            raise ValueError("router ngrams must be positive")


@dataclass(slots=True)
class LineRouter:
    """Score each line for identifying characters missed by the rules."""

    config: RouterTraining = field(default_factory=RouterTraining)
    weights: array = field(init=False, repr=False)
    fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.weights = array("d", [0.0]) * self.config.dimension

    def scores(
        self,
        text: str,
        rule_spans: Sequence[EntityPrediction],
    ) -> tuple[float, ...]:
        """Return one score between 0 and 1 per source line, in order."""
        if not self.fitted:
            raise RuntimeError("router must be fitted before scoring")
        return tuple(
            _sigmoid(sum(self.weights[index] for index in features) / math.sqrt(len(features)))
            for features in line_features(text, rule_spans, self.config)
        )

    def fit(
        self,
        documents: Sequence[Doc],
        rule_spans: Sequence[Sequence[EntityPrediction]],
        *,
        target_labels: frozenset[str],
        training: RouterTraining | None = None,
    ) -> None:
        """Train from the current weights with a fresh AdaGrad accumulator.

        Both positive and negative residual lines are required. Positive
        lines contain gold target characters that the fixed rules missed.
        """
        settings = training or self.config
        if settings.dimension != self.config.dimension or settings.ngrams != self.config.ngrams:
            raise ValueError("router feature dimensions and ngrams cannot change on fit")
        rows = _training_rows(documents, rule_spans, target_labels, settings)
        positives = sum(target for _, target in rows)
        negatives = len(rows) - positives
        if not positives or not negatives:
            raise ValueError("router training needs positive and negative residual lines")

        # Give positive and negative lines equal total weight in training.
        positive_weight = negatives / positives
        weights = array("d", self.weights)
        accumulated = array("d", [1e-3]) * settings.dimension
        order = list(range(len(rows)))
        generator = random.Random(settings.seed)
        for _epoch in range(settings.epochs):
            generator.shuffle(order)
            for row_index in order:
                features, target = rows[row_index]
                scale = 1.0 / math.sqrt(len(features))
                raw = sum(weights[index] for index in features) * scale
                gradient = (_sigmoid(raw) - float(target)) * (positive_weight if target else 1.0)
                feature_gradient = gradient * scale
                squared = feature_gradient * feature_gradient
                for index in features:
                    accumulated[index] += squared
                    weights[index] -= (
                        settings.learning_rate * feature_gradient / math.sqrt(accumulated[index])
                    )
        self.weights = weights
        self.config = settings
        self.fitted = True

    def to_disk(self, directory: str | Path) -> None:
        """Save weights and settings in a new or empty directory."""
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted router")
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"router directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "weights.bin").open("wb") as stream:
            self.weights.tofile(stream)
        (directory / "router.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "byteorder": sys.byteorder,
                    "weights": len(self.weights),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_disk(cls, directory: str | Path) -> LineRouter:
        """Load weights after checking their size and byte order."""
        directory = Path(directory)
        metadata = json.loads((directory / "router.json").read_text(encoding="utf-8"))
        if metadata["byteorder"] != sys.byteorder:
            raise ValueError("router weight byte order is incompatible")
        raw_config = metadata["config"]
        raw_config["ngrams"] = tuple(raw_config["ngrams"])
        router = cls(RouterTraining(**raw_config))
        weights = array("d")
        with (directory / "weights.bin").open("rb") as stream:
            weights.fromfile(stream, router.config.dimension)
            if stream.read(1):
                raise ValueError("router weight file is too large")
        if len(weights) != router.config.dimension or metadata["weights"] != len(weights):
            raise ValueError("router weight file has the wrong size")
        router.weights = weights
        router.fitted = True
        return router


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    digits = "".join("0" if char.isdigit() else char for char in plain)
    return _SPACE_RE.sub(" ", digits).strip()


def _shape(text: str) -> str:
    value = "".join(
        "A"
        if char.isupper()
        else "a"
        if char.islower()
        else "0"
        if char.isdigit()
        else " "
        if char.isspace()
        else char
        for char in text
    )
    value = _UPPER_RE.sub("A", value)
    value = _LOWER_RE.sub("a", value)
    value = _DIGIT_RE.sub("0", value)
    return _SPACE_RE.sub(" ", value).strip()


def _bucket(value: str, dimension: int) -> int:
    # Index 0 is the bias feature and is excluded from the hash buckets.
    return 1 + zlib.crc32(value.encode("utf-8")) % (dimension - 1)


def _add_tokens(features: set[int], tokens: Sequence[str], prefix: str, dimension: int) -> None:
    features.update(_bucket(f"{prefix}:word:{token}", dimension) for token in tokens)
    features.update(
        _bucket(f"{prefix}:bigram:{left}_{right}", dimension) for left, right in pairwise(tokens)
    )


def _line_rule_spans(
    lines: Sequence[Line], rule_spans: Sequence[EntityPrediction]
) -> tuple[tuple[EntityPrediction, ...], ...]:
    return tuple(
        tuple(span for span in rule_spans if span.start < line.end and line.start < span.end)
        for line in lines
    )


def line_features(
    text: str,
    rule_spans: Sequence[EntityPrediction],
    config: RouterTraining,
) -> tuple[tuple[int, ...], ...]:
    """Hash line text, shapes, position, neighboring words, and rule matches."""
    if any(span.end > len(text) for span in rule_spans):
        raise ValueError("rule span lies outside its document")
    lines = split_lines(text)
    contents = [text[line.start : line.end] for line in lines]
    normalised = [_normalise(content) for content in contents]
    tokens = [_TOKEN_RE.findall(content) for content in normalised]
    rules = _line_rule_spans(lines, rule_spans)
    rows = []
    for index, (line, content) in enumerate(zip(lines, contents, strict=True)):
        features = {0}
        lexical = f"^{normalised[index]}$"
        shape = f"^{_shape(content)}$"
        for size in config.ngrams:
            features.update(
                _bucket(f"current:char:{size}:{lexical[pos : pos + size]}", config.dimension)
                for pos in range(max(0, len(lexical) - size + 1))
            )
            features.update(
                _bucket(f"current:shape:{size}:{shape[pos : pos + size]}", config.dimension)
                for pos in range(max(0, len(shape) - size + 1))
            )
        _add_tokens(features, tokens[index], "current", config.dimension)
        for prefix, neighbour in (("previous", index - 1), ("next", index + 1)):
            if 0 <= neighbour < len(lines):
                _add_tokens(features, tokens[neighbour], prefix, config.dimension)

        line_rules = rules[index]
        features.update(
            _bucket(f"current:rule:{span.label}", config.dimension) for span in line_rules
        )
        features.add(_bucket(f"current:rule-count:{min(8, len(line_rules))}", config.dimension))
        covered = {
            position
            for span in line_rules
            for position in range(max(line.start, span.start), min(line.end, span.end))
            if text[position].isalnum()
        }
        alphanumeric = sum(text[position].isalnum() for position in range(line.start, line.end))
        coverage_bucket = min(10, round(10 * len(covered) / max(1, alphanumeric)))
        features.add(_bucket(f"current:rule-coverage:{coverage_bucket}", config.dimension))
        features.add(
            _bucket(
                f"position:{min(19, index * 20 // max(1, len(lines)))}",
                config.dimension,
            )
        )
        features.add(_bucket(f"length:{min(20, len(normalised[index]) // 5)}", config.dimension))
        rows.append(tuple(sorted(features)))
    return tuple(rows)


def _alphanumeric_positions(text: str, spans: Sequence[EntityPrediction]) -> set[int]:
    return {
        position
        for span in spans
        for position in range(span.start, span.end)
        if text[position].isalnum()
    }


def _training_rows(
    documents: Sequence[Doc],
    rule_predictions: Sequence[Sequence[EntityPrediction]],
    target_labels: frozenset[str],
    config: RouterTraining,
) -> list[tuple[tuple[int, ...], bool]]:
    """Label lines by gold alphanumeric characters left uncovered by rules."""
    if len(documents) != len(rule_predictions):
        raise ValueError("documents and rule predictions have different lengths")
    rows: list[tuple[tuple[int, ...], bool]] = []
    for document, rules in zip(documents, rule_predictions, strict=True):
        text = document.text
        lines = split_lines(text)
        features = line_features(text, rules, config)
        gold = tuple(
            EntityPrediction(span.start_char, span.end_char, span.label_)
            for span in document.ents
            if span.label_ in target_labels
        )
        residual = _alphanumeric_positions(text, gold) - _alphanumeric_positions(text, rules)
        rows.extend(
            (
                feature_row,
                any(position in residual for position in range(line.start, line.end)),
            )
            for line, feature_row in zip(lines, features, strict=True)
        )
    return rows


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))

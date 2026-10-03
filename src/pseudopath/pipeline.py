"""The public PseudoPath pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generic, Self, TypeVar

from spacy.tokens import Doc, Span

from ._calibration import _gold_prefixes, calibrate_profiles
from ._router import LineRouter, RouterTraining, _training_rows
from ._routing import EntityPrediction, route_document
from ._rules import _FixedRules
from .adapters import EdsNLPAdapter, NERAdapter, SpacyNERAdapter
from .training import Tok2VecTraining, TransformerTraining, _training_metadata

_RUNTIME_PACKAGES = ("pseudopath", "edsnlp", "spacy")
_PROFILES = frozenset({"all-lines", "prudent", "balanced", "fast"})
_FORMAT_VERSION = 4

ModelT = TypeVar("ModelT")


def _routing_value(value: str | float) -> str | float:
    if isinstance(value, str):
        if value not in _PROFILES:
            raise ValueError(f"unknown routing profile: {value}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("routing must be a named profile or a threshold between 0 and 1")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("routing threshold must be between 0 and 1")
    return threshold


def _gold_labels(documents: Sequence[Doc]) -> frozenset[str]:
    return frozenset(span.label_ for document in documents for span in document.ents)


def _predictions(
    rows: Sequence[Sequence[EntityPrediction]], count: int
) -> Sequence[Sequence[EntityPrediction]]:
    if len(rows) != count:
        raise RuntimeError("NER model returned an incorrect number of documents")
    return rows


class PseudoPath(Generic[ModelT]):
    """Combine a NER model, fixed rules, and a trained line router.

    Call ``fit`` or load a saved instance before predicting. Predictions are
    spaCy spans on the input document. Rules inspect the whole document,
    while the model processes the lines selected by the routing mode.
    """

    def __init__(
        self,
        model: ModelT,
        *,
        adapter: NERAdapter[ModelT],
        routing: str | float = "all-lines",
    ) -> None:
        self.model = model
        self.adapter = adapter
        self._rules = _FixedRules()
        self._router: LineRouter | None = None
        self._thresholds: dict[str, float] = {}
        self._routing = _routing_value(routing)
        self._target_labels: frozenset[str] | None = None
        self._recall_groups: dict[str, frozenset[str]] | None = None
        self._last_training: dict[str, Any] | None = None
        self._model_labels: frozenset[str] = frozenset()
        self._preset: str | None = None
        self._failed = False
        self._ready = False

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        seed: int = 42,
        model_config: Mapping[str, Any] | None = None,
        routing: str | float = "all-lines",
    ) -> Self:
        """Create a built-in CamemBERT, DistilCamemBERT or Tok2Vec pipeline.

        ``checkpoint`` selects transformer weights or a saved spaCy pipeline.
        ``model_config`` supplies component configuration overrides.
        A new instance must be fitted before it can predict entities.
        """
        from ._presets import create_preset

        model, adapter = create_preset(
            name,
            checkpoint=checkpoint,
            device=device,
            seed=seed,
            model_config=model_config,
        )
        instance = cls(model, adapter=adapter, routing=routing)
        instance._preset = name
        return instance

    def make_doc(self, text: str) -> Doc:
        """Tokenize text for inference and register ``doc._.context``."""
        return self._rules.make_doc(text)

    @property
    def routing(self) -> str | float:
        """Default profile or score threshold used by ``predict``."""
        return self._routing

    @routing.setter
    def routing(self, value: str | float) -> None:
        self._routing = _routing_value(value)

    def fit(
        self,
        train_docs: Sequence[Doc],
        validation_docs: Sequence[Doc],
        *,
        training: object | None = None,
        router: RouterTraining | None = None,
        reset_router: bool = False,
        target_labels: Sequence[str] | None = None,
        recall_groups: Mapping[str, Sequence[str]] | None = None,
        work_dir: str | Path | None = None,
    ) -> Self:
        """Train the model and router, calibrate profiles, and return this instance.

        Documents must have gold entities in ``doc.ents``. Later calls
        continue model and router weights. Set ``reset_router=True`` to train
        a new router. ``target_labels`` and ``recall_groups`` set the labels
        and groups used for calibration on this call.

        ``work_dir`` keeps intermediate files in a new or empty directory.
        If omitted, these files are removed after fitting. After a failure
        that changes model state, reload an artifact or create a new instance.
        """
        if not isinstance(reset_router, bool):
            raise TypeError("reset_router must be a bool")
        if self._failed:
            raise RuntimeError(
                "fit failed previously. Reload the artifact or create a new instance"
            )
        train = tuple(train_docs)
        validation = tuple(validation_docs)
        if not train or not validation:
            raise ValueError("train_docs and validation_docs must be nonempty")
        if not all(isinstance(doc, Doc) for doc in (*train, *validation)):
            raise TypeError("training data must contain spaCy Docs")
        labels = _gold_labels(train)
        if not labels:
            raise ValueError("training data contains no annotated entities")
        targets = frozenset(target_labels) if target_labels is not None else labels
        if not targets or not targets <= labels:
            raise ValueError("target_labels must be present in the training corpus")
        if self._model_labels and not labels <= self._model_labels:
            raise ValueError("a subsequent fit cannot introduce new model labels")
        groups = (
            {name: frozenset(group) for name, group in recall_groups.items()}
            if recall_groups is not None
            else {"target": targets}
        )
        if not groups or any(
            not name or not group or not group <= targets for name, group in groups.items()
        ):
            raise ValueError("recall_groups must be nonempty subsets of target_labels")
        if router is not None:
            if not isinstance(router, RouterTraining):
                raise TypeError("router must be RouterTraining")
            if (
                self._router is not None
                and not reset_router
                and (router.dimension, router.ngrams)
                != (self._router.config.dimension, self._router.config.ngrams)
            ):
                raise ValueError("router feature dimensions and ngrams cannot change on fit")
        expected_training = (
            Tok2VecTraining
            if isinstance(self.adapter, SpacyNERAdapter)
            else TransformerTraining
            if isinstance(self.adapter, EdsNLPAdapter)
            else None
        )
        if training is not None and expected_training is not None:
            if not isinstance(training, expected_training):
                raise TypeError(f"adapter requires {expected_training.__name__}")
        train_rules = tuple(self._rules.predict(doc) for doc in train)
        validation_rules = tuple(self._rules.predict(doc) for doc in validation)
        settings = router or (self._router.config if self._router is not None else RouterTraining())
        rows = _training_rows(train, train_rules, targets, settings)
        positives = sum(target for _, target in rows)
        if not positives or positives == len(rows):
            raise ValueError("router training needs positive and negative residual lines")
        _gold_prefixes(validation, targets)
        for group in groups.values():
            _gold_prefixes(validation, group)
        if work_dir is None:
            workspace = TemporaryDirectory(prefix="pseudopath-fit-")
            manager = workspace
        else:
            manager = nullcontext(str(work_dir))
        with manager as location:
            path = Path(location)
            if work_dir is not None:
                if path.exists() and any(path.iterdir()):
                    raise FileExistsError(f"fit work directory is not empty: {path}")
                path.mkdir(parents=True, exist_ok=True)
            # Training can change weights before failing, so invalidate old results.
            self._ready = False
            self._thresholds = {}
            self._failed = True
            self.model = self.adapter.fit(
                self.model, train, validation, training=training, work_dir=path / "model"
            )
            if self._router is None or reset_router:
                settings = router or (
                    self._router.config if self._router is not None else RouterTraining()
                )
                self._router = LineRouter(settings)
            self._router.fit(
                train,
                train_rules,
                target_labels=targets,
                training=router,
            )
            self._thresholds = calibrate_profiles(
                validation,
                validation_rules,
                self._router,
                self._predict_packed,
                target_labels=targets,
                recall_groups=groups,
            )
        self._model_labels = self._model_labels or labels
        self._target_labels = targets
        self._recall_groups = groups
        self._last_training = {
            "model": (
                _training_metadata(training)
                if type(training) in (Tok2VecTraining, TransformerTraining)
                else None
            ),
            "router": asdict(self._router.config),
            "reset_router": reset_router,
        }
        self._ready = True
        self._failed = False
        return self

    def _predict_packed(self, text: str) -> Sequence[EntityPrediction]:
        rows = _predictions(self.adapter.predict(self.model, (text,)), 1)
        return rows[0]

    def predict(
        self,
        document: str | Doc,
        *,
        routing: str | float | None = None,
    ) -> tuple[Span, ...]:
        """Return non-overlapping spaCy spans in document order.

        A string is tokenized with ``make_doc``. For a supplied ``Doc``, spans
        belong to that document and its annotations remain unchanged.
        ``routing`` overrides the default for this call. Entity boundaries
        must align with the source tokens, or prediction raises ``ValueError``.
        """
        if not self._ready or self._router is None:
            raise RuntimeError("PseudoPath must be fitted or loaded before prediction")
        if isinstance(document, str):
            source = self.make_doc(document)
        elif isinstance(document, Doc):
            source = document
        else:
            raise TypeError("document must be a string or spaCy Doc")
        rules = self._rules.predict(source)
        selection = self._routing if routing is None else _routing_value(routing)
        if selection == "all-lines":
            threshold = None
        elif isinstance(selection, str):
            threshold = self._thresholds[selection]
        else:
            threshold = selection
        result = route_document(
            source.text,
            rules,
            self._predict_packed,
            router=self._router,
            threshold=threshold,
        )
        spans = []
        for entity in result.entities:
            span = source.char_span(
                entity.start,
                entity.end,
                label=entity.label,
                alignment_mode="strict",
            )
            if span is None:
                raise ValueError("predicted entity is not aligned with the EDS tokenizer")
            spans.append(span)
        return tuple(spans)

    def to_disk(self, path: str | Path) -> None:
        """Save the fitted model, router, profiles, and metadata.

        The destination must be new or empty. Intermediate checkpoints and
        optimizer state are not saved.
        """
        if not self._ready or self._router is None:
            raise RuntimeError("PseudoPath must be fitted before serialization")
        manifest = {
            "format_version": _FORMAT_VERSION,
            "runtime_versions": {name: package_version(name) for name in _RUNTIME_PACKAGES},
            "routing": self._routing,
            "preset": self._preset,
            "target_labels": sorted(self._target_labels or ()),
            "recall_groups": {
                name: sorted(group) for name, group in (self._recall_groups or {}).items()
            },
            "model_labels": sorted(self._model_labels),
            "thresholds": self._thresholds,
            "last_training": self._last_training,
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        destination = Path(path)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        self.adapter.to_disk(self.model, destination / "model")
        self._router.to_disk(destination / "router")
        (destination / "manifest.json").write_text(payload, encoding="utf-8")

    @classmethod
    def from_disk(
        cls,
        path: str | Path,
        *,
        adapter: NERAdapter[ModelT] | None = None,
        model_loader: Callable[[Path], ModelT] | None = None,
        device: str = "cpu",
        routing: str | float | None = None,
    ) -> Self:
        """Restore a fitted instance from a saved directory.

        Presets load with their built-in adapter. Custom models require both
        ``adapter`` and ``model_loader``. The loader receives the model
        subdirectory and handles device placement. ``routing`` can override
        the saved default without recalibrating the profiles.
        """
        source = Path(path)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported PseudoPath artifact format")
        thresholds = manifest.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != _PROFILES - {"all-lines"}:
            raise ValueError("artifact has incomplete calibrated profiles")
        for value in thresholds.values():
            if isinstance(value, str):
                raise ValueError("artifact thresholds must be numeric")
            _routing_value(value)
        preset = manifest.get("preset")
        if adapter is None and model_loader is None:
            if preset is None:
                raise ValueError("custom artifacts require both adapter and model_loader")
            from ._presets import load_preset

            model, adapter = load_preset(preset, source / "model", device=device)
        elif adapter is None or model_loader is None:
            raise ValueError("supply both adapter and model_loader")
        else:
            model = model_loader(source / "model")
            preset = None
        instance = cls(
            model,
            adapter=adapter,
            routing=manifest["routing"] if routing is None else routing,
        )
        instance._router = LineRouter.from_disk(source / "router")
        instance._target_labels = frozenset(manifest["target_labels"])
        instance._recall_groups = {
            name: frozenset(group) for name, group in manifest["recall_groups"].items()
        }
        instance._thresholds = dict(manifest["thresholds"])
        if set(instance._thresholds) != _PROFILES - {"all-lines"}:
            raise ValueError("artifact has incomplete calibrated profiles")
        instance._model_labels = frozenset(manifest["model_labels"])
        instance._last_training = manifest["last_training"]
        instance._preset = preset
        instance._ready = True
        return instance

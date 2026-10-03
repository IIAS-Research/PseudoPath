"""Train a user-supplied spaCy NER pipeline."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from .._routing import EntityPrediction
from ..training import Tok2VecTraining, _training_metadata
from ._common import labels_in, score_predictions


def _activate_device(model: Language, device: str) -> None:
    if device == "cpu":
        available = spacy.require_cpu()
    elif device == "cuda":
        available = spacy.require_gpu(0)
    else:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if not available:
        raise RuntimeError(f"spaCy device is unavailable: {device}")
    expected = "gpu" if device == "cuda" else "cpu"
    for _name, pipe in model.pipeline:
        if hasattr(pipe, "model") and pipe.model.ops.device_type != expected:
            raise ValueError("select the spaCy device before creating or loading the model")


def _validate_model(model: Language) -> None:
    if not isinstance(model, Language):
        raise TypeError("SpacyNERAdapter requires a spaCy Language model")
    if "ner" not in model.pipe_names:
        raise ValueError("spaCy model must have a ner component")


def _is_initialized(model: Language) -> bool:
    parameters = [
        bool(node.has_param(name))
        for _pipe_name, pipe in model.pipeline
        if hasattr(pipe, "model")
        for node in pipe.model.walk()
        for name in node.param_names
    ]
    if not parameters:
        raise ValueError("spaCy model has no trainable parameters")
    if any(parameters) and not all(parameters):
        raise ValueError("spaCy model is only partially initialized")
    return all(parameters)


def _aligned_example(model: Language, source: Doc) -> Example:
    """Copy gold entities onto the model's tokens without changing the source.

    Expand unaligned boundaries to full tokens. If expansions overlap, merge
    their ranges and keep the longest aligned span's label. Ties prefer
    earlier spans, then the original annotation order.
    """
    tokenized = model.make_doc(source.text)
    candidates = []
    for order, entity in enumerate(source.ents):
        start, end = entity.start_char, entity.end_char
        while start < end and source.text[start].isspace():
            start += 1
        while end > start and source.text[end - 1].isspace():
            end -= 1
        if start == end:
            raise ValueError("training entity is empty after trimming whitespace")
        aligned = tokenized.char_span(start, end, label=entity.label_, alignment_mode="strict")
        if aligned is None:
            aligned = tokenized.char_span(start, end, label=entity.label_, alignment_mode="expand")
        if aligned is None:
            raise ValueError(
                "training entity cannot be aligned with the model tokenizer: "
                f"{entity.start_char}:{entity.end_char}"
            )
        candidates.append((aligned, order))

    components = []
    for candidate in candidates:
        span, _order = candidate
        if not components or span.start >= max(current[0].end for current in components[-1]):
            components.append([candidate])
        else:
            components[-1].append(candidate)

    aligned_entities = []
    for component in components:
        if len(component) == 1:
            aligned_entities.append(component[0][0])
            continue
        winner, _order = min(
            component,
            key=lambda item: (-(item[0].end - item[0].start), item[0].start, item[1]),
        )
        merged = tokenized.char_span(
            min(item[0].start_char for item in component),
            max(item[0].end_char for item in component),
            label=winner.label_,
            alignment_mode="strict",
        )
        if merged is None:
            raise RuntimeError("merged training entity is not tokenizer-aligned")
        aligned_entities.append(merged)

    return Example.from_dict(
        tokenized,
        {
            "entities": [
                (entity.start_char, entity.end_char, entity.label_) for entity in aligned_entities
            ]
        },
    )


def _examples(model: Language, documents: Sequence[Doc], indices: Sequence[int]):
    return [_aligned_example(model, documents[index]) for index in indices]


class SpacyNERAdapter:
    """Train, predict, and save a spaCy pipeline with a NER component.

    Select the device before creating or loading the supplied model.
    ``batch_size`` sets the number of texts processed together at prediction.
    """

    def __init__(self, *, device: Literal["cpu", "cuda"] = "cpu", batch_size: int = 32) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = device
        self.batch_size = batch_size
        self.selected_step: int | None = None
        self.selected_score: float | None = None
        self.training_summary: dict | None = None

    def fit(
        self,
        model: Language,
        train_docs: Sequence[Doc],
        validation_docs: Sequence[Doc],
        *,
        training: Tok2VecTraining | None,
        work_dir: Path,
    ) -> Language:
        """Train with a fresh optimizer and restore the selected checkpoint in place."""
        _validate_model(model)
        settings = training or Tok2VecTraining()
        if not isinstance(settings, Tok2VecTraining):
            raise TypeError("SpacyNERAdapter requires Tok2VecTraining")
        train = tuple(train_docs)
        validation = tuple(validation_docs)
        if not train or not validation:
            raise ValueError("Tok2Vec training and validation data must be nonempty")
        if any(not isinstance(document, Doc) for document in (*train, *validation)):
            raise TypeError("Tok2Vec training data must contain spaCy Docs")
        observed_labels = labels_in(train)
        if not observed_labels:
            raise ValueError("Tok2Vec training data contains no entities")
        selection_labels = frozenset(settings.selection_labels or observed_labels)

        workspace = Path(work_dir)
        if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
            raise FileExistsError(f"Tok2Vec work directory is not empty: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)
        checkpoints = workspace / "checkpoints"
        checkpoints.mkdir()

        _activate_device(model, self.device)
        spacy.util.fix_random_seed(settings.seed)
        if _is_initialized(model):
            current_labels = frozenset(model.get_pipe("ner").labels)
            if not observed_labels <= current_labels:
                raise ValueError("a later Tok2Vec fit cannot introduce new labels")
            optimizer = model.create_optimizer()
            model.resume_training(sgd=optimizer)
        else:
            for label in sorted(observed_labels):
                model.get_pipe("ner").add_label(label)
            optimizer = model.initialize(lambda: _examples(model, train, tuple(range(len(train)))))

        model_labels = frozenset(model.get_pipe("ner").labels)
        if not selection_labels <= model_labels:
            missing = sorted(selection_labels - model_labels)
            raise ValueError(f"selection_labels are not supported by Tok2Vec: {missing}")

        # spaCy changes Example objects during updates. Rebuild them for each batch.
        _examples(model, train, tuple(range(len(train))))
        _examples(model, validation, tuple(range(len(validation))))

        optimizer.learn_rate = settings.learning_rate
        optimizer.L2 = settings.l2
        optimizer.L2_is_weight_decay = True
        optimizer.grad_clip = settings.grad_clip

        generator = random.Random(settings.seed)
        order = list(range(len(train)))
        cursor = len(order)
        best_step: int | None = None
        best_score = float("-inf") if settings.greater_is_better else float("inf")
        checkpoint_rows = []
        last_losses: dict[str, float] = {}

        for step in range(1, settings.steps + 1):
            if cursor >= len(order):
                generator.shuffle(order)
                cursor = 0
            indices = order[cursor : cursor + settings.batch_size]
            cursor += len(indices)
            losses: dict[str, float] = {}
            model.update(
                _examples(model, train, indices),
                sgd=optimizer,
                drop=settings.dropout,
                losses=losses,
            )
            last_losses = {name: float(value) for name, value in sorted(losses.items())}

            if step % settings.validation_interval and step != settings.steps:
                continue
            predictions = self.predict(model, tuple(document.text for document in validation))
            score = score_predictions(
                validation, predictions, metric=settings.selection_metric, labels=selection_labels
            )
            checkpoint = checkpoints / f"step-{step:08d}"
            model.to_disk(checkpoint)
            checkpoint_rows.append(
                {"step": step, "score": score, "path": str(checkpoint.relative_to(workspace))}
            )
            if settings.checkpoint_selection == "last" or (
                score > best_score if settings.greater_is_better else score < best_score
            ):
                best_step = step
                best_score = score

        if best_step is None:
            raise RuntimeError("Tok2Vec training produced no checkpoint")
        model.from_disk(checkpoints / f"step-{best_step:08d}")
        self.training_summary = {
            "training": _training_metadata(settings),
            "labels": sorted(model_labels),
            "selection_labels": sorted(selection_labels),
            "checkpoints": checkpoint_rows,
            "selected_step": best_step,
            "selected_score": best_score,
            "last_losses": last_losses,
        }
        (workspace / "training.json").write_text(
            json.dumps(self.training_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.selected_step = best_step
        self.selected_score = best_score
        return model

    def predict(
        self, model: Language, texts: Sequence[str]
    ) -> Sequence[Sequence[EntityPrediction]]:
        """Return entity offsets for each text, in input order."""
        _validate_model(model)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Tok2Vec prediction inputs must be strings")
        documents = tuple(model.pipe(texts, batch_size=self.batch_size))
        if len(documents) != len(texts):
            raise RuntimeError("Tok2Vec returned an incorrect number of documents")
        predictions = []
        for text, document in zip(texts, documents, strict=True):
            if document.text != text:
                raise RuntimeError("Tok2Vec changed the input text")
            predictions.append(
                tuple(
                    EntityPrediction(span.start_char, span.end_char, span.label_)
                    for span in document.ents
                )
            )
        return tuple(predictions)

    def to_disk(self, model: Language, path: Path) -> None:
        """Save the pipeline in a new or empty directory."""
        _validate_model(model)
        destination = Path(path)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"spaCy model output directory is not empty: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.to_disk(destination)

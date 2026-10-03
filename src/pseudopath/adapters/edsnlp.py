"""Train and run a user-supplied EDS-NLP transformer NER pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .._entity_cleanup import clean_and_select
from .._reader import read_training_docs
from .._routing import EntityPrediction
from ..training import TransformerTraining
from ._common import labels_in, score_predictions

if TYPE_CHECKING:
    from edsnlp import Pipeline
    from spacy.tokens import Doc


def _neural_only(model: Pipeline) -> None:
    """Check that the pipeline contains normalizer followed by ner."""
    if tuple(model.pipe_names) != ("normalizer", "ner"):
        raise ValueError("EDS-NLP model must contain only normalizer and ner")


def _transformer(model: Pipeline):
    from edsnlp.pipes.trainable.embeddings.transformer.transformer import Transformer

    modules = [
        module
        for _name, pipe in model.torch_components()
        for _module_name, module in pipe.named_component_modules()
        if isinstance(module, Transformer)
    ]
    if len(modules) != 1:
        raise ValueError("EDS-NLP model must contain exactly one transformer")
    return modules[0]


def _encoder_blocks(transformer) -> tuple:
    model = transformer.transformer
    base = getattr(model, "base_model", model)
    encoder = getattr(base, "encoder", None)
    layers = (
        getattr(base, "layers", None)
        or getattr(encoder, "layer", None)
        or getattr(encoder, "layers", None)
    )
    if layers is None:
        raise ValueError("transformer encoder layers are unavailable")
    return tuple(layers)


def _optimizer(model: Pipeline, settings: TransformerTraining):
    """Use separate learning rates for the NER head and selected encoder layers."""
    import torch
    from edsnlp.training.optimizer import LinearSchedule, ScheduledOptimizer

    transformer = _transformer(model)
    transformer_parameters = set(transformer.parameters())
    if settings.trainable_transformer_layers == "all":
        selected_transformer = transformer_parameters
    else:
        blocks = _encoder_blocks(transformer)
        count = settings.trainable_transformer_layers
        if count > len(blocks):
            raise ValueError("trainable_transformer_layers exceeds the model depth")
        selected_transformer = (
            {parameter for block in blocks[-count:] for parameter in block.parameters()}
            if count
            else set()
        )
    task_parameters = set(model.parameters()) - transformer_parameters
    if not task_parameters:
        raise ValueError("EDS-NLP NER head has no trainable parameters")
    groups = [
        {
            "params": list(task_parameters),
            "lr": settings.task_lr,
            "schedules": [
                LinearSchedule(
                    total_steps=settings.steps,
                    warmup_rate=settings.warmup_fraction,
                    start_value=settings.task_lr,
                    path="lr",
                )
            ],
        }
    ]
    if selected_transformer:
        groups.append(
            {
                "params": list(selected_transformer),
                "lr": settings.transformer_lr,
                "schedules": [
                    LinearSchedule(
                        total_steps=settings.steps,
                        warmup_rate=settings.warmup_fraction,
                        start_value=0,
                        path="lr",
                    )
                ],
            }
        )
    return ScheduledOptimizer(
        torch.optim.AdamW(
            groups,
            betas=settings.adam_betas,
            eps=settings.adam_epsilon,
            weight_decay=settings.weight_decay,
        )
    )


def _training_docs(model: Pipeline, documents: Sequence[Doc], max_length: int) -> tuple[Doc, ...]:
    """Copy gold entities onto the model's tokens, then split training chunks."""
    from spacy.util import filter_spans

    aligned = []
    for source in documents:
        document = model.make_doc(source.text)
        spans = []
        for entity in source.ents:
            span = document.char_span(
                entity.start_char,
                entity.end_char,
                label=entity.label_,
                alignment_mode="expand",
            )
            if span is None:
                raise ValueError("training entity cannot align with the EDS tokenizer")
            spans.append(span)
        document.spans["pseudo-ml"] = spans
        document.ents = filter_spans(spans)
        aligned.append(document)
    chunks = read_training_docs(model, tuple(aligned), max_length)
    if not chunks:
        raise ValueError("training documents contain no usable text")
    return chunks


class EdsNLPAdapter:
    """Train, predict, and save an EDS-NLP NER pipeline supplied by the caller.

    The pipeline must contain ``normalizer`` followed by ``ner``, with one
    Transformer. NER predictions must be written to ``doc.spans["pseudo-ml"]``.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.device = device
        self.selected_step: int | None = None
        self.selected_score: float | None = None

    def fit(
        self,
        model: Pipeline,
        train_docs: Sequence[Doc],
        validation_docs: Sequence[Doc],
        *,
        training: TransformerTraining | None = None,
        work_dir: Path,
    ) -> Pipeline:
        """Train with a fresh optimizer and load the selected validation checkpoint."""
        import edsnlp
        from confit.utils.random import set_seed
        from edsnlp.training.trainer import GenericScorer, TrainingData, train

        settings = training or TransformerTraining()
        if not isinstance(settings, TransformerTraining):
            raise TypeError("EdsNLPAdapter requires TransformerTraining")
        _neural_only(model)
        labels = labels_in(train_docs)
        if not labels:
            raise ValueError("training documents contain no entities")
        ner = model.get_pipe("ner")
        model_labels = frozenset(ner.labels) if ner.labels is not None else labels
        if not labels <= model_labels:
            raise ValueError("training labels are outside the EDS-NLP NER head")
        selection_labels = frozenset(settings.selection_labels or labels)
        if not selection_labels <= model_labels:
            missing = sorted(selection_labels - model_labels)
            raise ValueError(f"selection_labels are not supported by EDS-NLP: {missing}")
        set_seed(settings.seed)
        if ner.labels is None:
            # Create label weights before collecting parameters for the optimizer.
            ner.update_labels(sorted(labels))
        if self.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")

        prepared = _training_docs(model, train_docs, settings.max_length)
        training_data = TrainingData(
            data=edsnlp.data.from_iterable(prepared),
            batch_size=f"{settings.batch_words} words",
            shuffle="dataset",
            sub_batch_size=f"{settings.grad_accumulation_tokens} tokens",
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = work_dir / "checkpoints"
        checkpoint_dir.mkdir()
        best_step = -1
        best_score = float("-inf") if settings.greater_is_better else float("inf")

        def capture(step: int) -> None:
            nonlocal best_step, best_score
            model.train(False)
            try:
                predictions = self.predict(model, tuple(doc.text for doc in validation_docs))
                score = score_predictions(
                    validation_docs,
                    predictions,
                    metric=settings.selection_metric,
                    labels=selection_labels,
                )
                checkpoint = checkpoint_dir / f"step-{step:08d}"
                model.to_disk(checkpoint)
            finally:
                model.train(True)
            if settings.checkpoint_selection == "last" or (
                score > best_score if settings.greater_is_better else score < best_score
            ):
                best_score = score
                best_step = step

        def on_validation(row: dict) -> None:
            capture(int(row["step"]))

        train(
            nlp=model,
            train_data=[training_data],
            val_data=[],
            seed=settings.seed,
            max_steps=settings.steps,
            optimizer=_optimizer(model, settings),
            validation_interval=settings.validation_interval,
            max_grad_norm=settings.grad_clip,
            scorer=GenericScorer(speed=False),
            cpu=self.device == "cpu",
            output_dir=work_dir,
            save_model=False,
            logger=False,
            on_validation_callback=on_validation,
        )
        if settings.steps % settings.validation_interval:
            capture(settings.steps)
        if best_step < 0:
            raise RuntimeError("training created no validation checkpoint")
        selected = edsnlp.load(checkpoint_dir / f"step-{best_step:08d}")
        _neural_only(selected)
        selected.to(self.device)
        self.selected_step = best_step
        self.selected_score = best_score
        return selected

    def predict(
        self, model: Pipeline, texts: Sequence[str]
    ) -> tuple[tuple[EntityPrediction, ...], ...]:
        """Return cleaned entity offsets for each text, in input order."""
        import torch

        _neural_only(model)
        model.train(False)
        inputs = [model.make_doc(text) for text in texts]
        with torch.inference_mode():
            outputs = tuple(model.pipe(inputs))
        if len(outputs) != len(texts):
            raise RuntimeError("NER pipeline returned the wrong number of documents")
        return tuple(
            tuple(
                EntityPrediction(span.start_char, span.end_char, span.label_)
                for span in clean_and_select(document.spans["pseudo-ml"])
            )
            for document in outputs
        )

    def to_disk(self, model: Pipeline, path: str | Path) -> None:
        """Save the pipeline at the supplied path."""
        _neural_only(model)
        model.to_disk(Path(path))

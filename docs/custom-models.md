# Custom models

The integrated presets cover common workflows. For another model, create or
load it with your own library and supply an adapter that implements the small
`NERAdapter` contract. The adapter connects your model to the rules, line
router, calibration, and span merging.

## Define an adapter

Implement training, prediction, and saving:

```python
from collections.abc import Sequence
from pathlib import Path

from spacy.tokens import Doc
from pseudopath import EntityPrediction


class MyAdapter:
    def fit(
        self,
        model: object,
        train_docs: Sequence[Doc],
        validation_docs: Sequence[Doc],
        *,
        training: object | None,
        work_dir: Path,
    ) -> object:
        # Train or update the supplied model and return it.
        raise NotImplementedError

    def predict(
        self, model: object, texts: Sequence[str]
    ) -> Sequence[Sequence[EntityPrediction]]:
        # Return predictions for the texts in the same order.
        raise NotImplementedError

    def to_disk(self, model: object, path: str | Path) -> None:
        # Save the model under path.
        raise NotImplementedError
```

`PseudoPath.fit(training=...)` passes the training settings to the adapter
as-is. `work_dir` is available for intermediate training files.
An adapter may update the supplied model in
place, but `fit` must return the model PseudoPath should use thereafter.
Leave the supplied documents' gold annotations and patient context unchanged.
Retokenize into new documents when your model requires different tokens.
Training settings can be any object your adapter accepts. Record them in your
[experiment history](reproducibility.md#saved-state-and-experiment-records).

### Prediction contract

`EntityPrediction(start, end, label)` uses a zero-based, half-open character
range `[start, end)` in each input text. The outer sequence returned by
`predict` must have the same length and order as `texts`. When routing is
active, a text can be a packed selection of physical lines rather than a
full document. Predictions that cross an inserted separator are discarded
during projection. Offsets must align with the output document's EDS tokens
so PseudoPath can return spaCy spans. Unaligned predictions raise an error.

## Train and save

Use your adapter with a model that you create externally:

```python
from pseudopath import PseudoPath

model = load_my_model()  # Load with your model library.
pipeline = PseudoPath(model=model, adapter=MyAdapter())
pipeline.fit(train_docs, validation_docs, training=my_settings)
pipeline.to_disk("pseudopath-artifact")
```

Here `train_docs`, `validation_docs`, and `my_settings` are your corpus and
model settings.

## Restore a model

Supply the adapter and a callable that loads the saved model:

```python
restored = PseudoPath.from_disk(
    "pseudopath-artifact",
    adapter=MyAdapter(),
    model_loader=load_my_model_from_disk,
)
```

The loader receives the saved model subdirectory and returns a model object.
It may capture device settings or other initialization parameters.
Install your model library's dependencies and record its checkpoint revision.
When you save an instance loaded with this pair, supply the pair again to
restore it, including instances originally created from a preset.

## External EDS-NLP models, including EDS-Pseudo

`EdsNLPAdapter` accepts an EDS-NLP pipeline containing exactly `normalizer`
and `ner`, in that order, with one Transformer. The NER component must write
to `doc.spans["pseudo-ml"]`, including an empty group when no entities are found
(use `span_setter={"pseudo-ml": True}`). A pipeline that only writes `doc.ents`
does not satisfy this contract. Prepare the neural-only pipeline yourself,
then use the existing adapter:

```python
import edsnlp
from pseudopath import (
    PATIENT_IDENTIFIER_LABELS, EdsNLPAdapter, PseudoPath, TransformerTraining, read_jsonl
)

model = edsnlp.load("/path/to/prepared/neural-only-pipeline")
pipeline = PseudoPath(model, adapter=EdsNLPAdapter(device="cpu"))
pipeline.fit(
    read_jsonl("train.jsonl"),
    read_jsonl("validation.jsonl"),
    training=TransformerTraining(
        max_length=256,
        task_lr=1e-5,
        transformer_lr=1e-6,
        trainable_transformer_layers=1,
        selection_labels=PATIENT_IDENTIFIER_LABELS,
    ),
)
```

For EDS-Pseudo, obtain checkpoint access and install its dependencies before
preparing the neural pipeline.
Do not pass its original rule and merge components: PseudoPath already applies
its fixed rules. A loaded NER head must support all training labels. Configure
Transformer window and stride on the model itself. See
[architecture overrides](models.md#architecture-overrides).
A fresh compatible EDS-NLP head can infer labels on its first `fit`.

## External spaCy models

Use `SpacyNERAdapter` for a compatible spaCy NER pipeline.
Select `spacy.require_cpu()` or
`spacy.require_gpu(0)` before construction or loading. The adapter does
not move the parser's mixed CPU/GPU layers. Import `edsnlp` before loading
an EDS-language spaCy model so that its language and tokenizer are registered.

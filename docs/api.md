# API reference

This page describes the public API: inputs, return values, state changes, and
constraints. Import its classes and functions from `pseudopath`. For a first
run, see the [quickstart](quickstart.md). For the component workflows, see
[architecture](architecture.md).

## Create an instance

```text
PseudoPath.from_preset(
    name, *, checkpoint=None, device="cpu", seed=42,
    model_config=None, routing="all-lines"
) -> PseudoPath
```

`name` is `"camembert"`, `"distilcamembert"`, or `"tok2vec"`.
Transformer presets create a fresh NER head on a public backbone or supplied
checkpoint. For Tok2Vec, `checkpoint` may supply an existing spaCy NER pipeline.
`model_config` overrides component settings when constructing a model.
`device` is `"cpu"` or `"cuda"`. An unavailable device raises an error.
See [models and configuration](models.md) for defaults and supported overrides.

For an external model:

```text
PseudoPath(model, *, adapter, routing="all-lines") -> PseudoPath
```

Create or load the model with its own library, then supply a compatible
adapter. Adapters do not construct or download models. `pipeline.model` holds the
working model, which may be replaced by the adapter during training.
`pipeline.adapter` holds the integration object.

Creation does not train the router or calibrate profiles. Prediction requires
a successful `fit` or an instance restored from a fitted artifact.

## Documents and entity values

```text
read_jsonl(path, *, text_key="text") -> tuple[Doc, ...]
pipeline.make_doc(text) -> Doc
```

`read_jsonl` reads JSON Lines records with text and gold entities:

```json
{"text": "Luc consulte.", "entities": [{"start": 0, "end": 3, "label": "PRENOM"}]}
```

Offsets are zero-based character positions. `end` is exclusive. The function
returns spaCy documents with entities in `doc.ents`, preserving exact gold
boundaries. Entities must not overlap. An optional `context` field supplies
[known patient terms](#patient-context). Use `text_key` for another text field, such as
`"note_text"`.

`make_doc` creates an unannotated inference document with the EDS tokenizer
and the patient-context extension. You may then set `doc._.context`.

`EntityPrediction(start, end, label)` is an immutable character-offset value
used by adapters and custom validation metrics. It is not the public
prediction result: `pipeline.predict` returns spaCy `Span` objects with `.text`,
`.label_`, `.start_char`, and `.end_char`.

## Train the model and router

```text
pipeline.fit(
    train_docs, validation_docs, *, training=None, router=None,
    reset_router=False, target_labels=None, recall_groups=None, work_dir=None
) -> PseudoPath
```

`fit` returns the same instance. It trains the NER model, trains the router on
target entities missed by the fixed rules, and calibrates routing profiles
against validation data. It updates model and router state without replacing
the supplied documents' gold annotations or patient context.

| Parameter | Meaning |
| --- | --- |
| `train_docs` | Nonempty sequence of annotated spaCy `Doc` objects used for learning. |
| `validation_docs` | Nonempty sequence used for checkpoint selection and routing calibration. |
| `training` | Adapter-specific settings, passed unchanged. Built-in adapters use `TransformerTraining` or `Tok2VecTraining`, with defaults when omitted. |
| `router` | `RouterTraining` settings. Omitted settings use the existing router configuration, or defaults on the first fit. |
| `reset_router` | `False` continues router weights. `True` creates a new router before training it. |
| `target_labels` | Labels used for residual routing and calibration. Defaults to all labels in the current training corpus. |
| `recall_groups` | Named, nonempty subsets of target labels, calibrated separately. Defaults to one group containing all target labels. |
| `work_dir` | New or empty directory retaining intermediate training files. Omitted means temporary files. |

Training must contain annotated entities and all requested target labels.
Router training needs both positive and negative residual lines. Validation
must contain gold alphanumeric characters for every recall group. See
[training and routing](training.md) for these requirements and calibration budgets.

### Successive fits

Later calls continue model and router weights, unless `reset_router=True`
starts a new router. Built-in adapters create a fresh optimizer for each call.
Labels must remain within the model's initial label set. Targets and recall
groups are resolved per call. See
[successive training phases](training.md#one-or-two-training-phases) for reset
settings and examples.

### Learning settings and checkpoint selection

`TransformerTraining`, `Tok2VecTraining`, and `RouterTraining` are immutable
settings objects. See [training controls](models.md#training-controls) for
their fields and [checkpoint selection](models.md#checkpoint-selection) for
the score definitions, defaults, and custom metric contract.

### Training diagnostics

Both built-in adapters expose `selected_step` and `selected_score` on
`pipeline.adapter`: the selected update within the latest successful adapter fit
and its validation score. They are `None` on a new adapter and are not restored
by `from_disk`. A later router or calibration failure does not clear them.

`SpacyNERAdapter.training_summary` contains additional backend-specific details.
Custom adapters are not required to provide these diagnostics.

## Predict entities

```text
pipeline.predict(document, *, routing=None) -> tuple[Span, ...]
```

`document` is a string or a spaCy `Doc`. A string creates a new document.
Passing a `Doc` returns spans belonging to that same document. Neither form
assigns predictions to `doc.ents` or changes supplied annotations or patient
context. An empty prediction is an empty tuple.

Prediction combines model output on selected lines with fixed rules on the
whole document. Returned entities are ordered and do not overlap. Overlapping
ranges are united to preserve all detected characters. The union label comes
from the longest original span, then the leftmost span. The model wins
identical-boundary ties. Offsets are relative to the original document.

Use `make_doc` for compatible inference tokenization. If a predicted span
cannot be represented exactly on an input document's tokens, `predict` raises
`ValueError` rather than changing offsets or retokenizing the input.

To annotate a document, assign the result explicitly:

```python
doc.ents = pipeline.predict(doc)
```

### Routing

`pipeline.routing` is the default mode. Assigning it changes that default.
`predict(..., routing=...)` overrides it for one call without changing it.
Accepted values are `"all-lines"`, `"prudent"`, `"balanced"`, `"fast"`, or a
finite numeric threshold in `[0, 1]`.

Named profiles use calibrated thresholds. A numeric threshold selects lines
with a router score at least that value. Fixed rules remain active in every mode. See
[training and routing](training.md#routing-modes) for budgets and limitations.

### Patient context

Supply known patient terms as a dictionary of entity labels and lists of
strings. For example, use `NOM` for family names, `PRENOM` for given names,
and `MAIL` for email addresses. Each document has its own context.

For a fitted or loaded instance, set `doc._.context` before prediction:

```python
doc = pipeline.make_doc("Luc Martin consulte.")
doc._.context = {
    "NOM": ["Martin"],
    "PRENOM": ["Luc", "Lucas"],
    "MAIL": ["luc.martin@example.org"],
}
entities = pipeline.predict(doc)
```

The fixed rules match these terms in the note and combine their matches with
model predictions. Context is optional. Pass the `Doc` to `predict` to use
it, since a plain string carries no patient context.

For annotated JSONL files, add a `context` field to each record:

```json
{"text": "Luc Martin consulte.", "entities": [{"start": 0, "end": 3, "label": "PRENOM"}, {"start": 4, "end": 10, "label": "NOM"}], "context": {"NOM": ["Martin"], "PRENOM": ["Luc"]}}
```

`read_jsonl` copies this field to `doc._.context`. Supply it in both training
and validation records when fitting with patient context. The rules use it
to determine which identifying characters remain for the router to learn.

Context is supplied per document and is not stored in the model artifact.
The aliases `EMAIL` and `NOM_NAISS` produce `MAIL` and `NOM` entities,
respectively.

## Use another model

`NERAdapter` is a protocol with three operations: `fit`, `predict`, and
`to_disk`. Inheritance is not required. The adapter trains the supplied model,
returns character-offset predictions in input order, and saves model state.
It must leave supplied gold annotations and patient context unchanged.

`EdsNLPAdapter` handles compatible EDS-NLP Transformer NER pipelines.
`SpacyNERAdapter` handles compatible spaCy NER pipelines. Other model families
implement the same contract and install their own dependencies. See
[custom models](custom-models.md) for signatures, backend requirements, and
external EDS-Pseudo integration.

## Save and restore

```text
pipeline.to_disk(path) -> None
PseudoPath.from_disk(
    path, *, adapter=None, model_loader=None, device="cpu", routing=None
) -> PseudoPath
```

Saving requires a fitted instance and a new or empty destination. It stores
the selected model, trained router, calibrated profiles, and descriptive
metadata. It is not an optimizer checkpoint or a complete experiment record.

Preset artifacts reload directly:

```python
restored = PseudoPath.from_disk("pseudopath-artifact", device="cpu")
```

Custom artifacts require both `adapter` and a callable `model_loader`. The
loader receives the saved model directory and owns device placement.
Supplying this pair also opts into custom loading for a preset artifact.
After saving that instance, supply the pair again when loading it.
`routing=` overrides the restored default without recalibration.

Loading rejects unsupported artifact formats. Recorded library versions are
informational, so use a compatible environment. See
[saved state and experiment records](reproducibility.md#saved-state-and-experiment-records)
for the metadata retained and the information to record separately.

## Errors and state

Invalid arguments and incompatible inputs raise errors. The package does not
switch devices, expand prediction boundaries, or replace missing model state
automatically. `predict` and `to_disk` require a fitted or loaded instance.

Once a fit starts modifying model state, a failure invalidates the instance:
its weights may have changed. Create a new instance or reload a saved artifact
before continuing. There is no automatic rollback or retry. Argument errors
detected before training do not invalidate an already fitted instance.

# Quickstart

## Install

PseudoPath requires Python 3.11 or newer. Install it from PyPI:

```bash
python -m pip install pseudopath
```

For CamemBERT or DistilCamemBERT, install the optional Transformer stack:

```bash
python -m pip install 'pseudopath[transformer]'
```

The fixed rules are included in the package.

## Prepare annotations

Create `train.jsonl` and `validation.jsonl`. Each line contains the original
text and entities with zero-based, half-open character offsets:

```json
{"text": "Luc Martin consulte.\nAucune autre mention.", "entities": [{"start": 0, "end": 3, "label": "PRENOM"}, {"start": 4, "end": 10, "label": "NOM"}]}
```

`read_jsonl(path)` returns annotated spaCy `Doc` objects and preserves the
gold character boundaries, even when they differ from normal token boundaries.
The adapters retokenize documents for model training. For a corpus using
`note_text`, call `read_jsonl(path, text_key="note_text")` instead.

If patient names or other identifying terms are available as structured data,
add the optional `context` field. See [patient context](api.md#patient-context)
for the JSONL format and how to supply context at prediction.

You can also pass your own `Doc` sequences directly to `fit`, with reference
entities in `doc.ents`. Gold entities must not overlap. Use nonempty corpora
with both identifying mentions and ordinary lines. See
[training data requirements](training.md#data-requirements) before fitting
your own corpus.

## Train, predict, and save

The Tok2Vec preset works with the base installation:

```python
from pseudopath import PseudoPath, Tok2VecTraining, read_jsonl

pipeline = PseudoPath.from_preset("tok2vec", routing="balanced")
pipeline.fit(
    read_jsonl("train.jsonl"),
    read_jsonl("validation.jsonl"),
    training=Tok2VecTraining(steps=100, validation_interval=20),
)

for span in pipeline.predict("Luc Martin consulte."):
    print(span.text, span.label_, span.start_char, span.end_char)

pipeline.to_disk("pseudopath-artifact")
restored = PseudoPath.from_disk("pseudopath-artifact")
restored.predict("Luc Martin consulte.", routing="all-lines")
```

`fit` orchestrates model training, router training, and profile calibration.
It returns the same instance. The 100-step schedule only demonstrates the API.
Use representative data and suitable settings to evaluate model quality.

### Use a Transformer

With the Transformer extra installed, change the preset and training settings:

```python
from pseudopath import PseudoPath, TransformerTraining, read_jsonl

pipeline = PseudoPath.from_preset("distilcamembert", device="cuda")
pipeline.fit(
    read_jsonl("train.jsonl"),
    read_jsonl("validation.jsonl"),
    training=TransformerTraining(steps=1000, validation_interval=100),
)
```

The first preset call may download public backbone weights. A fresh NER head
learns your training labels. The model is not ready for prediction until
`fit` completes. Use `"camembert"` for public CamemBERT-base or
`checkpoint="/path/to/local/checkpoint"` for local Transformer weights.
See [models and configuration](models.md#integrated-presets) for preset details.

## Choose routing

The first example uses `"balanced"`. Use `"prudent"` for a smaller validation
recall-loss budget, `"fast"` for a larger one, or `"all-lines"` to process the
whole document. Pass `routing=` to `predict` to change it for one call.
See [routing modes](training.md#routing-modes) for budgets and thresholds.

## Predict on a document

`predict` returns a tuple of spaCy `Span` objects. If you pass a `Doc`,
the spans belong to that same document, but its annotations stay unchanged:

```python
doc = restored.make_doc("Luc Martin consulte.")
spans = restored.predict(doc, routing=0.5)
assert all(span.doc is doc for span in spans)
```

To annotate the document, assign the returned entities explicitly:

```python
doc.ents = spans
```

Use `make_doc` for inference documents when possible. Predicted offsets
must align with the input document's tokens. Otherwise `predict` raises
an error instead of changing the span boundary.

See [training and routing](training.md) for two-phase runs,
[models and configuration](models.md) for configuration, and
[custom models](custom-models.md) for external model integration.

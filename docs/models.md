# Models and configuration

## Integrated presets

`PseudoPath.from_preset` creates a model and its adapter:

| Preset | Starting model | Training settings |
| --- | --- | --- |
| `"camembert"` | Public `almanach/camembert-base` backbone and a fresh NER head | `TransformerTraining` |
| `"distilcamembert"` | Public `cmarkea/distilcamembert-base` backbone and a fresh NER head | `TransformerTraining` |
| `"tok2vec"` | Fresh spaCy Tok2Vec/NER pipeline | `Tok2VecTraining` |

```python
from pseudopath import PseudoPath

pipeline = PseudoPath.from_preset(
    "distilcamembert",
    checkpoint="/path/to/local/checkpoint",
    device="cuda",
    seed=42,
)
```

Omit `checkpoint` to use the public default backbone. A Transformer checkpoint
can be a Hugging Face model ID or local directory. The first use may download
weights. For Tok2Vec, `checkpoint` can point to an existing compatible spaCy
pipeline. `device` selects CPU or CUDA before construction or loading.
`seed` seeds model creation. Training settings have their own seed.

To use a trained EDS-Pseudo checkpoint, obtain access and prepare its neural
pipeline, then use `EdsNLPAdapter`. See [external models](custom-models.md).

See [installation](quickstart.md#install) for the dependencies of each preset.

## Architecture overrides

The Transformer heads use a TextCNN with kernel size 3 and a joint CRF.
The backbone processes overlapping windows of 128 tokens with stride 96 and
an added newline token. Tok2Vec uses ORTH/SHAPE hash embeddings of width 96,
a four-layer Maxout encoder, and a transition-based NER head connected through
a Tok2Vec listener. `fit` initializes a fresh head from the training labels.

`model_config` contains partial component configurations. Overrides are
merged into the preset rather than requiring the full architecture:

```python
pipeline = PseudoPath.from_preset(
    "tok2vec",
    model_config={
        "tok2vec": {"model": {"encode": {"depth": 2}}},
        "ner": {"model": {"hidden_width": 32}},
    },
)
```

For a Transformer, configure the head and its embedding under `"ner"`:

```python
pipeline = PseudoPath.from_preset(
    "camembert",
    model_config={
        "ner": {
            "embedding": {
                "kernel_sizes": [3, 5],
                "embedding": {"window": 256, "stride": 192},
            }
        }
    },
)
```

Transformer `window` and `stride` are model architecture settings. Configure
them only under `model_config["ner"]["embedding"]["embedding"]` when creating
the model. They are not fields of `TransformerTraining`. Its `max_length`
instead limits the training-document chunks prepared by the reader.
Keep embedding, encoder, and listener widths consistent when changing Tok2Vec
width. Architecture overrides apply to newly constructed models, not a loaded
Tok2Vec checkpoint.

## Training configuration

Pass `TransformerTraining(...)` or `Tok2VecTraining(...)` to
`fit(training=...)`. Configure each phase directly:

```python
from pseudopath import Tok2VecTraining

settings = Tok2VecTraining(
    steps=1200,
    validation_interval=100,
    learning_rate=5e-4,
)
```

Use `fit(training=settings)` to apply these settings. For successive phases,
see [training and routing](training.md#one-or-two-training-phases). Example
experiment settings are listed in [reproducibility](reproducibility.md#example-training-protocols).

## Checkpoint selection

Both model training classes default to `checkpoint_selection="best"` and
`selection_metric="f2"`. Built-in metrics are `"f2"`, `"f1"`, `"precision"`,
and `"recall"`: all use micro exact entity matches, including label and
character offsets. `greater_is_better=True` maximizes the score. Set it to
`False` to minimize a custom error metric. Ties keep the earliest checkpoint.
`checkpoint_selection="last"` keeps the final checkpoint instead.

For example, select the checkpoint with the highest exact entity recall:

```python
from pseudopath import Tok2VecTraining

settings = Tok2VecTraining(selection_metric="recall")
```

`selection_labels` filters both gold and predictions before scoring. If
omitted, all training labels are used. Explicit labels must be supported by
the NER head. The metric sees model predictions only, without rules or routing.
It selects a validation checkpoint. It does not change the neural training
loss or the router's character-recall calibration.

A custom metric receives `(gold, predictions)`, each a sequence of per-document
`EntityPrediction(start, end, label)` sequences, and returns a finite float.
For example, minimize the number of false-positive entities:

```python
def false_positives(gold, predictions):
    return float(sum(
        len(set(predicted) - set(expected))
        for expected, predicted in zip(gold, predictions)
    ))


settings = Tok2VecTraining(
    selection_metric=false_positives,
    greater_is_better=False,
)
```

Pass these settings to `fit(training=settings)`. Keep custom metric code in
your [experiment record](reproducibility.md#saved-state-and-experiment-records)
and supply the function again for later fits.

## Training controls

| Class | Fields | Meaning |
| --- | --- | --- |
| `TransformerTraining` | `steps`, `validation_interval`, `seed` | Updates, validation frequency, and training seed. |
| `TransformerTraining` | `batch_words`, `grad_accumulation_tokens` | Word-based batches and token-based accumulation. |
| `TransformerTraining` | `max_length` | Maximum token length of document chunks prepared for training. |
| `TransformerTraining` | `task_lr`, `transformer_lr`, `trainable_transformer_layers` | Learning rates and final encoder layers to train, or `"all"`. Use zero backbone learning rate when training zero layers. |
| `TransformerTraining` | `warmup_fraction`, `weight_decay`, `adam_betas`, `adam_epsilon`, `grad_clip` | Schedule, regularization, and clipping. |
| `Tok2VecTraining` | `steps`, `validation_interval`, `seed`, `batch_size` | Updates, validation frequency, training seed, and batch size. |
| `Tok2VecTraining` | `learning_rate`, `dropout`, `l2`, `grad_clip` | Optimizer and regularization. |
| Both model settings | `selection_labels`, `checkpoint_selection`, `selection_metric`, `greater_is_better` | Validation labels, checkpoint policy, metric, and score direction. |
| `RouterTraining` | `dimension`, `epochs`, `learning_rate`, `seed`, `ngrams` | Hashed features, logistic training, seed, and character n-gram sizes. |

Pass `RouterTraining(...)` to `fit(router=...)` to configure the router.
Router `dimension` and `ngrams` cannot change while continuing its weights.
Use `reset_router=True` to change them.

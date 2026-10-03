# Training and routing

`PseudoPath.fit(train_docs, validation_docs, ...)` trains the selected or
supplied NER model, trains the line router on residual errors left by the
fixed rules, and calibrates three routing profiles against validation
documents.

## Data requirements

Pass nonempty sequences of spaCy `Doc` objects with gold entities in
`doc.ents`. The training corpus must contain all target labels and both
positive and negative residual lines. A positive residual line contains
a target entity missed by the fixed rules. A negative one contains no such
entity. Validation must contain gold alphanumeric characters for every
recall group. See the [quickstart](quickstart.md#prepare-annotations) for
the JSONL format.

## One or two training phases

A second `fit` starts from the current *model* weights. By default, it also
continues the router weights. Set `reset_router=True` to train a new router
on the second corpus while continuing the model. The first phase produces a
usable model and router. The second can adapt the model and train a new router
for local data. A reset keeps the previous router settings unless you pass a
new `router=`.

For example, use separate template and local corpora for the two phases:

```python
from pseudopath import PATIENT_IDENTIFIER_LABELS, PseudoPath, Tok2VecTraining, read_jsonl

pipeline = PseudoPath.from_preset("tok2vec")
pipeline.fit(
    read_jsonl("template_train.jsonl"),
    read_jsonl("template_validation.jsonl"),
    training=Tok2VecTraining(
        steps=2000, validation_interval=200, checkpoint_selection="last"
    ),
)
pipeline.fit(
    read_jsonl("local_train.jsonl"),
    read_jsonl("local_validation.jsonl"),
    training=Tok2VecTraining(
        steps=4000, validation_interval=400, selection_labels=PATIENT_IDENTIFIER_LABELS
    ),
    reset_router=True,
)
```

The four files contain your template and local annotations in the
[JSONL format](quickstart.md). Later phases cannot introduce labels absent
from the model's first phase. Built-in adapters start each phase with new
optimizer state and apply its configured checkpoint selection. Custom adapters
define their own optimizer policy. Each successful fit replaces the calibrated
thresholds. See
[models and configuration](models.md) for configurable training settings.

Model checkpoint selection is separate from router calibration. Use
`selection_metric` in the model training settings to choose a built-in exact
entity score or your own validation metric. See [checkpoint selection](models.md#checkpoint-selection).

## Calibration targets

By default, every label in the current training corpus is a calibration target.
`target_labels` limits that set. Labels and groups are resolved again for each
fit. Supply custom targets and groups for every phase that needs them.
`recall_groups` applies the same budget separately to subsets of those
targets. Every group must have gold alphanumeric characters in the validation
corpus, or its character recall is undefined.

For a corpus containing patient identifiers and hospital names, give the
12 patient identifiers and `HOPITAL` separate recall constraints:

```python
from pseudopath import PATIENT_IDENTIFIER_LABELS

target_labels = (*PATIENT_IDENTIFIER_LABELS, "HOPITAL")
recall_groups = {
    "identifiers": PATIENT_IDENTIFIER_LABELS,
    "hospital": ("HOPITAL",),
}

# Pass target_labels=target_labels and recall_groups=recall_groups
# to the local fit call, when all 13 labels occur in its training corpus.
```

A template pretraining phase can use its own labels and groups.

## Routing modes

The `"all-lines"` reference sends the whole document to the model and
merges its output with the fixed rules. The other profiles are calibrated on
*alphanumeric gold characters covered by predictions*, not exact entity
matches. The allowed loss against `"all-lines"`, on the validation corpus,
is:

| Profile | Maximum character-recall loss |
| --- | ---: |
| `"prudent"` | 0.1 percentage point |
| `"balanced"` | 0.5 percentage point |
| `"fast"` | 1.0 percentage point |

The budget also applies to each `recall_groups` entry. Among acceptable
thresholds, calibration minimizes characters sent to the model. These are
validation constraints, not guarantees on new documents.

At inference, set `pipeline.routing` or pass `routing=` to `predict`. Either
accepts one of the four named modes or a finite numeric threshold between
0 and 1. A numeric threshold bypasses profile calibration: each physical
line with a router score at least that threshold goes to the model.
Unselected lines still receive fixed-rule predictions.

## Working directories

`fit(work_dir=...)` retains intermediate model checkpoints in a new or
empty directory. Without it, temporary training files are removed when the
fit call ends. Save the fitted instance separately with `to_disk`.
See [save and restore](api.md#save-and-restore) for artifact contents.

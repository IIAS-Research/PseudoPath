# Reproducibility

For comparable runs, keep the following experiment records alongside the
saved model and router:

1. Exact train, validation, and test splits, annotation offsets, and selected
   labels. Clinical data are not included in the package.
2. Starting model: preset or custom architecture, checkpoint source and
   revision, overrides, initialization seed, and model-library versions.
3. Every `fit` phase in order: model and router settings, seeds,
   `target_labels`, `recall_groups`, and `reset_router`, including the
   checkpoint metric, score direction, and source code of any custom metric.
4. Selected model checkpoint, calibrated thresholds, source commit, package
   versions, and evaluation code and metric definitions.
5. Device and hardware for throughput measurements. Different machines
   are not directly comparable.

## Saved state and experiment records

`to_disk` stores the fitted model, router, calibrated thresholds, and a
version 4 manifest. The manifest includes labels, router settings, explicitly
supplied native settings from the last fit, runtime versions, and the preset
name when applicable. See [save and restore](api.md#save-and-restore) for
loading behavior.

Keep the full phase history, starting checkpoint revisions, and custom training
settings in your experiment record. A callable checkpoint metric is recorded
as `selection_metric="custom"`, so retain its source code and supply it again
for later fits. Use `fit(work_dir=...)` to retain intermediate checkpoints.

## Example training protocols

The following settings illustrate Transformer pretraining and adaptation.
Use `selection_labels=PATIENT_IDENTIFIER_LABELS` to score the 12 patient
identifiers, or choose labels for your corpus:

| Model and phase | `max_length` | `task_lr` | `transformer_lr` | `trainable_transformer_layers` |
| --- | ---: | ---: | ---: | --- |
| User-supplied EDS-Pseudo adaptation | 256 | 1e-5 | 1e-6 | 1 |
| DistilCamemBERT pretraining | 256 | 5e-5 | 5e-5 | `"all"` |
| DistilCamemBERT adaptation | 384 | 5e-5 | 5e-6 | 4 |

Other Transformer controls use their standard defaults.

Model checkpoint scores use exact entity matches, while routing calibration uses character recall
relative to the same model in `"all-lines"` mode. See
[checkpoint selection](models.md#checkpoint-selection) and
[routing modes](training.md#routing-modes) for their definitions.

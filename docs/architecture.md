# Architecture

PseudoPath combines a NER model, fixed rules, and a trained line router.
The router selects which lines the model processes, while the rules inspect
the full document. This page describes the components and the training and
prediction workflows. See the [API reference](api.md) for method signatures.

## Workflow overview

The following diagram shows training, configuration, and inference.
Open it to inspect the steps at full size.

```{figure} _static/figures/figure1-workflow.svg
:alt: Two-phase PseudoPath workflow: annotated data train NER models and a residual-line router, validation calibrates routing profiles, and inference combines full-document rules with neural predictions on selected packed lines.
:width: 100%
:target: _static/figures/figure1-workflow.svg
:name: pseudopath-workflow
:figclass: pseudopath-workflow

PseudoPath workflow. **A:** adapt candidate models,
train the residual-line router, calibrate profiles, and select a configuration.
**B:** apply rules to the full document, route and pack selected lines for the
chosen model, project predictions back, and combine them with rule entities.
```

In the package, `fit` configures one model at a time. Comparing the three
models shown in the diagram means fitting separate instances. See
[models and configuration](models.md) to select each starting model.

The figure shows the complete masking workflow. `predict` returns spaCy
entity spans. The calling application applies masking or replacement to
produce the illustrated masked note.

## Component responsibilities

| Component | Responsibility |
| --- | --- |
| `PseudoPath` | Hold the working model, router, and calibrated profiles. Coordinate training, prediction, and persistence. |
| NER model | Learn entities and predict them using its own model library. |
| `NERAdapter` | Translate between the model library and PseudoPath's training, prediction, and saving operations. |
| Fixed rules | Find entities from built-in patterns and optional patient context, independently of the model backend. |
| Line router | Score physical lines so the neural model can process only a selected part of a document. |
| Calibration | Choose profile thresholds from validation predictions and recall constraints. |
| Built-in training settings | Hold learning parameters as immutable values, separate from model architecture. |

Presets construct a model and select its adapter. To use a custom model,
create or load it with its own library, then pass it to `PseudoPath` with
an adapter implementing `NERAdapter`.

## Training flow

One fit coordinates three dependent steps:

1. Train the neural model and select its validation checkpoint.
2. Apply fixed rules to the annotated corpus, then train the router to distinguish
   lines containing target entities the rules missed from lines without them.
3. Calibrate router thresholds against validation predictions, using all-lines
   prediction as the reference for character-recall loss.

The selected model, trained router, and calibrated profiles form one fitted
instance. See [training and routing](training.md) for data requirements,
successive phases, and recall budgets.

## Prediction flow

Fixed rules inspect the whole document. In routed modes, the router selects
lines for the neural model, which receives their packed text. All-lines mode
sends the whole document instead. Model offsets are projected
back to the source document. Predictions crossing inserted separators are
discarded. The package merges model and rule entities, resolves overlaps,
and returns source-linked spaCy spans.

Prediction preserves the input document's annotations and patient context.
Returned spans refer to the source document and can be assigned to `doc.ents`
by the caller. Their boundaries must align with the source tokens.

## Model integration boundary

An adapter implements training, prediction, and saving for its model library.
Restoring a custom artifact also requires a model loader. See
[custom models](custom-models.md) for the contract.

PseudoPath uses spaCy documents and spans. Adapters return entity predictions
as character offsets, which PseudoPath converts to spans on the source document.

## Configuration and saved state

Choose the model architecture at construction and supply learning settings to
`fit`. See [models and configuration](models.md) for available controls.

Save the fitted model, router, profiles, and metadata with `to_disk`, then
restore them for prediction or further fitting. See
[save and restore](api.md#save-and-restore) for the artifact contract and
[reproducibility](reproducibility.md) for experiment records.

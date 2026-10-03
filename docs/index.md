---
hide-toc: true
---

# Clinical de-identification. Balance quality and speed.

PseudoPath detects **identifying entities in clinical notes**. **Choose your NER model**
and use a **trained router** to send it **only relevant lines**, balancing detection
quality with processing speed.

```{raw} html
<section id="flow-benchmark" class="flow-benchmark" aria-label="Performance comparison" data-config="_static/benchmarks.json">
  <p>Explore measured performance with JavaScript enabled.</p>
</section>
```

## Start with PseudoPath

Choose a CamemBERT, DistilCamemBERT, or Tok2Vec preset, train it on your
annotations, and return identifying entities as spaCy spans. Your application
can use these spans to mask or replace the detected text.

Install with Python 3.11 or newer:

```bash
python -m pip install pseudopath
```

Follow the [quickstart](quickstart.md) for optional Transformer dependencies
and a first training run. Then explore:

- [Models and configuration](models.md) to choose a preset and tune its training.
- [Training and routing](training.md) to adapt a model and choose recall budgets.
- [Custom models](custom-models.md) to integrate your own backend or checkpoint.
- [Architecture](architecture.md#workflow-overview) for the workflow diagram,
  and [reproducibility](reproducibility.md) to record and compare experiments.

```{toctree}
:hidden:
:caption: Use PseudoPath

quickstart
models
training
custom-models
```

```{toctree}
:hidden:
:caption: Understand and reproduce

architecture
reproducibility
```

```{toctree}
:hidden:
:caption: Reference

api
```

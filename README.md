# PseudoPath

Train a clinical NER model and route selected lines for inference.

Requires Python 3.11 or newer.

```bash
python -m pip install pseudopath
python -m pip install "pseudopath[transformer]"
```

Run the synthetic Tok2Vec example from the repository root:

```bash
python examples/train_tok2vec.py
python examples/predict.py
```

The examples use synthetic annotations. See [examples](examples/).

See LICENSE and THIRD_PARTY_NOTICES.md for license terms.

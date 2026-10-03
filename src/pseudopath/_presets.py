"""Build the integrated Transformer and Tok2Vec NER models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from confit import Config

from .adapters.edsnlp import EdsNLPAdapter
from .adapters.spacy import SpacyNERAdapter

_CHECKPOINTS = {
    "camembert": "almanach/camembert-base",
    "distilcamembert": "cmarkea/distilcamembert-base",
}
_TOK2VEC = {
    "tok2vec": {
        "model": {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 96,
                "attrs": ["ORTH", "SHAPE"],
                "rows": [5000, 2500],
                "include_static_vectors": False,
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 96,
                "depth": 4,
                "window_size": 1,
                "maxout_pieces": 3,
            },
        }
    },
    "ner": {
        "model": {
            "@architectures": "spacy.TransitionBasedParser.v2",
            "state_type": "ner",
            "extra_state_tokens": False,
            "hidden_width": 64,
            "maxout_pieces": 2,
            "use_upper": True,
            "nO": None,
            "tok2vec": {
                "@architectures": "spacy.Tok2VecListener.v1",
                "width": 96,
                "upstream": "*",
            },
        },
        "incorrect_spans_key": None,
        "moves": None,
        "update_with_oracle_cut_size": 100,
    },
}


def _validate_name(name: str) -> None:
    if name not in {*_CHECKPOINTS, "tok2vec"}:
        raise ValueError("preset must be 'camembert', 'distilcamembert', or 'tok2vec'")


def _config(defaults: dict, overrides: Mapping[str, Any] | None) -> Config:
    unknown = set(overrides or ()) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown model components: {sorted(unknown)}")
    return Config(defaults).merge(dict(overrides or {}))


def _spacy_device(device: str) -> None:
    import spacy

    if device == "cpu":
        spacy.require_cpu()
    elif device == "cuda":
        spacy.require_gpu(0)
    else:
        raise ValueError("device must be 'cpu' or 'cuda'")


def create_preset(
    name: str,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    model_config: Mapping[str, Any] | None = None,
) -> tuple[Any, EdsNLPAdapter | SpacyNERAdapter]:
    """Create a fresh NER head, or load a supplied Tok2Vec pipeline.

    ``model_config`` contains partial component settings. Use ``tok2vec`` and
    ``ner`` for spaCy, or ``ner`` for Transformers, whose nested ``embedding``
    settings control the CNN and encoder. A fresh head gets its labels on the first fit.
    Transformer checkpoints may be Hugging Face identifiers or local paths.
    """
    _validate_name(name)
    if name == "tok2vec":
        import edsnlp  # noqa: F401 - registers the EDS tokenizer
        import spacy

        if checkpoint is not None:
            if model_config:
                raise ValueError("model_config cannot change a saved Tok2Vec architecture")
            return load_preset(name, Path(checkpoint), device=device)
        config = _config(_TOK2VEC, model_config)
        _spacy_device(device)
        spacy.util.fix_random_seed(seed)
        model = spacy.blank("eds")
        model.add_pipe("tok2vec", config=config["tok2vec"])
        model.add_pipe("ner", config=config["ner"])
        return model, SpacyNERAdapter(device=device)

    # Keep the Transformer imports optional for the Tok2Vec preset.
    import edsnlp
    from confit.utils.random import set_seed

    adapter = EdsNLPAdapter(device=device)
    config = _config(
        {
            "ner": {
                "mode": "joint",
                "target_span_getter": {"pseudo-ml": True},
                "span_setter": {"pseudo-ml": True},
                "labels": None,
                "embedding": {
                    "@factory": "eds.text_cnn",
                    "kernel_sizes": [3],
                    "embedding": {
                        "@factory": "eds.transformer",
                        "model": str(_CHECKPOINTS[name] if checkpoint is None else checkpoint),
                        "window": 128,
                        "stride": 96,
                        "new_tokens": [[r"(?:\n\s*)*\n", "⏎"]],
                    },
                },
            }
        },
        model_config,
    )
    set_seed(seed)
    model = edsnlp.blank("eds")
    model.add_pipe("eds.normalizer", name="normalizer")
    model.add_pipe("eds.ner_crf", name="ner", config=config["ner"])
    model.to(device)
    return model, adapter


def load_preset(
    name: str, path: Path, *, device: str = "cpu"
) -> tuple[Any, EdsNLPAdapter | SpacyNERAdapter]:
    """Restore an integrated model from its PseudoPath artifact directory."""
    _validate_name(name)
    if name == "tok2vec":
        import edsnlp  # noqa: F401 - registers the EDS tokenizer
        import spacy

        _spacy_device(device)
        return spacy.load(path), SpacyNERAdapter(device=device)

    import edsnlp

    adapter = EdsNLPAdapter(device=device)
    model = edsnlp.load(path)
    model.to(device)
    return model, adapter

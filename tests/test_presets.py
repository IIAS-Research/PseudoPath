from pathlib import Path

import pytest
import spacy

from pseudopath import PseudoPath, RouterTraining
from pseudopath._presets import create_preset
from pseudopath.adapters.edsnlp import EdsNLPAdapter
from pseudopath.adapters.spacy import SpacyNERAdapter
from pseudopath.training import Tok2VecTraining, TransformerTraining


def test_tok2vec_overrides_train_and_reload(tmp_path: Path) -> None:
    flow = PseudoPath.from_preset(
        "tok2vec",
        model_config={
            "tok2vec": {"model": {"encode": {"depth": 1}}},
            "ner": {"model": {"hidden_width": 32}},
        },
    )
    model, adapter = flow.model, flow.adapter
    assert isinstance(adapter, SpacyNERAdapter)
    assert model.pipe_names == ["tok2vec", "ner"]
    assert model.config["components"]["tok2vec"]["model"]["encode"]["depth"] == 1
    assert model.config["components"]["ner"]["model"]["hidden_width"] == 32
    document = flow.make_doc("Luc consulte.\nRien à signaler.")
    document.ents = [document.char_span(0, 3, label="PRENOM")]
    flow.fit(
        [document],
        [document],
        training=Tok2VecTraining(
            steps=1, validation_interval=1, batch_size=1, selection_metric="f1"
        ),
        router=RouterTraining(dimension=1024, epochs=1),
        work_dir=tmp_path / "work",
    )
    flow.to_disk(tmp_path / "saved")
    restored = PseudoPath.from_disk(tmp_path / "saved")
    assert restored._preset == "tok2vec"
    assert restored._last_training["model"]["selection_metric"] == "f1"
    for routing in ("all-lines", "balanced", 0.5):
        expected = [
            (span.start_char, span.end_char, span.label_)
            for span in flow.predict(document.text, routing=routing)
        ]
        assert [
            (span.start_char, span.end_char, span.label_)
            for span in restored.predict(document.text, routing=routing)
        ] == expected
    custom = PseudoPath.from_disk(
        tmp_path / "saved", adapter=SpacyNERAdapter(), model_loader=spacy.load
    )
    custom.to_disk(tmp_path / "custom")
    with pytest.raises(ValueError, match="custom artifacts require both"):
        PseudoPath.from_disk(tmp_path / "custom")
    custom_restored = PseudoPath.from_disk(
        tmp_path / "custom", adapter=SpacyNERAdapter(), model_loader=spacy.load
    )
    assert custom_restored.adapter.predict(custom_restored.model, [document.text]) == (
        custom.adapter.predict(custom.model, [document.text])
    )
    fresh, _ = create_preset("tok2vec")
    assert fresh.config["components"]["tok2vec"]["model"]["encode"]["depth"] == 4


@pytest.mark.parametrize("name", ["camembert", "distilcamembert"])
def test_transformer_presets_accept_local_encoder_and_head_overrides(
    tmp_path: Path, name: str
) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("torch")
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "Luc": 4}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.normalizer = tokenizers.normalizers.NFC()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(tmp_path)
    transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(tmp_path)

    flow = PseudoPath.from_preset(
        name,
        checkpoint=tmp_path,
        model_config={
            "ner": {
                "embedding": {
                    "kernel_sizes": [3, 5],
                    "embedding": {"window": 8, "stride": 4},
                }
            }
        },
    )
    model, adapter = flow.model, flow.adapter
    assert isinstance(adapter, EdsNLPAdapter)
    assert model.pipe_names == ["normalizer", "ner"]
    ner = model.get_pipe("ner")
    assert ner.labels is None
    assert [layer.kernel_size[0] for layer in ner.embedding.module.convolutions] == [3, 5]
    assert (ner.embedding.embedding.window, ner.embedding.embedding.stride) == (8, 4)
    assert "⏎" in ner.embedding.embedding.tokenizer.get_vocab()

    document = flow.make_doc("Luc arrive.\nRien à signaler.")
    document.ents = [document.char_span(0, 3, label="PRENOM")]
    with pytest.raises(ValueError, match="selection_labels are not supported"):
        adapter.fit(
            model,
            [document],
            [document],
            training=TransformerTraining(selection_labels=("TYPO",)),
            work_dir=tmp_path / "invalid-labels",
        )
    assert ner.labels is None
    assert not (tmp_path / "invalid-labels").exists()
    scores = iter((-2.0, -1.0))

    def metric(gold, predictions):
        assert gold[0][0].label == "PRENOM"
        assert len(gold) == len(predictions) == 1
        return next(scores)

    greater_is_better = name == "camembert"
    flow.fit(
        [document],
        [document],
        training=TransformerTraining(
            steps=2,
            validation_interval=1,
            batch_words=32,
            grad_accumulation_tokens=32,
            trainable_transformer_layers=1,
            selection_metric=metric,
            greater_is_better=greater_is_better,
        ),
        router=RouterTraining(dimension=1024, epochs=1),
    )
    assert adapter.selected_step == (2 if greater_is_better else 1)
    assert adapter.selected_score == (-1.0 if greater_is_better else -2.0)
    flow.to_disk(tmp_path / "saved")
    restored = PseudoPath.from_disk(tmp_path / "saved")
    assert restored.model.get_pipe("ner").labels == ["PRENOM"]
    encoder = restored.model.get_pipe("ner").embedding.embedding
    assert (encoder.window, encoder.stride) == (8, 4)
    assert restored._last_training["model"]["steps"] == 2
    assert restored._last_training["model"]["selection_metric"] == "custom"
    assert restored.adapter.predict(restored.model, [document.text]) == flow.adapter.predict(
        flow.model, [document.text]
    )


def test_unknown_preset_and_component_fail_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="preset must be"):
        create_preset("eds-pseudo")
    with pytest.raises(ValueError, match="unknown model components"):
        create_preset("camembert", model_config={"typo": {}})

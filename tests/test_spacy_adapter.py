from pathlib import Path

import spacy

from pseudopath.adapters.spacy import SpacyNERAdapter
from pseudopath.training import Tok2VecTraining


def test_user_supplied_spacy_model_can_train_twice_and_reload(tmp_path: Path) -> None:
    model = spacy.blank("fr")
    model.add_pipe("ner")
    document = model.make_doc("Luc Martin consulte.")
    document.ents = (
        document.char_span(0, 3, label="PRENOM"),
        document.char_span(4, 10, label="NOM"),
    )
    adapter = SpacyNERAdapter()
    settings = Tok2VecTraining(steps=1, validation_interval=1, batch_size=1)

    assert (
        adapter.fit(model, (document,), (document,), training=settings, work_dir=tmp_path / "pre")
        is model
    )
    assert (
        adapter.fit(model, (document,), (document,), training=settings, work_dir=tmp_path / "fit")
        is model
    )
    assert frozenset(model.get_pipe("ner").labels) == {"PRENOM", "NOM"}

    adapter.to_disk(model, tmp_path / "saved")
    restored = spacy.load(tmp_path / "saved")
    assert adapter.predict(restored, (document.text,)) == adapter.predict(model, (document.text,))

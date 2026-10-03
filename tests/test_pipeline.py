from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pseudopath import PseudoPath
from pseudopath._router import RouterTraining
from pseudopath._routing import EntityPrediction


class FakeModel:
    def __init__(self, phases: int = 0) -> None:
        self.phases = phases

    @classmethod
    def from_disk(cls, path: Path) -> FakeModel:
        state = json.loads((path / "model.json").read_text())
        return cls(state["phases"])


class FakeAdapter:
    def __init__(self) -> None:
        self.training = None

    def fit(self, model, train_docs, validation_docs, *, training, work_dir: Path):
        assert train_docs and validation_docs
        self.training = training
        model.phases += 1
        return model

    def predict(self, model, texts):
        return tuple(
            (EntityPrediction(start, start + 3, "NOM"),) if (start := text.find("Luc")) >= 0 else ()
            for text in texts
        )

    def to_disk(self, model, path: Path) -> None:
        path.mkdir()
        (path / "model.json").write_text(json.dumps({"phases": model.phases}))


def annotated(flow: PseudoPath, text: str):
    doc = flow.make_doc(text)
    start = text.find("Luc")
    if start >= 0:
        span = doc.char_span(start, start + 3, label="NOM")
        assert span is not None
        doc.ents = (span,)
    return doc


def test_fit_predict_continue_and_reload(tmp_path):
    model = FakeModel()
    flow = PseudoPath(model, adapter=FakeAdapter())
    train = [annotated(flow, "Luc est présent\nligne ordinaire")]
    dev = [annotated(flow, "Luc reste ici\nligne ordinaire")]
    settings = RouterTraining(dimension=1024, epochs=2)

    flow.fit(train, dev, router=settings)
    assert model.phases == 1
    assert [
        (span.text, span.label_) for span in flow.predict("Luc reste ici", routing="all-lines")
    ] == [("Luc", "NOM")]
    flow.routing = "balanced"
    assert [(span.text, span.label_) for span in flow.predict(dev[0].text)] == [("Luc", "NOM")]
    flow.routing = "all-lines"
    source = flow.make_doc("Luc reste ici")
    assert flow.predict(source)[0].doc is source
    assert not source.ents
    source.ents = (source.char_span(4, 9, label="EXISTING"),)
    original_entities = source.ents
    predictions = flow.predict(source)
    assert [(span.text, span.label_) for span in predictions] == [("Luc", "NOM")]
    assert predictions[0].doc is source
    assert source.ents == original_entities

    first_router = flow._router
    flow.fit(train, dev, router=settings)
    assert model.phases == 2
    assert flow._router is first_router
    artifact = tmp_path / "artifact"
    flow.to_disk(artifact)
    with pytest.raises(ValueError, match="custom artifacts require both"):
        PseudoPath.from_disk(artifact)
    with pytest.raises(ValueError, match="supply both"):
        PseudoPath.from_disk(artifact, adapter=FakeAdapter())
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_versions"]["pseudopath"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest))
    restored = PseudoPath.from_disk(
        artifact, adapter=FakeAdapter(), model_loader=FakeModel.from_disk
    )
    assert restored.model.phases == 2
    assert [(span.text, span.label_) for span in restored.predict("Luc reste ici")] == [
        ("Luc", "NOM")
    ]
    manifest["format_version"] = -1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported PseudoPath artifact format"):
        PseudoPath.from_disk(artifact, adapter=FakeAdapter(), model_loader=FakeModel.from_disk)


def test_reset_router_starts_fresh_without_resetting_model():
    model = FakeModel()
    flow = PseudoPath(model, adapter=FakeAdapter())
    train = [annotated(flow, "Luc est présent\nligne ordinaire")]
    dev = [annotated(flow, "Luc reste ici\nligne ordinaire")]
    settings = RouterTraining(dimension=1024, epochs=2)

    flow.fit(train, dev, router=settings)
    first_router = flow._router
    assert first_router is not None
    first_weights = first_router.weights[:]

    flow.fit(train, dev, reset_router=True)
    assert model.phases == 2
    assert flow._router is not first_router
    assert flow._router.config == settings
    assert flow._router.weights == first_weights
    assert flow._last_training["reset_router"] is True

    new_settings = RouterTraining(dimension=2048, epochs=2)
    flow.fit(train, dev, router=new_settings, reset_router=True)
    assert model.phases == 3
    assert flow._router.config == new_settings


def test_invalid_router_settings_leave_fitted_pipeline_usable():
    model = FakeModel()
    flow = PseudoPath(model, adapter=FakeAdapter())
    train = [annotated(flow, "Luc est présent\nligne ordinaire")]
    dev = [annotated(flow, "Luc reste ici\nligne ordinaire")]
    settings = RouterTraining(dimension=1024, epochs=2)
    flow.fit(train, dev, router=settings)

    with pytest.raises(TypeError, match="RouterTraining"):
        flow.fit(train, dev, router=object())
    with pytest.raises(ValueError, match="feature dimensions"):
        flow.fit(train, dev, router=RouterTraining(dimension=2048, epochs=2))
    assert model.phases == 1
    assert [(span.text, span.label_) for span in flow.predict("Luc reste ici")] == [("Luc", "NOM")]
    flow.fit(train, dev, router=settings)
    assert model.phases == 2


def test_custom_training_config_is_forwarded_and_can_be_saved(tmp_path):
    @dataclass
    class CustomTraining:
        checkpoint: Path

    adapter = FakeAdapter()
    flow = PseudoPath(FakeModel(), adapter=adapter)
    train = [annotated(flow, "Luc est présent\nligne ordinaire")]
    dev = [annotated(flow, "Luc reste ici\nligne ordinaire")]
    settings = CustomTraining(checkpoint=tmp_path / "weights")
    flow.fit(train, dev, training=settings, router=RouterTraining(dimension=1024, epochs=2))
    assert adapter.training is settings
    flow.to_disk(tmp_path / "artifact")
    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["last_training"]["model"] is None


def test_invalid_threshold_is_not_clamped():
    with pytest.raises(ValueError, match="threshold"):
        PseudoPath(FakeModel(), adapter=FakeAdapter(), routing=1.1)

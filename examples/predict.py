"""Load any saved preset and compare routing profiles."""

from pseudopath import PseudoPath

pipeline = PseudoPath.from_disk("examples/outputs/tok2vec")
text = "Note pour Emma Girard\nTelephone : 0612345678."

for mode in ("all-lines", "prudent", "balanced", "fast", 0.5):
    entities = pipeline.predict(text, routing=mode)
    print(mode, [(span.text, span.label_) for span in entities])

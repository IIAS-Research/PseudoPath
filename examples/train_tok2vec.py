"""Train the Tok2Vec preset on a small synthetic corpus."""

from pseudopath import PseudoPath, Tok2VecTraining, read_jsonl

pipeline = PseudoPath.from_preset("tok2vec", routing="balanced")
pipeline.fit(
    read_jsonl("examples/data/train.jsonl"),
    read_jsonl("examples/data/validation.jsonl"),
    training=Tok2VecTraining(steps=100, validation_interval=100),
)
pipeline.to_disk("examples/outputs/tok2vec")

entities = pipeline.predict("Note pour Emma Girard\nAucun symptome signale.")
print([(span.text, span.label_) for span in entities])

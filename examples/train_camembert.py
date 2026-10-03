"""Train public CamemBERT-base with a fresh NER head on a synthetic corpus."""

from pseudopath import PseudoPath, TransformerTraining, read_jsonl

pipeline = PseudoPath.from_preset("camembert", routing="balanced")
pipeline.fit(
    read_jsonl("examples/data/train.jsonl"),
    read_jsonl("examples/data/validation.jsonl"),
    training=TransformerTraining(
        steps=2,
        validation_interval=2,
        batch_words=256,
        grad_accumulation_tokens=512,
        trainable_transformer_layers=1,
    ),
)
pipeline.to_disk("examples/outputs/camembert")

entities = pipeline.predict("Note pour Emma Girard\nAucun symptome signale.")
print([(span.text, span.label_) for span in entities])

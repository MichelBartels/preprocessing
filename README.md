# preprocessing
A simple Python library written in Rust allowing you to perform common NLP preprocessing steps much faster. Still WIP.

Simple example:
```py
from preprocessing import SQuADLoader, Tokenizer, StaticBatcher

texts = SQuADLoader("train-v2.0.json")
tokens = Tokenizer(texts, "bert-base-uncased")
batches = StaticBatcher(tokens, batch_size=8, seq_length=256)

for (input, target) in batches:
    output = model(input.input_ids)
    loss = loss_fn(output, target.start, target.end)
```

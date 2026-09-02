# Circuit Tracer

Circuit Tracer is a local tool for inspecting a one-block TensorFlow language
model. The first development ticket establishes the shared model and checkpoint
format used by both Google Colab training and the local inspection app.

## Supported Environment

- 64-bit Python 3.10
- TensorFlow 2.10.1
- Native Windows GPU: CUDA 11.2 and cuDNN 8.1
- CPU fallback when the CUDA libraries are unavailable

Select the project Python with pyenv and create an isolated environment:

```powershell
pyenv local 3.10.11
python -m venv .venv
& ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
```

Run the checks with:

```powershell
& ".venv\Scripts\python.exe" -m pip check
& ".venv\Scripts\python.exe" -m pytest
```

## Checkpoint Folder

Each checkpoint is one folder containing exactly these generated files:

```text
model.weights.h5
vocabulary.json
config.json
```

`vocabulary.json` keeps every token in its original order. The empty padding
token is entry 0 and `[UNK]` is entry 1. `config.json` records the architecture,
TensorFlow version, model sizes, and text-processing settings. The loader
refuses incomplete or inconsistent folders rather than loading uncertain data.

## Colab Training And Export

Use the repository version of the model instead of copying the model classes
into the notebook. Clone the repository and import from that checkout:

```python
!git clone https://github.com/khuongngoduc0310/Residual-Visualizer.git
%cd Residual-Visualizer

import tensorflow as tf

assert tf.__version__.startswith("2.10."), tf.__version__

from checkpoint import save_checkpoint
from model import ModelConfig, build_model, compile_for_training
from preprocess import (
    build_text_vectorizer,
    pad_punctuation,
    prepare_training_batch,
)
```

Process punctuation before creating the dataset. Adapt the tokenizer before
building the model so the output size uses the actual vocabulary length:

```python
MAX_TOKENS = 10_000
MAX_LEN = 80

processed_text = [pad_punctuation(text) for text in filtered_data]
text_ds = tf.data.Dataset.from_tensor_slices(processed_text)

vectorizer = build_text_vectorizer(
    max_tokens=MAX_TOKENS,
    output_sequence_length=MAX_LEN + 1,
)
vectorizer.adapt(text_ds.batch(32))
vocabulary = vectorizer.get_vocabulary()

config = ModelConfig(
    vocab_size=len(vocabulary),
    max_len=MAX_LEN,
    embedding_dim=256,
    num_heads=2,
    key_dim=128,
    feed_forward_dim=256,
    dropout_rate=0.1,
)
language_model = build_model(config)
compile_for_training(language_model)
```

Create next-token examples with sample weights. A weight of zero prevents the
right-side padding targets from contributing to training loss:

```python
train_ds = text_ds.shuffle(1_000).batch(32).map(
    lambda text: prepare_training_batch(text, vectorizer),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

language_model.fit(train_ds, epochs=10)
```

Right-side padding is required. The causal attention mask prevents real tokens
from looking forward at that padding, while the sample weights remove padding
from the loss.

Export all checkpoint files through the shared helper:

```python
save_checkpoint(
    "/content/circuit-tracer-checkpoint",
    language_model,
    vocabulary,
    config,
)
```

Export to a new or empty folder. The helper refuses to overwrite an existing
checkpoint, preventing a failed export from mixing old and new files.

The checkpoint must be exported with TensorFlow 2.10. A checkpoint produced by
a different TensorFlow minor version is rejected because weight compatibility
is not guaranteed.

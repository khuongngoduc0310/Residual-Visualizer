# Circuit Tracer

Circuit Tracer is a local tool for inspecting a one-block TensorFlow language
model. The first development ticket establishes the shared model and checkpoint
format used by both Google Colab training and the local inspection app.

## Supported Environment

- 64-bit Python 3.13.15
- TensorFlow 2.20.0
- Keras 3.13.2
- NumPy 2.1.3
- h5py 3.16.0
- protobuf 5.29.6
- Gradio 6.26.0
- WSL2/Linux with an NVIDIA driver for GPU inference

TensorFlow 2.20 does not provide native-Windows NVIDIA GPU support. Native
Windows can be used for CPU-only checks; use WSL2 for the local GPU app.

Inside WSL2, select the project Python with pyenv and create an isolated
environment:

```bash
pyenv local 3.13.15
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

Verify the NVIDIA and TensorFlow GPU paths:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For CPU-only development, install `requirements-dev.txt` instead. Run the
checks with:

```bash
python -m pip check
python -m pytest
```

## Run The Local App

Extract a Colab checkpoint before loading it. The app reads the checkpoint from
the machine running Python; it does not upload the folder through the browser.
For example, a checkpoint stored on the Windows drive is available in WSL2 at
`/mnt/c/Projects/Circuit Tracer/exports/checkpoint-20260902`.

Start the app inside WSL2:

```bash
source .venv/bin/activate
python app.py
```

Open `http://127.0.0.1:7860` in the Windows browser. Enter the extracted folder
path in **Checkpoint folder (server path)** and press **Load Model**. The app
validates all three files before showing the model details, reports CUDA GPU or
CPU based on TensorFlow's actual device visibility, and keeps the Gradio server
bound to localhost with public sharing disabled.

Loading a new checkpoint first releases the current model and clears the old
details. If the replacement fails, no model remains active. TensorFlow may keep
reserved GPU memory until the process exits even after the old model is
released.

## Checkpoint Folder

Each checkpoint is one folder containing exactly these generated files:

```text
model.weights.h5
vocabulary.json
config.json
```

`vocabulary.json` keeps every token in its original order. The empty padding
token is entry 0 and `[UNK]` is entry 1. `config.json` records the architecture,
TensorFlow and Keras versions, model sizes, and text-processing settings. The
loader refuses incomplete or inconsistent folders rather than loading uncertain
data.

## Colab Training And Export

Use the standard hosted Colab runtime. It currently provides Python 3.13.15 and
the exact package versions used by the checkpoint contract. Clone the
repository and import from that checkout instead of copying model classes into
the notebook:

```python
!git clone https://github.com/khuongngoduc0310/Residual-Visualizer.git
%cd Residual-Visualizer

import tensorflow as tf
import keras
import numpy as np
import h5py
import google.protobuf

assert tf.__version__ == "2.20.0", tf.__version__
assert keras.__version__ == "3.13.2", keras.__version__
assert np.__version__ == "2.1.3", np.__version__
assert h5py.__version__ == "3.16.0", h5py.__version__
assert google.protobuf.__version__ == "5.29.6", google.protobuf.__version__

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

The checkpoint must be exported with TensorFlow 2.20.0 and Keras 3.13.2. A
checkpoint produced by another runtime is rejected because weight
compatibility is not guaranteed. The format version is incremented when this
runtime contract changes, so older TensorFlow 2.10 checkpoints are rejected.

## Notebook

Run `notebook/compact_gpt_retrain_2.ipynb` in standard hosted Colab with a GPU
runtime. The notebook clones the repository, verifies the runtime versions,
trains the shared model, reloads the exported checkpoint, compares predictions
before and after loading, and downloads a ZIP containing the three checkpoint
files.

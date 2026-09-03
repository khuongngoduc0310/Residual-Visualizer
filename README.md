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
- Native Windows for CPU inference, or WSL2/Linux with an NVIDIA driver for GPU inference

TensorFlow 2.20 does not provide native-Windows NVIDIA GPU support. Native
Windows is the supported CPU path; use WSL2 for GPU inference.

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
nvcc --version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
```

The TensorFlow command must list at least one GPU. Then load a checkpoint in
the app, confirm the runtime says `CUDA GPU`, and complete one prompt analysis.
If the list is empty, check the NVIDIA driver and WSL2 CUDA passthrough before
debugging the app. The app intentionally falls back to CPU when no usable GPU
is visible.

For CPU-only development on Windows:

```powershell
py -3.13-64 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python scripts/verify_environment.py
python -m pip check
python -m pytest
```

`verify_environment.py` checks the supported Python bitness and direct package
versions. `pip check` verifies dependency metadata; both checks are required.

Run the app with one local-only command:

```powershell
python app.py
```

## Run The Local App

Extract a Colab checkpoint before loading it. The app reads the checkpoint from
the machine running Python; it does not upload the folder through the browser.
For example, a checkpoint stored on the Windows drive is available in WSL2 at
`/mnt/c/Projects/Circuit Tracer/exports/checkpoint-20260902`.

Start the app from the environment containing the dependencies:

```powershell
python app.py
```

Open `http://127.0.0.1:7860` in a browser. Enter the extracted folder
path in **Checkpoint folder (server path)** and press **Load Model**. The app
validates all three files before showing the model details, reports CUDA GPU or
CPU based on TensorFlow's actual device visibility, and keeps the Gradio server
bound to localhost with public sharing disabled.

For a quick UI check, use a desktop browser width of at least 1280px. Verify
that the control rail, model path, activation field, two supporting charts, and
output tabs are visible as a coherent workspace. A magnitude-bar click should
still update the token selector and the other two charts.

Loading a new checkpoint first releases the current model and clears the old
details. If the replacement fails, no model remains active. TensorFlow may keep
reserved GPU memory until the process exits even after the old model is
released.

## Analyze A Prompt

After loading a checkpoint, enter text in **Prompt** and press **Analyze
Prompt**. The prompt is processed exactly like training text: lowercased,
punctuation separated by spaces, and split on whitespace using the saved
vocabulary. The page shows every processed token with its 0-based position,
text, and numeric token ID, plus the processed token count against the model's
maximum sequence length.

Empty prompts and prompts longer than the model maximum are rejected rather
than silently shortened. Tokens missing from the vocabulary remain allowed and
are shown as `[UNK]` with a visible warning and count. The five most likely
next tokens are listed with their IDs and four-decimal probabilities, honestly
labeling the padding and unknown tokens when they appear. Results stay in
memory for the session only and are never written to disk.

## Inspect Internal Locations

The same Analyze Prompt run also captures every internal tensor of the model,
so you can move through the block without running inference again. Dropout is
disabled during this run, so repeated runs are deterministic. The captured
locations are:

| Location | Category |
| --- | --- |
| Token + position embeddings | Residual Stream |
| Attention update | Attention |
| Attention residual sum | Residual Stream |
| First normalized output | Residual Stream |
| FFN hidden activation | FFN |
| FFN update | FFN |
| FFN residual sum | Residual Stream |
| Final block output | Residual Stream |

After a successful analysis, the **Location** and **Token position** selectors
are filled in and default to the **Final block output** of the last processed
prompt token. The model diagram highlights the selected location, a short
plain-language explanation describes what the tensor contains, and three
coordinated Plotly views show token magnitudes, the token-by-dimension tensor,
and the selected token's dimension distribution. Token labels include their
positions, so repeated text remains distinguishable. Click a token-magnitude
bar or use the position-aware selector to change the selected token. Changing
either selector re-renders from the captured data only and never runs the model
again.

The heatmap reports its visible zero-centered range. The optional percentile
clipping control changes only the displayed color bounds; captured values and
the distribution remain unchanged. Normalized locations explain that their
token magnitudes may appear nearly flat because layer normalization fixes the
feature scale.

Captured tensors live in memory for the session only. They are cleared when a
new checkpoint is loaded, replaced by a fresh analysis, or dropped after a
failed analysis.

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

If a trusted TensorFlow 2.20.0/Keras 3.13.2 format-1 checkpoint is encountered,
the included one-time migration command can create a new format-2 folder without
altering the original:

```powershell
python scripts/migrate_checkpoint.py `
  "checkpoints\old-checkpoint" `
  "checkpoints\old-checkpoint-v2"
```

Use the new `-v2` folder in the app. The command refuses checkpoints with an
unknown runtime, architecture, vocabulary, or weight shape. Do not edit the
metadata by hand.

## Colab Training And Export

Use the standard hosted Colab runtime. It currently provides Python 3.13.15 and
the exact package versions used by the checkpoint contract. The notebook pins
the repository revision in its first setup cell; update that value only when
releasing a new compatible app. Clone the repository and import from that
checkout instead of copying model classes into the notebook:

```python
REPOSITORY_REVISION = "132461d"  # immutable app/chart implementation revision
!git clone https://github.com/khuongngoduc0310/Residual-Visualizer.git
%cd Residual-Visualizer
!git checkout {REPOSITORY_REVISION}

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
runtime contract changes; older TensorFlow 2.10 checkpoints are rejected.

## Troubleshooting

- **Python or package version mismatch:** create a fresh 64-bit Python 3.13
  environment and run `python scripts/verify_environment.py`. Do not repair a
  mismatched environment by guessing package versions.
- **CUDA unavailable:** this is expected on native Windows with TensorFlow
  2.20. Use the CPU requirements and continue locally, or run the GPU app in
  WSL2. Check `nvidia-smi` and `tf.config.list_physical_devices('GPU')` there.
- **Missing checkpoint file:** point the app at the extracted folder containing
  exactly `model.weights.h5`, `vocabulary.json`, and `config.json`, not at the
  ZIP or an outer directory.
- **Vocabulary or settings mismatch:** re-export all three files together with
  `save_checkpoint`; never combine files from different exports.
- **Unsupported checkpoint format:** export again with the current notebook and
  app revision. Older formats are rejected intentionally.
- **Prompt too long:** shorten the processed prompt to the maximum shown in the
  error. Prompts are rejected rather than silently truncated.
- **TensorFlow startup warnings:** oneDNN, CPU feature, and Keras deprecation
  messages are informational unless they are followed by an actual exception.

## Notebook

Run `notebook/compact_gpt_retrain_2.ipynb` in standard hosted Colab with a GPU
runtime. The notebook clones the repository, verifies the runtime versions,
trains the shared model, reloads the exported checkpoint, compares predictions
before and after loading, and downloads a ZIP containing the three checkpoint
files.

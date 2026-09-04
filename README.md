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
- Node.js 20+ with npm, used only to build the React frontend

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

The interface is a React application (in `frontend/`) served by the same
process that runs the Python model engine. The engine — checkpoint loading,
prompt analysis, and captured-tensor inspection — stays in Gradio behind the
scenes and is reached by the frontend over local JSON endpoints. Build the
frontend once, and again after any frontend change:

```powershell
cd frontend
npm install
npm run build
cd ..
```

Then start the app from the environment containing the Python dependencies.
`python app.py` serves the built React interface at the root URL and the model
engine underneath it; it stays bound to localhost with public sharing disabled:

```powershell
python app.py
```

Open `http://127.0.0.1:7860` in a browser. The **Server path** is pre-filled
with the included development checkpoint; replace it with your extracted
folder path and press **Load Model**. The app validates all three files before
showing the model details and reports CUDA GPU or CPU based on TensorFlow's
actual device visibility.

For frontend development, run the engine and the Vite dev server together:

```powershell
python app.py
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`; Vite proxies the model endpoints to the engine on
`127.0.0.1:7860`. The dev server hot-reloads frontend changes without touching
Python or the model.

For a quick UI check, use a desktop browser width of at least 1280px. Verify
that expanding the model diagram shows one residual line with the attention
and FFN branches, that analyzing a prompt selects the default **Layer norm ·
block output** node, and that clicking the node-strip chips and the ◀ Previous
/ Next ▶ controls move between captured states.

Loading a new checkpoint first releases the current model and clears the old
details. If the replacement fails, no model remains active. TensorFlow may keep
reserved GPU memory until the process exits even after the old model is
released.

## Analyze A Prompt

After loading a checkpoint, enter text in **Prompt** and press **Analyze
Prompt**. The prompt is processed exactly like training text: lowercased,
punctuation separated by spaces, and split on whitespace using the saved
vocabulary. Each processed token appears as a selectable chip showing its
0-based position and text, next to the processed token count against the
model's maximum sequence length.

Empty prompts and prompts longer than the model maximum are rejected rather
than silently shortened. Tokens missing from the vocabulary remain allowed and
are shown as `[UNK]` with a visible warning and count. The top five predicted
next tokens for the last position are listed inline; select the **Readout**
node for full top-K bars and per-token entropy. Results stay in memory for the
session only and are never written to disk.

## Trace The Residual Stream

The same Analyze Prompt run captures every internal tensor of the model, so you
can move through the model without running inference again. Dropout is disabled
during this run, so repeated runs are deterministic. The capture includes the
decomposed token and position embeddings, the residual-stream states, the
attention and FFN updates, the two layer norms, the causal attention pattern
(mean over heads), and the FFN hidden activation.

The page is organized around the residual stream, but it keeps the map on
screen: a compact **node strip** of chips sits directly above the captured
state, and the full wiring diagram is collapsed by default so the strip and
heatmap both fit the viewport. Press **Show model diagram** to expand the
wiring any time. The residual stream is one horizontal line from the
embeddings through the two "add" junctions and layer norms to the readout. The
attention and FFN blocks hang off that line as parallel branches: they read
the stream, compute, and write their output back at the add junction. Every
node below is a captured state you can select either from the node strip or
from its chip in the expanded diagram:

| Node | Kind | Notes |
| --- | --- | --- |
| Token embeddings | component | one half of the stream input |
| Position embeddings | component | the other half of the stream input |
| Residual stream input | stream | token + position embeddings |
| Causal attention pattern | pattern | query × key weights, mean over heads |
| Attention output → residual | update | what attention writes into the stream |
| Residual stream after attention | stream | stream input + attention update |
| Layer norm after attention | ln | normalization sitting on the line |
| FFN hidden (ReLU) | hidden | non-negative, sequential color scale |
| FFN output → residual | update | what the FFN writes into the stream |
| Residual stream after FFN | stream | normalized attention + FFN update |
| Layer norm block output | ln | the value the readout reads |
| Readout probabilities | readout | top-K next tokens + entropy |

Exactly one node's view is rendered at a time. Selecting a chip from the node
strip (or the expanded diagram), or stepping with ◀ Previous / Next ▶ or the
token chips, shows that node's captured tensor as one **row of square
heatmaps**: each token gets its
own square tile in a single heatmap with a clear gap between tiles, and the
selected token's tile is outlined in red. The plot always fits the panel
width; scroll the mouse wheel or use the mode bar to zoom in when you want a
closer look. A token's width-256 vector is laid out row-major as a 16×16 grid
(other widths use the factor pair closest to a square). Select a token from the
chips, the position dropdown, or by clicking any cell of its tile. Changing any
selection re-renders from captured data only and never runs the model again.

Color is normalized per node: each activation node's heatmap is scaled to its
own zero-centered range (the symmetric max-absolute-value of that tensor), so
every node uses its full colorbar and stream/update/norm nodes are not washed
out by each other's magnitudes. Normalized nodes explain that their magnitudes
may look nearly flat because layer normalization fixes the feature scale.

The readout node renders for the selected token: the most likely next tokens as
horizontal bars and a per-token entropy strip showing uncertainty across the
whole prompt. It is derived from the captured probabilities, whose ordering
matches the pre-softmax logits.

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
- **Frontend not built:** `python app.py` prints a notice when
  `frontend/dist` is missing. Build it once with `npm run build` inside
  `frontend/`, then start the app again. For iterative UI work, use `npm run
  dev` against a running `python app.py`.
- **TensorFlow startup warnings:** oneDNN, CPU feature, and Keras deprecation
  messages are informational unless they are followed by an actual exception.

## Notebook

Run `notebook/compact_gpt_retrain_2.ipynb` in standard hosted Colab with a GPU
runtime. The notebook clones the repository, verifies the runtime versions,
trains the shared model, reloads the exported checkpoint, compares predictions
before and after loading, and downloads a ZIP containing the three checkpoint
files.

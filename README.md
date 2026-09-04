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

The desktop workbench requires a browser viewport of at least `1920x1080` CSS
pixels. The exact model topology stays across the top, captured locations are
grouped on the left, the activation canvas occupies the flexible center, and
tensor metadata stays in the right inspector. The workbench fills wider and
taller viewports while its side panels remain stable. The page itself does not
scroll at supported sizes; analytical context and controls remain fixed while a
long token-by-dimension heatmap scrolls inside its canvas. Smaller viewports are
unsupported and may overflow rather than switching to a compact layout. The
**Locations** and **Metadata** buttons manually collapse or restore their
panels, and `Escape` closes an expanded canvas.

Checkpoint setup starts open and collapses after a successful load. A failed
load leaves it open and preserves the current research session. After a
successful analysis, the exact submitted prompt is shown as read-only session
context; use **Edit prompt** to return to the editor. Prompt tokens and the five
next-token predictions remain available in the right-side secondary-information
accordion.

Checkpoint loading validates and prepares a replacement before switching the
active model. A successful replacement resets the old capture and inspection
controls together; a failed replacement leaves the current model, capture, and
views unchanged. TensorFlow may keep reserved GPU memory until the process exits
even after an old model is released.

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
| Token embeddings | Embedding |
| Position embeddings | Embedding |
| Pre-attention residual | Residual Stream |
| Attention update | Attention |
| Attention residual sum | Residual Stream |
| First normalized output | Residual Stream |
| FFN hidden activation | FFN |
| FFN update | FFN |
| FFN residual sum | Residual Stream |
| Final block output | Residual Stream |

The topology diagram follows the computation rather than presenting a linear
list. Token and position embeddings enter the pre-attention residual, one
uninterrupted horizontal line carries the residual stream, and the attention
and feed-forward paths branch above it. Each branch returns through an explicit
addition node, followed by its computationally correct post-norm stage. Only
the ten captured tensors are buttons; vocabulary projection and softmax are
shown as non-interactive output stages. Selecting a diagram node or a grouped
navigator entry changes location in one click. Navigator rows show the exact
captured matrix shape, including the wider FFN hidden dimension.

Use **Pin current as A** to hold the current location as the comparison
reference. The next location selected from the diagram, navigator, or dropdown
immediately becomes **B**; no second confirmation or prompt run is needed. A
and B always come from the same in-memory prompt capture. The diagram and
navigator label both locations explicitly with monochrome A/B markers rather
than relying on color. Select A again or press **Unpin A** to leave comparison
mode.

After a successful analysis, the **Internal location** selector opens at the
**Final block output** with no token positions selected. The initial activation
field uses the default **Square** layout, with one row-major grid per processed
token. A 256-dimensional tensor is shown as an exact `16x16` grid; other widths
use the smallest square that preserves every source dimension, with unused
cells shown as non-data padding. Hover shows the token position, token text,
dimension index, and unrounded raw activation value. Long captures scroll
vertically inside the workspace, including the model maximum of 80 tokens.

Click any cell inside a token's overview grid to select or toggle that token; no
Ctrl, Shift, or other modifier is required. The first selected token opens token
detail automatically, and clearing the final token in **Token positions**
returns to the all-token Square overview.
Selections are identities by 0-based token position, so repeated token text is
unambiguous and the same positions stay selected while navigating locations.
Black outlines indicate selected structure without introducing another data
color.

The fixed diverging color scale maps negative values to purple, zero to white,
and positive values to orange. **Scale scope** defaults to **Location**, using
the exact symmetric range of the complete current location. **Capture** uses one
range across every captured location. **Selection** becomes available only after
one or more tokens are selected and uses those rows. The same resolved bounds
are passed to the overview and selected-token detail views so colors remain
comparable when changing views.

During comparison, raw A and B activations are shown side by side in both the
all-token overview and selected-token detail modes. Token selection is shared:
changing it updates both locations together. Their symmetric color bounds are
pooled according to the active **Scale scope**. When the complete A and B tensor
shapes match exactly, a full-width signed **B - A** delta is also shown with its
own symmetric zero-centered bounds. If the shapes differ, such as an FFN hidden
tensor whose width differs from the residual width, both raw tensors remain
available and the app reports their exact shapes and why delta is disabled.
Subtraction never broadcasts, truncates, projects, pads, or otherwise coerces
incompatible tensors.

**Square** is the default activation layout for both the all-token overview and
selected-token detail. It keeps dimensions in source order, shows unused cells
in gray as non-data padding, and explicitly warns that two-dimensional adjacency
is artificial. **Indexed** shows the unpadded all-token matrix when no tokens are
selected and one literal dimension row per selected token. Both layouts hover
the exact stored raw value rather than a rounded chart value.

Click a detail cell to pin one exact `(token position, dimension)` measurement;
clicking another detail cell replaces it. The black cell outline and metadata
show the pinned identity and its raw value at the current location. The identity
persists while navigating locations whose tensor width contains that dimension.
If a location is too narrow, the pin is cleared and the metadata explains the
incompatible dimension and shape. Overview clicks only select rows and cannot
create measurement pins.

The heatmap reports its visible zero-centered range. Full-range display is the
default. **Clip extremes** changes only the color bounds to the 1st-99th
percentile; captured values, hover values, statistics, and distributions remain
unchanged. While clipping is active, both the persistent range warning and the
chart subtitle disclose it. Selecting tokens opens magnitude, detail, and
distribution views; changing selectors re-renders stored capture data without
running the model again.

Captured tensors live in memory for the session only. They are cleared when a
new checkpoint is loaded or replaced by a successfully rendered fresh analysis;
token selections and pinned measurements reset with that successful
replacement. A failed analysis leaves the previous valid capture and inspection
views active. Selection, detail-mode, and measurement changes only re-render the
stored capture and never rerun inference.

Checkpoint replacement reports validation, model loading, and rendering stages;
prompt replacement reports validation, tracing, and rendering. Concise errors
appear beside the initiating control, with exception type and message available
in the adjacent collapsed technical-details section.

## Open A Second-Monitor Diagnostics Window

**Open diagnostics** in the metadata inspector opens a named, same-origin popup
for second-monitor use. The main workbench remains the sole controller: the
popup is read-only, has no state-changing controls, and never talks to the
server or to a second Gradio session.

- Clicking **Open diagnostics** again focuses the same named window; closing,
  reopening, resizing, or a blocked popup never damages the main session.
- The popup shows the current checkpoint, prompt, location, and selection
  context, an always-available token-magnitude chart (all tokens, selection
  highlighted), and a distribution chart for the selected tokens only. With no
  selection the distribution panel keeps a stable explanatory empty state.
- The two charts sit side by side when the window is wide and stack when it is
  narrow.
- The main workbench publishes one versioned, atomic payload over postMessage
  on every relevant change. The popup validates the exact origin and ignores
  stale revisions, so context and charts never mix revisions.
- Plotly is served from the installed local package (same origin, no CDN, no
  telemetry, no localStorage, no second Gradio session), and Gradio run history
  stays disabled so captured diagnostics are not persisted in browser history.
- Each chart card has **PNG** and **SVG** download buttons. They export the
  already-rendered figure entirely in the browser through the same local
  Plotly bundle (no Kaleido, no external service) and are disabled while the
  corresponding chart is not on screen.

## Export Research Artifacts

**Export artifacts…** in the metadata inspector (**Runtime and diagnostics**)
opens an accessible dialog. It never auto-runs: choose a scope and a format
explicitly, then press **Download**. Options that are not valid for the current
state are disabled with an inline reason.

Scopes:

- **Current selection** exports the selected token rows at the current
  location. It is disabled until at least one token position is selected.
- **Current location** exports every captured token at the current location.
- **Comparison** exports pinned **A** and the current location **B**. It is
  disabled until A is pinned and a different location is selected as B.

Formats:

- **CSV** is deterministic long-form data with one row per token position and
  source dimension (sorted by token position, then dimension). Columns are
  `token_position`, `token_text`, `token_id`, `location_key`,
  `source_dimension_index`, and `raw_value`. Comparison rows add a `side`
  column (`A`, `B`, and `delta` when included). A single `# `-prefixed comment
  line carries the same structured JSON metadata embedded in NPZ files.
- **NumPy .npz** preserves the source matrices as two-dimensional float64
  arrays (`values`, or `values_a`/`values_b`/`delta` for comparisons) and
  embeds the structured metadata as `metadata.json` inside the same archive.

The structured JSON metadata records the checkpoint path, architecture,
compute device, and model configuration; the processed prompt with every token
position/text/ID; the exported location key and label; the literal source
dimension indices; the current selection; any pinned measurement (with its raw
value when the dimension exists); and the comparison A/B sides. When a delta
is exported it is the exact signed `B - A`; incompatible A/B tensors (for
example an FFN hidden width that differs from the residual width) refuse the
delta export with a precise reason while raw A and B remain exportable. The
delta checkbox is disabled in that case.

Exported values are always the raw captured floats. Display clipping and
square-grid padding are display-only concerns and never alter or appear in the
raw data: there are no padded cells, and values are never clipped. CSV and NPZ
metadata mark this explicitly.

PNG/SVG figures are generated in the browser from the already-rendered Plotly
charts (no Kaleido, no CDN). The dialog offers the **active activation view**
(the chart(s) currently on the canvas, including comparison A/B/delta),
**token magnitudes**, and **selected-token distribution**; targets whose chart
is not currently rendered are disabled. The diagnostics window provides the
same PNG/SVG downloads for its two charts.

Raw files are generated strictly on request: each **Download** press builds one
artifact in memory and returns it as a temporary browser download; nothing is
written to disk and nothing is persisted automatically. Captured tensors and
selection state remain session-only, as elsewhere in the app. This is the only
way raw data leaves the app, so exports are easy to tell apart from automatic
persistence (there is none).

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

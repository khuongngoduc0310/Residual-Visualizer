# Agent Instructions

## Product Direction

- Build a Python app with Gradio for exploring language-model residual streams.
- The app must load a language model, show its model blocks, let the user select an internal location, and visualize the residual at that location.
- The first supported model is the configurable one-block, post-norm TensorFlow causal language model defined in `model.py`.

## Model And Checkpoint Constraints

- Keep `model.py` importable by both Colab training and the local app; importing it must not build, compile, train, or summarize a model.
- Preserve the checkpoint contract documented in `README.md`. Reject uncertain checkpoints instead of guessing missing settings.
- Training text is right-padded. Padding targets have zero sample weight, and real tokens rely on the causal mask to avoid future padding.

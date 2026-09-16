# Agent Instructions

## Product Direction

- Build a Python app with Gradio for exploring language-model residual streams.
- The app must load a language model, show its model blocks, let the user select an internal location, and visualize the residual at that location.
- The first supported model is the configurable-width, three-block, pre-norm TensorFlow causal language model with a final readout norm and padding-aware FFN activity L1 defined in `model.py`.

## Model And Checkpoint Constraints

- Keep `model.py` importable by both Colab training and the local app; importing it must not build, compile, train, or summarize a model.
- Preserve the checkpoint contract documented in `README.md`. Reject uncertain checkpoints instead of guessing missing settings.
- Training text is right-padded. Padding targets have zero sample weight, and real tokens rely on the causal mask to avoid future padding.
- FFN activity L1 must mask padding token positions explicitly because Keras sample weights do not apply to auxiliary layer losses.

## Engineering Design

- Prefer the smallest correct design. Add abstractions only for a concrete variation, reusable unit, or separate responsibility.
- Give each module one reason to change. Keep the dependency direction from core model modules through `engine.py` to the inspection view modules (`graph_view.py`, `readout_view.py`, `deembed_view.py`, `contribution_view.py`, `inspection_views.py`), then `server.py`, and finally the `app.py` entrypoint.
- Keep model and domain logic independent of Gradio, FastAPI, Plotly serialization, and process startup.
- Keep the inspection payload key set in `inspection_payload.py` and mirror it in `frontend/src/types.ts`. Regenerate `tests/fixtures/` with `tests/generate_fixtures.py` when the payload intentionally changes.
- Make dependencies explicit at system boundaries. Keep mutable state owned by one component rather than shared through module globals.
- Treat `ModelManager` as the owner of loaded-model and inspection-session state. Hold its lifecycle lock whenever an operation requires a consistent model and capture pair.
- Raise specific domain errors within core code and translate them into safe, actionable responses at the API boundary.
- Use types and immutable data structures to make contracts visible. Avoid hidden mutation of captured model data.
- Apply SOLID principles as design heuristics, not as reasons to introduce unnecessary classes, interfaces, or layers.
- Extract a helper when its name clarifies intent or it isolates a responsibility; otherwise keep related logic together.

## Change Discipline

- Inspect existing behavior, call sites, and tests before changing a contract.
- Preserve checkpoint formats, frontend endpoint names, and JSON payload shapes unless the task explicitly changes them.
- Make the smallest behavior-preserving refactor that establishes a clear ownership boundary.
- Add a regression test for every fixed bug and focused tests for new behavior. Test observable contracts rather than implementation details.
- Run focused tests while iterating, then run the complete Python suite with `.\.venv\Scripts\python.exe -m pytest`. Run frontend tests and the production build when frontend behavior or API contracts change.
- Run `.\.venv\Scripts\python.exe -m ruff check .` and `-m ruff format .` for Python, and `npm run lint` and `npm run format` in `frontend/`, before considering a change complete.
- Consider work complete only when the implementation, tests, and relevant documentation agree.

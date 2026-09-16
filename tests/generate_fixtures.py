"""Regenerate the cross-language payload fixtures.

Run from the repository root:

    .venv\\Scripts\\python.exe tests\\generate_fixtures.py

The fixtures in ``tests/fixtures`` are the shared contract between the Python
payload builders and the frontend TypeScript types. Regenerate them whenever a
payload intentionally changes, then review the diff.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support import VOCABULARY, block0, model_manager, write_checkpoint  # noqa: E402

import engine  # noqa: E402
import inspection_views  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def build_payloads(tmp_path: Path) -> dict[str, dict]:
    write_checkpoint(tmp_path)
    manager = model_manager()
    load = engine.load_model_payload(manager, str(tmp_path))
    analyze = engine.analyze_prompt_payload(manager, "hello , world")
    options = inspection_views.options_payload()
    readout = inspection_views.inspect_node_payload(manager, "readout", 2)
    deembed = inspection_views.inspect_node_payload(
        manager, block0("ffn_residual"), 1, "baseline", deembed=True
    )
    contribution = inspection_views.inspect_node_payload(
        manager, block0("attention_update"), 1, "baseline", vocab_contributions=True
    )
    ablation = engine.ablate_feature_payload(
        manager, block0("ffn_hidden"), [0], "zero", "all", None
    )
    return {
        "load": load,
        "analyze": analyze,
        "options": options,
        "inspect_readout": readout,
        "inspect_deembed": deembed,
        "inspect_contribution": contribution,
        "ablate": ablation,
    }


def write_fixtures(payloads: dict[str, dict]) -> None:
    FIXTURE_DIR.mkdir(exist_ok=True)
    schema = {}
    for name, payload in payloads.items():
        (FIXTURE_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        schema[name] = {key: type_name(value) for key, value in payload.items()}
    (FIXTURE_DIR / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        payloads = build_payloads(Path(directory) / "checkpoint")
    write_fixtures(payloads)
    print(f"Wrote {len(payloads)} fixtures to {FIXTURE_DIR}")
    print("Vocabulary:", VOCABULARY)


if __name__ == "__main__":
    main()

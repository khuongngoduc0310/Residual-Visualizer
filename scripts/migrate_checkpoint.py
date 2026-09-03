"""Migrate a verified format-1 checkpoint to the current format-2 contract."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import keras
import tensorflow as tf

from checkpoint import (
    CONFIG_FILENAME,
    SUPPORTED_KERAS_VERSION,
    SUPPORTED_TENSORFLOW_VERSION,
    TOKENIZER_SETTINGS,
    VOCABULARY_FILENAME,
    WEIGHTS_FILENAME,
    save_checkpoint,
)
from model import ARCHITECTURE_NAME, ModelConfig, build_model
from preprocess import validate_vocabulary


def migrate(source_directory: str, destination_directory: str) -> None:
    source = Path(source_directory).expanduser().resolve()
    destination = Path(destination_directory).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source checkpoint folder does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination folder must be empty: {destination}")

    document = json.loads((source / CONFIG_FILENAME).read_text(encoding="utf-8"))
    vocabulary = json.loads(
        (source / VOCABULARY_FILENAME).read_text(encoding="utf-8")
    )
    if document.get("format_version") != 1:
        raise ValueError("This utility only migrates format-1 checkpoints")
    if document.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError("Checkpoint architecture is not supported")
    if document.get("tensorflow_version") != SUPPORTED_TENSORFLOW_VERSION:
        raise ValueError(
            "Checkpoint was not exported with "
            f"TensorFlow {SUPPORTED_TENSORFLOW_VERSION}"
        )
    if tf.__version__ != SUPPORTED_TENSORFLOW_VERSION:
        raise ValueError(
            f"Current TensorFlow must be {SUPPORTED_TENSORFLOW_VERSION}; "
            f"found {tf.__version__}"
        )
    if keras.__version__ != SUPPORTED_KERAS_VERSION:
        raise ValueError(
            f"Current Keras must be {SUPPORTED_KERAS_VERSION}; "
            f"found {keras.__version__}"
        )
    if document.get("tokenizer") != TOKENIZER_SETTINGS:
        raise ValueError("Checkpoint tokenizer settings do not match the app")
    try:
        validate_vocabulary(vocabulary)
        config = ModelConfig.from_dict(document["model"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint metadata is invalid: {error}") from error
    if len(vocabulary) != config.vocab_size:
        raise ValueError("Vocabulary size does not match model settings")

    model = build_model(config)
    try:
        model.load_weights(str(source / WEIGHTS_FILENAME))
    except (OSError, ValueError) as error:
        raise ValueError("Checkpoint weights do not match model settings") from error

    save_checkpoint(destination, model, vocabulary, config)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/migrate_checkpoint.py SOURCE DESTINATION")
        return 2
    try:
        migrate(sys.argv[1], sys.argv[2])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Migration failed: {error}")
        return 1
    print(f"Migrated checkpoint to: {Path(sys.argv[2]).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

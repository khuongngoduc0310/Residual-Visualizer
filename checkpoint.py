import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import tensorflow as tf
from tensorflow.keras import Model, layers

from model import (
    ARCHITECTURE_NAME,
    ModelConfig,
    TokenAndPositionEmbedding,
    TransformerBlock,
    build_model,
)
from preprocess import (
    PADDING_TOKEN_ID,
    SPLIT,
    STANDARDIZE,
    UNKNOWN_TOKEN_ID,
    validate_vocabulary,
)


CHECKPOINT_FORMAT_VERSION = 1
SUPPORTED_TENSORFLOW_MINOR = (2, 10)
WEIGHTS_FILENAME = "model.weights.h5"
VOCABULARY_FILENAME = "vocabulary.json"
CONFIG_FILENAME = "config.json"
CHECKPOINT_FILENAMES = {
    WEIGHTS_FILENAME,
    VOCABULARY_FILENAME,
    CONFIG_FILENAME,
}
CHECKPOINT_DIGEST_ATTRIBUTE = "circuit_tracer_checkpoint_digest"

TOKENIZER_SETTINGS = {
    "standardize": STANDARDIZE,
    "split": SPLIT,
    "separate_punctuation": True,
    "padding_token_id": PADDING_TOKEN_ID,
    "unknown_token_id": UNKNOWN_TOKEN_ID,
}


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: Model
    vocabulary: List[str]
    config: ModelConfig


def save_checkpoint(
    directory,
    model: Model,
    vocabulary: List[str],
    config: ModelConfig,
) -> None:
    _require_supported_tensorflow(tf.__version__, "Current TensorFlow version")
    try:
        validate_vocabulary(vocabulary)
    except ValueError as error:
        raise CheckpointError(str(error)) from error
    if len(vocabulary) != config.vocab_size:
        raise CheckpointError(
            "The vocabulary size does not match config vocab_size "
            f"({len(vocabulary)} != {config.vocab_size})"
        )
    _validate_model(model, config)

    checkpoint_directory = Path(directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _require_empty_checkpoint_directory(checkpoint_directory)
    document = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "tensorflow_version": tf.__version__,
        "model": config.to_dict(),
        "tokenizer": TOKENIZER_SETTINGS,
    }
    _write_json(checkpoint_directory / CONFIG_FILENAME, document)
    _write_json(checkpoint_directory / VOCABULARY_FILENAME, vocabulary)
    weights_path = checkpoint_directory / WEIGHTS_FILENAME
    model.save_weights(str(weights_path))
    _write_checkpoint_digest(weights_path, document, vocabulary)


def load_checkpoint(directory) -> LoadedCheckpoint:
    _require_supported_tensorflow(tf.__version__, "Current TensorFlow version")
    checkpoint_directory = Path(directory)
    if not checkpoint_directory.is_dir():
        raise CheckpointError(
            f"Checkpoint folder does not exist: {checkpoint_directory}"
        )
    _reject_unexpected_files(checkpoint_directory)

    paths = {
        CONFIG_FILENAME: checkpoint_directory / CONFIG_FILENAME,
        VOCABULARY_FILENAME: checkpoint_directory / VOCABULARY_FILENAME,
        WEIGHTS_FILENAME: checkpoint_directory / WEIGHTS_FILENAME,
    }
    for filename, path in paths.items():
        if not path.is_file():
            raise CheckpointError(f"Missing checkpoint file: {filename}")

    document = _read_json(paths[CONFIG_FILENAME], "model settings")
    config = _parse_config(document)
    vocabulary = _read_json(paths[VOCABULARY_FILENAME], "vocabulary")
    try:
        validate_vocabulary(vocabulary)
    except ValueError as error:
        raise CheckpointError(str(error)) from error
    if len(vocabulary) != config.vocab_size:
        raise CheckpointError(
            "The vocabulary size does not match config vocab_size "
            f"({len(vocabulary)} != {config.vocab_size})"
        )
    _validate_checkpoint_digest(
        paths[WEIGHTS_FILENAME],
        document,
        vocabulary,
    )

    model = build_model(config)
    try:
        model.load_weights(str(paths[WEIGHTS_FILENAME]))
    except (OSError, ValueError) as error:
        raise CheckpointError(
            "The checkpoint weights do not match the saved model settings"
        ) from error
    return LoadedCheckpoint(model=model, vocabulary=vocabulary, config=config)


def _parse_config(document) -> ModelConfig:
    if not isinstance(document, dict):
        raise CheckpointError("Model settings must be a JSON object")
    expected_keys = {
        "format_version",
        "architecture",
        "tensorflow_version",
        "model",
        "tokenizer",
    }
    if set(document) != expected_keys:
        missing = sorted(expected_keys - set(document))
        unknown = sorted(set(document) - expected_keys)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise CheckpointError(f"Invalid config fields: {'; '.join(details)}")
    if document.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint format version: {document.get('format_version')}"
        )
    if document.get("architecture") != ARCHITECTURE_NAME:
        raise CheckpointError(
            f"Unsupported model architecture: {document.get('architecture')}"
        )
    saved_tensorflow_version = document.get("tensorflow_version")
    _require_supported_tensorflow(
        saved_tensorflow_version,
        "Checkpoint TensorFlow version",
    )
    if document.get("tokenizer") != TOKENIZER_SETTINGS:
        raise CheckpointError("Tokenizer settings do not match this app")
    try:
        return ModelConfig.from_dict(document.get("model"))
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"Invalid model settings: {error}") from error


def _validate_model(model: Model, config: ModelConfig) -> None:
    expected_layer_names = [
        "token_ids",
        "token_and_position_embedding",
        "transformer_block",
        "token_probabilities",
    ]
    if model.name != ARCHITECTURE_NAME or [
        layer.name for layer in model.layers
    ] != expected_layer_names:
        raise CheckpointError("The model architecture does not match config")

    embedding = model.get_layer("token_and_position_embedding")
    transformer = model.get_layer("transformer_block")
    output = model.get_layer("token_probabilities")
    if not isinstance(embedding, TokenAndPositionEmbedding):
        raise CheckpointError("The model embedding does not match config")
    if not isinstance(transformer, TransformerBlock):
        raise CheckpointError("The transformer block does not match config")
    if not isinstance(output, layers.Dense):
        raise CheckpointError("The model output does not match config")
    if tf.keras.activations.serialize(output.activation) != "softmax":
        raise CheckpointError("The model output activation must be softmax")
    if embedding.vocab_size != output.units:
        raise CheckpointError(
            "The embedding and output vocabulary sizes do not match"
        )
    attention_config = transformer.attn.get_config()
    if (
        embedding.vocab_size != embedding.token_emb.input_dim
        or embedding.embed_dim != embedding.token_emb.output_dim
        or embedding.max_len != embedding.pos_emb.input_dim
        or embedding.embed_dim != embedding.pos_emb.output_dim
        or transformer.num_heads != attention_config["num_heads"]
        or transformer.key_dim != attention_config["key_dim"]
        or transformer.embed_dim != attention_config["output_shape"]
        or transformer.ff_dim != transformer.ffn_1.units
        or transformer.embed_dim != transformer.ffn_2.units
    ):
        raise CheckpointError("The model layer sizes do not match config")
    if (
        transformer.ln_1.epsilon != transformer.layer_norm_epsilon
        or transformer.ln_2.epsilon != transformer.layer_norm_epsilon
        or transformer.dropout_1.rate != transformer.dropout_rate
        or transformer.dropout_2.rate != transformer.dropout_rate
        or tf.keras.activations.serialize(transformer.ffn_1.activation)
        != transformer.feed_forward_activation
    ):
        raise CheckpointError("The transformer layers do not match config")

    actual_config = ModelConfig(
        vocab_size=output.units,
        max_len=embedding.max_len,
        embedding_dim=embedding.embed_dim,
        num_heads=transformer.num_heads,
        key_dim=transformer.key_dim,
        feed_forward_dim=transformer.ff_dim,
        dropout_rate=transformer.dropout_rate,
        feed_forward_activation=transformer.feed_forward_activation,
        layer_norm_epsilon=transformer.layer_norm_epsilon,
    )
    if actual_config != config:
        raise CheckpointError(
            "The model does not match the supplied model settings"
        )


def _reject_unexpected_files(directory: Path) -> None:
    unexpected = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name not in CHECKPOINT_FILENAMES
    )
    if unexpected:
        raise CheckpointError(
            f"Unexpected checkpoint file: {', '.join(unexpected)}"
        )


def _require_empty_checkpoint_directory(directory: Path) -> None:
    _reject_unexpected_files(directory)
    if any(directory.iterdir()):
        raise CheckpointError("Checkpoint export folder must be empty")


def _read_json(path: Path, description: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Could not read {description}: {error}") from error


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_digest(document, vocabulary) -> str:
    payload = json.dumps(
        {"config": document, "vocabulary": vocabulary},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_checkpoint_digest(path: Path, document, vocabulary) -> None:
    try:
        with h5py.File(path, "a") as weights_file:
            weights_file.attrs[CHECKPOINT_DIGEST_ATTRIBUTE] = _checkpoint_digest(
                document,
                vocabulary,
            )
    except (OSError, ValueError) as error:
        raise CheckpointError(
            "Could not bind checkpoint settings to the model weights"
        ) from error


def _validate_checkpoint_digest(path: Path, document, vocabulary) -> None:
    try:
        with h5py.File(path, "r") as weights_file:
            saved_digest = weights_file.attrs.get(CHECKPOINT_DIGEST_ATTRIBUTE)
    except OSError as error:
        raise CheckpointError("Could not read checkpoint weights") from error
    if isinstance(saved_digest, bytes):
        saved_digest = saved_digest.decode("ascii", errors="replace")
    if saved_digest != _checkpoint_digest(document, vocabulary):
        raise CheckpointError(
            "The checkpoint files do not match each other"
        )


def _tensorflow_minor(version):
    if not isinstance(version, str):
        return None
    try:
        major, minor = version.split(".", maxsplit=2)[:2]
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _require_supported_tensorflow(version, description):
    if _tensorflow_minor(version) != SUPPORTED_TENSORFLOW_MINOR:
        raise CheckpointError(
            f"{description} must be TensorFlow 2.10; found {version}"
        )

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from checkpoint import LoadedCheckpoint


RESIDUAL_STREAM = "Residual Stream"
EMBEDDING = "Embedding"
ATTENTION = "Attention"
FFN = "FFN"

DEFAULT_LOCATION_KEY = "output_norm"


class InspectionError(ValueError):
    pass


@dataclass(frozen=True)
class LocationSpec:
    key: str
    label: str
    category: str
    explanation: str
    normalized: bool = False


LOCATIONS: Tuple[LocationSpec, ...] = (
    LocationSpec(
        key="token_embedding",
        label="Token embeddings",
        category=EMBEDDING,
        explanation="The learned embedding vector looked up for each prompt token.",
    ),
    LocationSpec(
        key="position_embedding",
        label="Position embeddings",
        category=EMBEDDING,
        explanation="The learned position vector added at each prompt position.",
    ),
    LocationSpec(
        key="embedding",
        label="Pre-attention residual",
        category=RESIDUAL_STREAM,
        explanation=(
            "The token and position embeddings summed for every prompt token. "
            "This is the residual stream as it enters the transformer block."
        ),
    ),
    LocationSpec(
        key="attention_update",
        label="Attention update",
        category=ATTENTION,
        explanation=(
            "What the causal attention heads add to the residual stream for "
            "each token, before any normalization."
        ),
    ),
    LocationSpec(
        key="attention_residual",
        label="Attention residual sum",
        category=RESIDUAL_STREAM,
        explanation=(
            "The residual stream after the attention update is added back to "
            "the input embeddings."
        ),
    ),
    LocationSpec(
        key="attention_norm",
        label="First normalized output",
        category=RESIDUAL_STREAM,
        explanation=(
            "The first post-norm layer normalization applied to the attention "
            "residual stream. Because layer normalization fixes each token's "
            "feature scale, its token-magnitude chart may look nearly flat."
        ),
        normalized=True,
    ),
    LocationSpec(
        key="ffn_hidden",
        label="FFN hidden activation",
        category=FFN,
        explanation=(
            "The raw hidden activation of the feed-forward network, expanded "
            "to the wider FFN width."
        ),
    ),
    LocationSpec(
        key="ffn_update",
        label="FFN update",
        category=FFN,
        explanation=(
            "What the feed-forward network adds to the normalized attention "
            "residual stream."
        ),
    ),
    LocationSpec(
        key="ffn_residual",
        label="FFN residual sum",
        category=RESIDUAL_STREAM,
        explanation=(
            "The residual stream after the FFN update is added back to the "
            "normalized attention output."
        ),
    ),
    LocationSpec(
        key="output_norm",
        label="Final block output",
        category=RESIDUAL_STREAM,
        explanation=(
            "The final layer-normalized block output. It is the residual "
            "stream state used to predict the next token. Because layer "
            "normalization fixes each token's feature scale, its token-"
            "magnitude chart may look nearly flat."
        ),
        normalized=True,
    ),
)

LOCATION_KEYS: Tuple[str, ...] = tuple(spec.key for spec in LOCATIONS)
LOCATION_BY_KEY: Dict[str, LocationSpec] = {
    spec.key: spec for spec in LOCATIONS
}


@dataclass(frozen=True)
class CapturedRun:
    locations: Dict[str, np.ndarray]
    probabilities: np.ndarray
    token_count: int


def location_spec(key: str) -> LocationSpec:
    try:
        return LOCATION_BY_KEY[key]
    except KeyError:
        raise InspectionError(f"Unknown internal location: {key}") from None


def location_choices() -> list:
    return [
        (f"{spec.category} \u00b7 {spec.label}", spec.key)
        for spec in LOCATIONS
    ]


def capture_locations(checkpoint: LoadedCheckpoint, token_ids) -> CapturedRun:
    """Run the loaded model once and return every internal tensor as a
    [sequence, width] NumPy array, all with dropout disabled."""
    if not isinstance(checkpoint, LoadedCheckpoint):
        raise TypeError("checkpoint must be a LoadedCheckpoint")

    ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
    if ids.shape.rank != 1:
        raise InspectionError("token ids must be a flat sequence")

    token_count = int(tf.size(ids))
    if token_count == 0:
        raise InspectionError("Prompt does not contain any tokens.")
    if token_count > checkpoint.config.max_len:
        raise InspectionError(
            f"Prompt has {token_count} tokens; the model accepts at most "
            f"{checkpoint.config.max_len}."
        )

    token_ids = ids[None, :]
    model = checkpoint.model
    embedding_layer = model.get_layer("token_and_position_embedding")
    token_embeddings = embedding_layer.token_emb(token_ids)
    position_embeddings = embedding_layer.pos_emb(tf.range(token_count))
    embeddings = token_embeddings + position_embeddings
    steps, _attention_scores = model.get_layer(
        "transformer_block"
    ).call_steps(embeddings, training=False)
    block_output = steps["output_norm"]
    probabilities = model.get_layer("token_probabilities")(block_output)

    locations = {
        "token_embedding": tf.squeeze(token_embeddings, axis=0),
        "position_embedding": position_embeddings,
        "embedding": tf.squeeze(embeddings, axis=0),
        **{
            key: tf.squeeze(tensor, axis=0)
            for key, tensor in steps.items()
        },
    }
    return CapturedRun(
        locations={key: value.numpy() for key, value in locations.items()},
        probabilities=tf.squeeze(probabilities, axis=0).numpy(),
        token_count=token_count,
    )

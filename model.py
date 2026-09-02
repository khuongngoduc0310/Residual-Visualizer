import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import layers, losses, models


ARCHITECTURE_NAME = "one_block_post_norm_causal_lm"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_len: int = 80
    embedding_dim: int = 256
    num_heads: int = 2
    key_dim: int = 128
    feed_forward_dim: int = 256
    dropout_rate: float = 0.1
    feed_forward_activation: str = "relu"
    layer_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        positive_integers = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "feed_forward_dim": self.feed_forward_dim,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.vocab_size < 2:
            raise ValueError("vocab_size must include PAD and UNK tokens")
        if self.num_heads * self.key_dim != self.embedding_dim:
            raise ValueError(
                "num_heads * key_dim must equal embedding_dim for this model"
            )
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be at least 0 and less than 1")
        if self.feed_forward_activation != "relu":
            raise ValueError("feed_forward_activation must be 'relu'")
        if (
            not math.isfinite(self.layer_norm_epsilon)
            or self.layer_norm_epsilon <= 0
        ):
            raise ValueError("layer_norm_epsilon must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        if not isinstance(values, dict):
            raise ValueError("model settings must be a JSON object")

        expected = {item.name for item in fields(cls)}
        supplied = set(values)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise ValueError(f"model settings are missing: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"model settings are unknown: {', '.join(unknown)}")
        return cls(**values)


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    destination = tf.range(n_dest)[:, None]
    source = tf.range(n_src)
    mask = tf.cast(destination >= source - n_src + n_dest, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    multiples = tf.concat(
        [
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.tile(mask, multiples)


@tf.keras.utils.register_keras_serializable(package="CircuitTracer")
class TransformerBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        embed_dim,
        ff_dim,
        dropout_rate=0.1,
        feed_forward_activation="relu",
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.feed_forward_activation = feed_forward_activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            output_shape=embed_dim,
            name="causal_attention",
        )
        self.dropout_1 = layers.Dropout(dropout_rate, name="attention_dropout")
        self.ln_1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="attention_layer_norm",
        )
        self.ffn_1 = layers.Dense(
            ff_dim,
            activation=feed_forward_activation,
            name="feed_forward_hidden",
        )
        self.ffn_2 = layers.Dense(embed_dim, name="feed_forward_output")
        self.dropout_2 = layers.Dropout(dropout_rate, name="feed_forward_dropout")
        self.ln_2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="output_layer_norm",
        )

    def call(self, inputs, training=None):
        steps, attention_scores = self.call_steps(inputs, training)
        return steps["output_norm"], attention_scores

    def call_steps(self, inputs, training=None):
        """Run every stage, returning each intermediate named after its
        diagram stage plus the attention scores."""
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size,
            seq_len,
            seq_len,
            tf.bool,
        )
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
            training=training,
        )
        attention_update = self.dropout_1(attention_output, training=training)
        attention_residual = inputs + attention_update
        normalized_attention = self.ln_1(attention_residual)

        feed_forward_hidden = self.ffn_1(normalized_attention)
        feed_forward_update = self.ffn_2(feed_forward_hidden)
        feed_forward_update = self.dropout_2(
            feed_forward_update,
            training=training,
        )
        feed_forward_residual = normalized_attention + feed_forward_update
        output = self.ln_2(feed_forward_residual)
        return {
            "attention_update": attention_update,
            "attention_residual": attention_residual,
            "attention_norm": normalized_attention,
            "ffn_hidden": feed_forward_hidden,
            "ffn_update": feed_forward_update,
            "ffn_residual": feed_forward_residual,
            "output_norm": output,
        }, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "feed_forward_activation": self.feed_forward_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="CircuitTracer")
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="token_embedding",
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_len,
            output_dim=embed_dim,
            name="position_embedding",
        )

    def call(self, token_ids):
        sequence_length = tf.shape(token_ids)[-1]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        return self.token_emb(token_ids) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


def build_model(config: ModelConfig) -> models.Model:
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")

    token_ids = layers.Input(shape=(None,), dtype=tf.int32, name="token_ids")
    embeddings = TokenAndPositionEmbedding(
        max_len=config.max_len,
        vocab_size=config.vocab_size,
        embed_dim=config.embedding_dim,
        name="token_and_position_embedding",
    )(token_ids)
    block_output, _ = TransformerBlock(
        num_heads=config.num_heads,
        key_dim=config.key_dim,
        embed_dim=config.embedding_dim,
        ff_dim=config.feed_forward_dim,
        dropout_rate=config.dropout_rate,
        feed_forward_activation=config.feed_forward_activation,
        layer_norm_epsilon=config.layer_norm_epsilon,
        name="transformer_block",
    )(embeddings)
    probabilities = layers.Dense(
        config.vocab_size,
        activation="softmax",
        name="token_probabilities",
    )(block_output)
    return models.Model(
        inputs=token_ids,
        outputs=probabilities,
        name=ARCHITECTURE_NAME,
    )


def compile_for_training(model: models.Model) -> None:
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(),
    )

import re
import string
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras import layers


PADDING_TOKEN = ""
UNKNOWN_TOKEN = "[UNK]"
PADDING_TOKEN_ID = 0
UNKNOWN_TOKEN_ID = 1
STANDARDIZE = "lower"
SPLIT = "whitespace"

_PUNCTUATION_PATTERN = re.compile(
    f"([{re.escape(string.punctuation)}\\n])"
)


def pad_punctuation(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    separated = _PUNCTUATION_PATTERN.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", separated).strip()


def build_text_vectorizer(
    *,
    max_tokens: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    vocabulary: Optional[Sequence[str]] = None,
) -> layers.TextVectorization:
    if vocabulary is not None:
        vocabulary = list(vocabulary)
        validate_vocabulary(vocabulary)
        if max_tokens is None:
            max_tokens = len(vocabulary)

    return layers.TextVectorization(
        standardize=STANDARDIZE,
        split=SPLIT,
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
        vocabulary=vocabulary,
    )


def prepare_training_batch(text, vectorizer):
    tokenized = vectorizer(tf.expand_dims(text, -1))
    inputs = tokenized[:, :-1]
    targets = tokenized[:, 1:]
    weights = tf.cast(targets != PADDING_TOKEN_ID, tf.float32)
    return inputs, targets, weights


def validate_vocabulary(vocabulary: Sequence[str]) -> None:
    if not isinstance(vocabulary, (list, tuple)):
        raise ValueError("vocabulary must be a list of strings")
    if len(vocabulary) < 2:
        raise ValueError("vocabulary must include PAD and UNK tokens")
    if not all(isinstance(token, str) for token in vocabulary):
        raise ValueError("vocabulary must contain only strings")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary tokens must be unique")
    if vocabulary[PADDING_TOKEN_ID] != PADDING_TOKEN:
        raise ValueError("vocabulary token 0 must be the empty PAD token")
    if vocabulary[UNKNOWN_TOKEN_ID] != UNKNOWN_TOKEN:
        raise ValueError("vocabulary token 1 must be [UNK]")

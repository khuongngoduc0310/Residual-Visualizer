from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from checkpoint import LoadedCheckpoint
from inspection import CapturedRun, InspectionError, capture_locations
from preprocess import (
    PADDING_TOKEN_ID,
    UNKNOWN_TOKEN_ID,
    build_text_vectorizer,
    pad_punctuation,
)


NEXT_TOKEN_COUNT = 5


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class TokenInfo:
    position: int
    text: str
    token_id: int


@dataclass(frozen=True)
class NextToken:
    rank: int
    text: str
    token_id: int
    probability: float


@dataclass(frozen=True)
class PromptAnalysis:
    tokens: Tuple[TokenInfo, ...]
    token_count: int
    unknown_count: int
    max_len: int
    next_tokens: Tuple[NextToken, ...]
    capture: CapturedRun


def display_text(token_id: int, vocabulary: List[str]) -> str:
    """Return how a token id should be shown, keeping PAD and UNK honest."""
    if token_id == PADDING_TOKEN_ID:
        return "(PAD)"
    if token_id == UNKNOWN_TOKEN_ID:
        return "[UNK]"
    return vocabulary[token_id]


def analyze_prompt(prompt: str, checkpoint: LoadedCheckpoint) -> PromptAnalysis:
    if not isinstance(prompt, str) or not prompt.strip():
        raise AnalysisError("Enter a prompt first.")

    vocabulary = checkpoint.vocabulary
    processed = pad_punctuation(prompt)
    vectorizer = build_text_vectorizer(vocabulary=vocabulary)
    ids = tf.cast(vectorizer(tf.constant([processed])), tf.int32)[0]

    try:
        captured = capture_locations(checkpoint, ids)
    except InspectionError as error:
        raise AnalysisError(str(error)) from error
    token_count = captured.token_count

    unknown_count = int(
        tf.reduce_sum(tf.cast(tf.equal(ids, UNKNOWN_TOKEN_ID), tf.int32))
    )

    probabilities = captured.probabilities[-1]
    top_k = min(NEXT_TOKEN_COUNT, checkpoint.config.vocab_size)
    top_indices = np.argsort(-probabilities)[:top_k]
    sorted_probabilities = probabilities[top_indices]

    tokens = tuple(
        TokenInfo(
            position=position,
            text=display_text(int(token_id), vocabulary),
            token_id=int(token_id),
        )
        for position, token_id in enumerate(ids.numpy())
    )
    next_tokens = tuple(
        NextToken(
            rank=rank,
            text=display_text(int(token_id), vocabulary),
            token_id=int(token_id),
            probability=float(probability),
        )
        for rank, (probability, token_id) in enumerate(
            zip(sorted_probabilities, top_indices),
            start=1,
        )
    )
    return PromptAnalysis(
        tokens=tokens,
        token_count=token_count,
        unknown_count=unknown_count,
        max_len=checkpoint.config.max_len,
        next_tokens=next_tokens,
        capture=captured,
    )

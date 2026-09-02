import numpy as np
import pytest
import tensorflow as tf

from analysis import AnalysisError, analyze_prompt, display_text
from checkpoint import load_checkpoint, save_checkpoint
from model import ModelConfig, build_model
from preprocess import PADDING_TOKEN_ID


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def tiny_config():
    return ModelConfig(
        vocab_size=len(VOCABULARY),
        max_len=6,
        embedding_dim=8,
        num_heads=2,
        key_dim=4,
        feed_forward_dim=8,
        dropout_rate=0.0,
    )


@pytest.fixture(scope="module")
def loaded_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkpoint")
    tf.keras.utils.set_random_seed(9)
    config = tiny_config()
    model = build_model(config)
    save_checkpoint(directory, model, VOCABULARY, config)
    return load_checkpoint(directory)


def test_tokenization_matches_training_rules(loaded_checkpoint):
    analysis = analyze_prompt("Hello, WORLD!", loaded_checkpoint)

    assert analysis.token_count == 4
    assert analysis.unknown_count == 0
    assert [token.text for token in analysis.tokens] == [
        "hello",
        ",",
        "world",
        "!",
    ]
    assert [token.token_id for token in analysis.tokens] == [2, 3, 4, 5]


def test_positions_are_zero_based(loaded_checkpoint):
    analysis = analyze_prompt("hello world !", loaded_checkpoint)

    assert [token.position for token in analysis.tokens] == [0, 1, 2]


def test_unknown_tokens_allowed_and_counted(loaded_checkpoint):
    analysis = analyze_prompt("hello mystery", loaded_checkpoint)

    assert [token.token_id for token in analysis.tokens] == [2, 1]
    assert analysis.unknown_count == 1
    assert analysis.tokens[1].text == "[UNK]"


def test_empty_and_whitespace_prompts_rejected(loaded_checkpoint):
    for prompt in ["", "   ", "\n \t "]:
        with pytest.raises(AnalysisError, match="Enter a prompt first"):
            analyze_prompt(prompt, loaded_checkpoint)


def test_prompt_over_max_len_rejected_without_shortening(loaded_checkpoint):
    with pytest.raises(AnalysisError, match="at most 6"):
        analyze_prompt("hello , world ! hello , world", loaded_checkpoint)


def test_results_contain_no_padding_tokens(loaded_checkpoint):
    analysis = analyze_prompt("hello , world", loaded_checkpoint)

    assert len(analysis.tokens) == analysis.token_count
    assert all(token.token_id != PADDING_TOKEN_ID for token in analysis.tokens)


def test_next_tokens_match_model_top_five_in_order(loaded_checkpoint):
    analysis = analyze_prompt("hello , world", loaded_checkpoint)
    ids = tf.constant(
        [[token.token_id for token in analysis.tokens]],
        dtype=tf.int32,
    )
    probabilities = loaded_checkpoint.model(ids, training=False).numpy()[0, -1, :]
    expected_ids = list(np.argsort(-probabilities)[:5])
    expected_probabilities = probabilities[expected_ids]

    assert [token.token_id for token in analysis.next_tokens] == expected_ids
    assert [token.rank for token in analysis.next_tokens] == [1, 2, 3, 4, 5]
    assert [token.probability for token in analysis.next_tokens] == pytest.approx(
        [float(value) for value in expected_probabilities]
    )
    probabilities_seen = [token.probability for token in analysis.next_tokens]
    assert probabilities_seen == sorted(probabilities_seen, reverse=True)


def test_display_text_labels_pad_and_unk():
    assert display_text(0, VOCABULARY) == "(PAD)"
    assert display_text(1, VOCABULARY) == "[UNK]"
    assert display_text(2, VOCABULARY) == "hello"


def test_analysis_is_deterministic(loaded_checkpoint):
    first = analyze_prompt("hello , world !", loaded_checkpoint)
    second = analyze_prompt("hello , world !", loaded_checkpoint)

    assert [token.token_id for token in first.next_tokens] == [
        token.token_id for token in second.next_tokens
    ]
    assert [token.probability for token in first.next_tokens] == pytest.approx(
        [token.probability for token in second.next_tokens]
    )

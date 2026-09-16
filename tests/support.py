"""Shared constants and helpers for the test suite.

Test modules import from this file directly; pytest puts the ``tests``
directory on ``sys.path`` when it collects them.
"""

import tensorflow as tf

import engine
from checkpoint import save_checkpoint
from model import ModelConfig, build_model

VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]

CONFIG_DEFAULTS = {
    "vocab_size": len(VOCABULARY),
    "max_len": 6,
    "embedding_dim": 8,
    "num_heads": 2,
    "key_dim": 4,
    "feed_forward_dim": 8,
    "dropout_rate": 0.0,
}


def tiny_config(**changes):
    """Build a small model config, overriding only the requested fields."""
    return ModelConfig(**{**CONFIG_DEFAULTS, **changes})


def fake_device(label="CPU", tf_device="/CPU:0", is_gpu=False):
    return engine.ComputeDevice(label=label, tf_device=tf_device, is_gpu=is_gpu)


def model_manager(**kwargs):
    """Build a manager pinned to a fake CPU device unless told otherwise."""
    kwargs.setdefault("device_detector", lambda: fake_device())
    return engine.ModelManager(**kwargs)


def block0(stage):
    """Key of a node in the first transformer block."""
    from inspection import block_node_key

    return block_node_key(0, stage)


def write_checkpoint(path, seed=9, config=None):
    """Create a tiny checkpoint on disk and return the config used."""
    tf.keras.utils.set_random_seed(seed)
    config = config or tiny_config()
    model = build_model(config)
    save_checkpoint(path, model, VOCABULARY, config)
    return config


def loaded_manager(path, seed=9):
    """Create a checkpoint at ``path`` and return a manager with it loaded."""
    write_checkpoint(path, seed=seed)
    manager = model_manager()
    manager.load(str(path))
    return manager


def analyze_fixture(path, prompt="hello , world"):
    """Return a loaded manager plus a successful analysis payload."""
    manager = loaded_manager(path)
    payload = engine.analyze_prompt_payload(manager, prompt)
    assert payload["ok"]
    return manager, payload

import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebook" / "compact_gpt_retrain_2.ipynb"
)


def test_training_notebook_uses_the_supported_three_block_configuration():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "ARCHITECTURE_NAME == 'three_block_pre_norm_causal_lm'" in source
    assert "EMBEDDING_DIM = 1024" in source
    assert "FEED_FORWARD_DIM = 1024" in source
    assert "NUM_HEADS = 8" in source
    assert "KEY_DIM = 128" in source
    assert "NUM_BLOCKS = 3" in source
    assert "FEED_FORWARD_ACTIVITY_L1 = 1e-5" in source
    assert "num_blocks=NUM_BLOCKS" in source
    assert "feed_forward_activity_l1=FEED_FORWARD_ACTIVITY_L1" in source
    assert "import matplotlib.pyplot as plt" in source

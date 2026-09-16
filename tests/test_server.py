"""Tests for the Gradio endpoint wiring."""

from support import model_manager

import server


def test_create_app_builds_endpoints_without_launching():
    demo = server.create_app(model_manager())

    assert isinstance(demo, server.gr.Blocks)
    api_names = {fn.api_name for fn in demo.fns.values()}
    assert {
        "load_checkpoint",
        "analyze_prompt",
        "ablate_feature",
        "clear_ablation",
        "inspect_node",
        "options",
    } <= api_names

import gc
import html
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import tensorflow as tf

import gradio as gr

import export as export_artifacts
from analysis import AnalysisError, PromptAnalysis, analyze_prompt
from charts import (
    display_bounds,
    render_activation_detail,
    render_activation_heatmap,
    render_activation_overview,
    signed_comparison,
    shared_display_bounds,
    render_token_distribution,
    render_token_magnitudes,
    render_token_magnitudes_all,
    tensor_statistics,
)
from checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from inspection import (
    DEFAULT_LOCATION_KEY,
    LOCATION_KEYS,
    location_choices,
    location_spec,
)
from model import ARCHITECTURE_NAME, ModelConfig


LOGGER = logging.getLogger(__name__)
LOCAL_SERVER_NAME = "127.0.0.1"
SHARE_PUBLICLY = False
LIGHT_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)


APP_CSS = """
html,
body,
.gradio-container {
    background: #ffffff !important;
    color: #111827 !important;
    margin: 0 !important;
    max-width: none !important;
    min-height: 100vh !important;
    padding: 0 !important;
}
.gradio-container {
    overflow: hidden !important;
}
.ct-page {
    box-sizing: border-box;
    height: 100vh;
    max-width: none;
    padding: 0.8rem 1rem;
    color: #111827;
    width: min(1440px, 100vw);
    margin: 0 auto;
}
.ct-header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 2rem;
    height: 4.4rem;
    margin-bottom: 0.7rem;
    border-bottom: 1px solid #d1d5db;
    padding: 0 0.2rem 0.7rem;
}
.ct-kicker {
    color: #1d4ed8;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.ct-title {
    color: #111827;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.45rem;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0.35rem 0 0;
}
.ct-subtitle {
    color: #4b5563;
    font-size: 0.95rem;
    margin: 0.55rem 0 0;
}
.ct-header-note {
    color: #4b5563;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-align: right;
    text-transform: uppercase;
}
.ct-console {
    align-items: flex-start;
    gap: 1.25rem;
    height: calc(100vh - 5.9rem);
    min-height: 0;
}
.ct-inspector {
    border-left: 1px solid #d1d5db;
    padding-left: 1rem;
}
.ct-rail {
    background: #f8fafc;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    height: 100%;
    min-height: 0;
    overflow-y: auto;
    padding: 0.8rem;
}
.ct-workspace {
    height: 100%;
    min-height: 0;
    overflow-y: auto;
    min-width: 0;
}
.ct-panel {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
}
.ct-panel + .ct-panel {
    margin-top: 1rem;
}
.ct-section-label {
    color: #4b5563;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 0.55rem;
    text-transform: uppercase;
}
.ct-panel-title {
    color: #111827;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.35rem;
    margin: 0 0 0.7rem;
}
.ct-status {
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 0.65rem 0.75rem;
    background: #ffffff;
    font-size: 0.86rem;
}
.ct-status-success {
    border-color: #9cc4b0;
    background: #eef8f1;
}
.ct-meta {
    color: #374151;
    font-size: 0.82rem;
    line-height: 1.55;
}
.ct-analysis-status {
    margin: 0.75rem 0;
}
.ct-two-up,
.ct-chart-row,
.ct-control-row {
    min-width: 0;
    gap: 0.75rem;
}
.ct-two-up > *,
.ct-chart-row > *,
.ct-control-row > * {
    min-width: 0;
}
.ct-chart-row {
    align-items: stretch;
}
.ct-chart-row .wrap {
    min-height: 330px;
}
.ct-chart-main .wrap {
    min-height: 430px;
}
.ct-table {
    min-width: 0;
}
.ct-table-wrap {
    overflow-x: auto;
}
.ct-diagram-wrap {
    overflow-x: auto;
}
.ct-visual-panel {
    position: relative;
}
.ct-expand-button {
    margin-left: auto;
    width: auto;
}
.ct-close-button {
    display: none;
}
body.ct-visual-expanded {
    overflow: hidden !important;
}
body.ct-visual-expanded #ct-visual-panel {
    background: #ffffff;
    border: 0;
    border-radius: 0;
    box-sizing: border-box;
    height: 100vh;
    inset: 0;
    overflow-y: auto;
    padding: 1.2rem 1.5rem 1.5rem;
    position: fixed;
    width: 100vw;
    z-index: 1000;
}
body.ct-visual-expanded #ct-visual-panel .ct-close-button {
    display: block;
    position: absolute;
    right: 1.5rem;
    top: 1rem;
    z-index: 2;
}
body.ct-visual-expanded #ct-visual-panel .ct-chart-main .wrap {
    min-height: 62vh;
}
.ct-diagram {
    display: flex;
    flex-wrap: nowrap;
    align-items: stretch;
    gap: 0.5rem;
    min-width: 1060px;
    padding: 0.4rem 0 0.75rem;
}
.ct-stage {
    flex: 1 0 112px;
    min-width: 112px;
    border: 1px solid #d1d5db;
    border-top: 3px solid #9ca3af;
    border-radius: 7px;
    padding: 0.65rem;
    background: #f8f6f1;
}
.ct-stage-label {
    color: #6c7482;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ct-stage-name {
    color: #172033;
    font-size: 0.8rem;
    font-weight: 700;
    line-height: 1.2;
    margin-top: 0.35rem;
}
.ct-stage-detail {
    color: #6c7482;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.63rem;
    line-height: 1.35;
    margin-top: 0.45rem;
}
.ct-stage[data-category="Attention"] { border-top-color: #b35c32; }
.ct-stage[data-category="Embedding"] { border-top-color: #7c3aed; }
.ct-stage[data-category="FFN"] { border-top-color: #6a7190; }
.ct-stage[data-category="Residual Stream"] { border-top-color: #3f8a72; }
.ct-stage.ct-selected {
    border-color: #b35c32;
    background: #fff1e9;
    box-shadow: 0 0 0 2px rgba(179, 92, 50, 0.2);
}
.ct-arrow {
    align-self: center;
    color: #a9a49a;
    font-size: 1rem;
}

/* Gradio follows the browser's dark preference, even when a light theme object is
   supplied. Keep this workbench deliberately light and target the nested elements
   Gradio 6 creates for Markdown and form components. */
.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}
.gradio-container .ct-page {
    flex-wrap: nowrap !important;
}
.ct-header,
.ct-header > .column,
.ct-rail,
.ct-workspace {
    flex-wrap: nowrap !important;
}
.ct-header {
    height: auto;
    min-height: 7.2rem;
}
.ct-header > .column {
    gap: 0.3rem !important;
}
.ct-kicker,
.ct-title,
.ct-subtitle,
.ct-header-note,
.ct-section-label,
.ct-panel-title,
.ct-status,
.ct-meta {
    min-height: 0 !important;
    overflow: visible !important;
}
.ct-kicker,
.ct-title,
.ct-subtitle,
.ct-header-note,
.ct-section-label,
.ct-panel-title {
    margin: 0 !important;
    padding: 0 !important;
}
.ct-kicker p,
.ct-title p,
.ct-subtitle p,
.ct-header-note p,
.ct-section-label p,
.ct-panel-title p,
.ct-status p,
.ct-meta p,
.ct-meta li,
.ct-meta strong {
    color: inherit !important;
}
.ct-kicker p,
.ct-header-note p,
.ct-section-label p {
    font-family: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    letter-spacing: inherit !important;
    line-height: 1.45 !important;
    margin: 0 !important;
    text-transform: inherit !important;
}
.ct-kicker p {
    color: #1d4ed8 !important;
}
.ct-header-note p,
.ct-section-label p {
    color: #4b5563 !important;
}
.ct-title p {
    color: #111827 !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 2.45rem !important;
    letter-spacing: -0.04em !important;
    line-height: 1 !important;
    margin: 0 !important;
}
.ct-subtitle p,
.ct-panel-title p,
.ct-status p,
.ct-meta p {
    margin: 0 !important;
}
.ct-subtitle p {
    color: #4b5563 !important;
}
.ct-status p {
    color: #111827 !important;
}
.ct-meta p,
.ct-meta li,
.ct-meta strong {
    color: #374151 !important;
}
.ct-status.prose {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.ct-meta.block,
.ct-meta.prose {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}
.ct-meta > * {
    background: transparent !important;
}
.ct-workspace .ct-section-label,
.ct-workspace .ct-panel-title,
.ct-workspace .ct-meta {
    background: #ffffff !important;
}
.ct-panel-title p {
    color: #111827 !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 1.35rem !important;
    line-height: 1.25 !important;
}
.ct-rail {
    min-width: 18rem !important;
}
.ct-rail > * {
    flex: 0 0 auto !important;
    max-width: 100%;
}
.ct-rail .form {
    flex: 0 0 auto !important;
}
.ct-workspace > .ct-panel {
    flex: 0 0 auto !important;
}
.ct-workspace > .ct-panel > .ct-panel {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.ct-diagram-wrap {
    background: #334155 !important;
    border: 0 !important;
    border-radius: 6px !important;
    color: #f8fafc !important;
    padding: 0.55rem 0.75rem 0 !important;
}
.ct-diagram-wrap > div {
    background: transparent !important;
}
.ct-page textarea,
.ct-page label.container,
.ct-control-row .container,
.ct-control-row .wrap,
.ct-control-row .wrap-inner,
.ct-control-row .secondary-wrap {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #111827 !important;
}
.ct-rail .form > .block,
.ct-control-row .block {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
}
.ct-page textarea::placeholder,
.ct-control-row input::placeholder {
    color: #64748b !important;
    opacity: 1;
}
.ct-page [data-testid="block-info"],
.ct-page .label-wrap,
.ct-control-row label,
.ct-control-row input,
.ct-control-row span {
    color: #111827 !important;
}
.ct-page .label-wrap {
    background: transparent !important;
}
.ct-rail button.label-wrap {
    color: #f8fafc !important;
}
.ct-control-row {
    align-items: end;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 0.7rem;
}
.ct-control-row > * {
    flex-basis: 9rem;
}
.ct-control-row > :first-child {
    flex-basis: 18rem;
}

@media (max-width: 900px) {
    html,
    body,
    .gradio-container {
        min-height: 100% !important;
        overflow: auto !important;
    }
    .ct-page {
        height: auto;
        min-height: 100vh;
        padding: 0.75rem;
    }
    .ct-header {
        align-items: flex-start;
        min-height: 0;
    }
    .ct-header-note {
        display: none !important;
    }
    .ct-console {
        flex-direction: column;
        height: auto;
    }
    .ct-rail,
    .ct-workspace {
        height: auto;
        min-width: 0 !important;
        overflow: visible;
        width: 100%;
    }
    .ct-control-row,
    .ct-chart-row,
    .ct-two-up {
        flex-direction: column;
    }
    .ct-control-row > *,
    .ct-control-row > :first-child {
        flex-basis: auto;
        width: 100%;
    }
}

/* Accessible export dialog. The dialog is native-modal, keyboard focusable,
   and never auto-opens; all downloads happen only from explicit actions. */
#ct-export-dialog {
    border: 1px solid #d1d5db;
    border-radius: 12px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28);
    color: #111827;
    max-height: min(720px, 92vh);
    max-width: min(560px, 94vw);
    padding: 0;
    width: 560px;
}
#ct-export-dialog::backdrop {
    background: rgba(15, 23, 42, 0.42);
}
.ct-export-head {
    border-bottom: 1px solid #e5e7eb;
    padding: 1rem 1.25rem 0.85rem;
}
.ct-export-head h2 {
    font: 600 1.1rem ui-sans-serif, system-ui, sans-serif;
    margin: 0 0 0.3rem;
}
.ct-export-head p {
    color: #4b5563;
    font-size: 0.85rem;
    line-height: 1.45;
    margin: 0;
}
.ct-export-body {
    display: grid;
    gap: 0.9rem;
    max-height: 56vh;
    overflow: auto;
    padding: 1rem 1.25rem;
}
.ct-export-body fieldset {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    margin: 0;
    min-width: 0;
    padding: 0.55rem 0.8rem 0.75rem;
}
.ct-export-body legend {
    color: #374151;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0 0.3rem;
    text-transform: uppercase;
}
.ct-export-option {
    display: flex;
    gap: 0.5rem;
    margin: 0.32rem 0;
}
.ct-export-option input {
    accent-color: #1d4ed8;
    flex: 0 0 auto;
    margin-top: 0.22rem;
}
.ct-export-option label {
    cursor: pointer;
    font-size: 0.92rem;
}
.ct-export-option small {
    color: #6b7280;
    display: block;
    font-size: 0.78rem;
    line-height: 1.35;
}
.ct-export-option input:disabled + label,
.ct-export-option input:disabled ~ label {
    color: #9ca3af;
    cursor: not-allowed;
}
.ct-export-delta {
    color: #374151;
    font-size: 0.9rem;
    margin-top: 0.35rem;
}
.ct-export-reason {
    border-top: 1px solid #e5e7eb;
    color: #4b5563;
    font-size: 0.86rem;
    line-height: 1.45;
    min-height: 1.2rem;
    padding: 0.7rem 1.25rem;
}
.ct-export-actions {
    border-top: 1px solid #e5e7eb;
    display: flex;
    gap: 0.6rem;
    justify-content: flex-end;
    padding: 0.8rem 1.25rem;
}
.ct-export-actions button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.42rem 0.95rem;
}
#ct-export-download {
    background: #1d4ed8;
    border: 1px solid #1d4ed8;
    color: #ffffff;
}
#ct-export-download:disabled {
    background: #d1d5db;
    border-color: #d1d5db;
    color: #6b7280;
    cursor: not-allowed;
}
#ct-export-cancel {
    background: #ffffff;
    border: 1px solid #9ca3af;
    color: #111827;
}
.ct-export-backdrop-note {
    color: #6b7280;
    font-size: 0.8rem;
    line-height: 1.45;
}
"""

# Issue #9 deliberately overrides the earlier stacked layout without disturbing
# the capture, chart, and replacement-session styling introduced by #7 and #8.
APP_CSS += """
.gradio-container {
    height: 100vh !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
.gradio-container .main.fillable {
    max-width: none !important;
    padding: 0 !important;
    width: 100% !important;
}
.gradio-container main.contain {
    max-width: 1440px !important;
    padding: 0 !important;
    width: 100% !important;
}
.gradio-container footer {
    display: none !important;
}
.ct-page {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.65rem !important;
    height: 100vh !important;
    max-width: 1440px !important;
    overflow: hidden !important;
    padding: 0.65rem 0.8rem !important;
    width: 100% !important;
}
.ct-header {
    align-items: center !important;
    border-bottom: 1px solid #cbd5e1;
    flex: 0 0 auto !important;
    gap: 1.25rem !important;
    margin: 0 !important;
    min-height: 3.7rem !important;
    padding: 0 0.15rem 0.55rem !important;
}
.ct-header .ct-title p {
    font-size: 2rem !important;
}
.ct-header-copy {
    flex: 1 1 auto !important;
    min-width: 18rem !important;
}
.ct-checkpoint-setup {
    flex: 0 1 34rem !important;
    min-width: 24rem !important;
}
.ct-checkpoint-setup > .label-wrap {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.ct-checkpoint-row {
    align-items: end !important;
    gap: 0.5rem !important;
}
.ct-checkpoint-row > :first-child {
    flex: 1 1 auto !important;
}
.ct-checkpoint-row > :last-child {
    flex: 0 0 7rem !important;
}
.ct-model-strip {
    flex: 0 0 10.4rem !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding: 0.55rem 0.7rem !important;
}
.ct-model-strip-heading {
    align-items: baseline !important;
    display: flex !important;
    flex: 0 0 auto !important;
    justify-content: space-between !important;
}
.ct-model-strip .ct-panel-title p {
    font-size: 1.05rem !important;
}
.ct-diagram-wrap {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #172033 !important;
    flex: 1 1 auto !important;
    height: 7.55rem !important;
    min-height: 7.55rem !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding: 0 !important;
}
.ct-diagram-wrap > div,
.ct-diagram-wrap .prose {
    height: 100% !important;
    min-height: 0 !important;
    overflow: visible !important;
}
.ct-diagram {
    height: 7.55rem;
    min-width: 1300px;
    position: relative;
    width: 1300px;
}
.ct-diagram-wires {
    height: 100%;
    inset: 0;
    overflow: visible;
    position: absolute;
    width: 100%;
}
.ct-backbone,
.ct-branch-wire,
.ct-input-wire {
    fill: none;
    stroke: #334155;
    stroke-width: 1.5;
    vector-effect: non-scaling-stroke;
}
.ct-backbone {
    stroke-width: 2.25;
}
.ct-diagram-caption {
    color: #64748b;
    font: 600 9px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.08em;
    position: absolute;
    text-transform: uppercase;
}
.ct-diagram-caption.ct-backbone-label { left: 184px; top: 103px; }
.ct-diagram-caption.ct-attention-label { left: 380px; top: 5px; }
.ct-diagram-caption.ct-ffn-label { left: 724px; top: 5px; }
.ct-stage {
    align-items: center;
    appearance: none;
    background: #ffffff;
    border: 1px solid #64748b;
    border-radius: 3px;
    color: #172033;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    height: 36px;
    justify-content: center;
    min-width: 0;
    padding: 3px 6px;
    position: absolute;
    text-align: center;
}
.ct-stage:hover:not(:disabled),
.ct-stage:focus-visible {
    background: #f1f5f9;
    border-color: #0f172a;
    outline: 2px solid #94a3b8;
    outline-offset: 1px;
}
.ct-stage:disabled {
    cursor: default;
    opacity: 0.64;
}
.ct-stage.ct-selected {
    background: #ffffff;
    border: 2px solid #111827;
    box-shadow: none;
}
.ct-stage.ct-pinned {
    box-shadow: 0 0 0 2px #ffffff, 0 0 0 3px #475569;
}
.ct-stage-label {
    color: #64748b;
    font: 700 7px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.ct-stage-name {
    color: #172033;
    font: 700 9px/1.15 ui-sans-serif, system-ui, sans-serif;
    margin-top: 2px;
}
.ct-stage-detail {
    display: none;
}
.ct-stage[data-stage="token_embedding"] { left: 14px; top: 14px; width: 124px; }
.ct-stage[data-stage="position_embedding"] { left: 14px; top: 67px; width: 124px; }
.ct-stage[data-stage="embedding"] { left: 181px; top: 67px; width: 126px; }
.ct-stage[data-stage="attention_update"] { left: 392px; top: 17px; width: 132px; }
.ct-stage[data-stage="attention_residual"] {
    border-radius: 50%;
    height: 34px;
    left: 535px;
    padding: 0;
    top: 68px;
    width: 34px;
}
.ct-stage[data-stage="attention_residual"] .ct-stage-label { display: none; }
.ct-stage[data-stage="attention_residual"] .ct-stage-name { font-size: 16px; margin: 0; }
.ct-stage[data-stage="attention_norm"] { left: 590px; top: 67px; width: 126px; }
.ct-stage[data-stage="ffn_hidden"] { left: 720px; top: 17px; width: 124px; }
.ct-stage[data-stage="ffn_update"] { left: 858px; top: 17px; width: 112px; }
.ct-stage[data-stage="ffn_residual"] {
    border-radius: 50%;
    height: 34px;
    left: 980px;
    padding: 0;
    top: 68px;
    width: 34px;
}
.ct-stage[data-stage="ffn_residual"] .ct-stage-label { display: none; }
.ct-stage[data-stage="ffn_residual"] .ct-stage-name { font-size: 16px; margin: 0; }
.ct-stage[data-stage="output_norm"] { left: 1032px; top: 67px; width: 116px; }
.ct-static-stage {
    background: #e2e8f0;
    border: 1px solid #94a3b8;
    border-radius: 3px;
    color: #334155;
    font: 700 9px/1.15 ui-sans-serif, system-ui, sans-serif;
    height: 36px;
    left: 1170px;
    padding: 7px 8px;
    position: absolute;
    text-align: center;
    top: 67px;
    width: 112px;
}
.ct-node-indicators {
    display: flex;
    gap: 2px;
    position: absolute;
    right: -5px;
    top: -7px;
}
.ct-location-indicator {
    align-items: center;
    background: #111827;
    border: 1px solid #ffffff;
    border-radius: 50%;
    color: #ffffff;
    display: none;
    font: 700 8px/14px ui-monospace, monospace;
    height: 14px;
    justify-content: center;
    width: 14px;
}
.ct-stage.ct-pinned .ct-indicator-a,
.ct-stage.ct-comparison-b .ct-indicator-b,
.ct-location-button.ct-pinned .ct-indicator-a,
.ct-location-button.ct-comparison-b .ct-indicator-b {
    display: inline-flex;
}
.ct-workbench {
    align-items: stretch !important;
    flex: 1 1 auto !important;
    gap: 0.65rem !important;
    min-height: 0 !important;
    overflow: hidden !important;
    flex-wrap: nowrap !important;
}
.ct-location-panel,
.ct-metadata-panel,
.ct-center {
    height: 100% !important;
    min-height: 0 !important;
    flex-wrap: nowrap !important;
}
.ct-location-panel {
    flex: 0 0 14.5rem !important;
    max-width: 14.5rem !important;
    min-width: 14.5rem !important;
    overflow: hidden !important;
    padding: 0.7rem !important;
}
.ct-location-panel > * {
    flex: 0 0 auto !important;
}
.ct-location-nav-host {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}
.ct-location-nav-host > div,
.ct-location-nav-host .prose {
    min-height: 0 !important;
}
.ct-location-group + .ct-location-group {
    border-top: 1px solid #e2e8f0;
    margin-top: 0.7rem;
    padding-top: 0.65rem;
}
.ct-location-group-title {
    color: #64748b;
    font: 700 9px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.1em;
    margin: 0 0 0.35rem;
    text-transform: uppercase;
}
.ct-location-button {
    appearance: none;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: #172033;
    cursor: pointer;
    display: grid;
    gap: 1px 5px;
    grid-template-columns: 1fr auto;
    margin: 2px 0;
    padding: 5px 6px;
    position: relative;
    text-align: left;
    width: 100%;
}
.ct-location-button:hover:not(:disabled),
.ct-location-button:focus-visible {
    background: #f1f5f9;
    border-color: #cbd5e1;
    outline: none;
}
.ct-location-button:disabled {
    cursor: default;
    opacity: 0.48;
}
.ct-location-button.ct-selected {
    border-color: #111827;
}
.ct-location-button.ct-pinned {
    box-shadow: inset 3px 0 #475569;
}
.ct-location-name {
    font-size: 0.76rem;
    font-weight: 650;
    line-height: 1.2;
}
.ct-location-shape {
    color: #64748b;
    font: 600 0.64rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: nowrap;
}
.ct-location-key {
    color: #94a3b8;
    font: 0.57rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    grid-column: 1 / -1;
}
.ct-location-button .ct-node-indicators {
    right: 1px;
    top: -4px;
}
.ct-center {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    gap: 0.55rem !important;
    min-width: 0 !important;
    overflow: hidden !important;
    flex-wrap: nowrap !important;
}
.ct-session-context,
.ct-canvas-toolbar,
.ct-canvas-panel {
    flex: 0 0 auto !important;
    margin: 0 !important;
}
.ct-session-context {
    padding: 0.55rem 0.7rem !important;
}
.ct-session-context .ct-session-context,
.ct-canvas-panel .ct-canvas-panel {
    border: 0 !important;
    padding: 0 !important;
}
.ct-prompt-editor-row,
.ct-prompt-readonly-row {
    align-items: end !important;
    gap: 0.5rem !important;
}
.ct-prompt-editor-row > :first-child,
.ct-prompt-readonly-row > :first-child {
    flex: 1 1 auto !important;
}
.ct-prompt-editor-row > :last-child,
.ct-prompt-readonly-row > :last-child {
    flex: 0 0 7rem !important;
}
.ct-analysis-status {
    border: 0 !important;
    color: #475569 !important;
    margin: 0.25rem 0 0 !important;
    padding: 0 !important;
}
.ct-canvas-toolbar {
    align-items: end !important;
    background: #f8fafc;
    border: 1px solid #d1d5db;
    border-radius: 7px;
    gap: 0.45rem !important;
    padding: 0.45rem !important;
}
.ct-canvas-toolbar > * {
    flex: 1 1 7rem !important;
    min-width: 6rem !important;
}
.ct-canvas-toolbar > :first-child {
    flex: 1.6 1 10rem !important;
}
.ct-panel-toggle {
    flex: 0 0 2.5rem !important;
    min-width: 2.5rem !important;
}
.ct-canvas-panel {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    min-height: 0 !important;
    overflow: hidden !important;
    padding: 0.55rem !important;
}
.ct-canvas-heading {
    align-items: baseline !important;
    display: flex !important;
    flex: 0 0 auto !important;
    justify-content: space-between !important;
    padding: 0 0.2rem 0.35rem !important;
}
.ct-canvas-scroll {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow: auto !important;
    overscroll-behavior: contain;
}
.ct-canvas-scroll > .block,
.ct-canvas-scroll .wrap {
    min-height: 430px;
}
.ct-comparison-row {
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    gap: 0.6rem !important;
    width: 100% !important;
}
.ct-comparison-row > * {
    flex: 1 1 50% !important;
    min-width: 0 !important;
}
.ct-delta-plot {
    width: 100% !important;
}
.ct-activation-plot {
    min-height: 0 !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
}
.ct-metadata-panel {
    flex: 0 0 18rem !important;
    max-width: 18rem !important;
    min-width: 18rem !important;
    overflow-y: auto !important;
    padding: 0.7rem !important;
}
.ct-metadata-panel > * {
    flex: 0 0 auto !important;
}
.ct-metadata-panel .ct-meta {
    background: transparent !important;
}
.ct-metadata-panel .ct-panel-title p {
    font-size: 1.05rem !important;
}
.ct-metadata-panel .ct-status {
    padding: 0.45rem 0.55rem !important;
}
.ct-secondary-plot .wrap {
    min-height: 260px !important;
}
.ct-technical-content {
    font-size: 0.72rem !important;
}
.ct-panel-toggle { display: block !important; }
body.ct-nav-collapsed #ct-location-panel,
body.ct-inspector-collapsed #ct-metadata-panel {
    display: none !important;
}
.ct-location-panel.ct-force-closed,
.ct-metadata-panel.ct-force-closed {
    display: none !important;
}
.ct-location-panel.ct-force-open,
.ct-metadata-panel.ct-force-open {
    display: flex !important;
}
body.ct-visual-expanded #ct-visual-panel {
    display: flex !important;
}
body.ct-visual-expanded #ct-visual-panel .ct-canvas-scroll {
    height: calc(100vh - 4rem) !important;
}
.ct-center.ct-visual-expanded {
    background: #ffffff;
    height: 100vh !important;
    inset: 0;
    padding: 1rem;
    position: fixed;
    width: 100vw !important;
    z-index: 1000;
}
.ct-center.ct-visual-expanded > :not(#ct-visual-panel) {
    display: none !important;
}
.ct-center.ct-visual-expanded #ct-visual-panel {
    display: flex !important;
    flex: 1 1 auto !important;
}
.ct-center.ct-visual-expanded #ct-visual-panel .ct-close-button {
    display: block !important;
}

@media (max-width: 1280px) and (min-width: 761px) {
    #ct-metadata-panel { display: none !important; }
    body.ct-inspector-open #ct-metadata-panel {
        background: #ffffff;
        bottom: 0.75rem;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.2);
        display: flex !important;
        position: fixed;
        right: 0.75rem;
        top: 12rem;
        width: 20rem;
        z-index: 60;
    }
    #ct-metadata-panel.ct-force-open {
        background: #ffffff;
        bottom: 0.75rem;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.2);
        display: flex !important;
        height: auto !important;
        max-height: calc(100vh - 19rem) !important;
        min-height: 0 !important;
        position: fixed;
        right: 0.75rem;
        top: 18.25rem;
        width: 20rem;
        z-index: 60;
    }
}
@media (max-width: 1080px) and (min-width: 761px) {
    #ct-location-panel { display: none !important; }
    body.ct-nav-open #ct-location-panel {
        background: #ffffff;
        bottom: 0.75rem;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.2);
        display: flex !important;
        left: 0.75rem;
        position: fixed;
        top: 12rem;
        width: 15.5rem;
        z-index: 60;
    }
    #ct-location-panel.ct-force-open {
        background: #ffffff;
        bottom: 0.75rem;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.2);
        display: flex !important;
        height: auto !important;
        max-height: calc(100vh - 19rem) !important;
        min-height: 0 !important;
        left: 0.75rem;
        position: fixed;
        top: 18.25rem;
        width: 15.5rem;
        z-index: 60;
    }
    .ct-header-note { display: none !important; }
}
@media (max-width: 900px) and (min-width: 761px) {
    .ct-header-copy .ct-subtitle { display: none !important; }
    .ct-checkpoint-setup { min-width: 20rem !important; }
    .ct-model-strip { flex-basis: 9.4rem !important; }
}
@media (max-width: 760px) {
    html,
    body,
    .gradio-container {
        min-height: 100% !important;
        overflow: auto !important;
    }
    .ct-page {
        height: auto !important;
        min-height: 100vh !important;
        overflow: visible !important;
    }
    .ct-header,
    .ct-workbench {
        flex-direction: column !important;
        overflow: visible !important;
    }
    .ct-checkpoint-setup,
    .ct-location-panel,
    .ct-metadata-panel,
    .ct-center {
        display: flex !important;
        flex: 0 0 auto !important;
        height: auto !important;
        min-width: 0 !important;
        width: 100% !important;
    }
    .ct-model-strip { flex-basis: 10.4rem !important; }
    .ct-canvas-scroll { max-height: 70vh !important; }
    .ct-canvas-toolbar { flex-direction: column !important; }
    .ct-canvas-toolbar > * { width: 100% !important; }
    .ct-comparison-row { flex-direction: column !important; }
    .ct-comparison-row > * { width: 100% !important; }
}
"""


@dataclass(frozen=True)
class ComputeDevice:
    label: str
    tf_device: str
    is_gpu: bool


@dataclass(frozen=True)
class LoadedState:
    checkpoint: LoadedCheckpoint
    checkpoint_path: Path
    device: ComputeDevice


@dataclass(frozen=True)
class InspectionSession:
    analysis: PromptAnalysis
    config: ModelConfig


@dataclass(frozen=True)
class ComparisonState:
    """The A capture selection; B is always the current inspection state."""
    location_key: str
    values: np.ndarray


@dataclass(frozen=True)
class MeasurementPin:
    """A source token/dimension identity traced across compatible locations."""
    token_position: int
    dimension: int


@dataclass(frozen=True)
class DiagnosticState:
    stage: str
    message: str
    error: Optional[str] = None


@dataclass(frozen=True)
class LoadResult:
    success: bool
    status: str
    metadata: str
    device: str
    diagram: str
    summary: str
    technical_details: str = ""
    generation: Optional[int] = None
    stale: bool = False


@dataclass(frozen=True)
class AnalysisResult:
    success: bool
    status: str
    token_count: str
    token_rows: list
    unknown_warning: str
    next_token_rows: list
    technical_details: str = ""


@dataclass(frozen=True)
class AnalysisCandidate:
    operation_id: int
    model_generation: int
    session: InspectionSession


@dataclass(frozen=True)
class AnalysisAttempt:
    candidate: Optional[AnalysisCandidate]
    status: str
    technical_details: str = ""
    stale: bool = False


def detect_compute_device() -> ComputeDevice:
    """Return a GPU only when this TensorFlow build and runtime expose one."""
    try:
        build_info = tf.sysconfig.get_build_info()
        is_cuda_build = bool(build_info.get("is_cuda_build", False))
    except (AttributeError, TypeError):
        is_cuda_build = False

    try:
        is_cuda_build = is_cuda_build or bool(tf.test.is_built_with_cuda())
    except (AttributeError, RuntimeError):
        pass

    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
    except (AttributeError, RuntimeError):
        physical_gpus = []

    if not is_cuda_build or not physical_gpus:
        return ComputeDevice(label="CPU", tf_device="/CPU:0", is_gpu=False)

    gpu_name = None
    try:
        details = tf.config.experimental.get_device_details(physical_gpus[0])
        gpu_name = details.get("device_name")
    except (AttributeError, RuntimeError, TypeError):
        pass
    if not gpu_name:
        gpu_name = physical_gpus[0].name
    return ComputeDevice(
        label=f"CUDA GPU: {gpu_name}",
        tf_device="/GPU:0",
        is_gpu=True,
    )


def format_model_summary(model) -> str:
    lines = []
    model.summary(expand_nested=True, print_fn=lines.append)
    return "\n".join(lines)


def _config_details(config: Optional[ModelConfig]) -> dict:
    if config is None:
        return {
            "embedding": "awaiting checkpoint",
            "attention": "awaiting checkpoint",
            "ffn": "awaiting checkpoint",
            "output": "awaiting checkpoint",
        }
    return {
        "embedding": f"sequence <= {config.max_len}, width {config.embedding_dim}",
        "attention": (
            f"{config.num_heads} heads x {config.key_dim}, "
            f"width {config.embedding_dim}"
        ),
        "ffn": f"hidden width {config.feed_forward_dim}",
        "output": f"softmax over {config.vocab_size} tokens",
    }


def render_model_diagram(
    config: Optional[ModelConfig] = None,
    selected_key: Optional[str] = None,
    pinned_key: Optional[str] = None,
    comparison_key: Optional[str] = None,
    capture_available: Optional[bool] = None,
) -> str:
    """Render the exact post-norm block topology and captured locations."""
    details = _config_details(config)
    capture_available = selected_key is not None if capture_available is None else capture_available
    stages = (
        ("token_embedding", "Embedding", "Token embeddings", details["embedding"]),
        ("position_embedding", "Embedding", "Position embeddings", "learned positions"),
        ("embedding", "Residual Stream", "Pre-attention residual", "token + position"),
        ("attention_update", "Attention", "Causal attention update", details["attention"]),
        ("attention_residual", "Residual Stream", "+", "attention residual sum"),
        ("attention_norm", "Residual Stream", "Post-norm 1", "first normalized output"),
        ("ffn_hidden", "FFN", "FFN hidden activation", details["ffn"]),
        ("ffn_update", "FFN", "FFN update", f"width {config.embedding_dim if config else '?'}"),
        ("ffn_residual", "Residual Stream", "+", "FFN residual sum"),
        ("output_norm", "Residual Stream", "Post-norm 2", "final block output"),
    )
    cards = []
    for key, category, name, detail in stages:
        classes = ["ct-stage"]
        if key == selected_key:
            classes.append("ct-selected")
        if key == pinned_key:
            classes.append("ct-pinned")
        if key == comparison_key:
            classes.append("ct-comparison-b")
        disabled = "" if capture_available else " disabled"
        current = "true" if key == selected_key else "false"
        cards.append(
            "<button type=\"button\" class=\"{}\" data-stage=\"{}\" "
            "data-location-key=\"{}\" data-category=\"{}\" aria-pressed=\"{}\"{}>"
            "<div class=\"ct-stage-label\">{}</div>"
            "<div class=\"ct-stage-name\">{}</div>"
            "<div class=\"ct-stage-detail\">{}</div>"
            "<span class=\"ct-node-indicators\" aria-hidden=\"true\">"
            "<span class=\"ct-location-indicator ct-indicator-a\">A</span>"
            "<span class=\"ct-location-indicator ct-indicator-b\">B</span>"
            "</span></button>".format(
                " ".join(classes),
                html.escape(key),
                html.escape(key),
                html.escape(category),
                current,
                disabled,
                html.escape(category),
                html.escape(name),
                html.escape(detail),
            )
        )
    current_key = html.escape(selected_key or "")
    pinned_location = html.escape(pinned_key or "")
    return (
        '<div class="ct-diagram" data-current-location="{}" '
        'data-pinned-location="{}" data-comparison-location="{}" '
        'aria-label="One-block post-norm model diagram">'
        '<svg class="ct-diagram-wires" viewBox="0 0 1300 121" '
        'preserveAspectRatio="none" aria-hidden="true">'
        '<defs><marker id="ct-arrowhead" markerWidth="7" markerHeight="7" '
        'refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" '
        'fill="#334155"/></marker></defs>'
        '<path class="ct-input-wire" d="M138 32 H162 V85 H181" marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-input-wire" d="M138 85 H181" marker-end="url(#ct-arrowhead)"/>'
        '<line class="ct-backbone" x1="170" y1="85" x2="1168" y2="85" '
        'marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-branch-wire" d="M307 85 H340 V35 H390" marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-branch-wire" d="M524 35 H552 V67" marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-branch-wire" d="M716 85 V35 H718" marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-branch-wire" d="M844 35 H856" marker-end="url(#ct-arrowhead)"/>'
        '<path class="ct-branch-wire" d="M970 35 H997 V67" marker-end="url(#ct-arrowhead)"/>'
        '</svg>'
        '<span class="ct-diagram-caption ct-attention-label">Attention branch</span>'
        '<span class="ct-diagram-caption ct-ffn-label">Feed-forward branch</span>'
        '<span class="ct-diagram-caption ct-backbone-label">Uninterrupted residual backbone</span>'
        '{}'
        '<div class="ct-static-stage" data-stage="vocabulary_projection" '
        'data-category="Output" title="{}">Vocabulary projection<br>+ softmax</div>'
        '</div>'
    ).format(
        current_key,
        pinned_location,
        html.escape(comparison_key or ""),
        "".join(cards),
        html.escape(details["output"]),
    )


LOCATION_GROUPS = (
    ("Embeddings", ("token_embedding", "position_embedding")),
    ("Attention branch", ("attention_update",)),
    ("Feed-forward branch", ("ffn_hidden", "ffn_update")),
    (
        "Residual backbone",
        ("embedding", "attention_residual", "attention_norm", "ffn_residual", "output_norm"),
    ),
)


def location_tensor_shape(
    key: str,
    config: Optional[ModelConfig],
    token_count: Optional[int] = None,
) -> str:
    """Return the captured matrix shape shown by the location navigator."""
    rows = str(token_count) if token_count is not None else "T"
    if config is None:
        width = "?"
    elif key == "ffn_hidden":
        width = str(config.feed_forward_dim)
    else:
        width = str(config.embedding_dim)
    return f"{rows} x {width}"


def render_location_navigator(
    config: Optional[ModelConfig] = None,
    token_count: Optional[int] = None,
    selected_key: Optional[str] = None,
    pinned_key: Optional[str] = None,
    comparison_key: Optional[str] = None,
) -> str:
    """Render grouped one-click navigation for captured tensors only."""
    groups = []
    capture_available = token_count is not None
    for group_name, keys in LOCATION_GROUPS:
        entries = []
        for key in keys:
            spec = location_spec(key)
            classes = ["ct-location-button"]
            if key == selected_key:
                classes.append("ct-selected")
            if key == pinned_key:
                classes.append("ct-pinned")
            if key == comparison_key:
                classes.append("ct-comparison-b")
            disabled = "" if capture_available else " disabled"
            entries.append(
                '<button type="button" class="{}" data-location-key="{}" '
                'aria-pressed="{}"{}>'
                '<span class="ct-location-name">{}</span>'
                '<span class="ct-location-shape">{}</span>'
                '<span class="ct-location-key">{}</span>'
                '<span class="ct-node-indicators" aria-hidden="true">'
                '<span class="ct-location-indicator ct-indicator-a">A</span>'
                '<span class="ct-location-indicator ct-indicator-b">B</span>'
                '</span></button>'.format(
                    " ".join(classes),
                    html.escape(key),
                    "true" if key == selected_key else "false",
                    disabled,
                    html.escape(spec.label),
                    html.escape(location_tensor_shape(key, config, token_count)),
                    html.escape(key),
                )
            )
        groups.append(
            '<section class="ct-location-group"><h3 class="ct-location-group-title">{}</h3>{}</section>'.format(
                html.escape(group_name), "".join(entries)
            )
        )
    return '<nav class="ct-location-nav" aria-label="Captured locations">{}</nav>'.format(
        "".join(groups)
    )


def _format_metadata(path: Path, checkpoint: LoadedCheckpoint) -> str:
    config = checkpoint.config
    return "\n".join(
        [
            f"**Checkpoint:** `{html.escape(str(path))}`",
            f"**Architecture:** `{html.escape(ARCHITECTURE_NAME)}`",
            f"**Vocabulary:** `{config.vocab_size:,}` tokens",
            f"**Maximum sequence length:** `{config.max_len}` tokens",
            f"**Model width:** `{config.embedding_dim}`",
            f"**Attention:** `{config.num_heads}` heads x `{config.key_dim}` key width",
            f"**FFN width:** `{config.feed_forward_dim}`",
        ]
    )


def _success_result(
    path: Path,
    state: LoadedState,
    generation: Optional[int] = None,
) -> LoadResult:
    return LoadResult(
        success=True,
        status="Model loaded successfully.",
        metadata=_format_metadata(path, state.checkpoint),
        device=f"**Compute device:** `{html.escape(state.device.label)}`",
        diagram=render_model_diagram(state.checkpoint.config),
        summary=format_model_summary(state.checkpoint.model),
        generation=generation,
    )


def _technical_error(error: BaseException) -> str:
    message = str(error).strip() or "No error message was provided."
    return f"{type(error).__name__}: {message}"


def _failure_result(
    message: str,
    state: Optional[LoadedState] = None,
    technical_details: str = "",
    stale: bool = False,
) -> LoadResult:
    if state is not None:
        return LoadResult(
            success=False,
            status=f"Replacement failed; previous model remains active. {message}",
            metadata=_format_metadata(state.checkpoint_path, state.checkpoint),
            device=f"**Compute device:** `{html.escape(state.device.label)}`",
            diagram=render_model_diagram(state.checkpoint.config),
            summary=format_model_summary(state.checkpoint.model),
            technical_details=technical_details,
            stale=stale,
        )
    return LoadResult(
        success=False,
        status=f"No model loaded. {message}",
        metadata="",
        device="**Compute device:** `not loaded`",
        diagram=render_model_diagram(),
        summary="",
        technical_details=technical_details,
        stale=stale,
    )


def _analysis_success_result(analysis: PromptAnalysis) -> AnalysisResult:
    unknown_warning = (
        f"**Warning:** this prompt contains `{analysis.unknown_count}` "
        "unknown token(s), mapped to `[UNK]`."
        if analysis.unknown_count
        else ""
    )
    return AnalysisResult(
        success=True,
        status=f"Analysis complete for `{analysis.token_count}` processed token(s).",
        token_count=(
            f"**Processed tokens:** `{analysis.token_count}` of "
            f"`{analysis.max_len}`"
        ),
        token_rows=[
            [token.position, token.text, token.token_id]
            for token in analysis.tokens
        ],
        unknown_warning=unknown_warning,
        next_token_rows=[
            [token.rank, token.text, token.token_id, f"{token.probability:.4f}"]
            for token in analysis.next_tokens
        ],
    )


def _analysis_failure_result(
    message: str,
    technical_details: str = "",
) -> AnalysisResult:
    return AnalysisResult(
        success=False,
        status=message,
        token_count="",
        token_rows=[],
        unknown_warning="",
        next_token_rows=[],
        technical_details=technical_details,
    )


def _token_choices(analysis: PromptAnalysis) -> list:
    return [
        (f"{token.position}: {token.text}", str(token.position))
        for token in analysis.tokens
    ]


def _selected_positions(token_value, token_count: int) -> list[int]:
    if token_value is None or token_value == "":
        return []
    raw = token_value if isinstance(token_value, (list, tuple)) else [token_value]
    positions = set()
    for value in raw:
        try:
            position = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= position < token_count:
            positions.add(position)
    return sorted(positions)


def toggle_token_position(token_value, clicked_position, token_count: int) -> list[int]:
    """Toggle one row without relying on browser multi-select modifier keys."""
    selected = set(_selected_positions(token_value, token_count))
    try:
        position = int(clicked_position)
    except (TypeError, ValueError):
        return sorted(selected)
    if not 0 <= position < token_count:
        return sorted(selected)
    if position in selected:
        selected.remove(position)
    else:
        selected.add(position)
    return sorted(selected)


def location_explanation(spec) -> str:
    """Markdown heading, category and plain-language explanation."""
    return (
        f"### {spec.category} \u00b7 {spec.label}\n\n{spec.explanation}"
    )


def _with_measurement_explanation(explanation: str, message: str) -> str:
    return f"{explanation}\n\n{message}" if message else explanation


def location_stats(spec, values: np.ndarray, token_positions: list[int]) -> str:
    """Markdown lines describing the full tensor and optional selection."""
    seq_len, width = values.shape
    captured = tensor_statistics(values.reshape(-1))
    lines = (
        f"**Shape:** `{seq_len} \u00d7 {width}`\n\n"
        f"**Captured range:** `{captured.minimum:.4f}` to `{captured.maximum:.4f}`\n\n"
        f"**Captured mean / std:** `{captured.mean:.4f}` / "
        f"`{captured.standard_deviation:.4f}`\n\n"
        f"**Selection count:** `{len(token_positions)}`"
    )
    if not token_positions:
        return lines + "\n\nOverview of every processed token."
    selected = tensor_statistics(values[token_positions].reshape(-1))
    return (
        lines
        + f"\n\n**Selected tokens ({len(token_positions)}):** norm `{selected.norm:.4f}`, "
        f"mean `{selected.mean:.4f}`, std `{selected.standard_deviation:.4f}`, "
        f"min `{selected.minimum:.4f}`, max `{selected.maximum:.4f}`"
    )


def all_heatmap_bounds(session: InspectionSession, clipped: bool) -> tuple[float, float]:
    locations = session.analysis.capture.locations
    all_values = np.concatenate(
        [locations[key].reshape(-1) for key in LOCATION_KEYS]
    ).reshape(-1, 1)
    return display_bounds(all_values, clipped)


def capture_heatmap_bounds(
    session: InspectionSession,
    location_key: str,
    token_positions: list[int],
    clipped: bool,
    scope: str,
) -> tuple[float, float]:
    """Resolve the color scale without changing captured tensors."""
    values = session.analysis.capture.locations[location_key]
    if scope == "capture":
        return all_heatmap_bounds(session, clipped)
    if scope == "selection" and token_positions:
        return display_bounds(values[token_positions], clipped)
    return display_bounds(values, clipped)


def comparison_heatmap_bounds(
    session: InspectionSession,
    comparison: ComparisonState,
    location_key: str,
    token_positions: list[int],
    clipped: bool,
    scope: str,
) -> tuple[float, float]:
    """Pool raw A/B values according to the active scale scope."""
    values_a = comparison.values
    values_b = session.analysis.capture.locations[location_key]
    if scope == "capture":
        return all_heatmap_bounds(session, clipped)
    if scope == "selection" and token_positions:
        return shared_display_bounds(
            values_a[token_positions], values_b[token_positions], clipped=clipped
        )
    return shared_display_bounds(values_a, values_b, clipped=clipped)


def diagnostics_json(
    manager: "ModelManager",
    stage: str = "idle",
    message: str = "",
    error: Optional[str] = None,
) -> str:
    """Return a stable, machine-readable snapshot for debugging the workbench."""
    state, session, comparison, generation, model_generation = manager.state_snapshot()
    payload = {
        "schema": "circuit-tracer.diagnostics.v1",
        "stage": stage,
        "message": message,
        "error": error,
        "generation": generation,
        "model_generation": model_generation,
        "model": {
            "loaded": state is not None,
            "checkpoint": str(state.checkpoint_path) if state else None,
            "architecture": ARCHITECTURE_NAME,
            "device": state.device.label if state else None,
        },
        "capture": {
            "available": session is not None,
            "token_count": session.analysis.token_count if session else 0,
            "locations": list(LOCATION_KEYS) if session else [],
        },
        "comparison": {
            "pinned": comparison is not None,
            "location": comparison.location_key if comparison else None,
            "shape": list(comparison.values.shape) if comparison else None,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


DIAGNOSTICS_WINDOW_SCHEMA = "circuit-tracer.diagnostics-window.v1"


def _plotly_spec(fig) -> Optional[dict]:
    """Return a compact {data, layout} spec, dropping the heavyweight template."""
    if fig is None:
        return None
    spec = json.loads(fig.to_json())
    layout = spec.get("layout") or {}
    layout.pop("template", None)
    spec["layout"] = layout
    return spec


def _token_labels(analysis) -> list[str]:
    return [f"{token.position}: {token.text}" for token in analysis.tokens]


def diagnostics_window_payload(
    manager: "ModelManager",
    location_key: Optional[str] = None,
    token_value=None,
) -> dict:
    """Return the atomic diagnostics-window content (revision added on publish).

    The content is a pure function of the manager snapshot plus the current
    view location/selection, so the popup never mixes context and charts
    across revisions. ``None`` inputs fall back to the recorded view.
    """
    state, session, comparison, generation, model_generation = manager.state_snapshot()
    recorded_location, recorded_positions, _ = manager.view_snapshot()
    location_key = location_key if location_key is not None else recorded_location
    if token_value is not None and session is not None:
        positions = _selected_positions(
            token_value, session.analysis.token_count
        )
    else:
        positions = list(recorded_positions)

    model = {
        "loaded": state is not None,
        "checkpoint": str(state.checkpoint_path) if state else None,
        "architecture": ARCHITECTURE_NAME,
        "device": state.device.label if state else None,
    }

    if session is None:
        return {
            "schema": DIAGNOSTICS_WINDOW_SCHEMA,
            "generation": generation,
            "model_generation": model_generation,
            "model": model,
            "capture": {
                "available": False,
                "prompt": None,
                "location": None,
                "selection": [],
                "token_count": 0,
            },
            "magnitudes": {"present": False, "figure": None},
            "distribution": {
                "present": False,
                "figure": None,
                "note": "No capture yet. Analyze a prompt to inspect magnitudes.",
            },
        }

    key = location_key or DEFAULT_LOCATION_KEY
    try:
        spec = location_spec(key)
    except Exception:
        key = DEFAULT_LOCATION_KEY
        spec = location_spec(key)
    values = session.analysis.capture.locations[spec.key]
    labels = _token_labels(session.analysis)
    token_count = session.analysis.token_count
    selection = [
        {"position": position, "label": labels[position]}
        for position in positions
        if 0 <= position < token_count
    ]
    prompt_tokens = [
        {"position": token.position, "text": token.text}
        for token in session.analysis.tokens
    ]

    magnitudes_figure = render_token_magnitudes_all(
        values, labels, positions
    )
    distribution = {
        "present": False,
        "figure": None,
        "note": (
            "Select one or more tokens in the workbench to show their "
            "dimension-value distribution."
        ),
    }
    if positions:
        selected = values[positions].reshape(-1)
        distribution["present"] = True
        distribution["figure"] = _plotly_spec(
            render_token_distribution(
                selected,
                positions[0],
                labels[positions[0]],
                title=(
                    f"Dimension distribution for {len(positions)} selected tokens"
                    if len(positions) > 1
                    else None
                ),
            )
        )

    return {
        "schema": DIAGNOSTICS_WINDOW_SCHEMA,
        "generation": generation,
        "model_generation": model_generation,
        "model": model,
        "capture": {
            "available": True,
            "prompt": {
                "tokens": prompt_tokens,
                "processed": " ".join(
                    token.text for token in session.analysis.tokens
                ),
            },
            "location": {
                "key": spec.key,
                "label": spec.label,
                "category": spec.category,
                "explanation": spec.explanation,
                "shape": location_tensor_shape(
                    spec.key, session.config, token_count
                ),
            },
            "selection": selection,
            "token_count": token_count,
        },
        "magnitudes": {
            "present": True,
            "figure": _plotly_spec(magnitudes_figure),
        },
        "distribution": distribution,
    }


def diagnostics_window_payload_json(
    manager: "ModelManager",
    location_key: Optional[str] = None,
    token_value=None,
) -> str:
    """Serialize one versioned atomic payload, bumping revision only on change."""
    content = diagnostics_window_payload(manager, location_key, token_value)
    with manager._lock:
        content_text = json.dumps(
            {key: value for key, value in content.items() if key != "revision"},
            sort_keys=True,
            separators=(",", ":"),
        )
        if content_text != manager._last_payload_content:
            manager._payload_revision += 1
            manager._last_payload_content = content_text
        content["revision"] = manager._payload_revision
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


INSPECT_AWAITING = (
    "Analyze a prompt to capture every internal location."
)
SCALE_SCOPE_CHOICES = [
    ("Selection", "selection"),
    ("Location", "location"),
    ("Capture", "capture"),
]
EMPTY_SELECTION_SCOPE_CHOICES = SCALE_SCOPE_CHOICES[1:]

CLICK_BRIDGE_JS = """
(() => {
  let clickSequence = 0;
  const setBridgeValue = (selector, value) => {
    const input = document.querySelector(selector);
    if (!input) return false;
    const prototype = input instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
    if (!setter) return false;
    setter.call(input, String(value));
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
    return true;
  };
  const bindOverlayButton = (id, action) => {
    const host = document.querySelector(`#${id}`);
    if (!host || host.dataset.overlayButtonBound === 'true') return;
    const button = host.querySelector('button') || host;
    host.dataset.overlayButtonBound = 'true';
    button.addEventListener('click', action);
  };
  const installOverlayControls = () => {
    bindOverlayButton('ct-expand-visuals', () => {
      document.body.classList.add('ct-visual-expanded');
    });
    bindOverlayButton('ct-close-visuals', () => {
      document.body.classList.remove('ct-visual-expanded');
    });
    bindOverlayButton('ct-toggle-nav', () => {
      if (window.matchMedia('(max-width: 1080px)').matches) {
        document.body.classList.toggle('ct-nav-open');
      } else {
        document.body.classList.toggle('ct-nav-collapsed');
      }
    });
    bindOverlayButton('ct-toggle-inspector', () => {
      if (window.matchMedia('(max-width: 1280px)').matches) {
        document.body.classList.toggle('ct-inspector-open');
      } else {
        document.body.classList.toggle('ct-inspector-collapsed');
      }
    });
  };
  const syncLocationIndicators = () => {
    const diagram = document.querySelector('.ct-diagram');
    if (!diagram) return;
    const current = diagram.dataset.currentLocation || '';
    const pinned = diagram.dataset.pinnedLocation || '';
    const comparison = diagram.dataset.comparisonLocation || '';
    document.querySelectorAll('[data-location-key]').forEach((node) => {
      const isCurrent = node.dataset.locationKey === current;
      node.classList.toggle('ct-selected', isCurrent);
      node.classList.toggle('ct-pinned', node.dataset.locationKey === pinned);
      node.classList.toggle('ct-comparison-b', node.dataset.locationKey === comparison);
      node.setAttribute('aria-pressed', isCurrent ? 'true' : 'false');
    });
  };
  const install = () => {
    syncLocationIndicators();
    const bindPlot = (hostSelector, handler) => {
      const graph = document.querySelector(hostSelector)?.querySelector('.js-plotly-plot');
      if (!graph || graph.dataset.clickBridgeInstalled === 'true') return;
      graph.dataset.clickBridgeInstalled = 'true';
      graph.on('plotly_click', handler);
    };
    const publish = (payload) => {
      payload.schema = 'circuit-tracer.click.v1';
      payload.sequence = ++clickSequence;
      setBridgeValue(
        '#ct-token-click input, #ct-token-click textarea', JSON.stringify(payload)
      );
    };
    bindPlot('#ct-magnitude-plot', (event) => {
      const point = event?.points?.[0];
      const position = point?.customdata ?? point?.pointIndex;
      if (position === undefined) return;
      publish({view: 'magnitude', token_position: Number(position)});
    });
    bindPlot('#ct-activation-plot', (event) => {
      const data = event?.points?.[0]?.customdata;
      if (!Array.isArray(data) || data[0] !== 'circuit-tracer.activation.v1') return;
      const view = data[1];
      const tokenPosition = Number(data[2]);
      const dimension = Number(data[4]);
      if (!Number.isInteger(tokenPosition) || !Number.isInteger(dimension)) return;
      if (view === 'overview') {
        publish({view, token_position: tokenPosition, dimension});
      } else if (view === 'detail' && dimension >= 0) {
        publish({view, token_position: tokenPosition, dimension});
      }
    });
  };
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    document.body.classList.remove(
      'ct-visual-expanded', 'ct-nav-open', 'ct-inspector-open'
    );
    document.querySelectorAll('.ct-force-open').forEach((panel) => {
      panel.classList.remove('ct-force-open');
    });
    document.querySelector('.ct-center')?.classList.remove('ct-visual-expanded');
  });
  install();
  new MutationObserver(install).observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
"""

LOCATION_NAV_JS = """
const bindLocationButtons = () => {
  element.querySelectorAll('button[data-location-key]').forEach((button) => {
    if (button.dataset.gradioLocationBound === 'true') return;
    button.dataset.gradioLocationBound = 'true';
    button.addEventListener('click', (event) => {
      if (button.disabled) return;
      event.preventDefault();
      trigger('click', {location: button.dataset.locationKey});
    });
  });
};
bindLocationButtons();
watch('value', bindLocationButtons);
"""

NAV_TOGGLE_JS = """
() => {
  const narrow = window.matchMedia('(max-width: 1080px)').matches;
  document.querySelectorAll('#ct-location-panel').forEach((panel) => {
    panel.classList.toggle('ct-force-open', narrow && !panel.classList.contains('ct-force-open'));
    panel.classList.toggle('ct-force-closed', !narrow && !panel.classList.contains('ct-force-closed'));
  });
  return [];
}
"""

INSPECTOR_TOGGLE_JS = """
() => {
  const narrow = window.matchMedia('(max-width: 1280px)').matches;
  document.querySelectorAll('#ct-metadata-panel').forEach((panel) => {
    panel.classList.toggle('ct-force-open', narrow && !panel.classList.contains('ct-force-open'));
    panel.classList.toggle('ct-force-closed', !narrow && !panel.classList.contains('ct-force-closed'));
  });
  return [];
}
"""

EXPAND_CANVAS_JS = """
() => {
  document.querySelector('.ct-center')?.classList.add('ct-visual-expanded');
  return [];
}
"""

CLOSE_CANVAS_JS = """
() => {
  document.querySelector('.ct-center')?.classList.remove('ct-visual-expanded');
  return [];
}
"""

DIAGNOSTICS_POPUP_NAME = "circuit-tracer-diagnostics"
DIAGNOSTICS_POPUP_PATH = "/ct-diagnostics"
PLOTLY_ASSET_PATH = "/ct-assets/plotly.min.js"
DIAGNOSTICS_SCHEMA = DIAGNOSTICS_WINDOW_SCHEMA
DIAGNOSTICS_REQUEST_SCHEMA = "circuit-tracer.diagnostics-window.request"


def diagnostics_popup_html() -> str:
    """Same-origin read-only popup page.

    The page never talks to the server itself: the main workbench (the sole
    controller) publishes one versioned atomic payload via postMessage. This
    page validates the exact origin and the monotonic revision before drawing
    read-only context plus two charts with the locally-served Plotly bundle
    (no CDN, no telemetry, no localStorage). The two chart cards sit side by
    side when the window is wide and stack when it is narrow.
    """
    schema = DIAGNOSTICS_SCHEMA
    request_schema = DIAGNOSTICS_REQUEST_SCHEMA
    plotly_path = PLOTLY_ASSET_PATH
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Circuit Tracer diagnostics</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #ffffff; color: #172033; }}
  body {{ font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }}
  header {{ border-bottom: 1px solid #e2e8f0; padding: 0.8rem 1rem; }}
  header h1 {{ margin: 0 0 0.2rem; font: 500 1.15rem Georgia, serif; color: #0f172a; }}
  header p {{ margin: 0; color: #64748b; font: 11px ui-monospace, monospace; }}
  main {{ display: grid; gap: 1rem; padding: 1rem; }}
  #context {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.7rem 0.9rem; }}
  #context h2 {{ margin: 0 0 0.4rem; font-size: 0.72rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b; }}
  #context .line {{ margin: 0.15rem 0; overflow-wrap: anywhere; }}
  #context .muted {{ color: #64748b; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 1rem; }}
  .chart {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.4rem; min-width: 0; }}
  .chart h2 {{ margin: 0.25rem 0.5rem; font-size: 0.78rem; color: #334155; }}
  .plot {{ min-height: 320px; width: 100%; }}
  .empty {{ border: 1px dashed #cbd5e1; border-radius: 6px; color: #64748b;
    margin: 0.5rem; padding: 1.5rem 1rem; text-align: center; }}
  .chart-head {{ align-items: center; display: flex; justify-content: space-between;
    gap: 0.5rem; }}
  .chart-head h2 {{ margin: 0.25rem 0.5rem; font-size: 0.78rem; color: #334155; }}
  .fig-export {{ display: flex; gap: 0.25rem; margin: 0 0.4rem; }}
  .fig-export button {{ background: #ffffff; border: 1px solid #cbd5e1;
    border-radius: 5px; color: #1d4ed8; cursor: pointer; font-size: 0.72rem;
    padding: 0.15rem 0.5rem; }}
  .fig-export button:hover {{ background: #eff6ff; }}
  .fig-export button:disabled {{ color: #9ca3af; cursor: not-allowed;
    background: #f8fafc; }}
</style>
</head>
<body>
<header>
  <h1>Circuit Tracer diagnostics</h1>
  <p>Read-only · synchronized from the main workbench · image downloads only</p>
</header>
<main>
  <section id="context"><h2>Session context</h2></section>
  <div class="charts">
    <section class="chart">
      <div class="chart-head">
        <h2>Token magnitudes</h2>
        <div class="fig-export">
          <button type="button" data-export-graph="mag" data-export-format="png" title="Download this chart as PNG">PNG</button>
          <button type="button" data-export-graph="mag" data-export-format="svg" title="Download this chart as SVG">SVG</button>
        </div>
      </div>
      <div class="plot" id="mag"></div>
      <div class="empty" id="mag-empty" hidden>No capture yet.</div>
    </section>
    <section class="chart">
      <div class="chart-head">
        <h2>Selected-token distribution</h2>
        <div class="fig-export">
          <button type="button" data-export-graph="dist" data-export-format="png" title="Download this chart as PNG">PNG</button>
          <button type="button" data-export-graph="dist" data-export-format="svg" title="Download this chart as SVG">SVG</button>
        </div>
      </div>
      <div class="plot" id="dist" hidden></div>
      <div class="empty" id="dist-empty"></div>
    </section>
  </div>
</main>
<script src="{plotly_path}"></script>
<script>
(() => {{
  const SCHEMA = "{schema}";
  const REQUEST_SCHEMA = "{request_schema}";
  const ORIGIN = window.location.origin;
  let lastRevision = -1;

  function render(payload) {{
    const ctx = payload.capture || {{}};
    const model = payload.model || {{}};
    const lines = [];
    if (model.loaded) {{
      lines.push(['checkpoint', model.checkpoint || '(loaded)']);
      lines.push(['architecture', model.architecture]);
      lines.push(['device', model.device || 'unknown']);
    }} else {{
      lines.push(['model', 'no model loaded']);
    }}
    if (ctx.available) {{
      lines.push(['prompt', ctx.prompt ? ctx.prompt.processed : '']);
      const loc = ctx.location || {{}};
      lines.push(['location', loc.label ? loc.label + ' — ' + loc.shape : '']);
      lines.push(['explanation', loc.explanation || '']);
      const sel = (ctx.selection || []).map(s => s.label);
      lines.push(['selected', sel.length ? sel.join(', ') : 'none']);
      lines.push(['token count', String(ctx.token_count)]);
    }} else {{
      lines.push(['capture', 'analyze a prompt first']);
    }}
    const el = document.getElementById('context');
    el.innerHTML = '<h2>Session context</h2>';
    for (const [k, v] of lines) {{
      const div = document.createElement('div');
      div.className = 'line';
      const key = document.createElement('span');
      key.className = 'muted';
      key.textContent = k + ': ';
      div.appendChild(key);
      div.appendChild(document.createTextNode(String(v || '')));
      el.appendChild(div);
    }}

    const mag = payload.magnitudes || {{}};
    const dist = payload.distribution || {{}};
    const magEmpty = document.getElementById('mag-empty');
    const distEmpty = document.getElementById('dist-empty');
    const magEl = document.getElementById('mag');
    const distEl = document.getElementById('dist');
    const hasPlotly = typeof window.Plotly !== 'undefined';

    if (mag.present && mag.figure && hasPlotly) {{
      magEl.hidden = false;
      magEmpty.hidden = true;
      window.Plotly.react(magEl, mag.figure.data || [], mag.figure.layout || {{}},
        {{responsive: true, displaylogo: false}});
    }} else {{
      magEl.hidden = true;
      magEmpty.hidden = false;
    }}

    if (dist.present && dist.figure && hasPlotly) {{
      distEl.hidden = false;
      distEmpty.hidden = true;
      window.Plotly.react(distEl, dist.figure.data || [], dist.figure.layout || {{}},
        {{responsive: true, displaylogo: false}});
    }} else {{
      distEl.hidden = true;
      distEmpty.hidden = false;
      distEmpty.textContent = dist.note || 'Select tokens to see their distribution.';
    }}

    document.querySelectorAll('.fig-export button').forEach((btn) => {{
      const plotEl = document.getElementById(btn.dataset.exportGraph || '');
      btn.disabled = !hasPlotly || !plotEl || plotEl.hidden;
    }});
    lastRevision = payload.revision;
  }}

  function handle(event) {{
    if (event.origin !== ORIGIN) return;
    let text;
    try {{ text = JSON.parse(event.data); }} catch (err) {{ return; }}
    if (!text || typeof text !== 'object') return;
    if (text.schema !== SCHEMA) return;
    if (typeof text.revision !== 'number' || text.revision <= lastRevision) return;
    render(text);
  }}

  window.addEventListener('message', handle);
  if (window.opener) {{
    try {{
      window.opener.postMessage(JSON.stringify({{schema: REQUEST_SCHEMA}}), ORIGIN);
    }} catch (err) {{ /* ignore */ }}
  }}
  window.addEventListener('resize', () => {{
    if (window.Plotly && typeof window.Plotly.Plots !== 'undefined') {{
      window.Plotly.Plots.resize(document.getElementById('mag'));
      window.Plotly.Plots.resize(document.getElementById('dist'));
    }}
  }});

  document.querySelectorAll('.fig-export button').forEach((btn) => {{
    btn.addEventListener('click', () => {{
      const graphId = btn.dataset.exportGraph || '';
      const format = btn.dataset.exportFormat || 'png';
      const el = document.getElementById(graphId);
      if (!el || el.hidden || typeof window.Plotly === 'undefined') return;
      try {{
        window.Plotly.downloadImage(el, {{
          format: format,
          filename: 'circuit-tracer_' + graphId + '_' + format,
        }});
      }} catch (err) {{ /* ignore a failed single download */ }}
    }});
  }});
}})();
</script>
</body>
</html>
"""


# Same-origin diagnostics popup. The main workbench (the sole controller)
# opens/focuses the one named popup and publishes a versioned atomic payload
# over postMessage; the popup only validates origin + monotonic revision and
# renders. This block runs once on the main page: it installs window-level
# Same-origin diagnostics popup. This is the "Open diagnostics" button
# handler: every explicit click opens (or focuses) the one named popup at the
# same-origin /ct-diagnostics page, then publishes the latest versioned atomic
# payload over postMessage. Window-level helpers and the refresh loop are
# installed only once; blocking, closing, reopening, or resizing the popup never
# damages the main workbench session.
OPEN_DIAGNOSTICS_JS = """
() => {
  const name = 'circuit-tracer-diagnostics';
  const origin = window.location.origin;
  const url = origin + '/ct-diagnostics';
  const payloadUrl = origin + '/ct-diagnostics-payload';
  const REQUEST_SCHEMA = 'circuit-tracer.diagnostics-window.request';

  if (!window.__ctDiagnosticsState) {
    window.__ctDiagnosticsState = { win: null, installed: false };
  }
  const state = window.__ctDiagnosticsState;

  const push = (win) => {
    if (!win || win.closed) return;
    fetch(payloadUrl, {cache: 'no-store'})
      .then((res) => { if (!res.ok) throw new Error('payload'); return res.text(); })
      .then((text) => { if (win && !win.closed) { try { win.postMessage(text, origin); } catch (error) { /* ignore */ } } })
      .catch(() => { /* server not ready yet */ });
  };

  const openOrFocus = () => {
    let win = state.win && !state.win.closed ? state.win : null;
    try {
      if (!win) win = window.open('', name);
      if (win && !win.closed && !win.__ctDiagnosticsDoc) {
        win.__ctDiagnosticsDoc = true;
        win.location.href = url;
      }
      if (!win || win.closed) win = window.open(url, name);
      state.win = win || null;
      if (win && win.focus) win.focus();
      return win;
    } catch (error) {
      state.win = null;
      return null;
    }
  };

  const publish = () => {
    const win = openOrFocus();
    if (win && !win.closed) push(win);
    return win;
  };

  if (!state.installed) {
    state.installed = true;
    window.ctOpenDiagnostics = publish;
    window.ctPublishDiagnostics = (win) => push(win);
    window.addEventListener('message', (event) => {
      if (event.origin !== origin) return;
      let msg;
      try { msg = JSON.parse(event.data); } catch (error) { return; }
      if (msg && msg.schema === REQUEST_SCHEMA && event.source) push(event.source);
    });
    // Only push while the user-visible popup is open; never re-open it here.
    window.setInterval(() => {
      const win = state.win;
      if (win && !win.closed) push(win);
    }, 800);
  }
  publish();
  return [];
}
"""


# Browser-side image export for the already-rendered Plotly figures. No
# Kaleido and no external CDN: the local /ct-assets/plotly.min.js bundle is
# loaded on demand only if the hosting page does not already expose Plotly,
# and Plotly.downloadImage draws from the live graph divs on screen.
EXPORT_IMAGE_JS = """
(() => {
  const EXPORT_PLOT_HOSTS = {
    activation: ['ct-activation-plot', 'ct-comparison-a-plot', 'ct-comparison-b-plot', 'ct-delta-plot'],
    magnitudes: ['ct-magnitude-plot'],
    distribution: ['ct-distribution-plot'],
  };
  const EXPORT_PLOT_LABELS = {
    activation: ['activation', 'comparison-a', 'comparison-b', 'delta'],
    magnitudes: ['token-magnitudes'],
    distribution: ['selected-token-distribution'],
  };
  const IMAGE_FORMATS = ['png', 'svg'];
  const EXPORT_ASSET_PATH = '/ct-assets/plotly.min.js';

  const report = (message) => {
    const reason = document.getElementById('ct-export-reason');
    if (reason) reason.textContent = message;
  };

  const revealElement = (element) => {
    let node = element;
    while (node && node !== document.documentElement) {
      if (node.nodeName === 'DETAILS' && !node.open) {
        node.open = true;
      }
      node = node.parentElement;
    }
  };

  const isRendered = (element) => {
    const rect = element.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1;
  };

  const ensurePlotly = () => new Promise((resolve) => {
    if (typeof window.Plotly !== 'undefined') {
      resolve(true);
      return;
    }
    const script = document.createElement('script');
    script.src = EXPORT_ASSET_PATH;
    script.async = true;
    script.onload = () => resolve(typeof window.Plotly !== 'undefined');
    script.onerror = () => resolve(false);
    document.head.appendChild(script);
  });

  window.ctDownloadExportImage = async function (target, format) {
    if (IMAGE_FORMATS.indexOf(format) < 0) {
      report('Unsupported image format: ' + format);
      return 0;
    }
    const graph = await ensurePlotly();
    if (!graph) {
      report('Plotly could not be loaded from the local asset; image export is unavailable.');
      return 0;
    }
    const hosts = EXPORT_PLOT_HOSTS[target] || [];
    const labels = EXPORT_PLOT_LABELS[target] || [];
    const matches = [];
    hosts.forEach((hostId, index) => {
      const host = document.getElementById(hostId);
      const plot = host ? host.querySelector('.js-plotly-plot') : null;
      if (!plot) return;
      revealElement(plot);
      matches.push({ plot: plot, label: labels[index] || hostId });
    });
    if (!matches.length) {
      report('No chart for that target is currently rendered on screen.');
      return 0;
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
    const rendered = matches.filter((match) => isRendered(match.plot));
    if (!rendered.length) {
      report('The chart is present but has no rendered size yet; open the chart panel and retry.');
      return 0;
    }
    rendered.forEach((match, index) => {
      setTimeout(() => {
        try {
          window.Plotly.downloadImage(match.plot, {
            format: format,
            filename: 'circuit-tracer_' + match.label + '_' + format,
          });
        } catch (error) {
          report('Image export failed for ' + match.label + ': ' + error.message);
        }
      }, index * 250);
    });
    report('Started image download for ' + rendered.length + ' chart(s) (' + format + ').');
    return rendered.length;
  };
})();
"""

# The modal export dialog. It asks the researcher to choose a scope and a
# format explicitly; options that are not valid for the current research state
# are disabled with an inline reason. It is a native, accessible <dialog>:
# labelled fieldsets, focus handled by the browser, Escape closes, and the
# status region is aria-live. Nothing is generated until "Download" is pressed.
EXPORT_DIALOG_HTML = """
<dialog id="ct-export-dialog" aria-labelledby="ct-export-title" aria-describedby="ct-export-intro">
  <div class="ct-export-head">
    <h2 id="ct-export-title">Export research artifacts</h2>
    <p id="ct-export-intro">
      Choose a scope and a format explicitly, then press Download.
      Raw CSV/NumPy files are generated on the server only when you press
      Download and are delivered as temporary browser downloads; nothing is
      written to disk automatically. PNG/SVG images are drawn in your browser
      from the charts already on screen (local Plotly, no external service).
    </p>
  </div>
  <div class="ct-export-body">
    <fieldset id="ct-export-scopes">
      <legend>Scope</legend>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-scope" value="selection" id="ct-scope-selection">
        <label for="ct-scope-selection">Current selection
          <small>Selected token positions at the current location.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-scope" value="location" id="ct-scope-location">
        <label for="ct-scope-location">Current location
          <small>Every captured token at the current location.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-scope" value="comparison" id="ct-scope-comparison">
        <label for="ct-scope-comparison">Comparison
          <small>Pinned A and current B, plus the signed B - A delta when the shapes match.</small>
        </label>
      </div>
    </fieldset>
    <fieldset id="ct-export-formats">
      <legend>Format</legend>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-format" value="csv" id="ct-format-csv">
        <label for="ct-format-csv">CSV
          <small>Deterministic long-form rows of raw values.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-format" value="npz" id="ct-format-npz">
        <label for="ct-format-npz">NumPy .npz
          <small>Source matrices plus a metadata.json entry.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-format" value="png" id="ct-format-png">
        <label for="ct-format-png">PNG
          <small>Browser image of the figure you choose below.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-format" value="svg" id="ct-format-svg">
        <label for="ct-format-svg">SVG
          <small>Vector browser image of the figure you choose below.</small>
        </label>
      </div>
    </fieldset>
    <fieldset id="ct-export-figures">
      <legend>Figure (for PNG/SVG)</legend>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-figure" value="activation" id="ct-figure-activation">
        <label for="ct-figure-activation">Active activation view
          <small>The activation chart(s) currently on the canvas.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-figure" value="magnitudes" id="ct-figure-magnitudes">
        <label for="ct-figure-magnitudes">Token magnitudes
          <small>Drawn when tokens are selected.</small>
        </label>
      </div>
      <div class="ct-export-option">
        <input type="radio" name="ct-export-figure" value="distribution" id="ct-figure-distribution">
        <label for="ct-figure-distribution">Selected-token distribution
          <small>Drawn when tokens are selected.</small>
        </label>
      </div>
    </fieldset>
    <label class="ct-export-delta" for="ct-export-delta">
      <input type="checkbox" id="ct-export-delta" name="ct-export-delta">
      Include the exact signed delta (B - A) in comparison exports
    </label>
  </div>
  <div id="ct-export-reason" class="ct-export-reason" role="status" aria-live="polite"></div>
  <div class="ct-export-actions">
    <button type="button" id="ct-export-cancel">Cancel</button>
    <button type="button" id="ct-export-download">Download</button>
  </div>
</dialog>
"""


# The main workbench is the sole controller of the export dialog. Every open
# click shows the modal and refreshes which scope/figure options are valid for
# the current research state from the same-origin /ct-export-options endpoint.
# Raw downloads POST to /ct-export and are saved from the returned blob;
# image downloads run through the browser-side Plotly exporter above.
OPEN_EXPORT_DIALOG_JS = """
() => {
  const ORIGIN = window.location.origin;
  const dialog = document.getElementById('ct-export-dialog');
  if (!dialog || typeof dialog.showModal !== 'function') return [];
  const state = (window.__ctExportDialogState = window.__ctExportDialogState || { bound: false });

  const report = (text) => {
    const reason = document.getElementById('ct-export-reason');
    if (reason) reason.textContent = text || '';
  };
  const checkedValue = (name) => {
    const input = document.querySelector('input[name="' + name + '"]:checked');
    return input ? input.value : '';
  };
  const radioByValue = (name, value) => {
    return document.querySelector('input[name="' + name + '"][value="' + value + '"]');
  };
  const isImageFormat = (format) => format === 'png' || format === 'svg';

  const formatIsActive = () => {
    const format = checkedValue('ct-export-format') || 'csv';
    return { format: format, image: isImageFormat(format) };
  };

  const saveBlob = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    setTimeout(() => { URL.revokeObjectURL(url); anchor.remove(); }, 1000);
  };

  const downloadRaw = async () => {
    const scope = checkedValue('ct-export-scope');
    const format = checkedValue('ct-export-format') || 'csv';
    const deltaBox = document.getElementById('ct-export-delta');
    const body = {
      scope: scope,
      format: format,
      include_delta: !!(deltaBox && deltaBox.checked),
    };
    report('Preparing ' + format.toUpperCase() + ' export for ' + scope + '…');
    try {
      const response = await fetch(ORIGIN + '/ct-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        let reason = 'Export failed (HTTP ' + response.status + ').';
        try {
          const detail = await response.json();
          if (detail && detail.reason) reason = detail.reason;
          else if (detail && detail.detail) reason = detail.detail;
        } catch (error) { /* keep default */ }
        throw new Error(reason);
      }
      const header = response.headers.get('Content-Disposition') || '';
      let filename = 'circuit-tracer_export_' + scope + '.' + format;
      const match = header.match(/filename\\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?/i);
      if (match) filename = (match[1] || match[2]).replace(/^utf-8''/i, '');
      const blob = await response.blob();
      saveBlob(blob, filename);
      report('Downloaded ' + filename + '. Raw values are unclipped; padding cells are never exported.');
    } catch (error) {
      report(String(error && error.message ? error.message : error));
    }
  };

  const downloadImage = async () => {
    const target = checkedValue('ct-export-figure') || 'activation';
    const format = checkedValue('ct-export-format') || 'png';
    await window.ctDownloadExportImage(target, format);
  };

  const download = () => {
    const active = formatIsActive();
    if (active.image) {
      downloadImage();
    } else {
      downloadRaw();
    }
  };

  const syncControls = () => {
    const options = state.options || null;
    const scopeField = document.getElementById('ct-export-scopes');
    const figureField = document.getElementById('ct-export-figures');
    const deltaBox = document.getElementById('ct-export-delta');
    const downloadBtn = document.getElementById('ct-export-download');
    if (!options) {
      if (downloadBtn) downloadBtn.disabled = true;
      return;
    }
    const active = formatIsActive();
    if (scopeField) scopeField.disabled = active.image;
    if (figureField) figureField.disabled = !active.image;

    const scopeValue = checkedValue('ct-export-scope');
    const scopeOk = (options.scopes || []).some((item) => item.value === scopeValue && item.available);
    const figureValue = checkedValue('ct-export-figure');
    const figureOk = (options.figures || []).some((item) => item.value === figureValue && item.available);

    const inComparison = scopeValue === 'comparison' && !active.image;
    if (deltaBox) {
      const deltaOk = !!(options.delta && options.delta.available);
      deltaBox.disabled = !inComparison || !deltaOk;
      deltaBox.checked = inComparison && deltaOk;
    }
    const anythingRaw = (options.scopes || []).some((item) => item.available);
    if (downloadBtn) {
      downloadBtn.disabled = !options.available || (active.image ? !figureOk : !scopeOk || !anythingRaw);
    }
    const scopeHint = (options.scopes || []).find((item) => item.value === scopeValue);
    if (scopeHint && !scopeHint.available && !active.image && scopeHint.reason) {
      report(scopeHint.reason);
    } else if (options.delta && options.delta.available === false && inComparison && options.delta.reason) {
      report(options.delta.reason);
    } else {
      report(options.message || 'Ready to export. Choose a scope and format, then press Download.');
    }
  };

  const chooseDefaults = () => {
    const options = state.options || null;
    const scope = checkedValue('ct-export-scope');
    if (!scope && options) {
      const preferred = options.default_scope || 'location';
      const firstAvailable = (options.scopes || []).find((item) => item.value === preferred && item.available)
        || (options.scopes || []).find((item) => item.available);
      const input = firstAvailable && radioByValue('ct-export-scope', firstAvailable.value);
      if (input) input.checked = true;
    }
    if (!checkedValue('ct-export-figure')) {
      const input = radioByValue('ct-export-figure', 'activation');
      if (input) input.checked = true;
    }
  };

  const refreshOptions = () => {
    report('Checking what is available to export…');
    fetch(ORIGIN + '/ct-export-options', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) throw new Error('Export options endpoint unavailable.');
        return response.json();
      })
      .then((options) => {
        state.options = options;
        (options.scopes || []).forEach((item) => {
          const input = radioByValue('ct-export-scope', item.value);
          if (input) {
            input.disabled = !item.available;
            const small = input.closest('.ct-export-option').querySelector('small');
            if (small && !item.available && item.reason) small.textContent = item.reason;
          }
        });
        (options.figures || []).forEach((item) => {
          const input = radioByValue('ct-export-figure', item.value);
          if (input) {
            input.disabled = !item.available;
            const small = input.closest('.ct-export-option').querySelector('small');
            if (small && !item.available && item.reason) small.textContent = item.reason;
          }
        });
        chooseDefaults();
        syncControls();
      })
      .catch((error) => report(String(error && error.message ? error.message : error)));
  };

  if (!state.bound) {
    state.bound = true;
    document.getElementById('ct-export-cancel').addEventListener('click', () => dialog.close());
    document.getElementById('ct-export-download').addEventListener('click', download);
    ['ct-export-format', 'ct-export-scope'].forEach((name) => {
      document.querySelectorAll('input[name="' + name + '"]').forEach((input) => {
        input.addEventListener('change', syncControls);
      });
    });
    dialog.addEventListener('close', () => report(''));
  }

  chooseDefaults();
  syncControls();
  refreshOptions();
  if (!dialog.open) dialog.showModal();
  return [];
}
"""


def _successful_inspection_outputs(
    session: InspectionSession,
    location_key: str = DEFAULT_LOCATION_KEY,
    token_position=None,
    clipped: bool = False,
    normalization: str = "location",
    mode: str = "square",
    comparison: Optional[ComparisonState] = None,
    measurement: Optional[MeasurementPin] = None,
    measurement_message: str = "",
):
    normalization = {"selected_point": "selection", "whole_model": "capture"}.get(
        normalization, normalization
    )
    if normalization not in {"selection", "location", "capture"}:
        normalization = "location"
    analysis = session.analysis
    spec = location_spec(location_key)
    token_positions = _selected_positions(token_position, analysis.token_count)
    location_dropdown = gr.update(choices=location_choices(), value=spec.key)
    token_dropdown = gr.update(
        choices=_token_choices(analysis),
        value=[str(position) for position in token_positions],
    )
    values = analysis.capture.locations[spec.key]
    labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    measurement_identity = (
        (measurement.token_position, measurement.dimension) if measurement else None
    )
    active_comparison = (
        comparison if comparison and comparison.location_key != spec.key else None
    )
    if active_comparison is None:
        bounds = capture_heatmap_bounds(
            session, spec.key, token_positions, clipped, normalization
        )
        activation = _render_activation_view(
            values,
            labels,
            token_positions,
            mode,
            clipped,
            bounds,
            spec.label,
            measurement_identity,
        )
        comparison_status = (
            f"A pinned at {location_spec(comparison.location_key).label}. "
            "Select another location to compare it as B."
            if comparison
            else "No comparison pinned."
        )
        return (
            location_dropdown,
            token_dropdown,
            _with_measurement_explanation(
                location_explanation(spec), measurement_message
            ),
            location_stats(spec, values, token_positions),
            _heatmap_range_message(bounds, clipped),
            (
                render_token_magnitudes(values, labels, token_positions)
                if token_positions else None
            ),
            gr.update(value=activation, visible=True),
            _selection_distribution(values, labels, token_positions),
            render_model_diagram(
                session.config,
                selected_key=spec.key,
                pinned_key=comparison.location_key if comparison else None,
                capture_available=True,
            ),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            comparison_status,
        )

    spec_a = location_spec(active_comparison.location_key)
    values_a = active_comparison.values
    raw_bounds = comparison_heatmap_bounds(
        session,
        active_comparison,
        spec.key,
        token_positions,
        clipped,
        normalization,
    )
    figure_a = _render_activation_view(
        values_a,
        labels,
        token_positions,
        mode,
        clipped,
        raw_bounds,
        f"A | {spec_a.label} | raw",
        measurement_identity,
    )
    figure_b = _render_activation_view(
        values,
        labels,
        token_positions,
        mode,
        clipped,
        raw_bounds,
        f"B | {spec.label} | raw",
        measurement_identity,
    )
    delta, disabled_reason = _comparison_delta(values_a, values, spec_a.label, spec.label)
    delta_figure = None
    range_message = _heatmap_range_message(raw_bounds, clipped, "Pooled raw A/B range")
    if delta is not None:
        delta_source = (
            delta[token_positions]
            if normalization == "selection" and token_positions
            else delta
        )
        delta_bounds = display_bounds(delta_source, clipped)
        delta_figure = _render_activation_view(
            delta,
            labels,
            token_positions,
            mode,
            clipped,
            delta_bounds,
            f"Signed delta | B - A | {spec.label} - {spec_a.label}",
            measurement_identity,
        )
        range_message += "\n\n" + _heatmap_range_message(
            delta_bounds, clipped, "Delta B - A range"
        )
        comparison_status = (
            f"**A:** {spec_a.label} `{values_a.shape[0]} × {values_a.shape[1]}`  \n"
            f"**B:** {spec.label} `{values.shape[0]} × {values.shape[1]}`  \n"
            "**Delta:** available; exact signed `B - A`."
        )
    else:
        comparison_status = disabled_reason
    return (
        location_dropdown,
        token_dropdown,
        _with_measurement_explanation(
            location_explanation(spec), measurement_message
        ),
        location_stats(spec, values, token_positions),
        range_message,
        render_token_magnitudes(values, labels, token_positions) if token_positions else None,
        gr.update(value=None, visible=False),
        _selection_distribution(values, labels, token_positions),
        render_model_diagram(
            session.config,
            selected_key=spec.key,
            pinned_key=active_comparison.location_key,
            comparison_key=spec.key,
            capture_available=True,
        ),
        gr.update(value=figure_a, visible=True),
        gr.update(value=figure_b, visible=True),
        gr.update(value=delta_figure, visible=delta_figure is not None),
        comparison_status,
    )


def _render_activation_view(
    values: np.ndarray,
    labels: list[str],
    token_positions: list[int],
    mode: str,
    clipped: bool,
    bounds: tuple[float, float],
    location_label: str,
    measurement_pin,
):
    if token_positions:
        return render_activation_detail(
            values,
            labels,
            token_positions,
            mode=mode,
            clipped=clipped,
            bounds=bounds,
            measurement_pin=measurement_pin,
            location_label=location_label,
        )
    return render_activation_overview(
        values,
        labels,
        clipped=clipped,
        bounds=bounds,
        measurement_pin=measurement_pin,
        location_label=location_label,
    )


def _selection_distribution(values, labels, token_positions):
    if not token_positions:
        return None
    return render_token_distribution(
        values[token_positions].reshape(-1),
        token_positions[0],
        labels[token_positions[0]],
        title=(
            f"Dimension distribution for token {token_positions[0]}: "
            f"{labels[token_positions[0]]}"
            if len(token_positions) == 1
            else f"Dimension distribution for {len(token_positions)} selected tokens"
        ),
    )


def _heatmap_range_message(
    bounds: tuple[float, float],
    clipped: bool,
    label: str = "Visible heatmap range",
) -> str:
    message = f"**{label}:** `{{:.4f}}` to `{{:.4f}}`".format(*bounds)
    if clipped:
        message += (
            "\n\n**Warning:** display colors are clipped to the 1st-99th "
            "percentile. Captured and hover values remain raw."
        )
    return message


def _comparison_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> tuple[Optional[np.ndarray], str]:
    if values_a.shape != values_b.shape:
        return None, (
            f"**Delta disabled:** A {label_a} has shape "
            f"`{values_a.shape[0]} × {values_a.shape[1]}`; B {label_b} has shape "
            f"`{values_b.shape[0]} × {values_b.shape[1]}`. Signed subtraction "
            "requires exactly equal shapes; no broadcasting, truncation, projection, "
            "or padding is applied. Raw A and B remain available side by side."
        )
    return signed_comparison(values_a, values_b), ""


def _report_progress(progress, value: float, description: str) -> None:
    if progress is not None:
        progress(value, desc=description)


class ModelManager:
    """Own the single server-side checkpoint used by the application."""

    def __init__(
        self,
        checkpoint_loader: Callable[[str], LoadedCheckpoint] = load_checkpoint,
        device_detector: Callable[[], ComputeDevice] = detect_compute_device,
        session_clearer: Callable[[], None] = tf.keras.backend.clear_session,
        collector: Callable[[], int] = gc.collect,
        prompt_analyzer: Callable[[str, LoadedCheckpoint], PromptAnalysis] = analyze_prompt,
    ) -> None:
        self._checkpoint_loader = checkpoint_loader
        self._device_detector = device_detector
        self._session_clearer = session_clearer
        self._collector = collector
        self._prompt_analyzer = prompt_analyzer
        self._lock = threading.RLock()
        self._compute_lock = threading.Lock()
        self._state: Optional[LoadedState] = None
        self._session: Optional[InspectionSession] = None
        self._comparison: Optional[ComparisonState] = None
        self._measurement_pin: Optional[MeasurementPin] = None
        self._current_location: Optional[str] = None
        self._next_operation_id = 0
        self._generation = 0
        self._model_generation = 0
        self._view_location_key: Optional[str] = None
        self._view_token_positions: list = []
        self._payload_revision = 0
        self._last_payload_content: Optional[str] = None
        self._last_payload_signature: Optional[str] = None
        self._last_payload_text: Optional[str] = None

    def _start_operation(self) -> int:
        with self._lock:
            self._next_operation_id += 1
            return self._next_operation_id

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def current_location(self) -> Optional[str]:
        with self._lock:
            return self._current_location

    @property
    def diagnostics_revision(self) -> int:
        with self._lock:
            return self._payload_revision

    @property
    def loaded_state(self) -> Optional[LoadedState]:
        with self._lock:
            return self._state

    @property
    def inspection_session(self) -> Optional[InspectionSession]:
        """Return the last capture; None when nothing has been analyzed."""
        with self._lock:
            return self._session

    @property
    def comparison_state(self) -> Optional[ComparisonState]:
        with self._lock:
            return self._comparison

    @property
    def measurement_pin(self) -> Optional[MeasurementPin]:
        with self._lock:
            return self._measurement_pin

    def state_snapshot(self):
        """Return related active state under one lock for coherent diagnostics."""
        with self._lock:
            return (
                self._state,
                self._session,
                self._comparison,
                self._generation,
                self._model_generation,
            )

    def set_view_state(self, location_key: Optional[str], token_positions) -> None:
        """Record the current location/selection so a later popup payload matches
        the visible main workbench without mixing revisions."""
        positions = sorted(
            {int(position) for position in (token_positions or [])}
        )
        with self._lock:
            self._view_location_key = location_key
            self._view_token_positions = positions

    def view_snapshot(self) -> tuple:
        with self._lock:
            return (
                self._view_location_key,
                list(self._view_token_positions),
                self._payload_revision,
            )

    def payload_text(
        self,
        location_key: Optional[str] = None,
        token_value=None,
    ) -> str:
        """Return a versioned payload, bumping the revision only on a change."""
        return diagnostics_window_payload_json(self, location_key, token_value)

    def payload_signature(self) -> str:
        """A cheap stable identity for the payload (no figure serialization)."""
        state, session, comparison, generation, model_generation = self.state_snapshot()
        location, positions, _ = self.view_snapshot()
        identity = (
            generation,
            model_generation,
            state.checkpoint_path if state else None,
            session.analysis.token_count if session else None,
            location,
            tuple(positions),
        )
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def current_payload(self) -> str:
        """Return the payload for the current view, cached by signature."""
        with self._lock:
            signature = self.payload_signature()
            if (
                signature == self._last_payload_signature
                and self._last_payload_text is not None
            ):
                return self._last_payload_text
            text = diagnostics_window_payload_json(self)
            self._last_payload_signature = signature
            self._last_payload_text = text
            return text

    def payload_status(self) -> dict:
        """Cheap revision/signature status used to decide when to re-publish."""
        with self._lock:
            signature = self.payload_signature()
            if (
                signature != self._last_payload_signature
                or self._last_payload_text is None
            ):
                text = diagnostics_window_payload_json(self)
                self._last_payload_signature = signature
                self._last_payload_text = text
            return {
                "revision": self._payload_revision,
                "signature": signature,
            }

    def is_current_generation(self, generation: Optional[int]) -> bool:
        with self._lock:
            return generation is not None and generation == self._generation

    def is_current_session(self, session: InspectionSession) -> bool:
        with self._lock:
            return self._session is session

    def store_inspection_session(self, session: InspectionSession) -> None:
        operation_id = self._start_operation()
        with self._lock:
            self._session = session
            self._comparison = None
            self._measurement_pin = None
            self._current_location = DEFAULT_LOCATION_KEY
            self._view_location_key = DEFAULT_LOCATION_KEY
            self._view_token_positions = []
            self._generation = operation_id

    def pin_measurement(
        self,
        location_key: str,
        token_position: int,
        dimension: int,
        expected_session: Optional[InspectionSession] = None,
    ) -> bool:
        """Pin an exact detail cell from the active in-memory capture."""
        with self._lock:
            if self._session is None:
                return False
            if expected_session is not None and self._session is not expected_session:
                return False
            try:
                values = self._session.analysis.capture.locations[location_key]
            except KeyError:
                return False
            if not (
                0 <= token_position < values.shape[0]
                and 0 <= dimension < values.shape[1]
            ):
                return False
            self._measurement_pin = MeasurementPin(token_position, dimension)
            return True

    def resolve_measurement_pin(
        self,
        location_key: str,
        expected_session: Optional[InspectionSession] = None,
    ) -> tuple[Optional[MeasurementPin], str]:
        """Resolve a pin at a location, clearing it instead of guessing compatibility."""
        with self._lock:
            pin = self._measurement_pin
            session = self._session
            if expected_session is not None and session is not expected_session:
                return None, ""
            if pin is None or session is None:
                return None, ""
            try:
                values = session.analysis.capture.locations[location_key]
                spec = location_spec(location_key)
            except Exception:
                return pin, ""
            if pin.token_position >= values.shape[0] or pin.dimension >= values.shape[1]:
                self._measurement_pin = None
                return (
                    None,
                    "**Pinned measurement cleared:** token position "
                    f"`{pin.token_position}`, dimension `{pin.dimension}` is not "
                    f"available at {spec.label} with shape "
                    f"`{values.shape[0]} \u00d7 {values.shape[1]}`.",
                )
            raw_value = np.format_float_positional(
                float(values[pin.token_position, pin.dimension]),
                unique=True,
                trim="-",
            )
            return (
                pin,
                "**Pinned measurement:** token position "
                f"`{pin.token_position}`, dimension `{pin.dimension}`, raw value "
                f"`{raw_value}` at {spec.label}.",
            )

    def pin_comparison(self, location_key: str) -> str:
        with self._lock:
            if self._session is None:
                return "Analyze a prompt before pinning comparison A."
            try:
                location_spec(location_key)
            except Exception as error:
                return str(error)
            values = self._session.analysis.capture.locations[location_key].copy()
            self._comparison = ComparisonState(location_key, values)
            self._current_location = location_key
            return f"Pinned A: {location_spec(location_key).label}. Select another location as B."

    def navigate_comparison(self, location_key: str, explicit: bool = False) -> bool:
        """Track B and clear comparison when A is deliberately selected again."""
        with self._lock:
            cleared = bool(
                self._comparison
                and location_key == self._comparison.location_key
                and (explicit or self._current_location != location_key)
            )
            if cleared:
                self._comparison = None
            self._current_location = location_key
            return cleared

    def clear_comparison(self) -> None:
        with self._lock:
            self._comparison = None

    def clear_session(self) -> None:
        """Drop the stored capture without unloading the current model."""
        operation_id = self._start_operation()
        with self._lock:
            self._session = None
            self._comparison = None
            self._measurement_pin = None
            self._current_location = None
            self._view_location_key = None
            self._view_token_positions = []
            self._generation = operation_id

    def _clear_unlocked(self) -> None:
        self._state = None
        self._session = None
        self._comparison = None
        self._measurement_pin = None
        self._current_location = None
        self._view_location_key = None
        self._view_token_positions = []
        self._model_generation += 1
        self._session_clearer()
        self._collector()

    def clear(self) -> None:
        operation_id = self._start_operation()
        with self._lock:
            self._clear_unlocked()
            self._generation = operation_id

    @contextmanager
    def use_loaded_state(self) -> Iterator[LoadedState]:
        """Hold the lifecycle lock while a future analysis uses the model."""
        with self._lock:
            if self._state is None:
                raise CheckpointError("Load a checkpoint before running analysis")
            yield self._state

    def load(self, directory, progress=None) -> LoadResult:
        operation_id = self._start_operation()
        with self._lock:
            active_state = self._state
        _report_progress(progress, 0.05, "Validating checkpoint")
        if not isinstance(directory, str) or not directory.strip():
            error = CheckpointError("Enter a checkpoint folder path first.")
            return _failure_result(
                str(error), active_state, _technical_error(error)
            )

        checkpoint_path = Path(directory.strip()).expanduser()
        candidate_state = None
        try:
            device = self._device_detector()
            _report_progress(progress, 0.35, "Loading model weights")
            with self._compute_lock, tf.device(device.tf_device):
                checkpoint = self._checkpoint_loader(str(checkpoint_path))
            candidate_state = LoadedState(
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            _report_progress(progress, 0.8, "Rendering model details")
            prepared_result = _success_result(checkpoint_path, candidate_state)

            with self._lock:
                if operation_id <= self._generation:
                    try:
                        self._collector()
                    except Exception:
                        LOGGER.exception("Could not collect a stale checkpoint candidate")
                    return _failure_result(
                        "Load was superseded by a newer session.",
                        self._state,
                        "StaleOperation: checkpoint candidate was not committed",
                        stale=True,
                    )
                self._state = candidate_state
                self._session = None
                self._comparison = None
                self._measurement_pin = None
                self._current_location = None
                self._view_location_key = None
                self._view_token_positions = []
                self._model_generation += 1
                self._generation = operation_id

            try:
                with self._compute_lock:
                    self._session_clearer()
                    self._collector()
            except Exception:
                LOGGER.exception("Loaded checkpoint, but old model cleanup failed")
            _report_progress(progress, 1.0, "Checkpoint ready")
            return LoadResult(
                **{
                    **prepared_result.__dict__,
                    "generation": operation_id,
                }
            )
        except CheckpointError as error:
            return _failure_result(
                f"Checkpoint could not be loaded: {error}",
                self.loaded_state,
                _technical_error(error),
            )
        except (OSError, ValueError, RuntimeError) as error:
            LOGGER.exception("Checkpoint loading failed")
            return _failure_result(
                f"Checkpoint could not be loaded: {type(error).__name__}. "
                "Check the folder and runtime configuration.",
                self.loaded_state,
                _technical_error(error),
            )
        except Exception as error:
            LOGGER.exception("Unexpected checkpoint loading failure")
            return _failure_result(
                "Checkpoint could not be loaded because of an unexpected runtime error.",
                self.loaded_state,
                _technical_error(error),
            )

    def prepare_analysis(self, prompt, progress=None) -> AnalysisAttempt:
        operation_id = self._start_operation()
        with self._lock:
            state = self._state
            model_generation = self._model_generation
        if state is None:
            error = CheckpointError("Load a checkpoint before running analysis")
            return AnalysisAttempt(None, str(error), _technical_error(error))

        try:
            _report_progress(progress, 0.05, "Validating prompt")
            with self._compute_lock:
                with self._lock:
                    if (
                        self._state is not state
                        or self._model_generation != model_generation
                        or operation_id <= self._generation
                    ):
                        return AnalysisAttempt(
                            None,
                            "Analysis was superseded by a newer session.",
                            "StaleOperation: analysis did not use the active model",
                            stale=True,
                        )
                _report_progress(progress, 0.25, "Tracing model internals")
                with tf.device(state.device.tf_device):
                    analysis = self._prompt_analyzer(prompt or "", state.checkpoint)
            return AnalysisAttempt(
                AnalysisCandidate(
                    operation_id=operation_id,
                    model_generation=model_generation,
                    session=InspectionSession(
                        analysis=analysis,
                        config=state.checkpoint.config,
                    ),
                ),
                "",
            )
        except (AnalysisError, CheckpointError) as error:
            return AnalysisAttempt(None, str(error), _technical_error(error))
        except Exception as error:
            LOGGER.exception("Prompt analysis failed")
            return AnalysisAttempt(
                None,
                "Prompt analysis failed because of an unexpected runtime error.",
                _technical_error(error),
            )

    def commit_analysis(self, candidate: AnalysisCandidate) -> bool:
        with self._lock:
            if (
                candidate.model_generation != self._model_generation
                or candidate.operation_id <= self._generation
            ):
                return False
            self._session = candidate.session
            self._comparison = None
            self._measurement_pin = None
            self._current_location = DEFAULT_LOCATION_KEY
            self._view_location_key = DEFAULT_LOCATION_KEY
            self._view_token_positions = []
            self._generation = candidate.operation_id
            return True


def load_model_callback(directory, manager: ModelManager, progress=None):
    result = manager.load(directory, progress)
    return (
        result.status,
        result.metadata,
        result.device,
        result.diagram,
        result.summary,
    )


def _skip_outputs(count: int):
    return tuple(gr.skip() for _ in range(count))


def load_and_reset_callback(directory, manager: ModelManager, progress=None):
    """Load a checkpoint and atomically clear capture-dependent UI state."""
    result = manager.load(directory, progress)
    if result.stale:
        return _skip_outputs(29)
    if not result.success:
        return (result.status, result.technical_details, *_skip_outputs(27))
    if not manager.is_current_generation(result.generation):
        return _skip_outputs(29)

    diagnostics = json.loads(
        diagnostics_json(manager, "ready", "Checkpoint loaded; capture reset.")
    )
    if not manager.is_current_generation(result.generation):
        return _skip_outputs(29)
    return (
        result.status,
        "",
        result.metadata,
        result.device,
        result.diagram,
        result.summary,
        "Model loaded. Enter a prompt to analyze.",
        "",
        "",
        "",
        [],
        [],
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(
            choices=EMPTY_SELECTION_SCOPE_CHOICES,
            value="location",
            interactive=True,
        ),
        gr.update(value="square"),
        False,
        INSPECT_AWAITING,
        "",
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        "No comparison pinned.",
        diagnostics,
        "",
    )


def analyze_prompt_callback(prompt, manager: ModelManager):
    attempt = manager.prepare_analysis(prompt)
    if attempt.candidate is None:
        previous = manager.inspection_session
        result = (
            _analysis_success_result(previous.analysis)
            if previous is not None
            else _analysis_failure_result(attempt.status)
        )
        return (
            attempt.status,
            result.token_count,
            result.unknown_warning,
            result.token_rows,
            result.next_token_rows,
        )

    result = _analysis_success_result(attempt.candidate.session.analysis)
    if not manager.commit_analysis(attempt.candidate):
        return _skip_outputs(5)
    return (
        result.status,
        result.token_count,
        result.unknown_warning,
        result.token_rows,
        result.next_token_rows,
    )


def analyze_and_inspect_callback(prompt, manager: ModelManager, progress=None):
    """Render a candidate capture, then atomically replace the active session."""
    attempt = manager.prepare_analysis(prompt, progress)
    if attempt.stale:
        return _skip_outputs(24)
    if attempt.candidate is None:
        return (attempt.status, attempt.technical_details, *_skip_outputs(22))

    candidate = attempt.candidate
    try:
        _report_progress(progress, 0.72, "Rendering capture")
        result = _analysis_success_result(candidate.session.analysis)
        inspection = _successful_inspection_outputs(candidate.session)
    except Exception as error:
        LOGGER.exception("Prompt result rendering failed")
        return (
            "Prompt analysis could not be displayed; previous capture remains active.",
            _technical_error(error),
            *_skip_outputs(22),
        )

    if not manager.commit_analysis(candidate):
        return _skip_outputs(24)
    _report_progress(progress, 1.0, "Analysis ready")
    diagnostics = json.loads(
        diagnostics_json(manager, "ready", "Prompt analysis committed.")
    )
    if not manager.is_current_session(candidate.session):
        return _skip_outputs(24)
    return (
        result.status,
        "",
        result.token_count,
        result.unknown_warning,
        result.token_rows,
        result.next_token_rows,
        inspection[0],
        inspection[1],
        gr.update(
            choices=EMPTY_SELECTION_SCOPE_CHOICES,
            value="location",
            interactive=True,
        ),
        gr.update(value="square"),
        False,
        *inspection[2:],
        diagnostics,
        "",
    )


def select_location_callback(
    location_key,
    token_value,
    clipped,
    normalization,
    manager: ModelManager,
    mode: str = "square",
):
    """Re-render the inspect panel from stored data; never reruns the model."""
    resolved_location = location_key or DEFAULT_LOCATION_KEY
    comparison_cleared = manager.navigate_comparison(resolved_location)
    _, session, comparison, generation, _ = manager.state_snapshot()
    if session is None:
        manager.set_view_state(None, [])
        outputs = (
            gr.update(value=None),
            INSPECT_AWAITING,
            "",
            "",
            None,
            None,
            None,
            render_model_diagram(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "No comparison pinned.",
        )
        return outputs if manager.is_current_generation(generation) else _skip_outputs(12)

    analysis = session.analysis
    positions = _selected_positions(token_value, analysis.token_count)
    measurement, measurement_message = manager.resolve_measurement_pin(
        resolved_location, expected_session=session
    )
    manager.set_view_state(resolved_location, positions)
    outputs = list(_successful_inspection_outputs(
        session,
        resolved_location,
        positions,
        bool(clipped),
        normalization or "location",
        mode or "square",
        comparison,
        measurement,
        measurement_message,
    )[1:])
    if comparison_cleared:
        outputs[-1] = "Comparison ended: A was selected again."
    return tuple(outputs) if manager.is_current_session(session) else _skip_outputs(12)


def select_bridged_location_callback(
    location_key,
    token_value,
    clipped,
    normalization,
    manager: ModelManager,
    mode: str = "square",
):
    """Resolve a diagram or navigator click through the structured JS bridge."""
    _, session, _, _, _ = manager.state_snapshot()
    if session is None:
        return _skip_outputs(13)
    try:
        location_spec(location_key)
    except Exception:
        return _skip_outputs(13)
    comparison_cleared = manager.navigate_comparison(location_key, explicit=True)
    _, session, comparison, _, _ = manager.state_snapshot()
    measurement, measurement_message = manager.resolve_measurement_pin(
        location_key, expected_session=session
    )
    positions = _selected_positions(token_value, session.analysis.token_count)
    manager.set_view_state(location_key, positions)
    outputs = list(_successful_inspection_outputs(
        session,
        location_key,
        positions,
        bool(clipped),
        normalization or "location",
        mode or "square",
        comparison,
        measurement,
        measurement_message,
    ))
    if comparison_cleared:
        outputs[-1] = "Comparison ended: A was selected again."
    return tuple(outputs) if manager.is_current_session(session) else _skip_outputs(13)


def select_clicked_token_callback(
    clicked_position,
    location_key,
    token_value,
    clipped,
    normalization,
    manager: ModelManager,
    mode: str = "square",
):
    """Apply a structured Plotly click using only the stored capture."""
    _, session, comparison, generation, _ = manager.state_snapshot()
    if session is None:
        outputs = (
            gr.update(value=None),
            INSPECT_AWAITING,
            "",
            "",
            None,
            None,
            None,
            render_model_diagram(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "No comparison pinned.",
        )
        return outputs if manager.is_current_generation(generation) else _skip_outputs(12)
    click = _parse_activation_click(clicked_position)
    location = location_key or DEFAULT_LOCATION_KEY
    positions = _selected_positions(token_value, session.analysis.token_count)
    if click is None:
        try:
            positions = [int(clicked_position)]
        except (TypeError, ValueError):
            pass
    elif click["view"] == "overview":
        positions = toggle_token_position(
            positions, click["token_position"], session.analysis.token_count
        )
    elif click["view"] == "detail":
        manager.pin_measurement(
            location,
            click["token_position"],
            click["dimension"],
            expected_session=session,
        )
    elif click["view"] == "magnitude":
        positions = [click["token_position"]]
    manager.set_view_state(location, positions)
    measurement, measurement_message = manager.resolve_measurement_pin(
        location, expected_session=session
    )
    outputs = _successful_inspection_outputs(
        session,
        location,
        positions,
        bool(clipped),
        normalization or "location",
        mode or "square",
        comparison,
        measurement,
        measurement_message,
    )
    return outputs[1:] if manager.is_current_session(session) else _skip_outputs(12)


def _parse_activation_click(value) -> Optional[dict]:
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "circuit-tracer.click.v1":
        return None
    view = payload.get("view")
    if view not in {"overview", "detail", "magnitude"}:
        return None
    try:
        token_position = int(payload["token_position"])
        dimension = int(payload.get("dimension", -1))
    except (KeyError, TypeError, ValueError):
        return None
    if token_position < 0 or (view == "detail" and dimension < 0):
        return None
    return {
        "view": view,
        "token_position": token_position,
        "dimension": dimension,
    }


def pin_comparison_callback(location_key, token_value, manager: ModelManager):
    message = manager.pin_comparison(location_key or DEFAULT_LOCATION_KEY)
    generation = manager.generation
    outputs = (message, diagnostics_json(manager, "inspect", message))
    return outputs if manager.is_current_generation(generation) else _skip_outputs(2)


def clear_comparison_callback(manager: ModelManager):
    manager.clear_comparison()
    generation = manager.generation
    outputs = (
        "Comparison A cleared.",
        diagnostics_json(manager, "inspect", "Comparison A cleared."),
    )
    return outputs if manager.is_current_generation(generation) else _skip_outputs(2)


def selection_scope_update(token_value):
    """Selection scaling is unavailable until the user chooses a token."""
    if token_value:
        return gr.update(choices=SCALE_SCOPE_CHOICES, interactive=True)
    return gr.update(
        choices=EMPTY_SELECTION_SCOPE_CHOICES,
        value="location",
        interactive=True,
    )


def activation_click_callback(
    clicked_position,
    location_key,
    token_value,
    clipped,
    normalization,
    manager: ModelManager,
    mode: str = "square",
):
    """Update activation views and scale choices as one chart-click response."""
    outputs = select_clicked_token_callback(
        clicked_position,
        location_key,
        token_value,
        clipped,
        normalization,
        manager,
        mode,
    )
    updated_tokens = outputs[0].get("value") if isinstance(outputs[0], dict) else token_value
    return (*outputs, selection_scope_update(updated_tokens))


def launch_kwargs() -> dict:
    return {
        "server_name": LOCAL_SERVER_NAME,
        "share": SHARE_PUBLICLY,
        "theme": LIGHT_THEME,
        "run_history": False,
    }


def create_app(manager: Optional[ModelManager] = None) -> gr.Blocks:
    manager = manager or ModelManager()

    def load_ui(directory, progress=gr.Progress(track_tqdm=False)):
        outputs = load_and_reset_callback(directory, manager, progress)
        if not (
            isinstance(outputs[0], str)
            and outputs[0] == "Model loaded successfully."
        ):
            return (*outputs, *_skip_outputs(6))
        state = manager.loaded_state
        return (
            *outputs,
            render_location_navigator(state.checkpoint.config),
            gr.update(open=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
        )

    def analyze_ui(prompt, progress=gr.Progress(track_tqdm=False)):
        outputs = analyze_and_inspect_callback(prompt, manager, progress)
        if not (
            isinstance(outputs[0], str)
            and outputs[0].startswith("Analysis complete")
        ):
            return (*outputs, *_skip_outputs(4))
        session = manager.inspection_session
        comparison = manager.comparison_state
        return (
            *outputs,
            render_location_navigator(
                session.config,
                session.analysis.token_count,
                DEFAULT_LOCATION_KEY,
                comparison.location_key if comparison else None,
            ),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=prompt),
        )

    def pin_ui(location, token, clipped, normalization, mode):
        _, diagnostics = pin_comparison_callback(
            location, token, manager
        )
        session = manager.inspection_session
        if session is None:
            return (*_skip_outputs(12), diagnostics)
        measurement, measurement_message = manager.resolve_measurement_pin(
            location or DEFAULT_LOCATION_KEY, expected_session=session
        )
        inspection = _successful_inspection_outputs(
            session,
            location or DEFAULT_LOCATION_KEY,
            token,
            bool(clipped),
            normalization or "location",
            mode or "square",
            manager.comparison_state,
            measurement,
            measurement_message,
        )
        return (*inspection[1:], diagnostics)

    def unpin_ui(location, token, clipped, normalization, mode):
        _, diagnostics = clear_comparison_callback(manager)
        session = manager.inspection_session
        if session is None:
            return (*_skip_outputs(12), diagnostics)
        measurement, measurement_message = manager.resolve_measurement_pin(
            location or DEFAULT_LOCATION_KEY, expected_session=session
        )
        inspection = _successful_inspection_outputs(
            session,
            location or DEFAULT_LOCATION_KEY,
            token,
            bool(clipped),
            normalization or "location",
            mode or "square",
            measurement=measurement,
            measurement_message=measurement_message,
        )
        inspection = list(inspection[1:])
        inspection[-1] = "Comparison A unpinned."
        return (*inspection, diagnostics)

    def navigate_ui(token, clipped, normalization, mode, evt: gr.EventData):
        return select_bridged_location_callback(
            getattr(evt, "location", None),
            token,
            clipped,
            normalization,
            manager,
            mode,
        )

    with gr.Blocks(title="Circuit Tracer") as demo:
        with gr.Column(elem_classes="ct-page"):
            with gr.Row(elem_classes="ct-header"):
                with gr.Column(elem_classes="ct-header-copy"):
                    gr.Markdown("RESIDUAL STREAM / INSPECTION", elem_classes="ct-kicker")
                    gr.Markdown("Circuit Tracer", elem_classes="ct-title")
                    gr.Markdown(
                        "A visual workbench for tracing one-block language-model activations.",
                        elem_classes="ct-subtitle",
                    )
                gr.Markdown("LOCAL ANALYSIS / NO TELEMETRY", elem_classes="ct-header-note")
                with gr.Accordion(
                    "Checkpoint setup",
                    open=True,
                    elem_classes="ct-checkpoint-setup",
                ) as checkpoint_setup:
                    with gr.Row(elem_classes="ct-checkpoint-row"):
                        checkpoint_folder = gr.Textbox(
                            label="Checkpoint folder (server path)",
                            placeholder="C:\\...\\checkpoint-v2",
                        )
                        load_button = gr.Button("Load model", variant="primary")
                    status = gr.Markdown(
                        "No model loaded. Enter an extracted checkpoint folder.",
                        elem_classes="ct-status",
                    )
                    with gr.Accordion("Load error details", open=False):
                        load_error_details = gr.Textbox(
                            value="", label="Technical details", interactive=False
                        )
            with gr.Group(elem_classes="ct-panel ct-model-strip"):
                with gr.Row(elem_classes="ct-model-strip-heading"):
                    gr.Markdown("MODEL TOPOLOGY", elem_classes="ct-section-label")
                    gr.Markdown(
                        "Post-norm causal transformer / captured nodes are selectable",
                        elem_classes="ct-meta",
                    )
                diagram = gr.HTML(
                    render_model_diagram(),
                    js_on_load=LOCATION_NAV_JS,
                    elem_classes="ct-diagram-wrap",
                )
            with gr.Row(elem_classes="ct-workbench"):
                with gr.Column(
                    elem_id="ct-location-panel",
                    elem_classes="ct-panel ct-location-panel",
                ):
                    gr.Markdown("CAPTURED LOCATIONS", elem_classes="ct-section-label")
                    gr.Markdown(
                        "Choose a tensor without rerunning the model.",
                        elem_classes="ct-meta",
                    )
                    location_navigator = gr.HTML(
                        render_location_navigator(),
                        js_on_load=LOCATION_NAV_JS,
                        elem_classes="ct-location-nav-host",
                    )
                with gr.Column(elem_classes="ct-center"):
                    with gr.Group(elem_classes="ct-panel ct-session-context"):
                        gr.Markdown("ANALYSIS CONTEXT", elem_classes="ct-section-label")
                        with gr.Group(visible=True) as prompt_editor:
                            with gr.Row(elem_classes="ct-prompt-editor-row"):
                                prompt_text = gr.Textbox(
                                    label="Prompt",
                                    lines=2,
                                    max_lines=3,
                                    placeholder="Enter a prompt to trace...",
                                )
                                analyze_button = gr.Button(
                                    "Analyze prompt", variant="primary"
                                )
                        with gr.Group(visible=False) as prompt_context:
                            with gr.Row(elem_classes="ct-prompt-readonly-row"):
                                analyzed_prompt = gr.Textbox(
                                    label="Analyzed prompt",
                                    lines=1,
                                    max_lines=2,
                                    interactive=False,
                                )
                                edit_prompt_button = gr.Button("Edit prompt")
                        analysis_status = gr.Markdown(
                            "Load a model, then enter a prompt.",
                            elem_classes="ct-status ct-analysis-status",
                        )
                    with gr.Row(elem_classes="ct-canvas-toolbar"):
                        token_dropdown = gr.Dropdown(
                            choices=[],
                            value=None,
                            interactive=True,
                            multiselect=True,
                            label="Token positions",
                        )
                        normalization_dropdown = gr.Dropdown(
                            choices=EMPTY_SELECTION_SCOPE_CHOICES,
                            value="location",
                            interactive=True,
                            label="Scale scope",
                            info="Selection appears after choosing tokens.",
                        )
                        mode_dropdown = gr.Dropdown(
                            choices=[("Square", "square"), ("Indexed", "indexed")],
                            value="square",
                            label="Detail mode",
                            info=(
                                "Square uses artificial 2D adjacency. In detail, "
                                "click a cell to pin its token/dimension."
                            ),
                            interactive=True,
                        )
                        heatmap_clip = gr.Checkbox(label="Clip extremes", value=False)
                        toggle_nav_button = gr.Button(
                            "Locations",
                            elem_id="ct-toggle-nav",
                            elem_classes="ct-panel-toggle",
                            size="sm",
                        )
                        toggle_inspector_button = gr.Button(
                            "Metadata",
                            elem_id="ct-toggle-inspector",
                            elem_classes="ct-panel-toggle",
                            size="sm",
                        )
                    with gr.Group(
                        elem_id="ct-visual-panel",
                        elem_classes="ct-panel ct-visual-panel ct-canvas-panel",
                    ):
                        with gr.Row(elem_classes="ct-canvas-heading"):
                            gr.Markdown("ACTIVATION CANVAS", elem_classes="ct-section-label")
                            expand_canvas_button = gr.Button(
                                "Expand canvas",
                                elem_id="ct-expand-visuals",
                                size="sm",
                            )
                        close_canvas_button = gr.Button(
                            "Close expanded view",
                            elem_id="ct-close-visuals",
                            elem_classes="ct-close-button",
                        )
                        with gr.Group(elem_classes="ct-chart-main ct-canvas-scroll"):
                            heatmap_plot = gr.Plot(
                                label="Activation field",
                                elem_id="ct-activation-plot",
                                elem_classes="ct-activation-plot",
                            )
                            with gr.Row(elem_classes="ct-comparison-row"):
                                comparison_a_plot = gr.Plot(
                                    label="Raw comparison A",
                                    visible=False,
                                    elem_id="ct-comparison-a-plot",
                                    elem_classes="ct-activation-plot",
                                )
                                comparison_b_plot = gr.Plot(
                                    label="Raw comparison B",
                                    visible=False,
                                    elem_id="ct-comparison-b-plot",
                                    elem_classes="ct-activation-plot",
                                )
                            delta_plot = gr.Plot(
                                label="Signed B minus A delta",
                                visible=False,
                                elem_id="ct-delta-plot",
                                elem_classes="ct-activation-plot ct-delta-plot",
                            )
                with gr.Column(
                    elem_id="ct-metadata-panel",
                    elem_classes="ct-panel ct-metadata-panel",
                ):
                    gr.Markdown("METADATA INSPECTOR", elem_classes="ct-section-label")
                    gr.Markdown("Current tensor", elem_classes="ct-panel-title")
                    location_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        interactive=True,
                        label="Internal location",
                    )
                    inspect_explanation = gr.Markdown(
                        INSPECT_AWAITING, elem_classes="ct-meta"
                    )
                    inspect_stats = gr.Markdown("", elem_classes="ct-meta")
                    inspect_range = gr.Markdown("", elem_classes="ct-meta")
                    with gr.Row():
                        pin_button = gr.Button("Pin current as A", variant="secondary")
                        unpin_button = gr.Button("Unpin A")
                    comparison_status = gr.Markdown(
                        "No comparison pinned.", elem_classes="ct-status"
                    )
                    with gr.Accordion("Prompt tokens and predictions", open=False):
                        token_count_line = gr.Markdown("", elem_classes="ct-meta")
                        unknown_warning = gr.Markdown("", elem_classes="ct-meta")
                        with gr.Tabs():
                            with gr.Tab("Tokens"):
                                token_table = gr.Dataframe(
                                    headers=["Position", "Token", "ID"],
                                    interactive=False,
                                    label="Prompt tokens",
                                    elem_classes="ct-table",
                                )
                            with gr.Tab("Next-token ranking"):
                                next_token_table = gr.Dataframe(
                                    headers=["Rank", "Token", "ID", "Probability"],
                                    interactive=False,
                                    label="Five most likely next tokens",
                                    elem_classes="ct-table",
                                )
                    with gr.Accordion("Selected-token views", open=False):
                        magnitude_plot = gr.Plot(
                            label="Token magnitudes",
                            elem_id="ct-magnitude-plot",
                            elem_classes="ct-secondary-plot",
                        )
                        distribution_plot = gr.Plot(
                            label="Selected token distribution",
                            elem_id="ct-distribution-plot",
                            elem_classes="ct-secondary-plot",
                        )
                    with gr.Accordion("Runtime and diagnostics", open=False):
                        with gr.Accordion("Analysis error details", open=False):
                            analysis_error_details = gr.Textbox(
                                value="", label="Technical details", interactive=False
                            )
                        export_button = gr.Button(
                            "Export artifacts\u2026",
                            elem_id="ct-export-open",
                            size="sm",
                            elem_classes="ct-export-open",
                        )
                        device = gr.Markdown(
                            "**Compute device:** `not loaded`",
                            elem_classes="ct-meta",
                        )
                        metadata = gr.Markdown("", elem_classes="ct-meta")
                        diagnostics_button = gr.Button(
                            "Open diagnostics",
                            elem_id="ct-open-diagnostics",
                            size="sm",
                        )
                        diagnostics = gr.JSON(
                            value=json.loads(diagnostics_json(manager)),
                            label="Diagnostics",
                            open=False,
                            elem_id="ct-diagnostics",
                        )
                        with gr.Accordion("Technical model summary", open=False):
                            summary = gr.Textbox(
                                value="",
                                lines=12,
                                max_lines=24,
                                label="model.summary()",
                                interactive=False,
                                elem_classes="ct-technical-content",
                            )
            token_click = gr.Textbox(value="", visible="hidden", elem_id="ct-token-click")
            export_dialog_host = gr.HTML(
                value=EXPORT_DIALOG_HTML,
                elem_id="ct-export-dialog-host",
                elem_classes="ct-export-dialog-host",
            )
            load_button.click(
                fn=load_ui,
                inputs=checkpoint_folder,
                outputs=[
                    status, load_error_details, metadata, device, diagram, summary,
                    analysis_status, analysis_error_details, token_count_line,
                    unknown_warning, token_table, next_token_table,
                    location_dropdown, token_dropdown, normalization_dropdown,
                    mode_dropdown, heatmap_clip, inspect_explanation, inspect_stats,
                    inspect_range, magnitude_plot, heatmap_plot, distribution_plot,
                    comparison_a_plot, comparison_b_plot, delta_plot,
                    comparison_status, diagnostics, token_click,
                    location_navigator, checkpoint_setup, prompt_editor,
                    prompt_context, analyzed_prompt, prompt_text,
                ],
                show_progress="full",
            )
            analyze_button.click(
                fn=analyze_ui,
                inputs=prompt_text,
                outputs=[
                    analysis_status,
                    analysis_error_details,
                    token_count_line,
                    unknown_warning,
                    token_table,
                    next_token_table,
                    location_dropdown,
                    token_dropdown,
                    normalization_dropdown,
                    mode_dropdown,
                    heatmap_clip,
                    inspect_explanation,
                    inspect_stats,
                    inspect_range,
                    magnitude_plot,
                    heatmap_plot,
                    distribution_plot,
                    diagram,
                    comparison_a_plot,
                    comparison_b_plot,
                    delta_plot,
                    comparison_status,
                    diagnostics,
                    token_click,
                    location_navigator,
                    prompt_editor,
                    prompt_context,
                    analyzed_prompt,
                ],
                show_progress="full",
            )
            selection_outputs = [
                token_dropdown,
                inspect_explanation,
                inspect_stats,
                inspect_range,
                magnitude_plot,
                heatmap_plot,
                distribution_plot,
                diagram,
                comparison_a_plot,
                comparison_b_plot,
                delta_plot,
                comparison_status,
            ]
            for location_surface in (diagram, location_navigator):
                location_surface.click(
                    fn=navigate_ui,
                    inputs=[
                        token_dropdown,
                        heatmap_clip,
                        normalization_dropdown,
                        mode_dropdown,
                    ],
                    outputs=[location_dropdown, *selection_outputs],
                    show_progress="minimal",
                )
            for dropdown in (location_dropdown, token_dropdown, normalization_dropdown, mode_dropdown):
                dropdown.input(
                    fn=lambda location, token, clipped, normalization, mode: select_location_callback(
                        location, token, clipped, normalization, manager, mode
                    ),
                    inputs=[location_dropdown, token_dropdown, heatmap_clip, normalization_dropdown, mode_dropdown],
                    outputs=selection_outputs,
                    show_progress="minimal",
                )
            heatmap_clip.input(
                fn=lambda location, token, clipped, normalization, mode: select_location_callback(
                    location, token, clipped, normalization, manager, mode
                ),
                inputs=[location_dropdown, token_dropdown, heatmap_clip, normalization_dropdown, mode_dropdown],
                outputs=selection_outputs,
                show_progress="minimal",
            )
            token_click.input(
                fn=lambda clicked, location, token, clipped, normalization, mode: activation_click_callback(
                    clicked, location, token, clipped, normalization, manager, mode
                ),
                inputs=[token_click, location_dropdown, token_dropdown, heatmap_clip, normalization_dropdown, mode_dropdown],
                outputs=[*selection_outputs, normalization_dropdown],
                show_progress="minimal",
            )
            token_dropdown.input(
                fn=selection_scope_update,
                inputs=token_dropdown,
                outputs=normalization_dropdown,
                show_progress="hidden",
            )
            pin_button.click(
                fn=pin_ui,
                inputs=[
                    location_dropdown, token_dropdown, heatmap_clip,
                    normalization_dropdown, mode_dropdown,
                ],
                outputs=[*selection_outputs, diagnostics],
                show_progress="minimal",
            )
            unpin_button.click(
                fn=unpin_ui,
                inputs=[
                    location_dropdown, token_dropdown, heatmap_clip,
                    normalization_dropdown, mode_dropdown,
                ],
                outputs=[*selection_outputs, diagnostics],
                show_progress="minimal",
            )
            edit_prompt_button.click(
                fn=lambda: (
                    gr.update(visible=True),
                    gr.update(visible=False),
                ),
                inputs=[],
                outputs=[prompt_editor, prompt_context],
                show_progress="hidden",
            )
            for button, script in (
                (toggle_nav_button, NAV_TOGGLE_JS),
                (toggle_inspector_button, INSPECTOR_TOGGLE_JS),
                (expand_canvas_button, EXPAND_CANVAS_JS),
                (close_canvas_button, CLOSE_CANVAS_JS),
                (diagnostics_button, OPEN_DIAGNOSTICS_JS),
                (export_button, OPEN_EXPORT_DIALOG_JS),
            ):
                button.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js=script,
                    queue=False,
                    show_progress="hidden",
                )
    return demo


def _manager_export_context(manager: "ModelManager") -> export_artifacts.ExportContext:
    """Build a plain export context from the active in-memory research state."""
    state, session, comparison, _generation, _model_generation = manager.state_snapshot()
    if state is None:
        raise export_artifacts.ExportError(
            "Load a checkpoint before exporting artifacts."
        )
    if session is None:
        raise export_artifacts.ExportError(
            "Analyze a prompt before exporting artifacts."
        )
    recorded_location, recorded_positions, _revision = manager.view_snapshot()
    location_key = recorded_location or DEFAULT_LOCATION_KEY
    spec = location_spec(location_key)
    values = session.analysis.capture.locations[spec.key]
    token_count = int(values.shape[0])
    positions = tuple(
        sorted(
            {
                int(position)
                for position in recorded_positions
                if 0 <= int(position) < token_count
            }
        )
    )
    tokens = tuple(
        {
            "position": token.position,
            "text": token.text,
            "token_id": token.token_id,
        }
        for token in session.analysis.tokens
    )
    pinned = None
    if manager.measurement_pin is not None:
        pinned = (
            manager.measurement_pin.token_position,
            manager.measurement_pin.dimension,
        )
    comparison_a = None
    if comparison is not None:
        comparison_a = export_artifacts.ExportComparison(
            a_location_key=comparison.location_key,
            a_location_label=location_spec(comparison.location_key).label,
            a_values=comparison.values,
        )
    return export_artifacts.ExportContext(
        checkpoint_path=str(state.checkpoint_path),
        architecture=ARCHITECTURE_NAME,
        device=state.device.label,
        config=session.config.to_dict(),
        processed_prompt=" ".join(
            str(token.text) for token in session.analysis.tokens
        ),
        tokens=tokens,
        location_key=spec.key,
        location_label=spec.label,
        values=values,
        selected_positions=positions,
        pinned_measurement=pinned,
        comparison=comparison_a,
    )


def _shape_list(values) -> list:
    return [int(values.shape[0]), int(values.shape[1])]


def export_options_payload(manager: "ModelManager") -> dict:
    """Current validity of every export scope/figure/delta choice."""
    context = None
    base_message = ""
    try:
        context = _manager_export_context(manager)
    except export_artifacts.ExportError as error:
        base_message = str(error)

    if context is None:
        scopes = [
            {
                "value": scope,
                "label": export_artifacts.SCOPE_LABELS[scope],
                "available": False,
                "reason": base_message or "Nothing is available to export yet.",
            }
            for scope in export_artifacts.EXPORT_SCOPES
        ]
        figures = [
            {
                "value": target,
                "label": export_artifacts.FIGURE_LABELS[target],
                "available": False,
                "reason": base_message or "Nothing is available to export yet.",
            }
            for target in export_artifacts.FIGURE_TARGETS
        ]
        return {
            "schema": export_artifacts.EXPORT_OPTIONS_SCHEMA,
            "available": False,
            "message": base_message or "Load a model and analyze a prompt first.",
            "default_scope": export_artifacts.DEFAULT_SCOPE,
            "capture": {"available": False, "token_count": 0},
            "location": None,
            "selection": {"count": 0},
            "comparison": {
                "pinned": False,
                "active": False,
                "a_key": None,
                "b_key": None,
                "a_shape": None,
                "b_shape": None,
                "shapes_equal": False,
                "reason": base_message,
            },
            "delta": {"available": False, "reason": base_message or "No comparison is active."},
            "scopes": scopes,
            "figures": figures,
        }

    scopes = []
    for scope in export_artifacts.EXPORT_SCOPES:
        available, reason = export_artifacts.scope_reason(context, scope)
        scopes.append(
            {
                "value": scope,
                "label": export_artifacts.SCOPE_LABELS[scope],
                "available": bool(available),
                "reason": reason,
            }
        )
    figures = []
    for target in export_artifacts.FIGURE_TARGETS:
        available, reason = export_artifacts.figure_reason(context, target)
        figures.append(
            {
                "value": target,
                "label": export_artifacts.FIGURE_LABELS[target],
                "available": bool(available),
                "reason": reason,
            }
        )

    comparison_active = bool(
        context.comparison is not None
        and context.location_key != context.comparison.a_location_key
    )
    shapes_equal = False
    if comparison_active:
        shapes_equal = (
            np.asarray(context.comparison.a_values).shape
            == np.asarray(context.values).shape
        )
    delta_reason = ""
    if context.comparison is not None and not shapes_equal and comparison_active:
        delta_reason = export_artifacts.delta_disabled_reason(
            context.comparison.a_values,
            context.values,
            context.comparison.a_location_label,
            context.location_label,
        )
    elif context.comparison is None:
        delta_reason = "Pin a location as A, then select a different location as B."
    elif not comparison_active:
        delta_reason = "Select a location other than pinned A to compare."

    return {
        "schema": export_artifacts.EXPORT_OPTIONS_SCHEMA,
        "available": True,
        "message": "Ready to export raw values or charts.",
        "default_scope": export_artifacts.DEFAULT_SCOPE,
        "capture": {"available": True, "token_count": context.token_count},
        "location": {
            "key": context.location_key,
            "label": context.location_label,
            "shape": _shape_list(context.values),
        },
        "selection": {"count": len(context.selected_positions)},
        "comparison": {
            "pinned": context.comparison is not None,
            "active": comparison_active,
            "a_key": context.comparison.a_location_key
            if context.comparison
            else None,
            "b_key": context.location_key,
            "a_shape": _shape_list(context.comparison.a_values)
            if context.comparison
            else None,
            "b_shape": _shape_list(context.values),
            "shapes_equal": shapes_equal,
            "reason": delta_reason,
        },
        "delta": {
            "available": bool(comparison_active and shapes_equal),
            "reason": delta_reason,
        },
        "scopes": scopes,
        "figures": figures,
    }


def _local_only_rejection(request):
    """Reject data requests that do not target the local app listener.

    The data routes below disclose the active prompt and checkpoint path, so
    they refuse any ``Host`` that is not the loopback interface. The main page
    and the same-origin diagnostics/export popups always use the loopback
    host and are unaffected.
    """
    from fastapi.responses import Response

    host = request.headers.get("host", "")
    host_name = host.rsplit(":", 1)[0].strip("[]")
    if host_name in ("127.0.0.1", "localhost", "::1"):
        return None
    return Response(
        content=json.dumps({"reason": "Request must target the local app."}),
        status_code=403,
        media_type="application/json",
    )


def register_export_http(app, manager: "ModelManager") -> None:
    """Attach same-origin routes for the export dialog.

    ``/ct-export-options`` reports which scope/figure choices are valid for the
    current research state so the dialog can disable the rest. ``/ct-export``
    builds one raw CSV or NPZ artifact strictly on request and returns it as a
    temporary browser download; it never writes a persistent file. Figure
    (PNG/SVG) downloads happen entirely in the browser from rendered charts.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response

    _export_request_limit = 64 * 1024

    @app.get("/ct-export-options", include_in_schema=False)
    def export_options_route(request: Request):
        rejected = _local_only_rejection(request)
        if rejected is not None:
            return rejected
        return Response(
            content=json.dumps(export_options_payload(manager), sort_keys=True),
            media_type="application/json",
        )

    @app.post("/ct-export", include_in_schema=False)
    async def create_export(request: Request):
        rejected = _local_only_rejection(request)
        if rejected is not None:
            return rejected
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > _export_request_limit:
            return JSONResponse(
                status_code=413,
                content={"reason": "Export request is too large."},
            )
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"reason": "Export request body must be JSON."},
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"reason": "Export request body must be a JSON object."},
            )
        scope = body.get("scope")
        format_ = body.get("format")
        include_delta = bool(body.get("include_delta", False))
        if scope not in export_artifacts.EXPORT_SCOPES:
            return JSONResponse(
                status_code=400,
                content={"reason": "Unknown export scope: {}".format(scope)},
            )
        if format_ not in export_artifacts.RAW_FORMATS:
            return JSONResponse(
                status_code=400,
                content={
                    "reason": "Only CSV and .npz are generated by the server; "
                    "PNG/SVG images are downloaded in the browser."
                },
            )
        try:
            context = _manager_export_context(manager)
            artifact = export_artifacts.serialize_raw(
                context,
                scope,
                format_,
                include_delta=include_delta,
            )
        except export_artifacts.ExportError as error:
            return JSONResponse(status_code=400, content={"reason": str(error)})
        except Exception:
            LOGGER.exception("Raw artifact export failed")
            return JSONResponse(
                status_code=500,
                content={"reason": "Export failed because of an unexpected error."},
            )
        return Response(
            content=artifact.content,
            media_type=artifact.media_type,
            headers={
                "Content-Disposition": (
                    'attachment; filename="{}"'.format(artifact.filename)
                ),
                "X-Export-Filename": artifact.filename,
            },
        )


def register_diagnostics_http(app, manager: "ModelManager") -> None:
    """Attach the same-origin diagnostics popup page and payload endpoints.

    ``app`` is any FastAPI/Starlette-compatible app (including the Gradio
    FastAPI app). These routes serve local assets only (no CDN, no second
    Gradio session); the main workbench remains the only controller.
    """
    import plotly as _plotly

    from fastapi import Request
    from fastapi.responses import HTMLResponse, Response

    plotly_asset_path = (
        Path(_plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    )

    @app.get(PLOTLY_ASSET_PATH, include_in_schema=False)
    def serve_plotly():
        return Response(
            content=plotly_asset_path.read_bytes(),
            media_type="application/javascript",
        )

    @app.get(DIAGNOSTICS_POPUP_PATH, include_in_schema=False)
    def diagnostics_page():
        return HTMLResponse(content=diagnostics_popup_html())

    @app.get("/ct-diagnostics-status", include_in_schema=False)
    def diagnostics_status(request: Request):
        rejected = _local_only_rejection(request)
        if rejected is not None:
            return rejected
        return Response(
            content=json.dumps(manager.payload_status()),
            media_type="application/json",
        )

    @app.get("/ct-diagnostics-payload", include_in_schema=False)
    def diagnostics_payload(request: Request):
        rejected = _local_only_rejection(request)
        if rejected is not None:
            return rejected
        return Response(
            content=manager.current_payload(),
            media_type="application/json",
        )

    register_export_http(app, manager)


def main() -> None:
    manager = ModelManager()
    demo = create_app(manager)
    demo.ct_manager = manager
    from gradio.routes import App as GradioApp

    server_app = GradioApp()
    register_diagnostics_http(server_app, manager)
    demo.launch(
        css=APP_CSS,
        js=CLICK_BRIDGE_JS + EXPORT_IMAGE_JS,
        _app=server_app,
        **launch_kwargs(),
    )


if __name__ == "__main__":
    main()

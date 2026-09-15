import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App, DEFAULT_CHECKPOINT } from "./App";
import { NodeView } from "./components/NodeView";
import type {
  AnalyzePayload,
  GraphNode,
  InspectPayload,
  LoadPayload,
  OptionsPayload,
} from "./types";

vi.mock("./components/PlotlyFigure", () => ({
  PlotlyFigure: (props: { "data-testid"?: string }) => (
    <div data-testid={props["data-testid"]} />
  ),
}));

const engine = vi.hoisted(() => ({
  getOptions: vi.fn(),
  loadCheckpoint: vi.fn(),
  analyzePrompt: vi.fn(),
  ablateFeature: vi.fn(),
  clearAblation: vi.fn(),
  inspectNode: vi.fn(),
}));

vi.mock("./api/client", () => engine);

const scrollToMock = vi.fn();
const matchMediaMock = vi.fn().mockReturnValue({ matches: false });
Object.defineProperty(HTMLElement.prototype, "scrollTo", {
  configurable: true,
  value: scrollToMock,
});
Object.defineProperty(HTMLElement.prototype, "clientWidth", {
  configurable: true,
  get() {
    return this.classList.contains("ct-graph-scroll") ? 600 : 0;
  },
});
Object.defineProperty(window, "matchMedia", {
  configurable: true,
  value: matchMediaMock,
});

const blockStages: Array<{
  stage: string;
  label: string;
  kind: GraphNode["kind"];
  family: GraphNode["family"];
  featureAxis?: boolean;
}> = [
  {
    stage: "attention_input_norm",
    label: "Layer norm - attention input",
    kind: "ln",
    family: "norm",
  },
  {
    stage: "attention_pattern",
    label: "Causal attention pattern",
    kind: "pattern",
    family: "pattern",
    featureAxis: false,
  },
  {
    stage: "attention_update",
    label: "Attention output → residual",
    kind: "update",
    family: "updates",
  },
  {
    stage: "attention_residual",
    label: "Residual stream · after attention",
    kind: "stream",
    family: "stream_raw",
  },
  {
    stage: "ffn_input_norm",
    label: "Layer norm - FFN input",
    kind: "ln",
    family: "norm",
  },
  {
    stage: "ffn_hidden",
    label: "FFN hidden (ReLU)",
    kind: "hidden",
    family: "hidden",
  },
  {
    stage: "ffn_update",
    label: "FFN output → residual",
    kind: "update",
    family: "updates",
  },
  {
    stage: "ffn_residual",
    label: "Residual stream · after FFN",
    kind: "stream",
    family: "stream_raw",
  },
];

function blockKey(blockIndex: number, stage: string): string {
  return `blocks.${blockIndex}.${stage}`;
}

const nodeKeys = [
  "token_embeddings",
  "position_embeddings",
  "embedding",
  ...Array.from({ length: 3 }, (_, blockIndex) =>
    blockStages.map(({ stage }) => blockKey(blockIndex, stage)),
  ).flat(),
  "output_norm",
  "readout",
];

function graphNode(
  key: string,
  label: string,
  kind: GraphNode["kind"],
  family: GraphNode["family"],
  featureAxis = true,
): GraphNode {
  const trace = nodeKeys;
  const index = trace.indexOf(key);
  const blockMatch = /^blocks\.(\d+)\.(.+)$/.exec(key);
  const blockIndex = blockMatch ? Number(blockMatch[1]) : null;
  const stage = blockMatch?.[2] ?? null;
  return {
    key,
    label,
    kind,
    family,
    explanation: `${label} explanation.`,
    normalized: kind === "ln",
    feature_axis: featureAxis,
    deembeddable:
      key === "embedding" ||
      key === "output_norm" ||
      stage === "attention_residual" ||
      stage === "ffn_residual",
    vocab_contributable:
      stage === "attention_update" || stage === "ffn_update",
    trace_index: index,
    trace_count: trace.length,
    prev_key: index > 0 ? trace[index - 1] : null,
    next_key: index < trace.length - 1 ? trace[index + 1] : null,
    block_index: blockIndex,
    stage,
    width_source:
      stage === "ffn_hidden"
        ? "ffn"
        : kind === "pattern" || kind === "readout"
          ? "none"
          : "model",
  };
}

const optionsFixture: OptionsPayload = {
  graph: {
    nodes: [
      graphNode("token_embeddings", "Token embeddings", "component", "components"),
      graphNode(
        "position_embeddings",
        "Position embeddings",
        "component",
        "components",
      ),
      graphNode("embedding", "Residual stream · input", "stream", "stream_raw"),
      ...Array.from({ length: 3 }, (_, blockIndex) =>
        blockStages.map(({ stage, label, kind, family, featureAxis }) =>
          graphNode(
            blockKey(blockIndex, stage),
            `Block ${blockIndex + 1} - ${label}`,
            kind,
            family,
            featureAxis,
          ),
        ),
      ).flat(),
      graphNode("output_norm", "Layer norm - readout input", "ln", "norm"),
      graphNode(
        "readout",
        "Readout · next-token probabilities",
        "readout",
        "readout",
        false,
      ),
    ],
    spine: [
      "embedding",
      ...Array.from({ length: 3 }, (_, blockIndex) => [
        blockKey(blockIndex, "attention_residual"),
        blockKey(blockIndex, "ffn_residual"),
      ]).flat(),
      "output_norm",
    ],
    spine_links: [
      "attention-add",
      "ffn-add",
      "attention-add",
      "ffn-add",
      "attention-add",
      "ffn-add",
      "layer-norm",
      "readout",
    ],
    branches: Array.from({ length: 3 }, (_, blockIndex) => [
      {
        key: `block_${blockIndex}_attention`,
        label: `Block ${blockIndex + 1} causal multi-head attention`,
        reads:
          blockIndex === 0
            ? "embedding"
            : blockKey(blockIndex - 1, "ffn_residual"),
        adds_before: blockKey(blockIndex, "attention_residual"),
        path: [
          blockKey(blockIndex, "attention_input_norm"),
          blockKey(blockIndex, "attention_update"),
        ],
        observables: [blockKey(blockIndex, "attention_pattern")],
        kind: "attention" as const,
        block_index: blockIndex,
        side: "above" as const,
      },
      {
        key: `block_${blockIndex}_ffn`,
        label: `Block ${blockIndex + 1} feed-forward network`,
        reads: blockKey(blockIndex, "attention_residual"),
        adds_before: blockKey(blockIndex, "ffn_residual"),
        path: [
          blockKey(blockIndex, "ffn_input_norm"),
          blockKey(blockIndex, "ffn_hidden"),
          blockKey(blockIndex, "ffn_update"),
        ],
        observables: [],
        kind: "ffn" as const,
        block_index: blockIndex,
        side: "below" as const,
      },
    ]).flat(),
    components: ["token_embeddings", "position_embeddings"],
    trace: nodeKeys,
    default_node: "output_norm",
  },
  locations: [],
  ablation_nodes: [
    {
      key: "blocks.0.ffn_hidden",
      label: "Block 1 - FFN hidden (ReLU)",
      kind: "hidden",
      family: "hidden",
    },
    {
      key: "blocks.2.ffn_residual",
      label: "Block 3 - Residual stream · after FFN",
      kind: "stream",
      family: "stream_raw",
    },
  ],
};

const loadFixture: LoadPayload = {
  ok: true,
  status: "Model loaded successfully.",
  loaded: true,
  meta: {
    architecture: "three_block_pre_norm_causal_lm",
    path: "C:\\ckpt",
    vocab_size: 6,
    max_len: 6,
    embedding_dim: 8,
    num_heads: 2,
    key_dim: 4,
    feed_forward_dim: 12,
    dropout_rate: 0,
    num_blocks: 3,
    feed_forward_activity_l1: 0.00001,
  },
  device_label: "CPU",
  summary: "three_block_pre_norm_causal_lm\n",
};

const analyzeFixture: AnalyzePayload = {
  ok: true,
  status: "Analysis complete for 2 processed token(s).",
  token_count: 2,
  max_len: 6,
  unknown_count: 0,
  tokens: [
    { position: 0, text: "hello", token_id: 2 },
    { position: 1, text: ",", token_id: 3 },
  ],
  next_tokens: [
    { rank: 1, text: "world", token_id: 4, probability: 0.25 },
  ],
};

function nodeInfo(key: string): GraphNode {
  const fallback = graphNode(key, key, "stream", "stream_raw");
  const node = optionsFixture.graph.nodes.find((item) => item.key === key);
  return node ?? fallback;
}

function inspectFixture(
  key: string | null,
  view: "baseline" | "ablated" | "diff" = "baseline",
  deembed = false,
  vocabContributions = false,
): InspectPayload {
  const nodeKey = key ?? "output_norm";
  const node = nodeInfo(nodeKey);
  const ablationIsThirdBlockResidual =
    nodeKey === "blocks.2.ffn_residual";
  return {
    ok: true,
    state: "ready",
    message: "",
    view,
    ablation:
      view === "baseline"
        ? null
        : {
            node_key: ablationIsThirdBlockResidual
              ? "blocks.2.ffn_residual"
              : "blocks.0.ffn_hidden",
            node_label:
              ablationIsThirdBlockResidual
                ? "Block 3 - Residual stream · after FFN"
                : "Block 1 - FFN hidden (ReLU)",
            dims: [0, 2],
            mode: "zero",
            scope: "token",
            position: 1,
            baseline_values: [0, 0],
          },
    node,
    selected_position: 1,
    token_choices: [
      { position: 0, text: "hello" },
      { position: 1, text: "," },
    ],
    shape: { seq_len: 2, width: 8 },
    capture: { min: -1, mean: 0, max: 1 },
    scale: { lower: -1, upper: 1 },
    tile: node.feature_axis ? { rows: 2, cols: 4 } : null,
    figure_kind: node.kind === "readout" ? "readout_topk" : "activation",
    map_figure: node.feature_axis ? { data: [{}], layout: {} } : null,
    pattern_figure: null,
    readout_figure: node.kind === "readout" ? { data: [{}], layout: {} } : null,
    entropy_figure: null,
    readout_rows: [],
    readout_compare:
      view === "baseline" || node.kind !== "readout"
        ? null
        : {
            base_top: [],
            ablated_top: [],
            movers: [],
            has_effect: false,
          },
    readout_compare_figure:
      view === "baseline" || node.kind !== "readout"
        ? null
        : { data: [{}], layout: {} },
    position_effects: [],
    deembed_present: deembed,
    deembed_top: deembed
      ? [
          {
            rank: 1,
            text: "world",
            token_id: 4,
            probability: view === "ablated" ? 0.2 : 0.4,
          },
        ]
      : [],
    deembed_movers:
      deembed && view === "ablated"
        ? [
            {
              token_id: 4,
              text: "world",
              baseline_probability: 0.4,
              ablated_probability: 0.2,
              delta: -0.2,
              highlighted: false,
            },
          ]
        : [],
    deembed_figure: deembed ? { data: [{}], layout: {} } : null,
    deembed_has_effect: deembed && view === "ablated" ? true : null,
    deembed_state_changed: deembed && view === "ablated" ? true : null,
    vocab_contribution_present: vocabContributions,
    vocab_contribution_promoted: vocabContributions
      ? [
          {
            rank: 1,
            text: "world",
            token_id: 4,
            logit_contribution: view === "ablated" ? 0.8 : 1.2,
          },
        ]
      : [],
    vocab_contribution_suppressed: vocabContributions
      ? [
          {
            rank: 1,
            text: "hello",
            token_id: 2,
            logit_contribution: view === "ablated" ? -0.4 : -0.7,
          },
        ]
      : [],
    vocab_contribution_movers:
      vocabContributions && view === "ablated"
        ? [
            {
              token_id: 4,
              text: "world",
              baseline_contribution: 1.2,
              ablated_contribution: 0.8,
              delta: -0.4,
            },
          ]
        : [],
    vocab_contribution_figure: vocabContributions
      ? { data: [{}], layout: {} }
      : null,
    vocab_contribution_has_effect:
      vocabContributions && view === "ablated" ? true : null,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  matchMediaMock.mockReturnValue({ matches: false });
  engine.getOptions.mockResolvedValue(optionsFixture);
  engine.loadCheckpoint.mockResolvedValue(loadFixture);
  engine.analyzePrompt.mockResolvedValue(analyzeFixture);
  engine.ablateFeature.mockResolvedValue({
    ok: true,
    status: "Ablated blocks.0.ffn_hidden dimensions 0, 2.",
    ablation: {
      node_key: "blocks.0.ffn_hidden",
      node_label: "Block 1 - FFN hidden (ReLU)",
      dims: [0, 2],
      mode: "zero",
      scope: "token",
      position: 1,
      baseline_values: [0, 0],
    },
    strongest_position: 1,
  });
  engine.clearAblation.mockResolvedValue({ ok: true, status: "Ablation cleared." });
  engine.inspectNode.mockImplementation(
    async (
      key: string | null,
      _position: number | null,
      view: "baseline" | "ablated" | "diff" = "baseline",
      _highlightToken: string | null = null,
      deembed = false,
      vocabContributions = false,
    ) => inspectFixture(key, view, deembed, vocabContributions),
  );
});

describe("App", () => {
  it("loads the stream graph on start", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(engine.getOptions).toHaveBeenCalledTimes(1));
    await user.click(
      screen.getByRole("button", { name: "Show model diagram" }),
    );
    expect(await screen.findByTestId("residual-graph")).toBeInTheDocument();
    const graph = screen.getByTestId("residual-graph");
    expect(Number(graph.getAttribute("width"))).toBeGreaterThan(1580);
    expect(Number(graph.getAttribute("width"))).toBeLessThan(2000);
    expect(graph.parentElement).toHaveClass("ct-graph-scroll");
    expect(graph.parentElement?.parentElement).toHaveClass("ct-graph-frame");
    expect(graph).toHaveTextContent(
      "Residual stream · input",
    );
    expect(graph).toHaveTextContent(
      "Block 1 - Layer norm - attention input",
    );
    expect(graph).toHaveTextContent(
      "Causal attention pattern",
    );
    expect(graph).toHaveTextContent(
      "Block 3 - Layer norm - FFN input",
    );
    expect(graph.querySelectorAll("[data-branch]")).toHaveLength(6);
    expect(optionsFixture.graph.branches).toHaveLength(6);
    expect(graph).toHaveTextContent("BLOCK 1 · ATTENTION");
    expect(graph).toHaveTextContent("BLOCK 3 · FEED-FORWARD");
    expect(
      graph.querySelector('[data-node="blocks.0.attention_residual"] text'),
    ).toHaveTextContent("After attention");
    expect(
      graph.querySelector('[data-node="blocks.2.ffn_hidden"] text'),
    ).toHaveTextContent("Hidden · ReLU");
    expect(graph.querySelector('[data-node="output_norm"]')).toHaveAttribute(
      "aria-label",
      "Final layer norm: Layer norm - readout input",
    );
    await waitFor(() => expect(scrollToMock).toHaveBeenCalled());
    expect(scrollToMock.mock.lastCall?.[0].left).toBeCloseTo(1262.5);
  });

  it("honors reduced motion when revealing the selected node", async () => {
    matchMediaMock.mockReturnValue({ matches: true });
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(engine.getOptions).toHaveBeenCalledTimes(1));
    await user.click(
      screen.getByRole("button", { name: "Show model diagram" }),
    );

    await waitFor(() => expect(scrollToMock).toHaveBeenCalled());
    expect(scrollToMock.mock.lastCall?.[0]).toMatchObject({ behavior: "auto" });
  });

  it("loads a checkpoint and shows its runtime details", async () => {
    const user = userEvent.setup();
    render(<App />);

    const pathInput = screen.getByLabelText(/Server path/);
    expect(pathInput).toHaveValue(DEFAULT_CHECKPOINT);
    await user.clear(pathInput);
    await user.type(pathInput, "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));

    await waitFor(() =>
      expect(engine.loadCheckpoint).toHaveBeenCalledWith("C:\\ckpt"),
    );
    expect(await screen.findByTestId("load-status")).toHaveTextContent(
      "Model loaded successfully.",
    );
    expect(screen.getByTestId("model-meta")).toHaveTextContent(
      /Transformer blocks\s*3/,
    );
    expect(screen.getByTestId("model-meta")).toHaveTextContent(
      /FFN activity L1\s*0.00001/,
    );
    expect(await screen.findByTestId("model-summary")).toHaveTextContent(
      "three_block_pre_norm_causal_lm",
    );
  });

  it("analyzes a prompt and inspects the default node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));

    await waitFor(() =>
      expect(engine.analyzePrompt).toHaveBeenCalledWith("hello ,"),
    );
    expect(await screen.findByText(/world/)).toBeInTheDocument();
    await waitFor(() => expect(engine.inspectNode).toHaveBeenCalled());
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Layer norm - readout input",
    );
    expect(screen.getByTestId("node-primary-plot")).toBeInTheDocument();
  });

  it("renders and selects graph nodes from blocks 1 and 3", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm - readout input",
      ),
    );
    await user.click(
      screen.getByRole("button", { name: "Show model diagram" }),
    );

    const graph = await screen.findByTestId("residual-graph");
    const block1Chip = graph.querySelector(
      '[data-node="blocks.0.attention_residual"]',
    );
    const block3Chip = graph.querySelector(
      '[data-node="blocks.2.ffn_hidden"]',
    );
    expect(block1Chip).not.toBeNull();
    expect(block3Chip).not.toBeNull();
    await user.click(block1Chip as Element);

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.0.attention_residual",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Block 1 - Residual stream · after attention",
    );
    await waitFor(() =>
      expect(scrollToMock.mock.lastCall?.[0].left).toBeCloseTo(185),
    );

    await user.click(block3Chip as Element);
    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.2.ffn_hidden",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Block 3 - FFN hidden (ReLU)",
    );
    await waitFor(() =>
      expect(scrollToMock.mock.lastCall?.[0].left).toBeCloseTo(971.25),
    );
  });

  it("chooses a node from the strip above the map", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm - readout input",
      ),
    );

    const strip = screen.getByTestId("node-strip");
    expect(strip).toBeInTheDocument();
    const chip = strip.querySelector(
      '[data-node="blocks.1.attention_residual"]',
    );
    expect(chip).not.toBeNull();
    await user.click(chip as Element);

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.1.attention_residual",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Block 2 - Residual stream · after attention",
    );
  });

  it("steps to the next node in the trace", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm - readout input",
      ),
    );

    await user.click(
      screen.getByRole("button", { name: /next node in the stream/i }),
    );

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "readout",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Readout · next-token probabilities",
    );
  });

  it("selecting a token re-inspects the same node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm - readout input",
      ),
    );

    const tokenSelect = await screen.findByTestId("token-select");
    await user.selectOptions(tokenSelect, "0");

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "output_norm",
        0,
      ),
    );
  });

  it("ablates a feature and keeps the current node in ablated view", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await waitFor(() => expect(screen.getByTestId("load-status")).toHaveTextContent("Model loaded"));
    expect(screen.getByText("valid range: 0–11")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm - readout input",
      ),
    );

    const dimensions = screen.getByTestId("ablation-dim-input");
    await user.clear(dimensions);
    await user.type(dimensions, "0, 2, 2");
    await user.click(screen.getByTestId("ablate-button"));

    await waitFor(() =>
      expect(engine.ablateFeature).toHaveBeenCalledWith(
        "blocks.0.ffn_hidden",
        [0, 2],
        "zero",
        "token",
        1,
      ),
    );
    expect(await screen.findByTestId("view-toggle")).toBeInTheDocument();
    expect(screen.getByTestId("ablation-status")).toHaveTextContent("Ablated");
    expect(engine.inspectNode).toHaveBeenLastCalledWith(
      "output_norm",
      1,
      "ablated",
      null,
    );
  });

  it("projects a residual state through the output matrix", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() => screen.getByTestId("deembed-toggle"));

    await user.click(screen.getByTestId("deembed-toggle"));

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "output_norm",
        1,
        "baseline",
        null,
        true,
      ),
    );
    expect(await screen.findByTestId("deembed-results")).toBeInTheDocument();
    expect(screen.getByTestId("deembed-table")).toHaveTextContent("world");
  });

  it("shows signed vocabulary contributions for update nodes", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));

    const updateChip = (
      await screen.findByTestId("node-strip")
    ).querySelector('[data-node="blocks.0.attention_update"]');
    expect(updateChip).not.toBeNull();
    await user.click(updateChip as Element);

    expect(await screen.findByTestId("vocab-contribution-toggle")).toBeInTheDocument();
    expect(screen.queryByTestId("deembed-toggle")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("vocab-contribution-toggle"));

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.0.attention_update",
        1,
        "baseline",
        null,
        false,
        true,
      ),
    );
    expect(
      await screen.findByTestId("vocab-contribution-results"),
    ).toHaveTextContent("not probabilities");
    expect(
      screen.getByTestId("vocab-contribution-promoted-table"),
    ).toHaveTextContent("+1.200000");
    expect(
      screen.getByTestId("vocab-contribution-suppressed-table"),
    ).toHaveTextContent("-0.700000");

    await user.click(screen.getByTestId("ablate-button"));
    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.0.attention_update",
        1,
        "ablated",
        null,
        false,
        true,
      ),
    );
    expect(
      screen.getByTestId("vocab-contribution-comparison-table"),
    ).toHaveTextContent("-0.400000");

    await user.click(screen.getByRole("button", { name: "Difference" }));
    await waitFor(() =>
      expect(screen.getByTestId("vocab-contribution-toggle")).toBeDisabled(),
    );
    expect(screen.queryByTestId("vocab-contribution-results")).not.toBeInTheDocument();
  });

  it("rolls back contribution intent when the server rejects the probe", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    const updateChip = (
      await screen.findByTestId("node-strip")
    ).querySelector('[data-node="blocks.0.ffn_update"]');
    await user.click(updateChip as Element);
    await waitFor(() => screen.getByTestId("vocab-contribution-toggle"));
    engine.inspectNode.mockResolvedValueOnce({
      ...inspectFixture("blocks.0.ffn_update"),
      state: "error",
      message: "Contribution failed.",
      node: null,
    });

    await user.click(screen.getByTestId("vocab-contribution-toggle"));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Contribution failed.",
    );
    expect(screen.getByTestId("vocab-contribution-toggle")).not.toBeChecked();
    await user.selectOptions(screen.getByTestId("token-select"), "0");
    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.0.ffn_update",
        0,
      ),
    );
  });

  it("preserves enabled contributions when a refresh fails", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    const updateChip = (
      await screen.findByTestId("node-strip")
    ).querySelector('[data-node="blocks.0.attention_update"]');
    await user.click(updateChip as Element);
    await user.click(await screen.findByTestId("vocab-contribution-toggle"));
    await waitFor(() => screen.getByTestId("vocab-contribution-results"));
    engine.inspectNode.mockResolvedValueOnce({
      ...inspectFixture("blocks.0.attention_update"),
      state: "error",
      message: "Refresh failed.",
      node: null,
    });

    await user.selectOptions(screen.getByTestId("token-select"), "0");

    expect(await screen.findByRole("alert")).toHaveTextContent("Refresh failed.");
    expect(screen.getByTestId("vocab-contribution-toggle")).toBeChecked();
    expect(screen.getByTestId("vocab-contribution-results")).toBeInTheDocument();
    await user.selectOptions(screen.getByTestId("token-select"), "0");
    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.0.attention_update",
        0,
        "baseline",
        null,
        false,
        true,
      ),
    );
  });

  it("ignores stale inspection responses", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    const strip = await screen.findByTestId("node-strip");
    const firstUpdate = strip.querySelector(
      '[data-node="blocks.0.attention_update"]',
    );
    const lastUpdate = strip.querySelector('[data-node="blocks.2.ffn_update"]');
    let resolveSlow: (value: InspectPayload) => void = () => undefined;
    const slowResponse = new Promise<InspectPayload>((resolve) => {
      resolveSlow = resolve;
    });
    engine.inspectNode
      .mockImplementationOnce(() => slowResponse)
      .mockResolvedValueOnce(inspectFixture("blocks.2.ffn_update"));

    await user.click(firstUpdate as Element);
    await user.click(lastUpdate as Element);

    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Block 3 - FFN output → residual",
    );
    await act(async () => {
      resolveSlow(inspectFixture("blocks.0.attention_update"));
      await slowResponse;
    });
    expect(screen.getByTestId("node-label")).toHaveTextContent(
      "Block 3 - FFN output → residual",
    );
  });

  it("refreshes an applied ablation using the latest inspection intent", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    const ablationResult = {
      ok: true,
      status: "Ablated blocks.0.ffn_hidden dimension 0.",
      ablation: {
        node_key: "blocks.0.ffn_hidden",
        node_label: "Block 1 - FFN hidden (ReLU)",
        dims: [0],
        mode: "zero" as const,
        scope: "token" as const,
        position: 1,
        baseline_values: [0],
      },
      strongest_position: 1,
    };
    let resolveAblation: (value: typeof ablationResult) => void = () => undefined;
    const delayedAblation = new Promise<typeof ablationResult>((resolve) => {
      resolveAblation = resolve;
    });
    engine.ablateFeature.mockImplementationOnce(() => delayedAblation);

    await user.click(screen.getByTestId("ablate-button"));
    const updateChip = screen
      .getByTestId("node-strip")
      .querySelector('[data-node="blocks.2.ffn_update"]');
    await user.click(updateChip as Element);
    await user.click(await screen.findByTestId("vocab-contribution-toggle"));
    await act(async () => {
      resolveAblation(ablationResult);
      await delayedAblation;
    });

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.2.ffn_update",
        1,
        "ablated",
        null,
        false,
        true,
      ),
    );
  });

  it("keeps projected predictions visible when ablating the de-embedded node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));

    const ffnResidualChip = (
      await screen.findByTestId("node-strip")
    ).querySelector('[data-node="blocks.2.ffn_residual"]');
    expect(ffnResidualChip).not.toBeNull();
    await user.click(ffnResidualChip as Element);
    await waitFor(() => screen.getByTestId("deembed-toggle"));
    await user.click(screen.getByTestId("deembed-toggle"));
    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "blocks.2.ffn_residual",
        1,
        "baseline",
        null,
        true,
      ),
    );

    await user.selectOptions(
      screen.getByLabelText("Activation node"),
      "blocks.2.ffn_residual",
    );
    await user.click(screen.getByTestId("ablate-button"));

    await waitFor(() =>
      expect(engine.ablateFeature).toHaveBeenCalledWith(
        "blocks.2.ffn_residual",
        [0],
        "zero",
        "token",
        1,
      ),
    );
    expect(engine.inspectNode).toHaveBeenLastCalledWith(
      "blocks.2.ffn_residual",
      1,
      "ablated",
      null,
      true,
    );
    expect(screen.getByTestId("node-label")).toHaveTextContent(
      "Block 3 - Residual stream · after FFN",
    );
    expect(await screen.findByTestId("deembed-table")).toHaveTextContent(
      "world",
    );
    expect(screen.getByTestId("deembed-comparison-table")).toHaveTextContent(
      "Baseline",
    );
    expect(screen.getByTestId("deembed-comparison-table")).toHaveTextContent(
      "Ablated",
    );
  });

  it("shows readout movers and highlights a hypothesized token", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
    await user.click(screen.getByRole("button", { name: "Load model" }));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() => screen.getByTestId("node-label"));
    await user.click(screen.getByTestId("ablate-button"));
    await waitFor(() => screen.getByTestId("view-toggle"));

    const readoutChip = screen
      .getByTestId("node-strip")
      .querySelector('[data-node="readout"]');
    expect(readoutChip).not.toBeNull();
    await user.click(readoutChip as Element);

    expect(await screen.findByTestId("ablation-results")).toBeInTheDocument();
    const input = screen.getByTestId("hypothesized-token-input");
    await user.type(input, "world");
    await user.click(screen.getByTestId("highlight-token-button"));

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "readout",
        1,
        "ablated",
        "world",
      ),
    );
  });
});

describe("NodeView", () => {
  it("keeps update contributions visible when an ablation has no effect", () => {
    const inspect = {
      ...inspectFixture("blocks.0.ffn_update", "ablated", false, true),
      vocab_contribution_figure: null,
      vocab_contribution_movers: [],
      vocab_contribution_has_effect: false,
    };

    render(
      <NodeView
        inspect={inspect}
        onSelectPosition={vi.fn()}
        highlightToken=""
        onHighlightTokenChange={vi.fn()}
        onHighlightTokenSubmit={vi.fn()}
        onDeembedChange={vi.fn()}
        onVocabContributionsChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("vocab-contribution-no-effect")).toHaveTextContent(
      "did not change measurably",
    );
    expect(
      screen.getByTestId("vocab-contribution-promoted-table"),
    ).toHaveTextContent("world");
    expect(screen.queryByTestId("vocab-contribution-plot")).not.toBeInTheDocument();
  });

  it("explains an unchanged ablated de-embed result without hiding predictions", () => {
    const inspect = {
      ...inspectFixture("blocks.2.ffn_residual", "ablated", true),
      deembed_figure: null,
      deembed_movers: [],
      deembed_has_effect: false,
      deembed_state_changed: false,
    };

    render(
      <NodeView
        inspect={inspect}
        onSelectPosition={vi.fn()}
        highlightToken=""
        onHighlightTokenChange={vi.fn()}
        onHighlightTokenSubmit={vi.fn()}
        onDeembedChange={vi.fn()}
        onVocabContributionsChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("deembed-no-effect")).toHaveTextContent(
      "selected residual state did not change",
    );
    expect(screen.getByTestId("deembed-table")).toHaveTextContent("world");
    expect(screen.queryByTestId("deembed-plot")).not.toBeInTheDocument();
  });

  it("distinguishes a changed residual with no projected probability effect", () => {
    const inspect = {
      ...inspectFixture("blocks.2.ffn_residual", "ablated", true),
      deembed_figure: null,
      deembed_has_effect: false,
      deembed_state_changed: true,
    };

    render(
      <NodeView
        inspect={inspect}
        onSelectPosition={vi.fn()}
        highlightToken=""
        onHighlightTokenChange={vi.fn()}
        onHighlightTokenSubmit={vi.fn()}
        onDeembedChange={vi.fn()}
        onVocabContributionsChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("deembed-no-effect")).toHaveTextContent(
      "residual state changed",
    );
    expect(screen.getByTestId("deembed-table")).toHaveTextContent("world");
  });
});

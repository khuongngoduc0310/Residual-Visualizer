import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App, DEFAULT_CHECKPOINT } from "./App";
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

const nodeKeys = [
  "embedding",
  "attention_residual",
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
  const trace = ["embedding", "attention_update", "attention_residual", "output_norm", "readout"];
  const index = trace.indexOf(key);
  return {
    key,
    label,
    kind,
    family,
    explanation: `${label} explanation.`,
    normalized: kind === "ln",
    feature_axis: featureAxis,
    trace_index: index,
    trace_count: trace.length,
    prev_key: index > 0 ? trace[index - 1] : null,
    next_key: index < trace.length - 1 ? trace[index + 1] : null,
  };
}

const optionsFixture: OptionsPayload = {
  graph: {
    nodes: [
      graphNode("embedding", "Residual stream input", "stream", "stream_raw"),
      graphNode("attention_update", "Attention output → residual", "update", "updates"),
      graphNode("attention_residual", "Residual stream after attention", "stream", "stream_raw"),
      graphNode("output_norm", "Layer norm block output", "ln", "stream_norm"),
      graphNode("readout", "Readout probabilities", "readout", "readout", false),
    ],
    spine: ["embedding", "attention_residual", "output_norm"],
    spine_links: ["attention-add", "layer-norm", "readout"],
    branches: [],
    components: [],
    trace: nodeKeys,
    default_node: "output_norm",
  },
  locations: [],
  ablation_nodes: [
    {
      key: "ffn_hidden",
      label: "FFN hidden (ReLU)",
      kind: "hidden",
      family: "hidden",
    },
  ],
};

const loadFixture: LoadPayload = {
  ok: true,
  status: "Model loaded successfully.",
  loaded: true,
  meta: {
    architecture: "one_block_post_norm_causal_lm",
    path: "C:\\ckpt",
    vocab_size: 6,
    max_len: 6,
    embedding_dim: 8,
    num_heads: 2,
    key_dim: 4,
    feed_forward_dim: 8,
    dropout_rate: 0,
  },
  device_label: "CPU",
  summary: "one_block_post_norm_causal_lm\n",
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
): InspectPayload {
  const nodeKey = key ?? "output_norm";
  const node = nodeInfo(nodeKey);
  return {
    ok: true,
    state: "ready",
    message: "",
    view,
    ablation:
      view === "baseline"
        ? null
        : {
            node_key: "ffn_hidden",
            node_label: "FFN hidden (ReLU)",
            dim: 0,
            mode: "zero",
            scope: "token",
            position: 1,
            baseline_value: 0,
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
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  engine.getOptions.mockResolvedValue(optionsFixture);
  engine.loadCheckpoint.mockResolvedValue(loadFixture);
  engine.analyzePrompt.mockResolvedValue(analyzeFixture);
  engine.ablateFeature.mockResolvedValue({
    ok: true,
    status: "Ablated ffn_hidden dimension 0.",
    ablation: {
      node_key: "ffn_hidden",
      node_label: "FFN hidden (ReLU)",
      dim: 0,
      mode: "zero",
      scope: "token",
      position: 1,
      baseline_value: 0,
    },
    strongest_position: 1,
  });
  engine.clearAblation.mockResolvedValue({ ok: true, status: "Ablation cleared." });
  engine.inspectNode.mockImplementation(
    async (
      key: string | null,
      _position: number | null,
      view: "baseline" | "ablated" | "diff" = "baseline",
    ) => inspectFixture(key, view),
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
    expect(screen.getByTestId("residual-graph")).toHaveTextContent(
      "Residual stream input",
    );
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
    expect(screen.getByTestId("model-meta")).toHaveTextContent("CPU");
    expect(await screen.findByTestId("model-summary")).toHaveTextContent(
      "one_block_post_norm_causal_lm",
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
      "Layer norm block output",
    );
    expect(screen.getByTestId("node-primary-plot")).toBeInTheDocument();
  });

  it("clicks a graph chip to inspect that node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm block output",
      ),
    );
    await user.click(
      screen.getByRole("button", { name: "Show model diagram" }),
    );

    const graph = await screen.findByTestId("residual-graph");
    const chip = graph.querySelector('[data-node="attention_residual"]');
    expect(chip).not.toBeNull();
    await user.click(chip as Element);

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "attention_residual",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Residual stream after attention",
    );
  });

  it("chooses a node from the strip above the map", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm block output",
      ),
    );

    const strip = screen.getByTestId("node-strip");
    expect(strip).toBeInTheDocument();
    const chip = strip.querySelector('[data-node="attention_residual"]');
    expect(chip).not.toBeNull();
    await user.click(chip as Element);

    await waitFor(() =>
      expect(engine.inspectNode).toHaveBeenLastCalledWith(
        "attention_residual",
        expect.anything(),
      ),
    );
    expect(await screen.findByTestId("node-label")).toHaveTextContent(
      "Residual stream after attention",
    );
  });

  it("steps to the next node in the trace", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm block output",
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
      "Readout probabilities",
    );
  });

  it("selecting a token re-inspects the same node", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm block output",
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

    await user.click(screen.getByRole("button", { name: "Load model" }));
    await waitFor(() => expect(screen.getByTestId("load-status")).toHaveTextContent("Model loaded"));
    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));
    await waitFor(() =>
      expect(screen.getByTestId("node-label")).toHaveTextContent(
        "Layer norm block output",
      ),
    );

    await user.click(screen.getByTestId("ablate-button"));

    await waitFor(() =>
      expect(engine.ablateFeature).toHaveBeenCalledWith(
        "ffn_hidden",
        0,
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

  it("shows readout movers and highlights a hypothesized token", async () => {
    const user = userEvent.setup();
    render(<App />);

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

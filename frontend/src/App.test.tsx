import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "./App";
import type {
  AnalyzePayload,
  InspectPayload,
  LoadPayload,
  OptionsPayload,
} from "./types";

vi.mock("./components/PlotlyFigure", () => ({
  PlotlyFigure: () => null,
}));

const engine = vi.hoisted(() => ({
  getOptions: vi.fn(),
  loadCheckpoint: vi.fn(),
  analyzePrompt: vi.fn(),
  inspectLocation: vi.fn(),
}));

vi.mock("./api/client", () => engine);

const optionsFixture: OptionsPayload = {
  locations: [
    {
      key: "output_norm",
      label: "Final block output",
      category: "Residual Stream",
      explanation: "The final layer-normalized block output.",
      normalized: true,
    },
  ],
  default_location: "output_norm",
  default_stages: [
    {
      key: "embedding",
      name: "Token + position embeddings",
      category: "Residual Stream",
      detail: null,
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
  stages: [
    {
      key: "embedding",
      name: "Token + position embeddings",
      category: "Residual Stream",
      detail: "sequence <= 6, width 8",
    },
    {
      key: "output_norm",
      name: "Final layer normalization",
      category: "Residual Stream",
      detail: "post-norm block output",
    },
  ],
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

function inspectFixture(): InspectPayload {
  return {
    ok: true,
    state: "ready",
    message: "",
    location: {
      key: "output_norm",
      label: "Final block output",
      category: "Residual Stream",
      explanation: "The final layer-normalized block output.",
      normalized: true,
    },
    selected_position: 1,
    token_choices: [
      { position: 0, text: "hello" },
      { position: 1, text: "," },
    ],
    shape: { seq_len: 2, width: 8 },
    capture: { min: -1, mean: 0, max: 1 },
    selected_stats: {
      norm: 1,
      mean: 0,
      standard_deviation: 0.5,
      minimum: -1,
      maximum: 1,
    },
    heatmap: { lower: -1, upper: 1, clipped: false },
    magnitude: { data: [{}], layout: {} },
    heatmap_figure: { data: [{}], layout: {} },
    distribution: { data: [{}], layout: {} },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  engine.getOptions.mockResolvedValue(optionsFixture);
  engine.loadCheckpoint.mockResolvedValue(loadFixture);
  engine.analyzePrompt.mockResolvedValue(analyzeFixture);
  engine.inspectLocation.mockResolvedValue(inspectFixture());
});

describe("App", () => {
  it("loads the inspection options on start", async () => {
    render(<App />);
    await waitFor(() => expect(engine.getOptions).toHaveBeenCalledTimes(1));
    expect(
      await screen.findByRole("option", {
        name: /Residual Stream · Final block output/,
      }),
    ).toBeInTheDocument();
  });

  it("loads a checkpoint and shows its runtime details", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Server path/), "C:\\ckpt");
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

  it("analyzes a prompt and inspects the capture", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText(/Prompt/), "hello ,");
    await user.click(screen.getByRole("button", { name: "Analyze prompt" }));

    await waitFor(() =>
      expect(engine.analyzePrompt).toHaveBeenCalledWith("hello ,"),
    );
    expect(await screen.findByText("hello")).toBeInTheDocument();
    expect(await screen.findByText("0.2500")).toBeInTheDocument();
    await waitFor(() => expect(engine.inspectLocation).toHaveBeenCalled());
    expect(await screen.findByTestId("inspect-text")).toHaveTextContent(
      "Final block output",
    );
  });
});

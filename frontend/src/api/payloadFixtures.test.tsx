import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import contributionRaw from "../../../tests/fixtures/inspect_contribution.json?raw";
import readoutRaw from "../../../tests/fixtures/inspect_readout.json?raw";
import schemaRaw from "../../../tests/fixtures/schema.json?raw";
import { NodeView } from "../components/NodeView";
import type { InspectPayload } from "../types";

vi.mock("../components/PlotlyFigure", () => ({
  PlotlyFigure: () => <div data-testid="plotly-figure" />,
}));

const noop = () => {};

function renderNodeView(inspect: InspectPayload) {
  return render(
    <NodeView
      inspect={inspect}
      onSelectPosition={noop}
      highlightToken=""
      onHighlightTokenChange={noop}
      onHighlightTokenSubmit={noop}
      onDeembedChange={noop}
      onVocabContributionsChange={noop}
    />,
  );
}

describe("payload fixtures", () => {
  it("renders the committed vocabulary-contribution payload", () => {
    const inspect = JSON.parse(contributionRaw) as InspectPayload;

    renderNodeView(inspect);

    expect(screen.getByTestId("vocab-contribution-promoted-table")).toBeInTheDocument();
    expect(
      screen.getByTestId("vocab-contribution-suppressed-table"),
    ).toBeInTheDocument();
  });

  it("renders the committed readout payload", () => {
    const inspect = JSON.parse(readoutRaw) as InspectPayload;

    renderNodeView(inspect);

    expect(screen.getByTestId("readout-table")).toBeInTheDocument();
  });

  it("keeps fixture keys aligned with the shared schema", () => {
    const schema = JSON.parse(schemaRaw) as Record<string, Record<string, string>>;
    const inspect = JSON.parse(contributionRaw) as Record<string, unknown>;

    expect(Object.keys(inspect).sort()).toEqual(
      Object.keys(schema.inspect_contribution).sort(),
    );
  });
});

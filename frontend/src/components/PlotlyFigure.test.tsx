import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlotlyFigure } from "./PlotlyFigure";

const plotly = vi.hoisted(() => ({
  purge: vi.fn(),
  react: vi.fn(),
}));

vi.mock("plotly.js-dist-min", () => ({ default: plotly }));

describe("PlotlyFigure", () => {
  it("shows a visible error when Plotly rejects a figure", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    plotly.react.mockImplementationOnce((host: HTMLElement) => {
      Object.assign(host, {
        on: vi.fn(),
        removeAllListeners: vi.fn(),
      });
      return Promise.reject(new Error("invalid figure"));
    });

    render(
      <PlotlyFigure
        figure={{ data: [{}], layout: {} }}
        data-testid="failed-plot"
      />,
    );

    expect(screen.getByTestId("failed-plot")).toBeInTheDocument();
    expect(
      await screen.findByText("Chart could not be rendered."),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(plotly.react).toHaveBeenCalledWith(
        expect.any(HTMLElement),
        [{}],
        expect.objectContaining({ width: 640, height: 380 }),
        expect.objectContaining({ responsive: true }),
      ),
    );
    expect(warn).toHaveBeenCalledWith(
      "Plotly render failed",
      expect.any(Error),
    );
    warn.mockRestore();
  });
});

import type { InspectPayload } from "../types";
import { PlotlyFigure } from "./PlotlyFigure";

interface VisualsProps {
  inspect: InspectPayload | null;
  expanded: boolean;
  onToggleExpanded: () => void;
  onSelectPosition: (position: number) => void;
}

export function Visuals({
  inspect,
  expanded,
  onToggleExpanded,
  onSelectPosition,
}: VisualsProps) {
  return (
    <section
      className="ct-panel ct-visual-panel"
      data-expanded={expanded || undefined}
    >
      {expanded && (
        <button
          type="button"
          className="ct-close-button"
          onClick={onToggleExpanded}
        >
          Close expanded view
        </button>
      )}
      <div className="ct-chart-main">
        <PlotlyFigure
          figure={inspect?.heatmap_figure ?? null}
          className="ct-plot ct-plot-main"
          data-testid="heatmap-plot"
        />
      </div>
      <div className="ct-chart-row">
        <div className="ct-chart-box">
          <PlotlyFigure
            figure={inspect?.magnitude ?? null}
            onSelectPosition={onSelectPosition}
            className="ct-plot ct-plot-side"
            data-testid="magnitude-plot"
          />
        </div>
        <div className="ct-chart-box">
          <PlotlyFigure
            figure={inspect?.distribution ?? null}
            className="ct-plot ct-plot-side"
            data-testid="distribution-plot"
          />
        </div>
      </div>
      {inspect?.heatmap && (
        <p className="ct-plot-caption" data-testid="heatmap-range">
          Visible heatmap range: {inspect.heatmap.lower.toFixed(4)} to{" "}
          {inspect.heatmap.upper.toFixed(4)}
          {inspect.heatmap.clipped
            ? " (display clipped to the 1st-99th percentile)"
            : ""}
        </p>
      )}
      <button
        type="button"
        className="ct-expand-button"
        onClick={onToggleExpanded}
      >
        {expanded ? "Restore workspace" : "Expand visualizations"}
      </button>
    </section>
  );
}

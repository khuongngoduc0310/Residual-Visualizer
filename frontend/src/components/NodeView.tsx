import type { InspectPayload } from "../types";
import { PlotlyFigure } from "./PlotlyFigure";

interface NodeViewProps {
  inspect: InspectPayload;
  onSelectPosition: (position: number) => void;
}

export function NodeView({ inspect, onSelectPosition }: NodeViewProps) {
  const kind = inspect.node?.kind ?? null;
  const scale = inspect.scale;

  let primary = null;
  if (kind === "pattern") {
    primary = inspect.pattern_figure;
  } else if (kind === "readout") {
    primary = inspect.readout_figure;
  } else {
    primary = inspect.map_figure;
  }

  return (
    <div className="ct-node-view">
      <div className="ct-chart-main">
        <PlotlyFigure
          figure={primary}
          onSelectPosition={kind === "readout" ? undefined : onSelectPosition}
          className="ct-plot ct-plot-main"
          data-testid="node-primary-plot"
        />
      </div>

      {scale && (
        <p className="ct-plot-caption" data-testid="scale-caption">
          Visible range: {scale.lower.toFixed(4)} to {scale.upper.toFixed(4)}
          {inspect.tile
            ? ` · each square = one token's ${inspect.shape?.width ?? ""} dims in a ${inspect.tile.rows}×${inspect.tile.cols} grid`
            : ""}
          {" · scroll to zoom"}
        </p>
      )}

      {kind === "readout" ? (
        <div className="ct-readout-row">
          <div className="ct-chart-box">
            <PlotlyFigure
              figure={inspect.entropy_figure}
              onSelectPosition={onSelectPosition}
              className="ct-plot ct-plot-side"
              data-testid="entropy-plot"
            />
          </div>
          <div className="ct-readout-table">
            <h3 className="ct-subheading">Top next tokens for this position</h3>
            <div className="ct-table-wrap">
              <table className="ct-table" data-testid="readout-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Token</th>
                    <th>ID</th>
                    <th>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {inspect.readout_rows.map((row) => (
                    <tr key={row.rank}>
                      <td>{row.rank}</td>
                      <td>{row.text}</td>
                      <td>{row.token_id}</td>
                      <td>{row.probability.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

import type { CSSProperties } from "react";
import type { InspectPayload } from "../types";
import { PlotlyFigure } from "./PlotlyFigure";

const CELL_PX = 18;
const LABEL_RESERVE = 130;

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

  const showTokenRowMap =
    kind !== null &&
    kind !== "pattern" &&
    kind !== "readout" &&
    inspect.map_figure !== null &&
    inspect.shape !== null &&
    inspect.tile !== null;

  let stageStyle: CSSProperties | undefined;
  if (showTokenRowMap && inspect.shape && inspect.tile) {
    const columns = inspect.shape.seq_len * inspect.tile.cols;
    stageStyle = {
      width: `${Math.max(320, columns * CELL_PX)}px`,
      height: `${inspect.tile.rows * CELL_PX + LABEL_RESERVE}px`,
    };
  }

  return (
    <div className="ct-node-view">
      {showTokenRowMap ? (
        <div className="ct-map-scroll" data-testid="token-map">
          <div className="ct-map-stage" style={stageStyle}>
            <PlotlyFigure
              figure={primary}
              onSelectPosition={onSelectPosition}
              className="ct-plot ct-plot-map"
              data-testid="node-primary-plot"
            />
          </div>
        </div>
      ) : (
        <div className="ct-chart-main">
          <PlotlyFigure
            figure={primary}
            onSelectPosition={kind === "readout" ? undefined : onSelectPosition}
            className="ct-plot ct-plot-main"
            data-testid="node-primary-plot"
          />
        </div>
      )}

      {scale && (
        <p className="ct-plot-caption" data-testid="scale-caption">
          Visible range: {scale.lower.toFixed(4)} to {scale.upper.toFixed(4)}
          {scale.clipped
            ? " (display clipped to the 1st-99th percentile)"
            : ""}
          {scale.source === "family"
            ? ` · shared ${scale.family} scale`
            : ""}
          {inspect.tile
            ? ` · each square = one token's ${inspect.shape?.width ?? ""} dims in a ${inspect.tile.rows}×${inspect.tile.cols} grid`
            : ""}
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

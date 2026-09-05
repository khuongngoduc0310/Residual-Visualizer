import type { InspectPayload } from "../types";
import { PlotlyFigure } from "./PlotlyFigure";

interface NodeViewProps {
  inspect: InspectPayload;
  onSelectPosition: (position: number) => void;
  highlightToken: string;
  onHighlightTokenChange: (value: string) => void;
  onHighlightTokenSubmit: () => void;
}

export function NodeView({
  inspect,
  onSelectPosition,
  highlightToken,
  onHighlightTokenChange,
  onHighlightTokenSubmit,
}: NodeViewProps) {
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
        <>
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
              <h3 className="ct-subheading">
                {inspect.view === "diff"
                  ? "Ablation probability movement"
                  : "Top next tokens for this position"}
              </h3>
              <div className="ct-table-wrap">
                <table className="ct-table" data-testid="readout-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Token</th>
                      <th>ID</th>
                      <th>{inspect.view === "diff" ? "Delta" : "Probability"}</th>
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

          {inspect.readout_compare && (
            <section className="ct-ablation-results" data-testid="ablation-results">
              <div className="ct-panel-head">
                <div>
                  <h3 className="ct-subheading">Ablation evidence</h3>
                  <p className="ct-muted">
                    Negative Δ means the feature was promoting that token.
                  </p>
                </div>
                <form
                  className="ct-highlight-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    onHighlightTokenSubmit();
                  }}
                >
                  <label className="ct-field-label" htmlFor="hypothesized-token">
                    Hypothesized next token
                  </label>
                  <div className="ct-button-row">
                    <input
                      id="hypothesized-token"
                      className="ct-text-input"
                      value={highlightToken}
                      onChange={(event) =>
                        onHighlightTokenChange(event.target.value)
                      }
                      placeholder="e.g. the"
                      data-testid="hypothesized-token-input"
                    />
                    <button
                      type="submit"
                      className="ct-button ct-button-ghost"
                      data-testid="highlight-token-button"
                    >
                      Highlight
                    </button>
                  </div>
                </form>
              </div>
              {inspect.readout_compare_figure && (
                <div className="ct-chart-box">
                  <PlotlyFigure
                    figure={inspect.readout_compare_figure}
                    className="ct-plot ct-plot-side"
                    data-testid="readout-delta-plot"
                  />
                </div>
              )}
              <div className="ct-compare-columns">
                <div>
                  <h4 className="ct-subheading">Baseline top-K</h4>
                  {inspect.readout_compare.base_top.map((row) => (
                    <span className="ct-next-chip" key={row.rank}>
                      {row.rank}. {row.text} ({row.probability.toFixed(3)})
                    </span>
                  ))}
                </div>
                <div>
                  <h4 className="ct-subheading">Ablated top-K</h4>
                  {inspect.readout_compare.ablated_top.map((row) => (
                    <span className="ct-next-chip" key={row.rank}>
                      {row.rank}. {row.text} ({row.probability.toFixed(3)})
                    </span>
                  ))}
                </div>
              </div>
              <div className="ct-table-wrap">
                <table className="ct-table" data-testid="mover-table">
                  <thead>
                    <tr>
                      <th>Token</th>
                      <th>Baseline</th>
                      <th>Ablated</th>
                      <th>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspect.readout_compare.movers.map((row) => (
                      <tr
                        key={row.token_id}
                        className={row.highlighted ? "ct-mover-highlighted" : ""}
                      >
                        <td>{row.text}</td>
                        <td>{row.baseline_probability.toFixed(6)}</td>
                        <td>{row.ablated_probability.toFixed(6)}</td>
                        <td>{row.delta.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="ct-input-hint" data-testid="position-effects">
                Effect by position: {inspect.position_effects
                  .map(
                    (effect) =>
                      `${effect.position} ${effect.text} (${effect.effect.toFixed(4)})`,
                  )
                  .join(" · ")}
              </p>
            </section>
          )}
        </>
      ) : null}
    </div>
  );
}

import type { InspectPayload } from "../types";
import { PlotlyFigure } from "./PlotlyFigure";

interface NodeViewProps {
  inspect: InspectPayload;
  onSelectPosition: (position: number) => void;
  highlightToken: string;
  onHighlightTokenChange: (value: string) => void;
  onHighlightTokenSubmit: () => void;
  onDeembedChange: (enabled: boolean) => void;
  onVocabContributionsChange: (enabled: boolean) => void;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(6)}`;
}

export function NodeView({
  inspect,
  onSelectPosition,
  highlightToken,
  onHighlightTokenChange,
  onHighlightTokenSubmit,
  onDeembedChange,
  onVocabContributionsChange,
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

      {inspect.node?.deembeddable && (
        <label className="ct-deembed-control">
          <input
            type="checkbox"
            checked={inspect.deembed_present}
            onChange={(event) => onDeembedChange(event.target.checked)}
            disabled={inspect.view === "diff"}
            data-testid="deembed-toggle"
          />
          <span>
            De-embed at this token
            {inspect.view === "diff" ? " (available in baseline or ablated view)" : ""}
          </span>
        </label>
      )}

      {inspect.node?.vocab_contributable && (
        <label className="ct-deembed-control">
          <input
            type="checkbox"
            checked={inspect.vocab_contribution_present}
            onChange={(event) => onVocabContributionsChange(event.target.checked)}
            disabled={inspect.view === "diff"}
            data-testid="vocab-contribution-toggle"
          />
          <span>
            Show vocabulary contributions
            {inspect.view === "diff" ? " (available in baseline or ablated view)" : ""}
          </span>
        </label>
      )}

      {inspect.deembed_present && (
        <section className="ct-ablation-results" data-testid="deembed-results">
          <div className="ct-panel-head">
            <div>
              <h3 className="ct-subheading">Projected next-token readout</h3>
              <p className="ct-muted">
                This residual state is projected through the model&apos;s final output
                matrix.
              </p>
              {inspect.view === "ablated" && inspect.ablation ? (
                <p className="ct-input-hint" data-testid="deembed-context">
                  Ablation target: {inspect.ablation.node_label}; dimensions{" "}
                  {inspect.ablation.dims.join(", ")};{" "}
                  {inspect.ablation.scope === "all"
                    ? "all prompt tokens"
                    : `token ${inspect.ablation.position}`}
                  .
                </p>
              ) : null}
            </div>
          </div>
          {inspect.view === "ablated" && inspect.deembed_state_changed === false ? (
            <p className="ct-status" data-testid="deembed-no-effect">
              The selected residual state did not change under this ablation, so its
              projected prediction is unchanged.
            </p>
          ) : inspect.view === "ablated" && inspect.deembed_has_effect === false ? (
            <p className="ct-status" data-testid="deembed-no-effect">
              The residual state changed, but its projected probabilities did not change
              measurably.
            </p>
          ) : null}
          <div
            className={`ct-readout-row ${inspect.deembed_figure ? "" : "ct-readout-row-single"}`}
          >
            {inspect.deembed_figure ? (
              <div className="ct-chart-box">
                <PlotlyFigure
                  figure={inspect.deembed_figure}
                  className="ct-plot ct-plot-side"
                  data-testid="deembed-plot"
                />
              </div>
            ) : null}
            <div className="ct-readout-table">
              <h3 className="ct-subheading">
                {inspect.view === "ablated"
                  ? "Ablated projected next tokens"
                  : "Top projected next tokens"}
              </h3>
              <div className="ct-table-wrap">
                <table className="ct-table" data-testid="deembed-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Token</th>
                      <th>ID</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspect.deembed_top.map((row) => (
                      <tr key={row.rank}>
                        <td>{row.rank}</td>
                        <td>{row.text}</td>
                        <td>{row.token_id}</td>
                        <td>{row.probability.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {inspect.deembed_top.length === 0 ? (
                  <p className="ct-status ct-status-error" role="alert">
                    Projected prediction data was missing from the server response.
                  </p>
                ) : null}
              </div>
            </div>
          </div>
          {inspect.view === "ablated" && inspect.deembed_movers.length > 0 ? (
            <div className="ct-table-wrap">
              <h3 className="ct-subheading">Projected probability movement</h3>
              <table className="ct-table" data-testid="deembed-comparison-table">
                <thead>
                  <tr>
                    <th>Token</th>
                    <th>Baseline</th>
                    <th>Ablated</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {inspect.deembed_movers.map((row) => (
                    <tr key={row.token_id}>
                      <td>{row.text}</td>
                      <td>{row.baseline_probability.toFixed(6)}</td>
                      <td>{row.ablated_probability.toFixed(6)}</td>
                      <td>{row.delta.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      )}

      {inspect.vocab_contribution_present && (
        <section
          className="ct-ablation-results"
          data-testid="vocab-contribution-results"
        >
          <h3 className="ct-subheading">Direct logit attribution</h3>
          <p className="ct-muted">
            Signed contributions through the model&apos;s final normalization scale and
            vocabulary kernel. Positive values promote a token; negative values suppress
            it. These are not probabilities.
          </p>
          {inspect.view === "ablated" ? (
            <p className="ct-input-hint">
              Movement can come from the update, the final normalization context, or
              both; unchanged attribution does not imply an unchanged final prediction.
            </p>
          ) : null}
          {inspect.view === "ablated" && inspect.ablation ? (
            <p className="ct-input-hint" data-testid="vocab-contribution-context">
              Ablation target: {inspect.ablation.node_label}; dimensions{" "}
              {inspect.ablation.dims.join(", ")};{" "}
              {inspect.ablation.scope === "all"
                ? "all prompt tokens"
                : `token ${inspect.ablation.position}`}
              .
            </p>
          ) : null}
          {inspect.view === "ablated" &&
          inspect.vocab_contribution_has_effect === false ? (
            <p className="ct-status" data-testid="vocab-contribution-no-effect">
              This update&apos;s direct logit attribution did not change measurably
              under the ablation.
            </p>
          ) : null}
          {inspect.vocab_contribution_figure ? (
            <div className="ct-chart-box">
              <PlotlyFigure
                figure={inspect.vocab_contribution_figure}
                className="ct-plot ct-plot-contribution"
                data-testid="vocab-contribution-plot"
              />
            </div>
          ) : null}
          <div className="ct-contribution-columns">
            <div className="ct-readout-table">
              <h3 className="ct-subheading">Strongest promoted tokens</h3>
              <div className="ct-table-wrap">
                <table
                  className="ct-table"
                  data-testid="vocab-contribution-promoted-table"
                  aria-label="Strongest promoted token contributions"
                >
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Token</th>
                      <th>ID</th>
                      <th>Logit contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspect.vocab_contribution_promoted.map((row) => (
                      <tr key={row.token_id}>
                        <td>{row.rank}</td>
                        <td>{row.text}</td>
                        <td>{row.token_id}</td>
                        <td>{signed(row.logit_contribution)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {inspect.vocab_contribution_promoted.length === 0 ? (
                  <p className="ct-muted">No positive contributions.</p>
                ) : null}
              </div>
            </div>
            <div className="ct-readout-table">
              <h3 className="ct-subheading">Strongest suppressed tokens</h3>
              <div className="ct-table-wrap">
                <table
                  className="ct-table"
                  data-testid="vocab-contribution-suppressed-table"
                  aria-label="Strongest suppressed token contributions"
                >
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Token</th>
                      <th>ID</th>
                      <th>Logit contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspect.vocab_contribution_suppressed.map((row) => (
                      <tr key={row.token_id}>
                        <td>{row.rank}</td>
                        <td>{row.text}</td>
                        <td>{row.token_id}</td>
                        <td>{signed(row.logit_contribution)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {inspect.vocab_contribution_suppressed.length === 0 ? (
                  <p className="ct-muted">No negative contributions.</p>
                ) : null}
              </div>
            </div>
          </div>
          {inspect.view === "ablated" &&
          inspect.vocab_contribution_movers.length > 0 ? (
            <div className="ct-table-wrap">
              <h3 className="ct-subheading">Attribution movement</h3>
              <table
                className="ct-table"
                data-testid="vocab-contribution-comparison-table"
                aria-label="Ablated vocabulary attribution movement"
              >
                <thead>
                  <tr>
                    <th>Token</th>
                    <th>Baseline</th>
                    <th>Ablated</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {inspect.vocab_contribution_movers.map((row) => (
                    <tr key={row.token_id}>
                      <td>{row.text}</td>
                      <td>{signed(row.baseline_contribution)}</td>
                      <td>{signed(row.ablated_contribution)}</td>
                      <td>{signed(row.delta)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
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
                      onChange={(event) => onHighlightTokenChange(event.target.value)}
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
                Effect by position:{" "}
                {inspect.position_effects
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

import { useCallback, useEffect, useMemo, useState } from "react";
import * as engine from "./api/client";
import { CtApiError } from "./api/gradio";
import { ModelDiagram } from "./components/ModelDiagram";
import { Visuals } from "./components/Visuals";
import type {
  AnalyzePayload,
  InspectPayload,
  LoadPayload,
  OptionsPayload,
} from "./types";
import "./App.css";

const FALLBACK_LOCATION = "output_norm";
const AWAITING_COPY = "Analyze a prompt to capture every internal location.";

function errorMessage(error: unknown): string {
  return error instanceof CtApiError || error instanceof Error
    ? error.message
    : "An unexpected error occurred.";
}

function formatCount(value: number | null): string {
  return value === null ? "—" : String(value);
}

export function App() {
  const [options, setOptions] = useState<OptionsPayload | null>(null);

  const [checkpointPath, setCheckpointPath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loadResult, setLoadResult] = useState<LoadPayload | null>(null);
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadError, setLoadError] = useState<string>("");

  const [analysis, setAnalysis] = useState<AnalyzePayload | null>(null);
  const [analyzeBusy, setAnalyzeBusy] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string>("");

  const [inspectResult, setInspectResult] = useState<InspectPayload | null>(null);
  const [inspectBusy, setInspectBusy] = useState(false);
  const [inspectError, setInspectError] = useState<string>("");
  const [locationKey, setLocationKey] = useState<string | null>(null);
  const [tokenPosition, setTokenPosition] = useState<number | null>(null);
  const [clipped, setClipped] = useState(false);
  const [visualExpanded, setVisualExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getOptionsSafe()
      .then((loaded) => {
        if (!cancelled) setOptions(loaded);
      })
      .catch((error) => {
        if (!cancelled) setLoadError(errorMessage(error));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function getOptionsSafe(): Promise<OptionsPayload> {
    return engine.getOptions();
  }

  useEffect(() => {
    document.body.classList.toggle("ct-visual-expanded", visualExpanded);
    return () => document.body.classList.remove("ct-visual-expanded");
  }, [visualExpanded]);

  const locations = options?.locations ?? [];
  const defaultLocation = options?.default_location ?? FALLBACK_LOCATION;

  const selectedLocationKey = locationKey ?? defaultLocation;

  const selectableStages = useMemo(() => {
    if (!analysis?.ok) return new Set<string>();
    return new Set(locations.map((item) => item.key));
  }, [analysis?.ok, locations]);

  const runInspect = useCallback(
    async (
      key: string | null,
      position: number | null,
      clip: boolean,
    ): Promise<InspectPayload> => {
      setInspectBusy(true);
      setInspectError("");
      try {
        const result = await engine.inspectLocation(key, position, clip);
        setInspectResult(result);
        if (result.state === "ready" && result.location) {
          setLocationKey(result.location.key);
          setTokenPosition(result.selected_position);
        }
        return result;
      } catch (error) {
        const message = errorMessage(error);
        setInspectError(message);
        throw error;
      } finally {
        setInspectBusy(false);
      }
    },
    [],
  );

  async function handleLoad(): Promise<void> {
    setLoadBusy(true);
    setLoadError("");
    try {
      const result = await engine.loadCheckpoint(checkpointPath);
      setLoadResult(result);
      setAnalysis(null);
      setInspectResult(null);
      setInspectError("");
      setLocationKey(null);
      setTokenPosition(null);
      if (!result.ok) {
        setLoadError(result.status);
      }
    } catch (error) {
      setLoadResult(null);
      setLoadError(errorMessage(error));
    } finally {
      setLoadBusy(false);
    }
  }

  async function handleAnalyze(): Promise<void> {
    setAnalyzeBusy(true);
    setAnalyzeError("");
    setInspectError("");
    try {
      const result = await engine.analyzePrompt(prompt);
      setAnalysis(result);
      if (result.ok) {
        const key = defaultLocation;
        setLocationKey(key);
        setTokenPosition(null);
        await runInspect(key, null, clipped);
      } else {
        setInspectResult(null);
        setLocationKey(null);
        setTokenPosition(null);
        setAnalyzeError(result.status);
      }
    } catch (error) {
      setAnalysis(null);
      setInspectResult(null);
      setAnalyzeError(errorMessage(error));
    } finally {
      setAnalyzeBusy(false);
    }
  }

  function handleSelectLocation(key: string): void {
    void runInspect(key, tokenPosition, clipped).catch(() => undefined);
  }

  function handleSelectToken(position: number): void {
    void runInspect(selectedLocationKey, position, clipped).catch(
      () => undefined,
    );
  }

  function handleToggleClip(): void {
    const next = !clipped;
    setClipped(next);
    void runInspect(selectedLocationKey, tokenPosition, next).catch(
      () => undefined,
    );
  }

  const inspectReady = inspectResult?.state === "ready";
  const inspectMessage =
    inspectError ||
    (inspectResult?.state === "awaiting" || inspectResult?.state === "error"
      ? inspectResult.message
      : "");

  const modelSummary = loadResult?.loaded ? loadResult.summary : null;

  return (
    <div className={visualExpanded ? "ct-page ct-page-expanded" : "ct-page"}>
      <header className="ct-header">
        <div className="ct-heading">
          <p className="ct-kicker">RESIDUAL STREAM / INSPECTION</p>
          <h1 className="ct-title">Circuit Tracer</h1>
          <p className="ct-subtitle">
            A visual workbench for tracing one-block language-model activations.
          </p>
        </div>
        <p className="ct-header-note">
          LOCAL ANALYSIS
          <br />
          NO TELEMETRY
        </p>
      </header>

      <div className="ct-console">
        <aside className="ct-rail">
          <section className="ct-panel">
            <h2 className="ct-section-label">01 / LOAD CHECKPOINT</h2>
            <label className="ct-field-label" htmlFor="checkpoint-path">
              Server path
            </label>
            <input
              id="checkpoint-path"
              className="ct-text-input"
              type="text"
              placeholder={"C:\\...\\checkpoint-v2"}
              value={checkpointPath}
              onChange={(event) => setCheckpointPath(event.target.value)}
            />
            <div className="ct-button-row">
              <button
                type="button"
                className="ct-button ct-button-primary"
                onClick={() => void handleLoad()}
                disabled={loadBusy || checkpointPath.trim() === ""}
              >
                Load model
              </button>
              {loadBusy && <span className="ct-busy">Loading…</span>}
            </div>
            <p
              className={`ct-status ${loadResult?.loaded ? "ct-status-ok" : "ct-status-error"}`}
              data-testid="load-status"
            >
              {loadError || loadResult?.status || "No model loaded. Enter an extracted checkpoint folder."}
            </p>

            <h2 className="ct-section-label">Runtime</h2>
            <dl className="ct-meta ct-meta-grid" data-testid="model-meta">
              <div>
                <dt>Compute device</dt>
                <dd>{loadResult?.device_label ?? "not loaded"}</dd>
              </div>
              {loadResult?.loaded && loadResult.meta && (
                <>
                  <div>
                    <dt>Checkpoint</dt>
                    <dd>{loadResult.meta.path}</dd>
                  </div>
                  <div>
                    <dt>Architecture</dt>
                    <dd>{loadResult.meta.architecture}</dd>
                  </div>
                  <div>
                    <dt>Vocabulary</dt>
                    <dd>
                      {formatCount(loadResult.meta.vocab_size)} tokens
                    </dd>
                  </div>
                  <div>
                    <dt>Maximum sequence</dt>
                    <dd>{formatCount(loadResult.meta.max_len)} tokens</dd>
                  </div>
                  <div>
                    <dt>Model width</dt>
                    <dd>{formatCount(loadResult.meta.embedding_dim)}</dd>
                  </div>
                  <div>
                    <dt>Attention</dt>
                    <dd>
                      {formatCount(loadResult.meta.num_heads)} heads x{" "}
                      {formatCount(loadResult.meta.key_dim)} key width
                    </dd>
                  </div>
                  <div>
                    <dt>FFN width</dt>
                    <dd>{formatCount(loadResult.meta.feed_forward_dim)}</dd>
                  </div>
                </>
              )}
            </dl>

            <details className="ct-accordion">
              <summary>Technical model summary</summary>
              {modelSummary ? (
                <pre className="ct-summary" data-testid="model-summary">
                  {modelSummary}
                </pre>
              ) : (
                <p className="ct-muted">No model loaded.</p>
              )}
            </details>
          </section>

          <section className="ct-panel">
            <h2 className="ct-section-label">02 / ANALYZE PROMPT</h2>
            <label className="ct-field-label" htmlFor="prompt">
              Prompt
            </label>
            <textarea
              id="prompt"
              className="ct-text-input ct-textarea"
              rows={5}
              placeholder="Enter a prompt to trace..."
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
            <div className="ct-button-row">
              <button
                type="button"
                className="ct-button ct-button-primary"
                onClick={() => void handleAnalyze()}
                disabled={analyzeBusy || prompt.trim() === ""}
              >
                Analyze prompt
              </button>
              {analyzeBusy && <span className="ct-busy">Analyzing…</span>}
            </div>
            <p
              className={`ct-status ${analysis?.ok ? "ct-status-ok" : analysis ? "ct-status-error" : ""}`}
              data-testid="analysis-status"
            >
              {analyzeError ||
                analysis?.status ||
                "Load a model, then enter a prompt."}
            </p>
            {analysis?.ok && (
              <p className="ct-meta" data-testid="analysis-count">
                Processed tokens: {analysis.token_count} of {analysis.max_len}
                {analysis.unknown_count ? (
                  <span className="ct-warning">
                    {" "}
                    &middot; contains {analysis.unknown_count} unknown
                    token(s), mapped to [UNK]
                  </span>
                ) : null}
              </p>
            )}
          </section>
        </aside>

        <main className="ct-workspace">
          <section className="ct-panel">
            <h2 className="ct-section-label">MODEL PATH</h2>
            <h3 className="ct-panel-title">The one-block causal language model</h3>
            <div className="ct-diagram-wrap">
              <ModelDiagram
                stages={loadResult?.loaded ? loadResult.stages : (options?.default_stages ?? [])}
                selectedKey={inspectReady && inspectResult?.location ? inspectResult.location.key : null}
                selectableKeys={selectableStages}
                onSelect={handleSelectLocation}
              />
            </div>
          </section>

          <section className="ct-panel">
            <h2 className="ct-section-label">03 / INSPECT CAPTURED STATES</h2>
            <div className="ct-control-row">
              <label className="ct-select-wrap">
                <span className="ct-field-label">Internal location</span>
                <select
                  className="ct-select"
                  value={selectedLocationKey}
                  onChange={(event) => handleSelectLocation(event.target.value)}
                  disabled={!analysis?.ok}
                >
                  {locations.map((item) => (
                    <option key={item.key} value={item.key}>
                      {item.category} · {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ct-select-wrap">
                <span className="ct-field-label">Token position</span>
                <select
                  className="ct-select"
                  value={tokenPosition ?? ""}
                  onChange={(event) =>
                    event.target.value === ""
                      ? undefined
                      : handleSelectToken(Number(event.target.value))
                  }
                  disabled={!analysis?.ok}
                >
                  {inspectResult?.token_choices.map((token) => (
                    <option key={token.position} value={token.position}>
                      {token.position}: {token.text}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ct-checkbox">
                <input
                  type="checkbox"
                  checked={clipped}
                  onChange={handleToggleClip}
                  disabled={!analysis?.ok}
                />
                Clip extremes
              </label>
              {inspectBusy && <span className="ct-busy">Rendering…</span>}
            </div>
            <div className="ct-inspect-text" data-testid="inspect-text">
              {inspectReady && inspectResult?.location ? (
                <>
                  <h3 className="ct-location-heading">
                    {inspectResult.location.category} ·{" "}
                    {inspectResult.location.label}
                  </h3>
                  <p>{inspectResult.location.explanation}</p>
                  {inspectResult.location.normalized && (
                    <p className="ct-muted">
                      Layer normalization fixes each token&apos;s feature scale,
                      so its token-magnitude chart may look nearly flat.
                    </p>
                  )}
                </>
              ) : (
                <p className="ct-muted">{inspectMessage || AWAITING_COPY}</p>
              )}
            </div>
            <div className="ct-two-up">
              <div className="ct-stat-box" data-testid="capture-stats">
                {inspectReady && inspectResult ? (
                  <dl className="ct-meta ct-meta-grid">
                    <div>
                      <dt>Shape</dt>
                      <dd>
                        {inspectResult.shape?.seq_len} ×{" "}
                        {inspectResult.shape?.width}
                      </dd>
                    </div>
                    <div>
                      <dt>Capture range</dt>
                      <dd>
                        min {inspectResult.capture?.min.toFixed(4)}, mean{" "}
                        {inspectResult.capture?.mean.toFixed(4)}, max{" "}
                        {inspectResult.capture?.max.toFixed(4)}
                      </dd>
                    </div>
                  </dl>
                ) : null}
              </div>
              <div className="ct-stat-box" data-testid="token-stats">
                {inspectReady && inspectResult?.selected_stats ? (
                  <dl className="ct-meta ct-meta-grid">
                    <div>
                      <dt>Selected token</dt>
                      <dd>
                        norm {inspectResult.selected_stats.norm.toFixed(4)}, mean{" "}
                        {inspectResult.selected_stats.mean.toFixed(4)}, std{" "}
                        {inspectResult.selected_stats.standard_deviation.toFixed(4)}
                      </dd>
                    </div>
                    <div>
                      <dt>Range</dt>
                      <dd>
                        min {inspectResult.selected_stats.minimum.toFixed(4)}, max{" "}
                        {inspectResult.selected_stats.maximum.toFixed(4)}
                      </dd>
                    </div>
                  </dl>
                ) : null}
              </div>
            </div>
          </section>

          {visualExpanded ? (
            <div className="ct-overlay" role="dialog" aria-label="Expanded visualizations">
              <Visuals
                inspect={inspectResult}
                expanded
                onToggleExpanded={() => setVisualExpanded(false)}
                onSelectPosition={(position) => handleSelectToken(position)}
              />
            </div>
          ) : (
            <Visuals
              inspect={inspectResult}
              expanded={false}
              onToggleExpanded={() => setVisualExpanded(true)}
              onSelectPosition={(position) => handleSelectToken(position)}
            />
          )}

          <section className="ct-panel">
            <h2 className="ct-section-label">PROMPT OUTPUT</h2>
            <h3 className="ct-subheading">Prompt tokens</h3>
            <div className="ct-table-wrap">
              <table className="ct-table" data-testid="token-table">
                <thead>
                  <tr>
                    <th>Position</th>
                    <th>Token</th>
                    <th>ID</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis?.tokens.map((token) => (
                    <tr key={token.position}>
                      <td>{token.position}</td>
                      <td>{token.text}</td>
                      <td>{token.token_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3 className="ct-subheading">Five most likely next tokens</h3>
            <div className="ct-table-wrap">
              <table className="ct-table" data-testid="next-token-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Token</th>
                    <th>ID</th>
                    <th>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis?.next_tokens.map((token) => (
                    <tr key={token.rank}>
                      <td>{token.rank}</td>
                      <td>{token.text}</td>
                      <td>{token.token_id}</td>
                      <td>{token.probability.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useState } from "react";
import * as engine from "./api/client";
import { CtApiError } from "./api/gradio";
import { NodeView } from "./components/NodeView";
import { ResidualGraph } from "./components/ResidualGraph";
import type {
  AnalyzePayload,
  GraphNode,
  InspectPayload,
  LoadPayload,
  OptionsPayload,
} from "./types";
import "./App.css";

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
  const [nodeKey, setNodeKey] = useState<string | null>(null);
  const [tokenPosition, setTokenPosition] = useState<number | null>(null);
  const [clipped, setClipped] = useState(false);
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    let cancelled = false;
    engine
      .getOptions()
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

  useEffect(() => {
    document.body.classList.toggle("ct-focused", focused);
    return () => document.body.classList.remove("ct-focused");
  }, [focused]);

  const graph = options?.graph ?? null;
  const nodeByKey = useMemo(() => {
    const map = new Map<string, GraphNode>();
    for (const node of graph?.nodes ?? []) map.set(node.key, node);
    return map;
  }, [graph]);

  const defaultNodeKey = graph?.default_node ?? "output_norm";
  const effectiveKey = nodeKey ?? defaultNodeKey;
  const graphNode = nodeByKey.get(effectiveKey) ?? null;

  const runInspect = useCallback(
    async (
      key: string | null,
      position: number | null,
      clip: boolean,
    ): Promise<InspectPayload> => {
      setInspectBusy(true);
      setInspectError("");
      try {
        const result = await engine.inspectNode(key, position, clip);
        setInspectResult(result);
        if (result.state === "ready" && result.node) {
          setNodeKey(result.node.key);
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
      setNodeKey(null);
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
        setNodeKey(null);
        setTokenPosition(null);
        await runInspect(null, null, clipped);
      } else {
        setInspectResult(null);
        setNodeKey(null);
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

  function handleSelectNode(key: string): void {
    if (key === (nodeKey ?? defaultNodeKey)) return;
    void runInspect(key, tokenPosition, clipped).catch(() => undefined);
  }

  function handleSelectToken(position: number): void {
    if (position === tokenPosition) return;
    void runInspect(effectiveKey, position, clipped).catch(() => undefined);
  }

  function handleStep(delta: number): void {
    if (!graphNode) return;
    const target = delta < 0 ? graphNode.prev_key : graphNode.next_key;
    if (target) void runInspect(target, tokenPosition, clipped);
  }

  function handleToggleClip(): void {
    const next = !clipped;
    setClipped(next);
    void runInspect(effectiveKey, tokenPosition, next).catch(() => undefined);
  }

  const inspectReady = inspectResult?.state === "ready";
  const inspectMessage =
    inspectError ||
    (inspectResult?.state === "awaiting" || inspectResult?.state === "error"
      ? inspectResult.message
      : "");

  const modelSummary = loadResult?.loaded ? loadResult.summary : null;
  const hasAnalysis = Boolean(analysis?.ok);

  return (
    <div className={focused ? "ct-page ct-page-focused" : "ct-page"}>
      <header className="ct-header">
        <div className="ct-heading">
          <p className="ct-kicker">RESIDUAL STREAM / ONE-BLOCK TRANSFORMER</p>
          <h1 className="ct-title">Circuit Tracer</h1>
          <p className="ct-subtitle">
            Watch the residual stream flow, and the writes that shape it.
          </p>
        </div>
        <p className="ct-header-note">
          LOCAL ANALYSIS
          <br />
          NO TELEMETRY
        </p>
      </header>

      <div className="ct-explorer">
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
                    <dt>Vocabulary</dt>
                    <dd>{formatCount(loadResult.meta.vocab_size)} tokens</dd>
                  </div>
                  <div>
                    <dt>Max sequence</dt>
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
                      {formatCount(loadResult.meta.key_dim)}
                    </dd>
                  </div>
                  <div>
                    <dt>FFN width</dt>
                    <dd>{formatCount(loadResult.meta.feed_forward_dim)}</dd>
                  </div>
                  <div>
                    <dt>Checkpoint</dt>
                    <dd className="ct-mono-small">{loadResult.meta.path}</dd>
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
              rows={4}
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
              <>
                <p className="ct-meta" data-testid="analysis-count">
                  Processed tokens: {analysis.token_count} of {analysis.max_len}
                  {analysis.unknown_count ? (
                    <span className="ct-warning">
                      {" "}
                      · contains {analysis.unknown_count} unknown token(s),
                      mapped to [UNK]
                    </span>
                  ) : null}
                </p>
                <h3 className="ct-subheading">Tokens</h3>
                <div className="ct-token-chips" data-testid="token-chips">
                  {analysis.tokens.map((token) => (
                    <button
                      key={token.position}
                      type="button"
                      className={`ct-token-chip ${
                        token.position === tokenPosition ? "ct-token-chip-selected" : ""
                      }`}
                      onClick={() => handleSelectToken(token.position)}
                      title="Select this token position"
                    >
                      <span className="ct-token-pos">{token.position}</span>
                      <span className="ct-token-text">{token.text}</span>
                    </button>
                  ))}
                </div>
                <h3 className="ct-subheading">Predicted next (last token)</h3>
                <div className="ct-next-inline" data-testid="analysis-next">
                  {analysis.next_tokens.slice(0, 5).map((next) => (
                    <span key={next.rank} className="ct-next-chip">
                      {next.rank}. {next.text} ({next.probability.toFixed(3)})
                    </span>
                  ))}
                </div>
              </>
            )}
          </section>
        </aside>

        <main className="ct-workspace">
          <section className="ct-panel">
            <div className="ct-panel-head">
              <div>
                <h2 className="ct-section-label">THE MODEL</h2>
                <h3 className="ct-panel-title">
                  One residual line, two parallel writes
                </h3>
              </div>
              <button
                type="button"
                className="ct-button ct-button-ghost"
                onClick={() => setFocused((value) => !value)}
              >
                {focused ? "Show controls" : "Focus view"}
              </button>
            </div>
            {graph ? (
              <ResidualGraph
                graph={graph}
                selectedKey={inspectReady && inspectResult?.node ? inspectResult.node.key : graphNode?.key ?? defaultNodeKey}
                enabled={hasAnalysis}
                onSelect={handleSelectNode}
              />
            ) : (
              <p className="ct-muted">Loading the model graph…</p>
            )}
            <p className="ct-muted ct-graph-note">
              Click any chip to see its captured tensor. Boxes above and below
              the line read from the residual stream and write back at the
              “add” junctions; “LN” marks the layer norms on the line.
            </p>
          </section>

          <section className="ct-panel ct-node-panel">
            <div className="ct-node-head">
              <div className="ct-node-title">
                <h2 className="ct-section-label">CAPTURED STATE</h2>
                {graphNode && (
                  <h3 className="ct-panel-title" data-testid="node-label">
                    {graphNode.label}
                  </h3>
                )}
              </div>
              <div className="ct-step-row">
                <button
                  type="button"
                  className="ct-button ct-button-ghost"
                  onClick={() => handleStep(-1)}
                  disabled={!hasAnalysis || !graphNode?.prev_key}
                  aria-label="Previous node in the stream"
                >
                  ◀ Previous
                </button>
                <span className="ct-trace-count">
                  {graphNode ? `${graphNode.trace_index + 1} / ${graphNode.trace_count}` : ""}
                </span>
                <button
                  type="button"
                  className="ct-button ct-button-ghost"
                  onClick={() => handleStep(1)}
                  disabled={!hasAnalysis || !graphNode?.next_key}
                  aria-label="Next node in the stream"
                >
                  Next ▶
                </button>
              </div>
            </div>

            {inspectReady && inspectResult?.node ? (
              <div data-testid="inspect-text">
                <p className="ct-explanation">{inspectResult.node.explanation}</p>
                {inspectResult.node.normalized && (
                  <p className="ct-muted">
                    Layer normalization rescales every token, so magnitude
                    comparisons across tokens are not meaningful here.
                  </p>
                )}
              </div>
            ) : (
              <p className="ct-muted" data-testid="inspect-awaiting">
                {inspectMessage || AWAITING_COPY}
              </p>
            )}

            <div className="ct-control-row">
              <label className="ct-select-wrap">
                <span className="ct-field-label">Node in the trace</span>
                <select
                  className="ct-select"
                  value={effectiveKey}
                  onChange={(event) => handleSelectNode(event.target.value)}
                  disabled={!hasAnalysis}
                  data-testid="node-select"
                >
                  {(graph?.trace ?? []).map((key) => {
                    const node = nodeByKey.get(key);
                    if (!node) return null;
                    return (
                      <option key={key} value={key}>
                        {node.label}
                      </option>
                    );
                  })}
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
                  disabled={!hasAnalysis}
                  data-testid="token-select"
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
                  disabled={!hasAnalysis}
                />
                Clip extremes
              </label>
              {inspectBusy && <span className="ct-busy">Rendering…</span>}
            </div>

            {inspectReady && inspectResult ? (
              <div className="ct-stat-box ct-stat-box-single" data-testid="capture-stats">
                {inspectResult.shape && (
                  <dl className="ct-meta ct-meta-grid">
                    <div>
                      <dt>Shape</dt>
                      <dd>
                        {inspectResult.shape.seq_len} ×{" "}
                        {inspectResult.shape.width}
                      </dd>
                    </div>
                    {inspectResult.capture && (
                      <div>
                        <dt>Capture range</dt>
                        <dd>
                          min {inspectResult.capture.min.toFixed(4)}, mean{" "}
                          {inspectResult.capture.mean.toFixed(4)}, max{" "}
                          {inspectResult.capture.max.toFixed(4)}
                        </dd>
                      </div>
                    )}
                  </dl>
                )}
              </div>
            ) : null}

            {inspectReady && inspectResult ? (
              <NodeView
                inspect={inspectResult}
                onSelectPosition={handleSelectToken}
              />
            ) : null}
          </section>
        </main>
      </div>
    </div>
  );
}

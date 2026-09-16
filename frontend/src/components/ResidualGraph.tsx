import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { GraphNode, StreamGraph } from "../types";

const MIN_VIEW_WIDTH = 1500;
const VIEW_HEIGHT = 570;
const SPINE_Y = 285;
const CHIP_W = 140;
const CHIP_H = 60;
const SPINE_X0 = 300;
const SPINE_STEP = 185;
const READOUT_W = 155;

const COMPONENT_CY = [225, 345];

const KIND_FILL: Record<string, string> = {
  component: "#e0e7ff",
  stream: "#fef3c7",
  update: "#dcfce7",
  ln: "#f1f5f9",
  hidden: "#fce7f3",
  pattern: "#dbeafe",
  readout: "#ede9fe",
};

const KIND_STROKE: Record<string, string> = {
  component: "#6366f1",
  stream: "#d97706",
  update: "#16a34a",
  ln: "#94a3b8",
  hidden: "#db2777",
  pattern: "#2563eb",
  readout: "#7c3aed",
};

const KIND_LEGEND: Array<{ kind: string; label: string }> = [
  { kind: "component", label: "Embeddings" },
  { kind: "stream", label: "Residual stream" },
  { kind: "update", label: "Writes" },
  { kind: "ln", label: "Layer norm" },
  { kind: "hidden", label: "FFN hidden" },
  { kind: "pattern", label: "Attention pattern" },
  { kind: "readout", label: "Readout" },
];

function wrapLines(text: string, width: number, size: number): string[] {
  const charsPerLine = Math.max(4, Math.floor(width / (size * 0.6)));
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= charsPerLine || !current) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  if (lines.length <= 2) return lines;

  const visible = lines.slice(0, 2);
  const lastLine = visible[1];
  visible[1] = `${lastLine.slice(0, Math.max(1, charsPerLine - 1)).trimEnd()}…`;
  return visible;
}

function diagramLabel(node: GraphNode): string {
  switch (node.stage) {
    case "attention_input_norm":
    case "ffn_input_norm":
      return "Layer norm";
    case "attention_pattern":
      return "Causal attention pattern";
    case "attention_update":
      return "Attention output";
    case "attention_residual":
      return "After attention";
    case "ffn_hidden":
      return "Hidden · ReLU";
    case "ffn_update":
      return "FFN output";
    case "ffn_residual":
      return "After FFN";
    default:
      break;
  }

  if (node.key === "output_norm") return "Final layer norm";
  if (node.key === "readout") return "Next-token probabilities";
  return node.label;
}

function linkKind(link: string): "add" | "ln" | "softmax" {
  if (link.endsWith("-add")) return "add";
  if (link === "layer-norm") return "ln";
  return "softmax";
}

interface ResidualGraphProps {
  graph: StreamGraph;
  selectedKey: string;
  enabled: boolean;
  onSelect: (key: string) => void;
}

export function ResidualGraph({
  graph,
  selectedKey,
  enabled,
  onSelect,
}: ResidualGraphProps) {
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const nodeByKey = useMemo(() => {
    const map = new Map<string, GraphNode>();
    for (const node of graph.nodes) map.set(node.key, node);
    return map;
  }, [graph]);

  const spineXs = useMemo(
    () => graph.spine.map((_key, index) => SPINE_X0 + index * SPINE_STEP),
    [graph],
  );
  const lastChipX = spineXs[spineXs.length - 1] ?? SPINE_X0;
  const readoutCx = lastChipX + SPINE_STEP;
  const viewWidth = Math.max(MIN_VIEW_WIDTH, readoutCx + READOUT_W / 2 + 5);

  useEffect(() => {
    const scrollContainer = scrollRef.current;
    if (!scrollContainer) return;

    const revealSelected = () => {
      const selectedNode = Array.from(
        scrollContainer.querySelectorAll<SVGGElement>("[data-node]"),
      ).find((element) => element.dataset.node === selectedKey);
      const svg = scrollContainer.querySelector("svg");
      const centerX = Number(selectedNode?.dataset.centerX);
      if (!selectedNode || !svg || !Number.isFinite(centerX)) return;

      const renderedWidth = svg.getBoundingClientRect().width || viewWidth;
      const maxScrollLeft = Math.max(0, renderedWidth - scrollContainer.clientWidth);
      const targetLeft = Math.min(
        maxScrollLeft,
        Math.max(
          0,
          centerX * (renderedWidth / viewWidth) - scrollContainer.clientWidth / 2,
        ),
      );
      if (typeof scrollContainer.scrollTo === "function") {
        const reduceMotion = window.matchMedia?.(
          "(prefers-reduced-motion: reduce)",
        ).matches;
        scrollContainer.scrollTo({
          left: targetLeft,
          behavior: reduceMotion ? "auto" : "smooth",
        });
      }
    };

    revealSelected();
    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(revealSelected);
    observer.observe(scrollContainer);
    return () => observer.disconnect();
  }, [selectedKey, viewWidth]);

  function spineIndex(key: string): number {
    const index = graph.spine.indexOf(key);
    return index >= 0 ? index : -1;
  }

  function renderChip(
    key: string,
    cx: number,
    cy: number,
    width: number,
    height: number,
    node: GraphNode,
    fontSize: number,
  ): ReactNode {
    const selectable = enabled;
    const selected = key === selectedKey;
    const hovered = key === hoverKey;
    const x = cx - width / 2;
    const y = cy - height / 2;
    const displayLabel = diagramLabel(node);
    const lines = wrapLines(displayLabel, width - 14, fontSize);
    const lineCount = Math.max(1, lines.length);
    const fill = KIND_FILL[node.kind] ?? "#f8fafc";
    const stroke = selected
      ? "#1d4ed8"
      : hovered
        ? "#334155"
        : (KIND_STROKE[node.kind] ?? "#cbd5e1");
    return (
      <g
        key={key}
        role="button"
        aria-label={
          displayLabel === node.label ? node.label : `${displayLabel}: ${node.label}`
        }
        tabIndex={selectable ? 0 : -1}
        data-node={key}
        data-center-x={cx}
        data-selected={selected || undefined}
        opacity={selectable ? 1 : 0.55}
        onMouseEnter={() => setHoverKey(key)}
        onMouseLeave={() => setHoverKey(null)}
        onClick={() => selectable && onSelect(key)}
        onKeyDown={(event) => {
          if (selectable && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            onSelect(key);
          }
        }}
        style={{ cursor: selectable ? "pointer" : "default" }}
      >
        <title>{node.label}</title>
        {selected ? (
          <rect
            x={x - 3}
            y={y - 3}
            width={width + 6}
            height={height + 6}
            rx={11}
            fill="none"
            stroke="#93c5fd"
            strokeWidth={5}
            opacity={0.55}
          />
        ) : null}
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          rx={9}
          fill={fill}
          stroke={stroke}
          strokeWidth={selected ? 3 : hovered ? 2 : 1.5}
        />
        {lines.map((line, index) => {
          const lineY = cy + (index - (lineCount - 1) / 2) * (fontSize + 3);
          return (
            <text
              key={`${line}-${index}`}
              x={cx}
              y={lineY}
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight={selected ? 700 : 500}
              fill="#1e293b"
            >
              {line}
            </text>
          );
        })}
      </g>
    );
  }

  function renderArrow(x1: number, y1: number, x2: number, y2: number, key?: string) {
    return (
      <line
        key={key}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#94a3b8"
        strokeWidth={1.5}
        markerEnd="url(#ct-arrowhead)"
      />
    );
  }

  function renderArrowLabel(
    x: number,
    y: number,
    text: string,
    anchor: "middle" | "start" | "end" = "middle",
  ) {
    return (
      <text
        x={x}
        y={y}
        textAnchor={anchor}
        fontSize={10}
        fontWeight={600}
        fill="#64748b"
      >
        {text}
      </text>
    );
  }

  const items: ReactNode[] = [];
  items.push(
    <defs key="defs">
      <marker
        id="ct-arrowhead"
        markerWidth="8"
        markerHeight="8"
        refX="6"
        refY="3"
        orient="auto"
      >
        <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
      </marker>
    </defs>,
  );

  items.push(
    <line
      key="spine-line"
      x1={185}
      y1={SPINE_Y}
      x2={lastChipX + CHIP_W / 2}
      y2={SPINE_Y}
      stroke="#334155"
      strokeWidth={3}
    />,
  );

  // The decomposed embeddings feed the stream input chip on the left.
  graph.components.forEach((key, index) => {
    const node = nodeByKey.get(key);
    if (!node || index >= COMPONENT_CY.length) return;
    const cx = 120;
    const cy = COMPONENT_CY[index];
    items.push(renderChip(key, cx, cy, 130, 42, node, 10));
    items.push(
      renderArrow(
        cx + 65,
        cy,
        spineXs[0] - CHIP_W / 2,
        SPINE_Y,
        `component-arrow-${key}`,
      ),
    );
  });

  // The stream-state chips along the central line.
  graph.spine.forEach((key, index) => {
    const node = nodeByKey.get(key);
    if (!node) return;
    items.push(renderChip(key, spineXs[index], SPINE_Y, CHIP_W, CHIP_H, node, 11));
  });

  // Operation markers between residual states are add junctions. The final
  // output norm and softmax remain on the main path after the blocks.
  const renderOperationPill = (
    index: number,
    midX: number,
    kind: "add" | "ln" | "softmax",
  ) => {
    if (kind === "add") {
      const radius = 6.5;
      return (
        <g key={`pill-${index}`}>
          <circle
            cx={midX}
            cy={SPINE_Y}
            r={radius}
            fill="#111827"
            stroke="#111827"
            strokeWidth={1}
          />
          <line
            x1={midX - 4}
            y1={SPINE_Y}
            x2={midX + 4}
            y2={SPINE_Y}
            stroke="#ffffff"
            strokeWidth={1.6}
          />
          <line
            x1={midX}
            y1={SPINE_Y - 4}
            x2={midX}
            y2={SPINE_Y + 4}
            stroke="#ffffff"
            strokeWidth={1.6}
          />
          <text
            x={midX}
            y={SPINE_Y + 18}
            textAnchor="middle"
            fontSize={10}
            fontWeight={600}
            fill="#334155"
          >
            add
          </text>
        </g>
      );
    }
    const label = kind === "ln" ? "LN" : "softmax";
    const pillW = kind === "softmax" ? 58 : 44;
    const pillY = SPINE_Y - 58;
    return (
      <g key={`pill-${index}`}>
        <rect
          x={midX - pillW / 2}
          y={pillY}
          width={pillW}
          height={20}
          rx={6}
          fill="#e2e8f0"
          stroke="#94a3b8"
          strokeWidth={1}
        />
        <text
          x={midX}
          y={pillY + 13.5}
          textAnchor="middle"
          fontSize={9.5}
          fontWeight={700}
          fill="#334155"
        >
          {label}
        </text>
        <line
          x1={midX}
          y1={pillY + 20}
          x2={midX}
          y2={SPINE_Y - 2}
          stroke="#94a3b8"
          strokeWidth={1.2}
          strokeDasharray="2 2"
        />
      </g>
    );
  };

  // Pre-norm branches read a raw residual state through their first path node,
  // then write back at an add junction while the residual line bypasses them.
  graph.branches.forEach((branch) => {
    const readIndex = spineIndex(branch.reads);
    const addIndex = spineIndex(branch.adds_before) - 1;
    if (readIndex < 0 || addIndex < 0) return;
    const readX = spineXs[readIndex];
    const addX = (spineXs[addIndex] + spineXs[addIndex + 1]) / 2;
    const isAttention = branch.kind === "attention";
    const isAbove = branch.side === "above";
    const containerY = isAbove ? 40 : 365;
    const containerH = isAttention ? 150 : 158;
    const centerX = (readX + addX) / 2;
    const containerW = isAttention ? 330 : 280;
    const containerLeft = centerX - containerW / 2;
    const innerW = containerW - 40;
    const pathNodes = branch.path
      .map((key) => nodeByKey.get(key))
      .filter((node): node is GraphNode => node !== undefined);
    const observableNodes = branch.observables
      .map((key) => nodeByKey.get(key))
      .filter((node): node is GraphNode => node !== undefined);
    const pathX = observableNodes.length ? centerX - 62 : centerX;
    const pathWidth = observableNodes.length ? 165 : innerW;
    const firstPathY = containerY + 36;
    const lastPathY = firstPathY + (pathNodes.length - 1) * 44;

    items.push(
      <g key={`branch-${branch.key}`} data-branch={branch.key}>
        <rect
          x={containerLeft}
          y={containerY}
          width={containerW}
          height={containerH}
          rx={12}
          fill={isAttention ? "#f8fbff" : "#fffafd"}
          stroke={isAttention ? "#bfdbfe" : "#e5d5e1"}
          strokeWidth={1.3}
        />
        <title>{branch.label}</title>
        <text
          x={centerX}
          y={isAbove ? containerY - 9 : containerY + containerH + 19}
          textAnchor="middle"
          fontSize={10.5}
          fontWeight={700}
          fill="#475569"
          letterSpacing="0.08em"
        >
          {`BLOCK ${branch.block_index + 1} · ${isAttention ? "ATTENTION" : "FEED-FORWARD"}`}
        </text>
        {pathNodes.slice(0, -1).map((node, pathIndex) => (
          <g key={`path-arrow-${node.key}`}>
            {renderArrow(
              pathX,
              containerY + 54 + pathIndex * 44,
              pathX,
              containerY + 62 + pathIndex * 44,
            )}
          </g>
        ))}
        {pathNodes.map((node, childIndex) =>
          renderChip(
            node.key,
            pathX,
            containerY + 36 + childIndex * 44,
            pathWidth,
            36,
            node,
            10,
          ),
        )}
        {observableNodes.map((node, observableIndex) => (
          <g key={`observable-${node.key}`}>
            <line
              x1={pathX + pathWidth / 2}
              y1={containerY + 80}
              x2={centerX + 57}
              y2={containerY + 80 + observableIndex * 42}
              stroke="#94a3b8"
              strokeWidth={1.2}
              strokeDasharray="3 3"
            />
            {renderChip(
              node.key,
              centerX + 104,
              containerY + 80 + observableIndex * 42,
              94,
              36,
              node,
              8.5,
            )}
          </g>
        ))}
        {isAbove
          ? renderArrow(readX, SPINE_Y - CHIP_H / 2, pathX, firstPathY + 20)
          : renderArrow(readX, SPINE_Y + CHIP_H / 2, pathX, firstPathY - 20)}
        {isAbove
          ? renderArrowLabel(readX - 10, 235, "reads", "end")
          : renderArrowLabel(readX - 10, 348, "reads", "end")}
        {renderArrow(pathX, isAbove ? lastPathY + 20 : lastPathY - 20, addX, SPINE_Y)}
        {isAbove
          ? renderArrowLabel(addX - 10, 246, "writes", "end")
          : renderArrowLabel(addX + 10, 335, "writes", "start")}
      </g>,
    );
  });

  // Operation markers between the stream-state chips. Drawn after the
  // branches so the dark "add" taps sit on top of the write-arrow tips.
  graph.spine.forEach((_key, index) => {
    if (index >= spineXs.length - 1) return;
    const link = graph.spine_links[index];
    if (!link) return;
    const midX = (spineXs[index] + spineXs[index + 1]) / 2;
    items.push(renderOperationPill(index, midX, linkKind(link)));
  });

  // Readout chip at the end of the line, reached by its own arrow.
  const readout = nodeByKey.get("readout");
  if (readout) {
    const readoutLeft = readoutCx - READOUT_W / 2;
    const arrowStart = lastChipX + CHIP_W / 2 + 2;
    const arrowEnd = readoutLeft - 3;
    items.push(renderArrow(arrowStart, SPINE_Y, arrowEnd, SPINE_Y, "readout-arrow"));
    items.push(
      renderOperationPill(
        graph.spine.length - 1,
        (arrowStart + arrowEnd) / 2 + 8,
        linkKind(graph.spine_links[graph.spine.length - 1] ?? "readout"),
      ),
    );
    items.push(
      renderChip("readout", readoutCx, SPINE_Y, READOUT_W, CHIP_H, readout, 10),
    );
  }

  const legendEntries = KIND_LEGEND.filter((entry) =>
    graph.nodes.some((node) => node.kind === entry.kind),
  );

  return (
    <>
      <div className="ct-graph-frame">
        <div
          ref={scrollRef}
          className="ct-graph-scroll"
          tabIndex={0}
          aria-label="Scrollable residual stream wiring diagram"
        >
          <svg
            className="ct-graph-svg"
            width={viewWidth}
            height={VIEW_HEIGHT}
            viewBox={`0 0 ${viewWidth} ${VIEW_HEIGHT}`}
            data-testid="residual-graph"
            role="group"
            aria-label="Residual stream wiring diagram"
          >
            {items}
          </svg>
        </div>
      </div>
      <p className="ct-graph-hint">Scroll horizontally to follow the model flow.</p>
      <div className="ct-graph-legend" data-testid="graph-legend">
        {legendEntries.map((entry) => (
          <span key={entry.kind} className="ct-graph-legend-item">
            <span
              className="ct-graph-legend-dot"
              style={{
                background: KIND_FILL[entry.kind] ?? "#f8fafc",
                borderColor: KIND_STROKE[entry.kind] ?? "#cbd5e1",
              }}
            />
            {entry.label}
          </span>
        ))}
        <span className="ct-graph-legend-item">
          <span className="ct-graph-legend-dot ct-graph-legend-selected" />
          selected node
        </span>
      </div>
    </>
  );
}

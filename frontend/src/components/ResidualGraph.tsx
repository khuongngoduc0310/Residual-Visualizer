import { useMemo } from "react";
import type { ReactNode } from "react";
import type { GraphNode, StreamGraph } from "../types";

const VIEW_WIDTH = 1500;
const VIEW_HEIGHT = 440;
const SPINE_Y = 235;
const CHIP_W = 170;
const CHIP_H = 66;

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

const LINK_BADGE: Record<string, { text: string; kind: "add" | "ln" }> = {
  "attention-add": { text: "attn \u2295", kind: "add" },
  "layer-norm": { text: "LN", kind: "ln" },
  "ffn-add": { text: "ffn \u2295", kind: "add" },
};

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
  return lines.slice(0, 2);
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
  const nodeByKey = useMemo(() => {
    const map = new Map<string, GraphNode>();
    for (const node of graph.nodes) map.set(node.key, node);
    return map;
  }, [graph]);

  const layout = useMemo(() => {
    const spineXs = graph.spine.map(
      (_key, index) => 300 + index * 210,
    );
    const readoutX = 300 + graph.spine.length * 210 + 70;
    return { spineXs, readoutX };
  }, [graph]);

  const { spineXs, readoutX } = layout;

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
    const x = cx - width / 2;
    const y = cy - height / 2;
    const lines = wrapLines(node.label, width - 14, fontSize);
    const fill = KIND_FILL[node.kind] ?? "#f8fafc";
    const stroke = selected ? "#dc2626" : KIND_STROKE[node.kind] ?? "#cbd5e1";
    const lineCount = Math.max(1, lines.length);
    return (
      <g
        role="button"
        aria-label={node.label}
        tabIndex={selectable ? 0 : -1}
        data-node={key}
        data-selected={selected || undefined}
        opacity={selectable ? 1 : 0.55}
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
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          rx={9}
          fill={fill}
          stroke={stroke}
          strokeWidth={selected ? 3 : 1.5}
        />
        {lines.map((line, index) => {
          const lineY = cy + (index - (lineCount - 1) / 2) * (fontSize + 3);
          return (
            <text
              key={line}
              x={cx}
              y={lineY}
              textAnchor="middle"
              fontSize={fontSize}
              fontWeight={selected ? 600 : 500}
              fill="#1e293b"
            >
              {line}
            </text>
          );
        })}
      </g>
    );
  }

  function renderArrow(x1: number, y1: number, x2: number, y2: number) {
    const markerEnd = "url(#ct-arrowhead)";
    return (
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#94a3b8"
        strokeWidth={1.5}
        markerEnd={markerEnd}
      />
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

  // The residual stream line itself.
  items.push(
    <line
      key="spine-line"
      x1={60}
      y1={SPINE_Y}
      x2={spineXs[spineXs.length - 1] + CHIP_W / 2 + 20}
      y2={SPINE_Y}
      stroke="#334155"
      strokeWidth={3}
    />,
  );

  // Link badges (attention/FFN add junctions and layer norms) between spine
  // chips, plus the final readout arrow.
  graph.spine.forEach((key, index) => {
    const node = nodeByKey.get(key);
    if (!node) return;
    const cx = spineXs[index];
    items.push(
      renderChip(key, cx, SPINE_Y, CHIP_W, CHIP_H, node, 11),
    );

    if (index >= graph.spine_links.length) return;
    const link = graph.spine_links[index];
    if (index < spineXs.length - 1) {
      const midX = (spineXs[index] + spineXs[index + 1]) / 2;
      const badge = LINK_BADGE[link];
      if (badge) {
        const text = badge.kind === "add" ? "add" : "LN";
        items.push(
          <g key={`link-${index}`}>
            <rect
              x={midX - 18}
              y={SPINE_Y - 12}
              width={36}
              height={24}
              rx={6}
              fill={badge.kind === "add" ? "#1e293b" : "#e2e8f0"}
              stroke={badge.kind === "add" ? "#1e293b" : "#cbd5e1"}
              strokeWidth={1}
            />
            <text
              x={midX}
              y={SPINE_Y + 4}
              textAnchor="middle"
              fontSize={10}
              fontWeight={700}
              fill={badge.kind === "add" ? "#ffffff" : "#334155"}
            >
              {text}
            </text>
          </g>,
        );
      }
    }
  });

  // Readout chip at the end of the line.
  const readout = nodeByKey.get("readout");
  if (readout) {
    const lastX = spineXs[spineXs.length - 1];
    items.push(renderArrow(lastX + CHIP_W / 2, SPINE_Y, readoutX - 75, SPINE_Y));
    items.push(
      renderChip("readout", readoutX, SPINE_Y, 150, CHIP_H, readout, 10),
    );
  }

  // Components converge on the stream input.
  graph.components.forEach((key, index) => {
    const node = nodeByKey.get(key);
    if (!node) return;
    const cy = 145 + index * 115;
    const cx = 130;
    items.push(renderChip(key, cx, cy, 150, 44, node, 10));
    items.push(
      renderArrow(
        cx + 75,
        cy,
        spineXs[0] - CHIP_W / 2,
        SPINE_Y,
      ),
    );
  });

  // Branch containers: attention reads the stream early, the FFN reads it
  // after the first layer norm. Each writes back at its add junction.
  graph.branches.forEach((branch) => {
    const readIndex = spineIndex(branch.reads);
    const addIndex = spineIndex(branch.adds_before) - 1;
    if (readIndex < 0 || addIndex < 0) return;
    const readX = spineXs[readIndex];
    const addX = (spineXs[addIndex] + spineXs[addIndex + 1]) / 2;
    const width = Math.max(220, Math.abs(addX - readX) + 60);
    const left = readX + (addX - readX) / 2 - width / 2;
    const isAttention = branch.key === "attention";
    const top = isAttention ? 62 : 305;
    const containerH = 118;
    const children = branch.nodes
      .map((key) => nodeByKey.get(key))
      .filter((node): node is GraphNode => node !== undefined);

    items.push(
      <g key={`branch-${branch.key}`}>
        <rect
          x={left}
          y={top}
          width={width}
          height={containerH}
          rx={10}
          fill="#f8fafc"
          stroke="#cbd5e1"
          strokeDasharray="4 3"
          strokeWidth={1.2}
        />
        <text
          x={left + width / 2}
          y={top - 8}
          textAnchor="middle"
          fontSize={10.5}
          fontWeight={600}
          fill="#64748b"
        >
          {branch.label}
        </text>
        {children.map((node, childIndex) =>
          renderChip(
            node.key,
            left + width / 2,
            top + 26 + childIndex * 40,
            width - 24,
            32,
            node,
            9.5,
          ),
        )}
        {isAttention ? (
          renderArrow(
            readX,
            SPINE_Y - CHIP_H / 2,
            readX,
            top + containerH,
          )
        ) : (
          renderArrow(readX, SPINE_Y + CHIP_H / 2, readX, top)
        )}
        {renderArrow(left + width / 2, top + containerH, addX, SPINE_Y - 12)}
      </g>,
    );
  });

  return (
    <div className="ct-graph-scroll">
      <svg
        className="ct-graph-svg"
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        data-testid="residual-graph"
        role="group"
        aria-label="Residual stream wiring diagram"
      >
        {items}
      </svg>
    </div>
  );
}


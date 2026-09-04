import type { Stage } from "../types";

const CATEGORY_CLASS: Record<string, string> = {
  Attention: "ct-cat-attention",
  FFN: "ct-cat-ffn",
  "Residual Stream": "ct-cat-residual",
  Output: "ct-cat-output",
};

interface ModelDiagramProps {
  stages: Stage[];
  selectedKey: string | null;
  selectableKeys: Set<string>;
  onSelect?: (key: string) => void;
}

export function ModelDiagram({
  stages,
  selectedKey,
  selectableKeys,
  onSelect,
}: ModelDiagramProps) {
  const items: React.ReactNode[] = [];
  stages.forEach((stage, index) => {
    const selectable = selectableKeys.has(stage.key);
    const classes = [
      "ct-stage",
      CATEGORY_CLASS[stage.category] ?? "",
      stage.key === selectedKey ? "ct-selected" : "",
      selectable ? "ct-stage-selectable" : "",
    ]
      .filter(Boolean)
      .join(" ");
    items.push(
      <button
        key={stage.key}
        type="button"
        className={classes}
        data-stage={stage.key}
        disabled={!selectable}
        title={selectable ? "Inspect this location" : undefined}
        onClick={() => selectable && onSelect?.(stage.key)}
      >
        <span className="ct-stage-name">{stage.name}</span>
        <span className="ct-stage-detail">{stage.detail ?? "awaiting checkpoint"}</span>
      </button>,
    );
    if (index < stages.length - 1) {
      items.push(
        <span key={`${stage.key}-arrow`} className="ct-arrow" aria-hidden="true">
          &rarr;
        </span>,
      );
    }
  });
  return <div className="ct-diagram">{items}</div>;
}

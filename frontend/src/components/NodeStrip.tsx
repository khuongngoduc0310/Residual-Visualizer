export interface NodeStripItem {
  key: string;
  label: string;
}

interface NodeStripProps {
  items: NodeStripItem[];
  selectedKey: string | null;
  disabled?: boolean;
  onSelect: (key: string) => void;
}

export function NodeStrip({
  items,
  selectedKey,
  disabled = false,
  onSelect,
}: NodeStripProps) {
  if (items.length === 0) return null;
  return (
    <div
      className="ct-node-chips"
      data-testid="node-strip"
      role="group"
      aria-label="Choose a node in the trace"
    >
      {items.map((item) => {
        const selected = item.key === selectedKey;
        return (
          <button
            key={item.key}
            type="button"
            data-node={item.key}
            disabled={disabled}
            aria-pressed={selected}
            title={selected ? `${item.label} (shown)` : `Inspect ${item.label}`}
            className={`ct-node-chip ${selected ? "ct-node-chip-selected" : ""}`}
            onClick={() => onSelect(item.key)}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

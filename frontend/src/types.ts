export type NodeKind =
  | "component"
  | "stream"
  | "update"
  | "ln"
  | "hidden"
  | "pattern"
  | "readout";

export type NodeFamily =
  | "components"
  | "stream_raw"
  | "updates"
  | "stream_norm"
  | "hidden"
  | "pattern"
  | "readout";

export interface GraphNode {
  key: string;
  label: string;
  kind: NodeKind;
  family: NodeFamily;
  explanation: string;
  normalized: boolean;
  feature_axis: boolean;
  trace_index: number;
  trace_count: number;
  prev_key: string | null;
  next_key: string | null;
}

export interface GraphBranch {
  key: string;
  label: string;
  reads: string;
  adds_before: string;
  nodes: string[];
}

export interface StreamGraph {
  nodes: GraphNode[];
  spine: string[];
  spine_links: string[];
  branches: GraphBranch[];
  components: string[];
  trace: string[];
  default_node: string;
}

export interface LocationLite {
  key: string;
  label: string;
  kind: NodeKind;
  family: NodeFamily;
}

export interface OptionsPayload {
  graph: StreamGraph;
  locations: LocationLite[];
}

export interface ModelMeta {
  architecture: string | null;
  path: string | null;
  vocab_size: number | null;
  max_len: number | null;
  embedding_dim: number | null;
  num_heads: number | null;
  key_dim: number | null;
  feed_forward_dim: number | null;
  dropout_rate: number | null;
}

export interface LoadPayload {
  ok: boolean;
  status: string;
  loaded: boolean;
  meta: ModelMeta;
  device_label: string | null;
  summary: string | null;
}

export interface TokenRow {
  position: number;
  text: string;
  token_id: number;
}

export interface NextTokenRow {
  rank: number;
  text: string;
  token_id: number;
  probability: number;
}

export interface AnalyzePayload {
  ok: boolean;
  status: string;
  token_count: number | null;
  max_len: number | null;
  unknown_count: number | null;
  tokens: TokenRow[];
  next_tokens: NextTokenRow[];
}

export interface Shape {
  seq_len: number;
  width: number;
}

export interface CaptureStats {
  min: number;
  mean: number;
  max: number;
}

export interface ScaleInfo {
  lower: number;
  upper: number;
  clipped: boolean;
  family: NodeFamily;
  source: "family" | "matrix";
}

export interface FigureSpec {
  data: unknown[];
  layout: Record<string, unknown>;
}

export interface TokenChoice {
  position: number;
  text: string;
}

export type FigureKind =
  | "activation"
  | "hidden"
  | "pattern"
  | "readout_topk"
  | null;

export interface InspectPayload {
  ok: boolean;
  state: "awaiting" | "ready" | "error";
  message: string;
  node: GraphNode | null;
  selected_position: number | null;
  token_choices: TokenChoice[];
  shape: Shape | null;
  capture: CaptureStats | null;
  scale: ScaleInfo | null;
  tile: { rows: number; cols: number } | null;
  figure_kind: FigureKind;
  map_figure: FigureSpec | null;
  pattern_figure: FigureSpec | null;
  readout_figure: FigureSpec | null;
  entropy_figure: FigureSpec | null;
  readout_rows: NextTokenRow[];
}

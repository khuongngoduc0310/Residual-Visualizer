export interface Stage {
  key: string;
  name: string;
  category: string;
  detail: string | null;
}

export interface LocationOption {
  key: string;
  label: string;
  category: string;
  explanation: string;
  normalized: boolean;
}

export interface OptionsPayload {
  locations: LocationOption[];
  default_location: string;
  default_stages: Stage[];
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
  stages: Stage[];
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

export interface LocationInfo {
  key: string;
  label: string;
  category: string;
  explanation: string;
  normalized: boolean;
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

export interface SelectedStats {
  norm: number;
  mean: number;
  standard_deviation: number;
  minimum: number;
  maximum: number;
}

export interface HeatmapRange {
  lower: number;
  upper: number;
  clipped: boolean;
}

export interface FigureSpec {
  data: unknown[];
  layout: Record<string, unknown>;
}

export interface TokenChoice {
  position: number;
  text: string;
}

export interface InspectPayload {
  ok: boolean;
  state: "awaiting" | "ready" | "error";
  message: string;
  location: LocationInfo | null;
  selected_position: number | null;
  token_choices: TokenChoice[];
  shape: Shape | null;
  capture: CaptureStats | null;
  selected_stats: SelectedStats | null;
  heatmap: HeatmapRange | null;
  magnitude: FigureSpec | null;
  heatmap_figure: FigureSpec | null;
  distribution: FigureSpec | null;
}

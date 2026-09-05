import { callApi } from "./gradio";
import type {
  AnalyzePayload,
  AblationOperationPayload,
  InspectPayload,
  InspectView,
  LoadPayload,
  OptionsPayload,
} from "../types";

export function getOptions(): Promise<OptionsPayload> {
  return callApi("options", [""]);
}

export function loadCheckpoint(path: string): Promise<LoadPayload> {
  return callApi("load_checkpoint", [path]);
}

export function analyzePrompt(prompt: string): Promise<AnalyzePayload> {
  return callApi("analyze_prompt", [prompt]);
}

export function ablateFeature(
  nodeKey: string,
  dim: number,
  mode: "zero" | "mean",
  scope: "token" | "all",
  position: number | null,
): Promise<AblationOperationPayload> {
  return callApi("ablate_feature", [nodeKey, dim, mode, scope, position]);
}

export function clearAblation(): Promise<{ ok: boolean; status: string }> {
  return callApi("clear_ablation", []);
}

export function inspectNode(
  nodeKey: string | null,
  tokenPosition: number | null,
  view: InspectView = "baseline",
  highlightToken: string | null = null,
): Promise<InspectPayload> {
  return callApi("inspect_node", [
    nodeKey,
    tokenPosition,
    view,
    highlightToken,
  ]);
}

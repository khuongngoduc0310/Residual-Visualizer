import { callApi } from "./gradio";
import type {
  AnalyzePayload,
  InspectPayload,
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

export function inspectLocation(
  locationKey: string | null,
  tokenPosition: number | null,
  clipped: boolean,
): Promise<InspectPayload> {
  return callApi("inspect_location", [locationKey, tokenPosition, clipped]);
}

export class CtApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CtApiError";
  }
}

const GRADIO_BASE =
  import.meta.env.VITE_GRADIO_BASE ?? "/gradio";

function apiUrl(apiName: string): string {
  return `${GRADIO_BASE}/gradio_api/call/${apiName}`;
}

function parseSseResult(body: string): unknown[] {
  let result: unknown[] | null = null;
  for (const line of body.split(/\r?\n/)) {
    if (!line.startsWith("data: ")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line.slice("data: ".length));
    } catch {
      continue;
    }
    if (Array.isArray(parsed) && parsed.length > 0) {
      result = parsed;
    }
  }
  if (result === null) {
    throw new CtApiError("The server returned no result for this request.");
  }
  return result;
}

async function readEventResult(response: Response): Promise<unknown[]> {
  const body = await response.text();
  if (!response.ok) {
    const detail = readErrorDetail(body);
    throw new CtApiError(detail ?? `Request failed (${response.status}).`);
  }
  return parseSseResult(body);
}

function readErrorDetail(body: string): string | null {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (parsed.detail && typeof parsed.detail === "object") {
      const message = (parsed.detail as { message?: unknown }).message;
      if (typeof message === "string") return message;
    }
  } catch {
    // Not JSON; fall through to the generic message.
  }
  return null;
}

export async function callApi<Result>(
  apiName: string,
  args: unknown[],
): Promise<Result> {
  const postResponse = await fetch(apiUrl(apiName), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: args }),
  });
  const bodyText = await postResponse.text();
  let payload: { event_id?: string; detail?: unknown } = {};
  try {
    payload = JSON.parse(bodyText) as { event_id?: string; detail?: unknown };
  } catch {
    // Non-JSON rejection body; handled by the generic message below.
  }
  if (!postResponse.ok) {
    const detail = payload.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail && typeof detail === "object" &&
            typeof (detail as { message?: unknown }).message === "string"
          ? ((detail as { message: string }).message)
          : `The request was rejected by the server (${postResponse.status}).`;
    throw new CtApiError(message);
  }
  if (!payload.event_id) {
    throw new CtApiError("The request was rejected by the server.");
  }
  const getResponse = await fetch(`${apiUrl(apiName)}/${payload.event_id}`);
  const outputs = await readEventResult(getResponse);
  if (outputs.length === 0) {
    throw new CtApiError("The server returned no outputs.");
  }
  return outputs[0] as Result;
}

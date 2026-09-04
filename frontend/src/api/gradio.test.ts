import { afterEach, describe, expect, it, vi } from "vitest";
import { callApi, CtApiError } from "./gradio";

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function sseResponse(status: number, text: string): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => JSON.parse(text),
    text: async () => text,
  } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("callApi", () => {
  it("posts positional args and unwraps the single output payload", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(200, { event_id: "evt-1" }))
      .mockResolvedValueOnce(
        sseResponse(200, 'event: complete\ndata: [{"ok":true}]\n\n'),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await callApi<{ ok: boolean }>("options", [""]);

    expect(result).toEqual({ ok: true });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/gradio/gradio_api/call/options",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ data: [""] }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/gradio/gradio_api/call/options/evt-1",
    );
  });

  it("surfaces server rejections", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(
      jsonResponse(422, { detail: "invalid input" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(callApi("options", ["x"])).rejects.toThrow(
      new CtApiError("invalid input"),
    );
  });

  it("throws when the server never yields an output", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(200, { event_id: "evt-2" }))
      .mockResolvedValueOnce(sseResponse(200, "event: heartbeat\ndata: {}\n\n"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(callApi("options", [""])).rejects.toThrow(
      new CtApiError("The server returned no result for this request."),
    );
  });
});

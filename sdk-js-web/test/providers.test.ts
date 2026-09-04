import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Tests for what an `OrtSession` says about where it is going to run.
 *
 * ORT-Web falls back silently: a page that asks for `webgpu` on a device without
 * it runs on WASM and is told nothing, several times slower than intended. There
 * is no `getProviders()` in the browser to reconcile against, so the SDK narrows
 * the requested list by what the browser can actually offer — which is what
 * these tests drive, by handing it a `navigator` with and without WebGPU.
 *
 * Mirrors the Python SDK's `tests/test_session_providers.py`, where the same
 * question is answered by reading ORT's own effective list.
 */

class FakeTensor {
  constructor(
    public readonly type: string,
    public readonly data: Float32Array,
    public readonly dims: number[],
  ) {}
}

vi.mock("onnxruntime-web", () => ({
  Tensor: FakeTensor,
  InferenceSession: {
    create: vi.fn(() =>
      Promise.resolve({
        inputNames: ["images"],
        outputNames: ["logits"],
        inputMetadata: undefined,
        outputMetadata: undefined,
        run: () => Promise.resolve({}),
        release: () => Promise.resolve(),
      }),
    ),
  },
}));

const { detectProviders } = await import("../src/core/providers.js");
const { OrtSession } = await import("../src/core/session.js");

/**
 * Install a `navigator` that either exposes a WebGPU adapter or does not.
 *
 * @param adapter What `requestAdapter()` resolves to; `null` means the API is
 *   present but no adapter is available, which is the silent-fallback case.
 */
function withGpu(adapter: unknown): void {
  vi.stubGlobal("navigator", { gpu: { requestAdapter: () => Promise.resolve(adapter) } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("detectProviders", () => {
  it("keeps webgpu when an adapter exists", async () => {
    withGpu({});

    expect(await detectProviders(["webgpu", "wasm"])).toEqual(["webgpu", "wasm"]);
  });

  it("drops webgpu when the API is present but yields no adapter", async () => {
    withGpu(null);

    expect(await detectProviders(["webgpu", "wasm"])).toEqual(["wasm"]);
  });

  it("drops webgpu when there is no navigator at all", async () => {
    vi.stubGlobal("navigator", undefined);

    expect(await detectProviders(["webgpu", "wasm"])).toEqual(["wasm"]);
  });

  it("survives requestAdapter throwing", async () => {
    vi.stubGlobal("navigator", {
      gpu: {
        requestAdapter: () => {
          throw new Error("blocked");
        },
      },
    });

    expect(await detectProviders(["webgpu", "wasm"])).toEqual(["wasm"]);
  });

  it("keeps a provider it does not know how to test", async () => {
    withGpu(null);

    expect(await detectProviders(["something-new", "wasm"])).toEqual([
      "something-new",
      "wasm",
    ]);
  });
});

describe("OrtSession providers", () => {
  it("reports the narrowed list and keeps the request alongside it", async () => {
    withGpu(null);

    const session = await OrtSession.create(new ArrayBuffer(8), {
      providers: ["webgpu", "wasm"],
    });

    expect(session.requestedProviders).toEqual(["webgpu", "wasm"]);
    expect(session.providers).toEqual(["wasm"]);
  });

  it("warns when a provider the caller named cannot run", async () => {
    withGpu(null);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    await OrtSession.create(new ArrayBuffer(8), { providers: ["webgpu"] });

    expect(warn).toHaveBeenCalledOnce();
    expect(warn.mock.calls[0]?.[0]).toMatch(/webgpu/);
  });

  it("stays quiet when the default list falls back", async () => {
    withGpu(null);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const session = await OrtSession.create(new ArrayBuffer(8));

    expect(session.providers).toEqual(["wasm"]);
    expect(warn).not.toHaveBeenCalled();
  });

  it("keeps the request when nothing survives, rather than passing an empty list", async () => {
    vi.stubGlobal("navigator", undefined);
    vi.spyOn(console, "warn").mockImplementation(() => {});

    const session = await OrtSession.create(new ArrayBuffer(8), { providers: ["webgpu"] });

    expect(session.providers).toEqual(["webgpu"]);
  });
});

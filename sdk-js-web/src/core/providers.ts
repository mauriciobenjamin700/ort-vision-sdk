/**
 * Execution-provider defaults for ONNX Runtime Web sessions.
 *
 * Unlike Node ORT, the browser ORT cannot enumerate "available providers" at
 * runtime — `webgpu` either works or ORT silently falls back to the next
 * provider in the list. We just expose a sensible default order.
 */

/**
 * Default execution provider preference order for browser ORT.
 *
 * `webgpu` is tried first when available; ORT-Web falls back to `wasm`
 * automatically when WebGPU is not supported by the browser or device.
 */
export const DEFAULT_PROVIDERS: readonly string[] = ["webgpu", "wasm"];

/**
 * Resolve the execution providers to pass to `InferenceSession.create`.
 *
 * @param requested Explicit provider list in preference order; `undefined` returns the default.
 */
export function resolveProviders(requested?: readonly string[]): string[] {
  if (requested === undefined) {
    return [...DEFAULT_PROVIDERS];
  }
  if (requested.length === 0) {
    return [...DEFAULT_PROVIDERS];
  }
  return [...requested];
}

/**
 * Narrow a provider list to the ones this browser can actually offer.
 *
 * ORT-Web exposes no equivalent of Node's `session.getProviders()`, so there is
 * no way to ask which provider a session ended up on. What the browser *does*
 * answer is whether the underlying API exists at all — no `navigator.gpu`, or no
 * adapter behind it, means `webgpu` was never going to run — and that covers the
 * case that actually bites: a page asking for WebGPU on a device without it,
 * silently getting WASM, and being several times slower than intended with no
 * error to point at.
 *
 * This is best-effort by construction. A provider that survives here can still
 * fail inside ORT for a reason the browser does not surface (a missing shader
 * feature, an exhausted device), so a surviving entry means "not ruled out",
 * not "confirmed running". Anything this function does not know how to test is
 * kept rather than dropped: guessing a provider away would be worse than
 * admitting ignorance about it.
 *
 * @param requested Providers in preference order, already resolved.
 * @returns The subset that is not ruled out, in the same order.
 */
export async function detectProviders(
  requested: readonly string[],
): Promise<string[]> {
  const webgpu = await hasWebGpuAdapter();
  return requested.filter((provider) => {
    switch (provider) {
      case "webgpu":
        return webgpu;
      case "webnn":
        return typeof navigator !== "undefined" && "ml" in navigator;
      default:
        return true;
    }
  });
}

/**
 * Whether this environment exposes a WebGPU adapter.
 *
 * `navigator.gpu` existing is not enough — a browser can ship the API and still
 * hand back no adapter (a blocklisted driver, a headless context, a machine with
 * no suitable GPU), which is exactly the configuration where a page asks for
 * WebGPU and quietly runs on WASM.
 */
async function hasWebGpuAdapter(): Promise<boolean> {
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)
    ?.gpu;
  if (gpu === undefined) {
    return false;
  }
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

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

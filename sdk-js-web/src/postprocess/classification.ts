/**
 * Classification head postprocessing — softmax + top-k.
 */

/** Apply numerically-stable softmax to a 1-D vector of logits. */
export function softmax(
  logits: Float32Array | readonly number[],
): Float32Array {
  const n = logits.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] as number;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp((logits[i] as number) - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) {
    out[i] = (out[i] as number) / sum;
  }
  return out;
}

export interface TopKResult {
  readonly indices: Int32Array;
  readonly values: Float32Array;
}

/**
 * Return the top-k entries of a 1-D probability vector, sorted descending.
 *
 * @param k Number of entries to return; `null` returns all entries.
 */
export function topK(
  probabilities: Float32Array,
  k: number | null,
): TopKResult {
  const n = probabilities.length;
  const kEff = k === null ? n : Math.min(Math.max(0, k), n);

  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => (probabilities[b] as number) - (probabilities[a] as number));

  const indices = new Int32Array(kEff);
  const values = new Float32Array(kEff);
  for (let i = 0; i < kEff; i++) {
    const j = idx[i] as number;
    indices[i] = j;
    values[i] = probabilities[j] as number;
  }
  return { indices, values };
}

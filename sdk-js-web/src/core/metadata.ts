/**
 * Read the metadata an exporter baked into a `.onnx` file.
 *
 * `onnxruntime-web` exposes input/output metadata but **not** the model's
 * custom metadata map, which is where Ultralytics writes `names`, `task` and
 * `imgsz`. The Python SDK gets it for free from
 * `InferenceSession.get_modelmeta().custom_metadata_map`; in the browser the
 * only way to the same information is to read it out of the file, so this
 * module walks just enough of the ModelProto wire format to collect
 * `metadata_props`.
 *
 * It never throws and never allocates unbounded: a truncated, hostile or
 * simply unexpected file yields an empty map, and every caller treats that as
 * "the model says nothing", falling back to what it was given.
 */

/** Field number of `metadata_props` in `ModelProto` (repeated StringStringEntryProto). */
const MODEL_METADATA_PROPS_FIELD = 14;

/** Field numbers of `key` and `value` in `StringStringEntryProto`. */
const ENTRY_KEY_FIELD = 1;
const ENTRY_VALUE_FIELD = 2;

/** Protobuf wire types this reader understands. */
const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LENGTH_DELIMITED = 2;
const WIRE_FIXED32 = 5;

/**
 * Hard ceiling on a single length-delimited field, as a guard against a corrupt
 * length turning into a huge slice. Model metadata values are strings — a class
 * name map for thousands of classes still fits well inside this.
 */
const MAX_FIELD_BYTES = 1 << 20;

/** A cursor over a byte range, tracking its own position. */
interface Cursor {
  readonly bytes: Uint8Array;
  readonly end: number;
  pos: number;
}

/**
 * Read a base-128 varint.
 *
 * @param cursor Cursor to advance.
 * @returns The value, or `null` when the varint is truncated or overlong
 *   (beyond the 64-bit range protobuf allows).
 */
function readVarint(cursor: Cursor): number | null {
  let result = 0;
  let shift = 0;
  while (cursor.pos < cursor.end) {
    const byte = cursor.bytes[cursor.pos]!;
    cursor.pos += 1;
    result += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) return result;
    shift += 7;
    if (shift > 63) return null;
  }
  return null;
}

/**
 * Skip a field whose contents are not needed.
 *
 * @param cursor Cursor to advance past the field's payload.
 * @param wireType Wire type read from the field's tag.
 * @returns `true` when the field was skipped, `false` when the stream is
 *   unreadable from here (unknown wire type or truncated payload).
 */
function skipField(cursor: Cursor, wireType: number): boolean {
  switch (wireType) {
    case WIRE_VARINT:
      return readVarint(cursor) !== null;
    case WIRE_FIXED64:
      cursor.pos += 8;
      return cursor.pos <= cursor.end;
    case WIRE_LENGTH_DELIMITED: {
      const length = readVarint(cursor);
      if (length === null) return false;
      cursor.pos += length;
      return cursor.pos <= cursor.end;
    }
    case WIRE_FIXED32:
      cursor.pos += 4;
      return cursor.pos <= cursor.end;
    default:
      return false;
  }
}

/**
 * Read a length-delimited payload as a byte range.
 *
 * @param cursor Cursor to advance past the payload.
 * @returns Start and end offsets of the payload, or `null` when the length is
 *   truncated, overruns the buffer, or exceeds {@link MAX_FIELD_BYTES}.
 */
function readLengthDelimited(cursor: Cursor): { start: number; end: number } | null {
  const length = readVarint(cursor);
  if (length === null || length > MAX_FIELD_BYTES) return null;
  const start = cursor.pos;
  const end = start + length;
  if (end > cursor.end) return null;
  cursor.pos = end;
  return { start, end };
}

/**
 * Decode one `StringStringEntryProto` into a key/value pair.
 *
 * @param bytes The model buffer.
 * @param start Offset the entry's payload starts at.
 * @param end Offset the entry's payload ends at.
 * @returns The pair, or `null` when either half is missing or undecodable.
 */
function readEntry(
  bytes: Uint8Array,
  start: number,
  end: number,
): readonly [string, string] | null {
  const cursor: Cursor = { bytes, end, pos: start };
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let key: string | null = null;
  let value: string | null = null;

  while (cursor.pos < end) {
    const tag = readVarint(cursor);
    if (tag === null) return null;
    const field = tag >>> 3;
    const wireType = tag & 0x07;
    if (
      wireType === WIRE_LENGTH_DELIMITED &&
      (field === ENTRY_KEY_FIELD || field === ENTRY_VALUE_FIELD)
    ) {
      const range = readLengthDelimited(cursor);
      if (range === null) return null;
      const text = decoder.decode(bytes.subarray(range.start, range.end));
      if (field === ENTRY_KEY_FIELD) key = text;
      else value = text;
      continue;
    }
    if (!skipField(cursor, wireType)) return null;
  }

  if (key === null || value === null) return null;
  return [key, value];
}

/**
 * Collect a model's custom metadata map straight out of its bytes.
 *
 * @param model The `.onnx` file contents.
 * @returns Key/value metadata — `names`, `task`, `imgsz`, ... for an
 *   Ultralytics export — or an empty object when the file carries none or
 *   cannot be walked.
 */
export function readModelMetadata(
  model: Uint8Array | ArrayBufferLike,
): Readonly<Record<string, string>> {
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  const cursor: Cursor = { bytes, end: bytes.length, pos: 0 };
  const metadata: Record<string, string> = {};

  while (cursor.pos < cursor.end) {
    const tag = readVarint(cursor);
    if (tag === null) break;
    const field = tag >>> 3;
    const wireType = tag & 0x07;
    if (field === MODEL_METADATA_PROPS_FIELD && wireType === WIRE_LENGTH_DELIMITED) {
      const range = readLengthDelimited(cursor);
      if (range === null) break;
      const entry = readEntry(bytes, range.start, range.end);
      if (entry) metadata[entry[0]] = entry[1];
      continue;
    }
    if (!skipField(cursor, wireType)) break;
  }

  return metadata;
}

/**
 * Read the class names an export baked into the model metadata.
 *
 * Ultralytics writes `names` as the Python `repr` of a `dict[int, str]` — e.g.
 * `"{0: 'deworm', 1: 'not_deworm'}"`. The value is parsed structurally (never
 * evaluated), and anything unparseable, non-`dict`, or not keyed by contiguous
 * integers from zero is rejected whole rather than half-applied: a partial name
 * map would silently mislabel predictions.
 *
 * @param metadata A model's custom metadata map.
 * @returns Class names in class-id order, or `null` when the model carries no
 *   usable `names` entry.
 */
export function modelNames(
  metadata: Readonly<Record<string, string>> | undefined,
): readonly string[] | null {
  const raw = metadata?.names?.trim();
  if (!raw || !raw.startsWith("{") || !raw.endsWith("}")) return null;

  const body = raw.slice(1, -1).trim();
  if (!body) return null;

  const names = new Map<number, string>();
  const entryPattern = /(-?\d+)\s*:\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")/g;
  let consumed = 0;
  for (const match of body.matchAll(entryPattern)) {
    const id = Number(match[1]);
    const text = match[2] ?? match[3];
    if (!Number.isInteger(id) || text === undefined) return null;
    names.set(id, unescapeQuoted(text));
    consumed += match[0].length;
  }
  if (names.size === 0) return null;

  const separators = body.length - consumed;
  if (separators > names.size * 3) return null;

  const ordered: string[] = [];
  for (let id = 0; id < names.size; id += 1) {
    const name = names.get(id);
    if (name === undefined) return null;
    ordered.push(name);
  }
  return ordered;
}

/**
 * Resolve the backslash escapes Python's `repr` emits inside a quoted string.
 *
 * @param text The quoted string's contents, escapes intact.
 * @returns The same text with `\\`, `\'`, `\"`, `\n`, `\r` and `\t` resolved.
 */
function unescapeQuoted(text: string): string {
  return text.replace(/\\(.)/g, (_, char: string) => {
    if (char === "n") return "\n";
    if (char === "r") return "\r";
    if (char === "t") return "\t";
    return char;
  });
}

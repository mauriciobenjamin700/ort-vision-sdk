/**
 * Static server for the browser benchmark, rooted at the repository.
 *
 * Two headers are the point of it. `Cross-Origin-Opener-Policy: same-origin`
 * plus `Cross-Origin-Embedder-Policy: require-corp` make the page
 * cross-origin isolated, which is what lets ONNX Runtime's WASM backend use
 * threads — without them it runs on one, 3.4x slower on the YOLO11n-seg this
 * bench was built around. `Cache-Control: no-store` keeps a re-run from
 * reading the model out of the HTTP cache and reporting a creation time no
 * first visit sees.
 *
 * An optional bandwidth cap delays every `.onnx` and `.wasm` response by its
 * size, to make download overlap measurable on localhost:
 *
 *   node bench/browser/server.mjs                 # http://localhost:8765/bench/browser/run.html
 *   node bench/browser/server.mjs --port 9000
 *   node bench/browser/server.mjs --bandwidth 5   # MB/s
 *
 * Binds 127.0.0.1. Reach it from an Android phone with
 * `adb reverse tcp:8765 tcp:8765` and open `http://localhost:8765/...` there:
 * `localhost` is a secure context, which WebGPU and Cache Storage require.
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".json": "application/json",
  ".map": "application/json",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
};

/**
 * Read `--name value` from the command line.
 *
 * @param {string} name Flag name without dashes.
 * @param {string} fallback Value when the flag is absent.
 * @returns {string} The value.
 */
function flag(name, fallback) {
  const at = process.argv.indexOf(`--${name}`);
  return at > 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1] : fallback;
}

const port = Number(flag("port", "8765"));
const bandwidthMBps = Number(flag("bandwidth", "0"));

createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url ?? "/", "http://x").pathname);
  const file = join(ROOT, normalize(pathname).replace(/^(\.\.[/\\])+/, ""));
  if (!file.startsWith(ROOT)) {
    response.writeHead(403).end();
    return;
  }
  try {
    if (!(await stat(file)).isFile()) throw new Error("not a file");
    const body = await readFile(file);
    if (bandwidthMBps > 0 && /\.(onnx|wasm)$/.test(file)) {
      await new Promise((done) => setTimeout(done, body.length / (bandwidthMBps * 1000)));
    }
    response.writeHead(200, {
      "Content-Type": TYPES[extname(file)] ?? "application/octet-stream",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cache-Control": "no-store",
    });
    response.end(body);
  } catch {
    response.writeHead(404).end("not found");
  }
}).listen(port, "127.0.0.1", () => {
  const cap = bandwidthMBps > 0 ? `, .onnx/.wasm capped at ${bandwidthMBps} MB/s` : "";
  console.log(`http://localhost:${port}/bench/browser/run.html${cap}`);
});

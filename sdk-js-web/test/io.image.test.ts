import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * When decoding reads the frame back, and when it may wait.
 *
 * A full-resolution `getImageData` is most of the `load` stage, so the decoder
 * defers it for sources that cannot contain a translucent pixel — an
 * alpha-less `VideoFrame`, a live `MediaStream` video, a JPEG — and reads it
 * right away for everything else, because the canvas may only stand in for
 * the RGB image when every pixel is opaque. These tests count the read-backs.
 *
 * Node has no canvas, video or `VideoFrame`; minimal stand-ins are installed.
 */

/** How many times `getImageData` ran across all stand-in canvases. */
let readBacks = 0;

class FakeContext {
  constructor(private readonly canvas: FakeCanvas) {}
  drawImage(): void {}
  getImageData(_x: number, _y: number, width: number, height: number) {
    readBacks++;
    const data = new Uint8ClampedArray(width * height * 4).fill(255);
    return { data, width, height };
  }
}

class FakeCanvas {
  constructor(
    public width: number,
    public height: number,
  ) {}
  getContext() {
    return new FakeContext(this);
  }
}

class FakeVideoFrame {
  constructor(
    public readonly format: string | null,
    public readonly displayWidth = 4,
    public readonly displayHeight = 2,
  ) {}
}

class FakeMediaStream {}

class FakeVideo {
  readyState = 4;
  videoWidth = 4;
  videoHeight = 2;
  constructor(public srcObject: unknown) {}
}

beforeEach(() => {
  readBacks = 0;
  vi.stubGlobal("OffscreenCanvas", FakeCanvas);
  vi.stubGlobal("VideoFrame", FakeVideoFrame);
  vi.stubGlobal("MediaStream", FakeMediaStream);
  vi.stubGlobal("HTMLVideoElement", FakeVideo);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const { loadImageSource } = await import("../src/io/image.js");
const { RGBImage } = await import("../src/types.js");

describe("loadImageSource read-back", () => {
  it("defers it for an alpha-less VideoFrame and hands over the canvas", async () => {
    const loaded = await loadImageSource(new FakeVideoFrame("I420") as never);

    expect(readBacks).toBe(0);
    expect(loaded.canvas).not.toBeNull();
    expect(loaded.owned).toBe(true);
    expect(loaded.image.data.length).toBe(4 * 2 * 3);
    expect(readBacks).toBe(1);
    expect(loaded.image.data).toBe(loaded.image.data);
    expect(readBacks).toBe(1);
  });

  it("reads right away for a VideoFrame that may carry alpha", async () => {
    await loadImageSource(new FakeVideoFrame("RGBA") as never);

    expect(readBacks).toBe(1);
  });

  it("reads right away for a VideoFrame of unknown format", async () => {
    await loadImageSource(new FakeVideoFrame(null) as never);

    expect(readBacks).toBe(1);
  });

  it("defers it for a live MediaStream video", async () => {
    await loadImageSource(new FakeVideo(new FakeMediaStream()) as never);

    expect(readBacks).toBe(0);
  });

  it("reads right away for a video playing a file", async () => {
    await loadImageSource(new FakeVideo(null) as never);

    expect(readBacks).toBe(1);
  });
});

describe("RGBImage.deferred", () => {
  it("materializes once, on first read", () => {
    const materialize = vi.fn(() => new Uint8Array(2 * 2 * 3).fill(9));
    const image = RGBImage.deferred(2, 2, materialize);

    expect(materialize).not.toHaveBeenCalled();
    expect(image).toBeInstanceOf(RGBImage);
    expect([image.width, image.height]).toEqual([2, 2]);
    expect(image.data[0]).toBe(9);
    expect(image.data).toBe(image.data);
    expect(materialize).toHaveBeenCalledTimes(1);
  });

  it("rejects a materialized buffer of the wrong length", () => {
    const image = RGBImage.deferred(2, 2, () => new Uint8Array(5));

    expect(() => image.data).toThrow("does not match");
  });
});
